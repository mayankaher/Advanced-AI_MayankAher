"""
Evaluation: Probability Calibration & Reliability Diagrams.
Implements:
  - Reliability diagram binning (confidence vs actual positive frequency)
  - Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
  - Benchmark comparison: Learned BN vs Expert Model vs Rule-Based Baseline on held-out dataset

Syllabus Topic: Calibration, Reliability Diagrams, Proper Scoring Rules.
"""
from __future__ import annotations
from typing import Dict, List, Any, Sequence, Tuple
import numpy as np
import pandas as pd
from backend.app.pgm.network import BayesNet
from backend.app.pgm.variable_elimination import VariableElimination
from backend.app.eval.metrics import compute_auroc, compute_brier_score, compute_safety_metrics
from backend.app.decision.triage import TriageRecommender


class CalibrationEvaluator:
    def __init__(self, net: BayesNet):
        self.net = net
        self.ve = VariableElimination(net)
        self.triage = TriageRecommender(net)

    def compute_reliability_curve(
        self,
        y_true: Sequence[int],
        y_prob: Sequence[float],
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Partitions predictions into n_bins between [0, 1].
        Computes mean confidence, empirical accuracy (fraction of true positives), and ECE.
        """
        y_true_arr = np.asarray(y_true, dtype=np.int32)
        y_prob_arr = np.asarray(y_prob, dtype=np.float64)
        N = len(y_prob_arr)

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bins_data = []
        ece = 0.0
        mce = 0.0

        for i in range(n_bins):
            low = bin_edges[i]
            high = bin_edges[i + 1]
            if i == n_bins - 1:
                mask = (y_prob_arr >= low) & (y_prob_arr <= high)
            else:
                mask = (y_prob_arr >= low) & (y_prob_arr < high)

            count = int(np.sum(mask))
            if count > 0:
                mean_conf = float(np.mean(y_prob_arr[mask]))
                mean_acc = float(np.mean(y_true_arr[mask]))
                gap = abs(mean_acc - mean_conf)
                ece += (count / N) * gap
                mce = max(mce, gap)
            else:
                mean_conf = float((low + high) / 2.0)
                mean_acc = 0.0
                gap = 0.0

            bins_data.append({
                "bin_index": i,
                "bin_range": f"{low:.1f}-{high:.1f}",
                "mean_predicted_prob": round(mean_conf, 4),
                "actual_positive_fraction": round(mean_acc, 4),
                "sample_count": count,
                "calibration_gap": round(gap, 4)
            })

        return {
            "bins": bins_data,
            "ece": round(float(ece), 4),
            "mce": round(float(mce), 4)
        }

    def evaluate_model_on_heldout(
        self,
        df: pd.DataFrame,
        expert_net: Optional[BayesNet] = None,
        max_patients: int = 500
    ) -> Dict[str, Any]:
        """
        Evaluates the ground truth/learned model, the perturbed expert model,
        and the rule-based baseline on held-out patients.
        """
        sub_df = df.head(max_patients)
        records = sub_df.to_dict(orient="records")

        # Ground truth binary targets for Critical
        y_crit_true = [1 if r["Severity"] == "Critical" else 0 for r in records]
        severities = [r["Severity"] for r in records]

        # 1. Main Bayesian Model
        main_probs = []
        main_levels = []
        for r in records:
            ev = {k: v for k, v in r.items() if k not in ("Condition", "Severity") and k in self.net.nodes}
            post_f, _ = self.ve.query(["Severity"], ev)
            dist = post_f.get_distribution({"Severity": self.net.states["Severity"]})
            p_crit = dist["Critical"]
            main_probs.append(p_crit)
            rec = self.triage.recommend(ev, p_sev=dist)
            main_levels.append(rec["best_level"])

        main_auroc = compute_auroc(y_crit_true, main_probs)
        main_brier = compute_brier_score(y_crit_true, main_probs)
        main_calib = self.compute_reliability_curve(y_crit_true, main_probs)
        main_safety = compute_safety_metrics(severities, main_levels)

        # 2. Perturbed Expert Model (if provided)
        expert_res = None
        if expert_net:
            ve_exp = VariableElimination(expert_net)
            triage_exp = TriageRecommender(expert_net)
            exp_probs = []
            exp_levels = []
            for r in records:
                ev = {k: v for k, v in r.items() if k not in ("Condition", "Severity") and k in expert_net.nodes}
                post_f, _ = ve_exp.query(["Severity"], ev)
                dist = post_f.get_distribution({"Severity": expert_net.states["Severity"]})
                exp_probs.append(dist["Critical"])
                rec = triage_exp.recommend(ev, p_sev=dist)
                exp_levels.append(rec["best_level"])

            exp_auroc = compute_auroc(y_crit_true, exp_probs)
            exp_brier = compute_brier_score(y_crit_true, exp_probs)
            exp_calib = self.compute_reliability_curve(y_crit_true, exp_probs)
            exp_safety = compute_safety_metrics(severities, exp_levels)

            expert_res = {
                "auroc": round(float(exp_auroc), 4),
                "brier_score": round(float(exp_brier), 4),
                "ece": exp_calib["ece"],
                "reliability_bins": exp_calib["bins"],
                "safety": exp_safety
            }

        # 3. Rule-Based ESI Baseline
        rule_scores = []
        rule_levels = []
        for r in records:
            ev = {k: v for k, v in r.items() if k not in ("Condition", "Severity")}
            # Rule heuristic
            sev_vital = any(ev.get(v) == "Severe" for v in ["HeartRate", "SystolicBP", "SpO2", "RespRate", "Temperature"])
            unresp = (ev.get("Consciousness") == "Unresponsive")
            abn_vitals = sum(1 for v in ["HeartRate", "SystolicBP", "SpO2", "RespRate", "Temperature"] if ev.get(v) == "Abnormal")

            if sev_vital or unresp:
                score = 0.95
                lvl = "P1_Immediate"
            elif abn_vitals >= 2 or ev.get("Consciousness") == "Altered":
                score = 0.50
                lvl = "P2_Urgent"
            elif abn_vitals >= 1:
                score = 0.15
                lvl = "P3_Standard"
            else:
                score = 0.01
                lvl = "P4_NonUrgent"

            rule_scores.append(score)
            rule_levels.append(lvl)

        rule_auroc = compute_auroc(y_crit_true, rule_scores)
        rule_brier = compute_brier_score(y_crit_true, rule_scores)
        rule_calib = self.compute_reliability_curve(y_crit_true, rule_scores)
        rule_safety = compute_safety_metrics(severities, rule_levels)

        return {
            "evaluated_sample_count": len(records),
            "ground_truth_critical_prevalence": round(float(np.mean(y_crit_true)), 4),
            "bayesian_model": {
                "auroc": round(float(main_auroc), 4),
                "brier_score": round(float(main_brier), 4),
                "ece": main_calib["ece"],
                "reliability_bins": main_calib["bins"],
                "safety": main_safety
            },
            "expert_model": expert_res,
            "rule_based_baseline": {
                "auroc": round(float(rule_auroc), 4),
                "brier_score": round(float(rule_brier), 4),
                "ece": rule_calib["ece"],
                "reliability_bins": rule_calib["bins"],
                "safety": rule_safety
            }
        }
