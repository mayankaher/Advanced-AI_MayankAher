"""
Multi-Patient Prioritization: Decision-Theoretic c*mu Queue Index.
Implements:
  - Expected harm rate: E[c | e] = sum_s P(s | e) * harm_rate[s]
  - Expected service time: E[1/mu | e] = sum_c P(c | e) * service_min[c]
  - Priority index: priority_i = E[c | e_i] / E[1/mu | e_i] (c*mu rule)
  - Side-by-side comparison with P(Critical)-only and FIFO orderings
  - Contextual clinical explanations for why rankings differ

Syllabus Topic: Multi-agent Decision Making, Queueing Control, c*mu Rule, Dynamic Scheduling.
"""
from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from backend.app.pgm.network import BayesNet
from backend.app.pgm.variable_elimination import VariableElimination
from backend.app.decision.utility import DecisionConfig
from backend.app.decision.triage import TriageRecommender


class PatientQueuePrioritizer:
    def __init__(self, net: BayesNet, config: Optional[DecisionConfig] = None):
        self.net = net
        self.config = config or DecisionConfig.from_spec(net.decision_spec)
        self.ve = VariableElimination(net)
        self.triage = TriageRecommender(net, self.config)

    def compute_patient_metrics(
        self,
        patient_id: str,
        evidence: Dict[str, str],
        arrival_time: float = 0.0,
        wait_minutes: float = 0.0
    ) -> Dict[str, Any]:
        """
        Computes posterior (Severity, Condition), expected harm rate, expected service time,
        and c*mu priority index for a single patient.
        """
        # Marginals for Severity and Condition
        post_sev, _ = self.ve.query(["Severity"], evidence)
        sev_dist = post_sev.get_distribution({"Severity": self.net.states["Severity"]})
        sev_vec = np.array([sev_dist["Low"], sev_dist["Moderate"], sev_dist["Critical"]], dtype=np.float64)

        post_cond, _ = self.ve.query(["Condition"], evidence)
        cond_dist = post_cond.get_distribution({"Condition": self.net.states["Condition"]})
        cond_vec = np.array([
            cond_dist[c] for c in self.net.states["Condition"]
        ], dtype=np.float64)

        # Expected harm rate: E[c]
        expected_harm = float(np.sum(sev_vec * self.config.harm_rate))

        # Expected service time: E[1/mu]
        cond_service = np.array([
            self.config.service_minutes.get(c, 20.0) for c in self.net.states["Condition"]
        ], dtype=np.float64)
        expected_service = float(np.sum(cond_vec * cond_service))

        # c*mu priority index
        priority_index = expected_harm / expected_service if expected_service > 0 else 0.0

        # Triage recommendation
        rec = self.triage.recommend(evidence, p_sev=sev_dist)

        return {
            "patient_id": patient_id,
            "evidence": evidence,
            "arrival_time": arrival_time,
            "wait_minutes": wait_minutes,
            "p_severity": {k: round(float(v), 4) for k, v in sev_dist.items()},
            "p_condition": {k: round(float(v), 4) for k, v in cond_dist.items()},
            "p_critical": round(float(sev_dist["Critical"]), 4),
            "expected_harm_rate": round(expected_harm, 5),
            "expected_service_minutes": round(expected_service, 2),
            "priority_index": round(priority_index, 5),
            "best_level": rec["best_level"],
            "best_eu": rec["best_eu"]
        }

    def rank_patients(
        self,
        patients: List[Dict[str, Any]],
        clinician_count: int = 3
    ) -> Dict[str, Any]:
        """
        Ranks multiple patients using three policies:
          1. Decision-theoretic c*mu index (optimal for delay-cost vs service time)
          2. P(Critical)-only priority (standard clinical triage)
          3. FIFO (arrival order)
        Returns detailed comparisons and explanations of discrepancies.
        """
        evaluated = []
        for i, p in enumerate(patients):
            pid = p.get("patient_id", f"P{i+1}")
            ev = p.get("evidence", {})
            arr = p.get("arrival_time", float(i))
            wait = p.get("wait_minutes", 0.0)
            metrics = self.compute_patient_metrics(pid, ev, arr, wait)
            evaluated.append(metrics)

        # 1. c*mu ranking (descending priority_index)
        cmu_ranked = sorted(evaluated, key=lambda x: x["priority_index"], reverse=True)
        for rank, p in enumerate(cmu_ranked, 1):
            p["cmu_rank"] = rank

        # 2. P(Critical) ranking (descending p_critical, then expected harm)
        pcrit_ranked = sorted(evaluated, key=lambda x: (x["p_critical"], x["priority_index"]), reverse=True)
        for rank, p in enumerate(pcrit_ranked, 1):
            p["pcrit_rank"] = rank

        # 3. FIFO ranking (ascending arrival_time)
        fifo_ranked = sorted(evaluated, key=lambda x: x["arrival_time"])
        for rank, p in enumerate(fifo_ranked, 1):
            p["fifo_rank"] = rank

        # Generate comparative explanations
        explanations = []
        for i, p in enumerate(cmu_ranked):
            pid = p["patient_id"]
            pcrit_r = p["pcrit_rank"]
            cmu_r = p["cmu_rank"]

            if cmu_r < pcrit_r:
                # Ranked higher in c*mu than in P(Critical)
                comp_patient = next((cp for cp in cmu_ranked if cp["pcrit_rank"] == cmu_r), None)
                comp_str = f" Patient {comp_patient['patient_id']}" if comp_patient else "others"
                exp = (
                    f"Patient {pid} is promoted from #{pcrit_r} to #{cmu_r} by c*mu rule because their expected "
                    f"service time ({p['expected_service_minutes']}m) is substantially shorter than{comp_str}, "
                    f"clearing clinician capacity faster to minimize aggregate ER delay harm."
                )
            elif cmu_r > pcrit_r:
                # Ranked lower in c*mu than in P(Critical)
                exp = (
                    f"Patient {pid} drops from #{pcrit_r} to #{cmu_r} under c*mu rule despite higher P(Critical) "
                    f"({p['p_critical']:.3f}) due to longer anticipated service duration ({p['expected_service_minutes']}m)."
                )
            else:
                exp = (
                    f"Patient {pid} rank #{cmu_r} is consistent across both c*mu and P(Critical) criteria."
                )
            explanations.append({"patient_id": pid, "explanation": exp})

        return {
            "clinician_count": clinician_count,
            "cmu_ranked": cmu_ranked,
            "pcrit_ranked": pcrit_ranked,
            "fifo_ranked": fifo_ranked,
            "explanations": explanations
        }
