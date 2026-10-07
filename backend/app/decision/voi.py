"""
Decision Theory: Value of Information (VOI) & Preposterior Analysis.
Implements:
  - Value of Information: VOI(T) = sum_o P(o | e) * max_a EU(a | e, T=o) - max_a EU(a | e)
  - Net VOI: VOI(T) - Cost(T)
  - Expected Value of Perfect Information (EVPI) about Severity
  - Preposterior outcome trees with transparent intermediate distributions

Syllabus Topic: Value of Information, Preposterior Analysis, EVPI.
"""
from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from backend.app.pgm.network import BayesNet
from backend.app.pgm.variable_elimination import VariableElimination
from backend.app.decision.utility import DecisionConfig
from backend.app.decision.triage import TriageRecommender


class VOIAnalyzer:
    def __init__(self, net: BayesNet, config: Optional[DecisionConfig] = None):
        self.net = net
        self.config = config or DecisionConfig.from_spec(net.decision_spec)
        self.ve = VariableElimination(net)
        self.triage = TriageRecommender(net, self.config)

    def analyze(self, evidence: Dict[str, str]) -> Dict[str, Any]:
        """
        Computes VOI and Net VOI for all unobserved test variables,
        along with preposterior outcomes and EVPI.
        """
        # Current baseline EU without any additional tests
        base_rec = self.triage.recommend(evidence)
        base_max_eu = base_rec["best_eu"]
        base_best_level = base_rec["best_level"]
        base_p_sev = base_rec["p_severity"]
        base_sev_vec = np.array([base_p_sev["Low"], base_p_sev["Moderate"], base_p_sev["Critical"]], dtype=np.float64)

        # 1. Expected Value of Perfect Information (EVPI) about Severity
        # EVPI = sum_s P(s | e) * max_a U[a, s] - max_a EU(a | e)
        U = self.config.get_effective_utility()
        max_u_per_severity = np.max(U, axis=0)  # max over levels for each severity state
        expected_u_perfect = float(np.sum(base_sev_vec * max_u_per_severity))
        evpi = max(0.0, expected_u_perfect - base_max_eu)

        test_nodes = ["ECG_Abnormal", "Troponin_High", "Lactate_High", "ChestXray_Abnormal"]
        test_results = []

        best_test_name = None
        highest_net_voi = float("-inf")

        for test in test_nodes:
            if test in evidence:
                # Already observed
                continue

            test_cost = self.config.test_costs.get(test, 1.0)
            states = self.net.states[test]  # ['No', 'Yes']

            # Compute marginal P(T | e)
            p_test_factor, _ = self.ve.query([test], evidence)
            p_test_dist = p_test_factor.get_distribution({test: states})

            expected_posterior_eu = 0.0
            outcomes = []

            for outcome_val in states:
                p_outcome = p_test_dist.get(outcome_val, 0.0)
                aug_ev = dict(evidence)
                aug_ev[test] = outcome_val

                # Posterior given outcome
                aug_rec = self.triage.recommend(aug_ev)
                aug_max_eu = aug_rec["best_eu"]
                aug_best_level = aug_rec["best_level"]
                aug_p_sev = aug_rec["p_severity"]

                expected_posterior_eu += p_outcome * aug_max_eu

                outcomes.append({
                    "outcome": outcome_val,
                    "p_outcome": round(float(p_outcome), 4),
                    "best_level": aug_best_level,
                    "best_eu": round(float(aug_max_eu), 3),
                    "p_severity": aug_p_sev
                })

            voi_gross = max(0.0, expected_posterior_eu - base_max_eu)
            net_voi = voi_gross - test_cost

            if net_voi > highest_net_voi:
                highest_net_voi = net_voi

            test_results.append({
                "test": test,
                "cost": test_cost,
                "gross_voi": round(float(voi_gross), 4),
                "net_voi": round(float(net_voi), 4),
                "outcomes": outcomes,
                "is_positive": net_voi > 0
            })

        # Recommendation
        test_results.sort(key=lambda x: x["net_voi"], reverse=True)
        recommended_test = None
        recommendation_reason = "No additional test has positive net VOI: proceed with current triage recommendation."

        if test_results and test_results[0]["net_voi"] > 0:
            recommended_test = test_results[0]["test"]
            recommendation_reason = (
                f"Perform '{recommended_test}' (Net VOI = +{test_results[0]['net_voi']:.3f}). "
                f"The expected information gain exceeds testing cost ({test_results[0]['cost']} utility units) "
                f"and may change the optimal triage decision."
            )

        return {
            "baseline_best_level": base_best_level,
            "baseline_best_eu": round(float(base_max_eu), 3),
            "evpi": round(float(evpi), 4),
            "recommended_test": recommended_test,
            "recommendation_reason": recommendation_reason,
            "test_analyses": test_results
        }
