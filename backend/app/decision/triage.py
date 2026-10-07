"""
Decision Theory: Single-Patient Triage Recommendation & Evidence Attribution.
Implements:
  - Expected Utility computation: EU(level) = sum_s P(s | e) * U[level, s]
  - argmax recommendation over 4 triage levels
  - Evidence contribution analysis: Leave-One-Out impact on P(Critical)

Syllabus Topic: Decision Problems, Expected Utility Maximization, Interpretability.
"""
from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from backend.app.pgm.network import BayesNet
from backend.app.pgm.variable_elimination import VariableElimination
from backend.app.decision.utility import DecisionConfig


class TriageRecommender:
    def __init__(self, net: BayesNet, config: Optional[DecisionConfig] = None):
        self.net = net
        self.config = config or DecisionConfig.from_spec(net.decision_spec)
        self.ve = VariableElimination(net)

    def compute_expected_utility(self, p_sev: np.ndarray | Dict[str, float]) -> np.ndarray:
        """
        EU = U_eff @ p_sev
        where U_eff is (4, 3) and p_sev is length 3 [Low, Moderate, Critical].
        """
        if isinstance(p_sev, dict):
            vec = np.array([p_sev["Low"], p_sev["Moderate"], p_sev["Critical"]], dtype=np.float64)
        else:
            vec = np.asarray(p_sev, dtype=np.float64)
        U_eff = self.config.get_effective_utility()
        return U_eff @ vec

    def recommend(
        self,
        evidence: Dict[str, str],
        p_sev: Optional[np.ndarray | Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates triage options for given evidence and posterior.
        If p_sev is not provided, computes exact posterior using Variable Elimination.
        """
        if p_sev is None:
            post_sev_factor, _ = self.ve.query(["Severity"], evidence)
            p_sev_dict = post_sev_factor.get_distribution({"Severity": self.net.states["Severity"]})
            p_sev_vec = np.array([p_sev_dict["Low"], p_sev_dict["Moderate"], p_sev_dict["Critical"]], dtype=np.float64)
        elif isinstance(p_sev, dict):
            p_sev_dict = p_sev
            p_sev_vec = np.array([p_sev["Low"], p_sev["Moderate"], p_sev["Critical"]], dtype=np.float64)
        else:
            p_sev_vec = np.asarray(p_sev, dtype=np.float64)
            p_sev_dict = {
                "Low": float(p_sev_vec[0]),
                "Moderate": float(p_sev_vec[1]),
                "Critical": float(p_sev_vec[2])
            }

        eu_values = self.compute_expected_utility(p_sev_vec)
        best_idx = int(np.argmax(eu_values))
        best_level = self.config.levels[best_idx]

        eu_by_level = {
            self.config.levels[i]: round(float(eu_values[i]), 3)
            for i in range(len(self.config.levels))
        }

        return {
            "p_severity": {k: round(float(v), 4) for k, v in p_sev_dict.items()},
            "eu_by_level": eu_by_level,
            "best_level": best_level,
            "best_eu": round(float(eu_values[best_idx]), 3)
        }

    def compute_evidence_contributions(
        self,
        evidence: Dict[str, str],
        baseline_p_crit: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        r"""
        Leave-One-Out (LOO) evidence attribution for P(Critical):
        For each observed variable X=x in evidence:
          Evaluate posterior P(Critical | evidence \ {X})
          Impact = P(Critical | evidence) - P(Critical | evidence \ {X})
        Positive impact means observing X shifted belief towards Critical.
        Negative impact means observing X lowered belief in Critical.
        """
        if not evidence:
            return []

        if baseline_p_crit is None:
            post_full, _ = self.ve.query(["Severity"], evidence)
            dist_full = post_full.get_distribution({"Severity": self.net.states["Severity"]})
            baseline_p_crit = dist_full["Critical"]

        contributions = []
        for var, val in evidence.items():
            loo_ev = {k: v for k, v in evidence.items() if k != var}
            post_loo, _ = self.ve.query(["Severity"], loo_ev)
            dist_loo = post_loo.get_distribution({"Severity": self.net.states["Severity"]})
            loo_p_crit = dist_loo["Critical"]

            diff = baseline_p_crit - loo_p_crit
            # Log odds shift
            eps = 1e-6
            odds_full = max(eps, min(1 - eps, baseline_p_crit)) / (1 - max(eps, min(1 - eps, baseline_p_crit)))
            odds_loo = max(eps, min(1 - eps, loo_p_crit)) / (1 - max(eps, min(1 - eps, loo_p_crit)))
            log_odds_ratio = np.log(odds_full / odds_loo)

            contributions.append({
                "variable": var,
                "value": val,
                "impact_on_p_critical": round(float(diff), 4),
                "log_odds_ratio": round(float(log_odds_ratio), 4),
                "loo_p_critical": round(float(loo_p_crit), 4)
            })

        # Sort by absolute impact descending
        contributions.sort(key=lambda x: abs(x["impact_on_p_critical"]), reverse=True)
        return contributions
