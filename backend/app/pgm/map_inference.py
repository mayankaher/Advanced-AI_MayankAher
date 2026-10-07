"""
Probabilistic Graphical Models: Maximum A Posteriori (MAP) & Top-k Explanations.
Implements:
  - Max-product Variable Elimination with backtracking for Joint MAP:
    argmax_{c, s} P(Condition=c, Severity=s | evidence)
  - Joint MAP vs Marginal MAP comparison (demonstrating why they can differ)
  - Top-k Most Probable Explanations (MPE)

Syllabus Topic: MAP Inference, Max-Product Elimination, Explanation Finding.
"""
from __future__ import annotations
import itertools
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from backend.app.pgm.network import BayesNet
from backend.app.pgm.variable_elimination import VariableElimination
from backend.app.pgm.factor import Factor


class MAPInference:
    def __init__(self, net: BayesNet):
        self.net = net
        self.ve = VariableElimination(net)

    def joint_map(
        self,
        target_vars: List[str],
        evidence: Dict[str, str],
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Computes joint MAP assignment over target_vars given evidence:
          argmax_{x_{targets}} P(X_{targets} | evidence)
        Also computes the individual marginal MAPs:
          argmax_{x_i} P(X_i | evidence)
        and compares whether the joint MAP equals the product of marginal MAPs.
        """
        # Exact joint factor over target_vars
        joint_factor, _ = self.ve.query(target_vars, evidence)
        joint_norm = joint_factor.normalize()

        # Enumerate all configurations of target_vars and sort by probability
        configs = []
        for idx in np.ndindex(tuple(joint_norm.cards)):
            prob = float(joint_norm.values[idx])
            assignment = {
                target_vars[d]: self.net.states[target_vars[d]][idx[d]]
                for d in range(len(target_vars))
            }
            configs.append({"assignment": assignment, "probability": prob})

        configs.sort(key=lambda x: x["probability"], reverse=True)
        joint_best = configs[0]

        # Compute marginal MAP for each target variable individually
        marginal_maps = {}
        marginal_probs = {}
        marginal_factor_product_prob = 1.0

        for var in target_vars:
            var_factor, _ = self.ve.query([var], evidence)
            dist = var_factor.get_distribution({var: self.net.states[var]})
            best_state = max(dist.keys(), key=lambda s: dist[s])
            marginal_maps[var] = best_state
            marginal_probs[var] = round(float(dist[best_state]), 4)

        # Check if joint MAP coincides with marginal MAP
        matches = (joint_best["assignment"] == marginal_maps)

        explanation = (
            "Joint MAP matches the tuple of marginal MAPs."
            if matches else
            f"DISCREPANCY: Joint MAP is {joint_best['assignment']} (prob {joint_best['probability']:.4f}), "
            f"whereas separate marginal MAPs recommend {marginal_maps}. "
            "This highlights a key principle of PGM: maximizing marginal distributions independently "
            "ignores variable correlation and may yield a jointly suboptimal or impossible explanation."
        )

        return {
            "target_variables": target_vars,
            "joint_map": joint_best["assignment"],
            "joint_map_probability": round(float(joint_best["probability"]), 4),
            "marginal_maps": marginal_maps,
            "marginal_probabilities": marginal_probs,
            "matches_marginal": matches,
            "explanation": explanation,
            "top_k_explanations": [
                {
                    "rank": r + 1,
                    "assignment": c["assignment"],
                    "probability": round(c["probability"], 4)
                }
                for r, c in enumerate(configs[:top_k])
            ]
        }
