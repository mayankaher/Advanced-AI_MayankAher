"""
Decision Theory: Sensitivity Analysis & Decision Thresholds.
Implements:
  - 1D sweeping of P(Critical) from 0 to 1 to find optimal triage regions
  - Analytical pairwise break-even probabilities
  - Sensitivity with respect to individual utility matrix entries

Syllabus Topic: Decision Theory, Sensitivity Analysis, Break-Even Analysis.
"""
from __future__ import annotations
from typing import Dict, List, Any, Tuple
import numpy as np
from backend.app.decision.utility import DecisionConfig


class SensitivityAnalyzer:
    def __init__(self, config: DecisionConfig):
        self.config = config

    def sweep_p_critical(
        self,
        base_p_low: float = 0.5,
        base_p_mod: float = 0.5,
        steps: int = 101
    ) -> Dict[str, Any]:
        """
        Sweeps P(Critical) in [0, 1]. For each p_crit, allocates remaining probability (1 - p_crit)
        between Low and Moderate in proportion to base_p_low : base_p_mod.
        Computes EU for each level and identifies the optimal level.
        """
        p_crit_grid = np.linspace(0.0, 1.0, steps)
        U = self.config.get_effective_utility()
        levels = self.config.levels

        ratio_sum = base_p_low + base_p_mod
        if ratio_sum <= 1e-6:
            r_low, r_mod = 0.5, 0.5
        else:
            r_low = base_p_low / ratio_sum
            r_mod = base_p_mod / ratio_sum

        curves: Dict[str, List[float]] = {lvl: [] for lvl in levels}
        optimal_level_at_step: List[str] = []

        for p_c in p_crit_grid:
            rem = max(0.0, 1.0 - p_c)
            p_vec = np.array([rem * r_low, rem * r_mod, p_c], dtype=np.float64)
            eu_vec = U @ p_vec
            best_idx = int(np.argmax(eu_vec))
            optimal_level_at_step.append(levels[best_idx])
            for i, lvl in enumerate(levels):
                curves[lvl].append(round(float(eu_vec[i]), 3))

        # Detect decision boundaries (intervals where level is optimal)
        regions: List[Dict[str, Any]] = []
        cur_lvl = optimal_level_at_step[0]
        start_p = p_crit_grid[0]

        for i in range(1, len(optimal_level_at_step)):
            if optimal_level_at_step[i] != cur_lvl:
                regions.append({
                    "level": cur_lvl,
                    "p_crit_min": round(float(start_p), 3),
                    "p_crit_max": round(float(p_crit_grid[i]), 3)
                })
                cur_lvl = optimal_level_at_step[i]
                start_p = p_crit_grid[i]
        regions.append({
            "level": cur_lvl,
            "p_crit_min": round(float(start_p), 3),
            "p_crit_max": 1.0
        })

        return {
            "p_crit_grid": [round(float(x), 3) for x in p_crit_grid],
            "curves": curves,
            "regions": regions
        }

    def compute_pairwise_breakeven(self) -> List[Dict[str, Any]]:
        """
        Computes the analytical break-even probability P*(Critical) between adjacent triage levels,
        assuming a standard 2-state or fixed Low/Moderate ratio.
        For level i and level j:
          EU_i(p) = EU_j(p) => (U_i - U_j) @ p = 0
        """
        U = self.config.get_effective_utility()
        levels = self.config.levels
        breakevens = []

        for i in range(len(levels) - 1):
            lvl_a = levels[i]
            lvl_b = levels[i + 1]
            diff = U[i] - U[i + 1]  # diff over [Low, Mod, Crit]
            # Assuming Moderate vs Critical boundary (with Low = 0):
            # diff[1]*(1 - p) + diff[2]*p = 0 => p* = -diff[1] / (diff[2] - diff[1])
            denom = diff[2] - diff[1]
            p_crit_star = -diff[1] / denom if abs(denom) > 1e-6 else None

            # Assuming Low vs Moderate boundary (with Crit = 0):
            denom_lm = diff[1] - diff[0]
            p_mod_star = -diff[0] / denom_lm if abs(denom_lm) > 1e-6 else None

            breakevens.append({
                "pair": f"{lvl_a} vs {lvl_b}",
                "crit_threshold_given_mod": round(float(p_crit_star), 4) if p_crit_star is not None and 0 <= p_crit_star <= 1 else None,
                "mod_threshold_given_low": round(float(p_mod_star), 4) if p_mod_star is not None and 0 <= p_mod_star <= 1 else None,
                "utility_delta_vector": [round(float(x), 2) for x in diff]
            })

        return breakevens
