"""
Evaluation: Clinical Classification & Safety Metrics.
Implements:
  - AUROC (Area Under Receiver Operating Characteristic Curve)
  - Brier Score: Mean squared error of probabilistic predictions
  - Under-triage Rate: Fraction of truly Critical patients assigned below P1
  - Over-triage Rate: Fraction of truly Low-acuity patients assigned to P1

Syllabus Topic: Evaluation Metrics, Probabilistic Calibration, Clinical Safety.
"""
from __future__ import annotations
from typing import List, Sequence, Dict, Any, Tuple
import numpy as np


def compute_auroc(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    """
    Computes exact AUROC using trapezoidal rule / Wilcoxon-Mann-Whitney U statistic:
    AUROC = (U - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    """
    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)

    n_pos = int(np.sum(y_true_arr == 1))
    n_neg = int(np.sum(y_true_arr == 0))

    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Rank with tie handling
    order = np.argsort(y_prob_arr)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_prob_arr) + 1)

    # Handle ties: average ranks
    sorted_probs = y_prob_arr[order]
    unique_vals, inverse_indices, counts = np.unique(sorted_probs, return_inverse=True, return_counts=True)
    if len(unique_vals) < len(y_prob_arr):
        # There are ties
        for val_idx, count in enumerate(counts):
            if count > 1:
                tied_mask = (sorted_probs == unique_vals[val_idx])
                tied_ranks = np.where(tied_mask)[0] + 1
                avg_rank = float(np.mean(tied_ranks))
                original_indices = order[tied_mask]
                ranks[original_indices] = avg_rank

    rank_sum_pos = float(np.sum(ranks[y_true_arr == 1]))
    u_stat = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    auroc = u_stat / (n_pos * n_neg)
    return float(np.clip(auroc, 0.0, 1.0))


def compute_brier_score(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    """
    Brier Score: MSE between predicted probabilities and binary outcomes.
    BS = (1 / N) * sum (y_prob_i - y_true_i)^2
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)
    return float(np.mean((y_prob_arr - y_true_arr) ** 2))


def compute_safety_metrics(
    true_severities: Sequence[str],
    assigned_levels: Sequence[str]
) -> Dict[str, float]:
    """
    Under-triage: truly Critical assigned below P1_Immediate (P2, P3, P4).
    Over-triage: truly Low assigned to P1_Immediate.
    """
    true_severities = list(true_severities)
    assigned_levels = list(assigned_levels)

    crit_total = sum(1 for s in true_severities if s == "Critical")
    crit_undertriaged = sum(
        1 for s, lvl in zip(true_severities, assigned_levels)
        if s == "Critical" and lvl != "P1_Immediate"
    )

    low_total = sum(1 for s in true_severities if s == "Low")
    low_overtriaged = sum(
        1 for s, lvl in zip(true_severities, assigned_levels)
        if s == "Low" and lvl == "P1_Immediate"
    )

    undertriage_rate = (crit_undertriaged / crit_total) if crit_total > 0 else 0.0
    overtriage_rate = (low_overtriaged / low_total) if low_total > 0 else 0.0

    return {
        "undertriage_rate": round(float(undertriage_rate), 4),
        "overtriage_rate": round(float(overtriage_rate), 4),
        "critical_count": crit_total,
        "undertriaged_count": crit_undertriaged,
        "low_count": low_total,
        "overtriaged_count": low_overtriaged
    }
