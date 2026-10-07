"""
Simulation: ER Queue Prioritization Policies.
Implements:
  1. FIFO: First In First Out (arrival order)
  2. Rule-Based ESI Baseline: Emergency Severity Index heuristic based on vital abnormalities
  3. Bayesian Network + c*mu Rule (Our decision-theoretic system)
  4. Oracle: Omniscient upper bound with knowledge of true latent severity

Syllabus Topic: Queue Scheduling Policies, Heuristics vs Probabilistic Optimization.
"""
from __future__ import annotations
from typing import Dict, List, Any
import numpy as np
from backend.app.decision.priority import PatientQueuePrioritizer


def evaluate_rule_based_priority(patient: Dict[str, Any]) -> float:
    """
    Rule-based Emergency Severity Index (ESI) heuristic:
      Level 1 (score 1000): Any Severe vital sign or Unresponsive consciousness
      Level 2 (score 100): >= 2 Abnormal vital signs or Altered consciousness
      Level 3 (score 10): 1 Abnormal vital sign
      Level 4 (score 1): All normal
    Ties broken by wait time (earlier arrival first).
    """
    ev = patient["evidence"]
    severe_count = 0
    abnormal_count = 0

    vitals = ["HeartRate", "SystolicBP", "SpO2", "RespRate", "Temperature"]
    for v in vitals:
        val = ev.get(v)
        if val == "Severe":
            severe_count += 1
        elif val == "Abnormal":
            abnormal_count += 1

    cons = ev.get("Consciousness", "Alert")
    if severe_count > 0 or cons == "Unresponsive":
        tier = 1000.0
    elif abnormal_count >= 2 or cons == "Altered":
        tier = 100.0
    elif abnormal_count >= 1:
        tier = 10.0
    else:
        tier = 1.0

    # Add small bonus for waiting time to break ties FIFO within tier
    wait_bonus = max(0.0, patient.get("wait_time", 0.0)) * 0.001
    return tier + wait_bonus


class PolicySelector:
    def __init__(self, prioritizer: PatientQueuePrioritizer):
        self.prioritizer = prioritizer

    def select_next_patient(
        self,
        queue: List[Dict[str, Any]],
        policy: str
    ) -> int:
        """
        Returns the index in the queue of the next patient to be treated according to policy:
          - 'fifo'
          - 'rule_based'
          - 'bn_cmu'
          - 'oracle'
        """
        if not queue:
            raise IndexError("Queue is empty")

        if policy == "fifo":
            # Earliest arrival time
            best_idx = 0
            best_time = queue[0]["arrival_time"]
            for i in range(1, len(queue)):
                if queue[i]["arrival_time"] < best_time:
                    best_time = queue[i]["arrival_time"]
                    best_idx = i
            return best_idx

        elif policy == "rule_based":
            best_idx = 0
            best_score = float("-inf")
            for i, pt in enumerate(queue):
                score = evaluate_rule_based_priority(pt)
                if score > best_score:
                    best_score = score
                    best_idx = i
            return best_idx

        elif policy == "bn_cmu":
            best_idx = 0
            best_index = float("-inf")
            for i, pt in enumerate(queue):
                # Compute or retrieve c*mu priority index
                if "cmu_index" not in pt:
                    m = self.prioritizer.compute_patient_metrics(pt["patient_id"], pt["evidence"])
                    pt["cmu_index"] = m["priority_index"]
                if pt["cmu_index"] > best_index:
                    best_index = pt["cmu_index"]
                    best_idx = i
            return best_idx

        elif policy == "oracle":
            # True severity: Critical (3) > Moderate (2) > Low (1)
            sev_weights = {"Critical": 3000.0, "Moderate": 200.0, "Low": 10.0}
            best_idx = 0
            best_score = float("-inf")
            for i, pt in enumerate(queue):
                score = sev_weights.get(pt.get("current_severity", pt["true_severity"]), 10.0)
                # Tie break by shorter service time
                cond = pt["true_condition"]
                serv_time = self.prioritizer.config.service_minutes.get(cond, 20.0)
                adj_score = score / serv_time + pt.get("wait_time", 0.0) * 0.001
                if adj_score > best_score:
                    best_score = adj_score
                    best_idx = i
            return best_idx

        else:
            raise ValueError(f"Unknown policy: {policy}")
