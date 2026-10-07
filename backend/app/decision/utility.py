"""
Decision Theory: Utility Matrix and Preferences.
Maintains decision configurations:
  - Triage levels (P1_Immediate, P2_Urgent, P3_Standard, P4_NonUrgent)
  - Utility matrix U[level, severity]
  - Harm rate per minute by severity (delay cost c)
  - Service minutes by condition (mean service time 1/mu)
  - Test costs in utility units
  - Risk-attitude transformations (risk-neutral vs risk-averse)
  - Validation: monotonicity checks & clinical consistency warnings

Syllabus Topic: Decision Theory, Utility Functions, Risk Preferences.
"""
from __future__ import annotations
from typing import Dict, List, Any, Tuple, Optional
import numpy as np


class DecisionConfig:
    def __init__(
        self,
        levels: Optional[List[str]] = None,
        utility_matrix: Optional[np.ndarray | List[List[float]]] = None,
        severity_order: Optional[List[str]] = None,
        harm_rate_per_min: Optional[np.ndarray | List[float]] = None,
        service_minutes_by_condition: Optional[Dict[str, float]] = None,
        test_costs: Optional[Dict[str, float]] = None,
        risk_aversion: float = 0.0  # 0.0 = risk-neutral; >0 = risk-averse
    ):
        self.levels = levels or ["P1_Immediate", "P2_Urgent", "P3_Standard", "P4_NonUrgent"]
        self.severity_order = severity_order or ["Low", "Moderate", "Critical"]
        
        # Default ground-truth utility matrix: U[level, severity]
        # Rows: P1, P2, P3, P4. Cols: Low, Moderate, Critical
        if utility_matrix is None:
            self.U = np.array([
                [-20.0, -6.0,   0.0],   # P1_Immediate
                [-8.0,  -2.0, -25.0],   # P2_Urgent
                [-2.0, -15.0, -80.0],   # P3_Standard
                [ 0.0, -40.0, -100.0]   # P4_NonUrgent
            ], dtype=np.float64)
        else:
            self.U = np.asarray(utility_matrix, dtype=np.float64)

        if harm_rate_per_min is None:
            self.harm_rate = np.array([0.01, 0.08, 0.60], dtype=np.float64)
        else:
            self.harm_rate = np.asarray(harm_rate_per_min, dtype=np.float64)

        self.service_minutes = service_minutes_by_condition or {
            "Cardiac": 25.0,
            "Respiratory": 20.0,
            "Infection": 20.0,
            "Trauma": 30.0,
            "Minor": 10.0
        }

        self.test_costs = test_costs or {
            "ECG_Abnormal": 0.5,
            "Troponin_High": 1.5,
            "Lactate_High": 1.0,
            "ChestXray_Abnormal": 2.0
        }
        self.risk_aversion = risk_aversion

    def get_effective_utility(self) -> np.ndarray:
        """
        Applies exponential risk-aversion transform if risk_aversion > 0:
        U_eff = -exp(-gamma * U) normalized back to U scale.
        When risk_aversion == 0, returns standard utility matrix.
        """
        if self.risk_aversion <= 1e-6:
            return self.U
        gamma = self.risk_aversion
        # U values are <= 0. Severe penalties are large negative values.
        # Risk aversion heavily penalizes severe negative outcomes (e.g. Critical undertriaged).
        transformed = -np.exp(-gamma * self.U)
        return transformed

    def validate_monotonicity(self) -> List[str]:
        """
        Validates clinical consistency of the utility matrix and produces warnings:
          - Ideal care (diagonal: P1 for Critical, P4 for Low) should have highest utility in column.
          - Under-triage of Critical patients (P4 or P3 for Critical) should be penalized heavily.
        """
        warnings = []
        # Check P1 is best for Critical (column 2)
        best_for_critical = np.argmax(self.U[:, 2])
        if best_for_critical != 0:
            warnings.append(f"Level '{self.levels[best_for_critical]}' has higher utility than P1 for Critical severity!")

        # Check P4 is best for Low (column 0)
        best_for_low = np.argmax(self.U[:, 0])
        if best_for_low != 3:
            warnings.append(f"Level '{self.levels[best_for_low]}' has higher utility than P4 for Low severity (over-triage incentive)!")

        # Monotonicity: under-triaging Critical should get progressively worse: U[P1, Crit] >= U[P2, Crit] >= U[P3, Crit] >= U[P4, Crit]
        crit_col = self.U[:, 2]
        if not (crit_col[0] >= crit_col[1] >= crit_col[2] >= crit_col[3]):
            warnings.append("Utility for Critical severity does not strictly decrease as triage priority decreases (under-triage penalty anomaly).")

        return warnings

    @classmethod
    def from_spec(cls, spec_decision: Dict[str, Any]) -> DecisionConfig:
        levels = spec_decision.get("levels")
        U_mat = spec_decision.get("utility_matrix_rows_levels_cols_severity")
        sev_order = spec_decision.get("severity_order")
        harm = spec_decision.get("harm_rate_per_min_by_severity")
        serv = spec_decision.get("service_minutes_by_condition")
        costs = spec_decision.get("test_cost_utility_units")
        return cls(
            levels=levels,
            utility_matrix=U_mat,
            severity_order=sev_order,
            harm_rate_per_min=harm,
            service_minutes_by_condition=serv,
            test_costs=costs
        )
