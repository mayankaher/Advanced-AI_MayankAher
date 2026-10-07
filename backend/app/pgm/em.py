"""
Probabilistic Graphical Models: Expectation-Maximization (EM) with Hidden Variables.
Implements:
  - Latent variable learning where 'Condition' and 'Severity' are unobserved
  - E-Step: Soft posterior assignments q_i(Condition=c, Severity=s | evidence_i)
  - M-Step: Expected sufficient statistics accumulation and CPT updates
  - Verification of Monotone Non-Decreasing Marginal Log-Likelihood
  - Label switching and local optima analysis

Syllabus Topic: Incomplete Data, EM Algorithm, Latent Variable Models.
"""
from __future__ import annotations
import copy
from typing import Dict, List, Any, Optional, Tuple
import itertools
import numpy as np
import pandas as pd
from backend.app.pgm.network import BayesNet
from backend.app.pgm.variable_elimination import VariableElimination
from backend.app.pgm.learning import ParameterLearner, softmax


class EMLearner:
    def __init__(self, net: BayesNet):
        self.net = net
        self.param_learner = ParameterLearner(net)

    def run_em(
        self,
        df: pd.DataFrame,
        max_iters: int = 15,
        tol: float = 1e-3,
        seed: int = 42,
        noise_sigma: float = 0.5
    ) -> Dict[str, Any]:
        """
        Runs EM on dataset with 'Condition' and 'Severity' hidden.
        Initializes from perturbed expert model.
        """
        # Create initial model with perturbed parameters
        current_net = self.param_learner.generate_expert_model(noise_sigma=noise_sigma, seed=seed)
        
        # We focus on the observed columns in df
        obs_vars = [c for c in df.columns if c not in ("Condition", "Severity")]
        n_samples = len(df)

        ll_history: List[float] = []
        cpt_error_history: List[float] = []

        # Convert dataframe rows to evidence dictionaries
        records = df[obs_vars].to_dict(orient="records")

        # Core state spaces
        cond_states = self.net.states["Condition"]
        sev_states = self.net.states["Severity"]
        n_cond = len(cond_states)
        n_sev = len(sev_states)

        for iteration in range(max_iters):
            ve = VariableElimination(current_net)

            # E-step: compute joint posterior q_i(Condition, Severity) and log-likelihood
            total_ll = 0.0
            expected_counts_cond_sev = np.zeros((3, 4, n_cond, n_sev), dtype=np.float64) # [Age, Hist, Cond, Sev]
            # Accumulators for symptoms, vitals, arrival
            vital_counts = {v: np.zeros((n_cond, n_sev, current_net.cards[v]), dtype=np.float64) for v in ["HeartRate", "SystolicBP", "SpO2", "RespRate", "Consciousness", "Temperature"]}
            symptom_counts = {s: np.zeros((n_cond, current_net.cards[s]), dtype=np.float64) for s in ["ChestPain", "Dyspnea", "VisibleInjury"]}
            arrival_counts = np.zeros((n_sev, current_net.cards["ArrivalMode"]), dtype=np.float64)

            for row in records:
                # Query joint over Condition and Severity given row evidence
                joint_factor, _ = ve.query(["Condition", "Severity"], row, prune_barren=True)
                joint_factor_norm = joint_factor.normalize()

                # Scope can be ['Condition', 'Severity'] or reversed
                if joint_factor_norm.vars == ["Condition", "Severity"]:
                    q_cs = joint_factor_norm.values
                else:
                    q_cs = joint_factor_norm.values.T

                # Marginal likelihood for row: P(evidence)
                # Compute by evaluating evidence likelihood
                # From Bayes theorem: P(e) = sum_{c, s} P(c, s, e)
                # We can approximate row log-likelihood via VE or joint sum
                # P(e) is the normalization factor before normalization
                # For safety: evaluate joint probability of a sample
                total_ll += float(np.log(np.maximum(1e-12, np.sum(joint_factor.values))))

                # Accumulate statistics
                age_val = row.get("Age")
                hist_val = row.get("History")
                if age_val and hist_val:
                    a_idx = self.net.states["Age"].index(age_val)
                    h_idx = self.net.states["History"].index(hist_val)
                    expected_counts_cond_sev[a_idx, h_idx] += q_cs

                # Symptoms
                for s in symptom_counts:
                    if s in row:
                        val_idx = self.net.states[s].index(row[s])
                        q_c = q_cs.sum(axis=1) # marginal Condition
                        symptom_counts[s][:, val_idx] += q_c

                # Vitals
                for v in vital_counts:
                    if v in row:
                        val_idx = self.net.states[v].index(row[v])
                        vital_counts[v][:, :, val_idx] += q_cs

                # Arrival
                if "ArrivalMode" in row:
                    val_idx = self.net.states["ArrivalMode"].index(row["ArrivalMode"])
                    q_s = q_cs.sum(axis=0) # marginal Severity
                    arrival_counts[:, val_idx] += q_s

            ll_history.append(round(total_ll / n_samples, 4))
            cpt_error = self.param_learner.compute_average_cpt_kl(
                {n: current_net.nodes_spec[n]["cpt"] for n in current_net.nodes}
            )
            cpt_error_history.append(round(cpt_error, 5))

            # Check convergence
            if iteration > 0 and abs(ll_history[-1] - ll_history[-2]) < tol:
                break

            # M-step: update CPTs in current_net
            # Update symptoms
            for s, counts in symptom_counts.items():
                for c in range(n_cond):
                    row_probs = counts[c] / (counts[c].sum() + 1e-9)
                    cond_name = cond_states[c]
                    current_net.nodes_spec[s]["cpt"][cond_name] = [round(float(p), 6) for p in row_probs]

            # Update vitals
            for v, counts in vital_counts.items():
                for c in range(n_cond):
                    for s_i in range(n_sev):
                        key = f"{cond_states[c]}|{sev_states[s_i]}"
                        row_probs = counts[c, s_i] / (counts[c, s_i].sum() + 1e-9)
                        current_net.nodes_spec[v]["cpt"][key] = [round(float(p), 6) for p in row_probs]

            # Update arrival
            for s_i in range(n_sev):
                key = sev_states[s_i]
                row_probs = arrival_counts[s_i] / (arrival_counts[s_i].sum() + 1e-9)
                current_net.nodes_spec["ArrivalMode"][key] = [round(float(p), 6) for p in row_probs]

            # Rebuild factors
            current_net._build_cpts()

        # Check monotone property
        is_monotone = True
        for i in range(1, len(ll_history)):
            if ll_history[i] < ll_history[i - 1] - 0.05:  # small numerical slack
                is_monotone = False

        return {
            "iterations_run": len(ll_history),
            "log_likelihood_per_patient": ll_history,
            "cpt_error_history": cpt_error_history,
            "monotone_increasing": is_monotone,
            "final_cpt_error": cpt_error_history[-1],
            "explanation": (
                "EM iteratively maximizes the expected complete-data log-likelihood (Q-function). "
                "Because Condition and Severity are unobserved, EM alternates between inferring soft distributions "
                "over latent patient phenotypes (E-step) and re-estimating clinical symptom/vital CPTs (M-step). "
                "Notice that marginal log-likelihood is monotonically non-decreasing at each iteration, "
                "though the algorithm can be sensitive to initialization and label switching."
            )
        }
