"""
Probabilistic Graphical Models: Temporal Inference (HMMs & Dynamic Bayesian Networks).
Implements:
  - Forward Filtering (Online belief updates over waiting time):
    alpha_t(s_t) proportional to P(y_t | s_t) * sum_{s_{t-1}} P(s_t | s_{t-1}) alpha_{t-1}(s_{t-1})
  - Backward Smoothing (Forward-Backward smoothing for retrospective auditing):
    gamma_t(s_t) proportional to alpha_t(s_t) * beta_t(s_t)
  - Viterbi Algorithm (Most Probable Trajectory of patient clinical deterioration)
  - Deterioration alerts and simulated dynamic vital updates

Syllabus Topic: Temporal Models, Hidden Markov Models, Filtering, Smoothing, Viterbi.
"""
from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from backend.app.pgm.network import BayesNet
from backend.app.pgm.variable_elimination import VariableElimination


class TemporalInference:
    def __init__(self, net: BayesNet):
        self.net = net
        self.ve = VariableElimination(net)
        # Severity transition matrix per 10-minute step: P(S_t | S_{t-1})
        # Rows: S_{t-1} in [Low, Moderate, Critical]. Cols: S_t
        raw_trans = net.decision_spec.get(
            "severity_transition_per_10min",
            [[0.97, 0.029, 0.001], [0.02, 0.93, 0.05], [0.0, 0.04, 0.96]]
        )
        self.T_step = np.asarray(raw_trans, dtype=np.float64)
        self.sev_states = ["Low", "Moderate", "Critical"]

    def compute_transition_for_duration(self, minutes: float) -> np.ndarray:
        """
        Computes transition matrix for an arbitrary time elapsed using matrix exponentiation:
        T(dt) = (T_10min) ^ (minutes / 10.0)
        """
        steps = max(0.1, minutes / 10.0)
        # Eigendecomposition of transition matrix for fractional matrix powers
        eigenvals, eigenvecs = np.linalg.eig(self.T_step)
        inv_vecs = np.linalg.inv(eigenvecs)
        # A^t = V * (Lambda^t) * V^-1
        lambda_t = np.diag(eigenvals ** steps)
        T_dt = np.real(eigenvecs @ lambda_t @ inv_vecs)
        # Row-normalize to guarantee valid stochastic matrix
        T_dt = np.maximum(0.0, T_dt)
        T_dt /= T_dt.sum(axis=1, keepdims=True)
        return T_dt

    def compute_observation_likelihood(
        self,
        vitals_and_symptoms: Dict[str, str],
        demographics: Optional[Dict[str, str]] = None
    ) -> np.ndarray:
        """
        Computes likelihood vector [P(y | Low), P(y | Moderate), P(y | Critical)]
        by conditioning the static BN on demographics + vitals with Severity forced to each state.
        """
        demo = demographics or {}
        emission_probs = np.zeros(3, dtype=np.float64)

        for s_idx, s_name in enumerate(self.sev_states):
            ev = dict(demo)
            ev.update(vitals_and_symptoms)
            ev["Severity"] = s_name
            # Compute probability of this full configuration or joint likelihood
            # Via VE: query empty targets gives total unnormalized probability Z
            # Or query single dummy variable to get joint Z
            joint_f, _ = self.ve.query(["Condition"], ev, prune_barren=True)
            # Sum of joint_f values before normalization is P(evidence)
            emission_probs[s_idx] = float(np.sum(joint_f.values))

        p_sum = emission_probs.sum()
        if p_sum > 0:
            emission_probs /= p_sum
        else:
            emission_probs = np.ones(3) / 3.0

        return emission_probs

    def filter_trajectory(
        self,
        readings: List[Dict[str, Any]],
        demographics: Optional[Dict[str, str]] = None,
        alert_threshold: float = 0.35
    ) -> Dict[str, Any]:
        """
        Runs forward filtering across a sequence of time-stamped clinical readings:
        readings = [
          {"minute": 0, "vitals": {...}},
          {"minute": 20, "vitals": {...}}, ...
        ]
        """
        if not readings:
            return {"timeline": [], "alerts": []}

        timeline: List[Dict[str, Any]] = []
        alerts: List[Dict[str, Any]] = []

        # Step 0: Initial prior or initial observation
        r0 = readings[0]
        obs0 = r0.get("vitals", {})
        dt0 = r0.get("minute", 0)

        # Baseline prior over Severity
        prior_factor, _ = self.ve.query(["Severity"], demographics or {})
        prior_dist = prior_factor.get_distribution({"Severity": self.sev_states})
        b_curr = np.array([prior_dist["Low"], prior_dist["Moderate"], prior_dist["Critical"]], dtype=np.float64)

        if obs0:
            lik0 = self.compute_observation_likelihood(obs0, demographics)
            b_curr = b_curr * lik0
            b_curr /= b_curr.sum()

        timeline.append({
            "minute": dt0,
            "belief": {self.sev_states[i]: round(float(b_curr[i]), 4) for i in range(3)},
            "p_critical": round(float(b_curr[2]), 4),
            "vitals": obs0,
            "alert": b_curr[2] >= alert_threshold
        })

        if b_curr[2] >= alert_threshold:
            alerts.append({
                "minute": dt0,
                "p_critical": round(float(b_curr[2]), 4),
                "message": f"Deterioration Alert at t={dt0}m: P(Critical)={b_curr[2]:.2f} exceeded threshold {alert_threshold:.2f}"
            })

        prev_minute = dt0

        for r in readings[1:]:
            curr_minute = r.get("minute", prev_minute + 15)
            dt = curr_minute - prev_minute
            vitals = r.get("vitals", {})

            # 1. Prediction step: b_pred = b_{t-1} @ T(dt)
            T_dt = self.compute_transition_for_duration(dt)
            b_pred = b_curr @ T_dt

            # 2. Update step: b_curr = b_pred * P(y_t | S_t)
            if vitals:
                lik = self.compute_observation_likelihood(vitals, demographics)
                b_curr = b_pred * lik
                b_curr /= b_curr.sum()
            else:
                b_curr = b_pred

            is_alert = (b_curr[2] >= alert_threshold)
            timeline.append({
                "minute": curr_minute,
                "belief": {self.sev_states[i]: round(float(b_curr[i]), 4) for i in range(3)},
                "p_critical": round(float(b_curr[2]), 4),
                "vitals": vitals,
                "alert": is_alert
            })

            if is_alert:
                alerts.append({
                    "minute": curr_minute,
                    "p_critical": round(float(b_curr[2]), 4),
                    "message": f"Deterioration Alert at t={curr_minute}m: P(Critical)={b_curr[2]:.2f} exceeded threshold {alert_threshold:.2f}"
                })

            prev_minute = curr_minute

        return {
            "timeline": timeline,
            "alerts": alerts,
            "alert_threshold": alert_threshold
        }

    def smooth_trajectory(
        self,
        readings: List[Dict[str, Any]],
        demographics: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Forward-Backward Smoothing:
        Returns smoothed marginals gamma_t = P(S_t | y_{1:T}).
        """
        T = len(readings)
        if T == 0:
            return []

        # Forward pass (alphas)
        alphas = []
        transitions = []
        likelihoods = []

        # Initial alpha_0
        r0 = readings[0]
        lik0 = self.compute_observation_likelihood(r0.get("vitals", {}), demographics)
        prior_factor, _ = self.ve.query(["Severity"], demographics or {})
        prior_dist = prior_factor.get_distribution({"Severity": self.sev_states})
        p0 = np.array([prior_dist["Low"], prior_dist["Moderate"], prior_dist["Critical"]], dtype=np.float64)

        a0 = p0 * lik0
        a0 /= a0.sum()
        alphas.append(a0)
        likelihoods.append(lik0)

        for t in range(1, T):
            dt = readings[t].get("minute", t * 15) - readings[t - 1].get("minute", (t - 1) * 15)
            trans = self.compute_transition_for_duration(dt)
            transitions.append(trans)
            lik = self.compute_observation_likelihood(readings[t].get("vitals", {}), demographics)
            likelihoods.append(lik)

            a_pred = alphas[-1] @ trans
            a_t = a_pred * lik
            a_t /= a_t.sum()
            alphas.append(a_t)

        # Backward pass (betas)
        betas = [None] * T
        betas[T - 1] = np.ones(3, dtype=np.float64)

        for t in range(T - 2, -1, -1):
            trans = transitions[t]
            lik_next = likelihoods[t + 1]
            # beta_t = T_t @ (lik_{t+1} * beta_{t+1})
            b_next = lik_next * betas[t + 1]
            betas[t] = trans @ b_next
            betas[t] /= betas[t].sum()

        smoothed = []
        for t in range(T):
            gamma = alphas[t] * betas[t]
            gamma /= gamma.sum()
            smoothed.append({
                "minute": readings[t].get("minute", t * 15),
                "smoothed_belief": {self.sev_states[i]: round(float(gamma[i]), 4) for i in range(3)},
                "smoothed_p_critical": round(float(gamma[2]), 4)
            })

        return smoothed

    def viterbi(
        self,
        readings: List[Dict[str, Any]],
        demographics: Optional[Dict[str, str]] = None
    ) -> List[str]:
        """
        Viterbi algorithm for Most Probable Path of severity states:
        argmax_{s_{1:T}} P(s_{1:T}, y_{1:T})
        """
        T = len(readings)
        if T == 0:
            return []

        prior_factor, _ = self.ve.query(["Severity"], demographics or {})
        prior_dist = prior_factor.get_distribution({"Severity": self.sev_states})
        p0 = np.array([prior_dist["Low"], prior_dist["Moderate"], prior_dist["Critical"]], dtype=np.float64)

        lik0 = self.compute_observation_likelihood(readings[0].get("vitals", {}), demographics)
        V = [np.log(np.maximum(1e-12, p0 * lik0))]
        backpointers = []

        for t in range(1, T):
            dt = readings[t].get("minute", t * 15) - readings[t - 1].get("minute", (t - 1) * 15)
            trans = self.compute_transition_for_duration(dt)
            lik = self.compute_observation_likelihood(readings[t].get("vitals", {}), demographics)

            log_trans = np.log(np.maximum(1e-12, trans))
            # For each state j in current step: max_i (V[t-1][i] + log_trans[i, j])
            scores = V[-1][:, np.newaxis] + log_trans
            best_prev = np.argmax(scores, axis=0)
            max_scores = np.max(scores, axis=0) + np.log(np.maximum(1e-12, lik))

            V.append(max_scores)
            backpointers.append(best_prev)

        # Traceback
        best_last = int(np.argmax(V[-1]))
        path = [best_last]
        for t in range(T - 2, -1, -1):
            best_last = int(backpointers[t][best_last])
            path.insert(0, best_last)

        return [self.sev_states[idx] for idx in path]
