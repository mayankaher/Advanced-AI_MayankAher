"""
Simulation: Synthetic ER Patient Stream Generator.
Generates realistic patient arrivals according to ground-truth Bayesian Network.

Syllabus Topic: Generative Models, Discrete-Event Simulation Inputs.
"""
from __future__ import annotations
from typing import Dict, List, Any, Optional
import numpy as np
from backend.app.pgm.network import BayesNet


class PatientGenerator:
    def __init__(self, net: BayesNet, rng_seed: int = 42):
        self.net = net
        self.rng = np.random.default_rng(rng_seed)

    def generate_patient(self, arrival_time_minutes: float, patient_id: str) -> Dict[str, Any]:
        """
        Samples a single patient from the ground-truth distribution and assigns arrival time.
        """
        sample = self.net.ancestral_sample(n_samples=1, rng=self.rng)[0]

        # Symptoms and vitals observed
        evidence = {
            k: v for k, v in sample.items()
            if k not in ("Condition", "Severity", "ECG_Abnormal", "Troponin_High", "Lactate_High", "ChestXray_Abnormal")
        }

        return {
            "patient_id": patient_id,
            "arrival_time": arrival_time_minutes,
            "true_condition": sample["Condition"],
            "true_severity": sample["Severity"],
            "current_severity": sample["Severity"],
            "evidence": evidence,
            "adverse_events": 0,
            "service_duration": 0.0,
            "wait_time": 0.0
        }

    def generate_arrival_stream(
        self,
        arrival_rate_per_hour: float = 7.0,
        horizon_hours: float = 24.0
    ) -> List[Dict[str, Any]]:
        """
        Generates Poisson arrivals with parameter lambda = arrival_rate_per_hour.
        Inter-arrival times are Exp(lambda).
        """
        rate_per_min = arrival_rate_per_hour / 60.0
        current_time = 0.0
        max_time = horizon_hours * 60.0
        patients = []
        count = 0

        while current_time < max_time:
            inter_arrival = self.rng.exponential(1.0 / rate_per_min)
            current_time += inter_arrival
            if current_time >= max_time:
                break
            count += 1
            pt = self.generate_patient(current_time, f"PT_{count:04d}")
            patients.append(pt)

        return patients
