"""
FastAPI Pydantic v2 Schemas for ER Triage System.
Strict validation of variable names, state spaces, evidence forms, and query options.
"""
from __future__ import annotations
from typing import Dict, List, Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator


class EvidenceModel(BaseModel):
    """
    Evidence dictionary mapping variable names to their observed categorical state.
    Missing variables represent unmeasured / latent evidence.
    """
    evidence: Dict[str, str] = Field(default_factory=dict, description="Observed clinical evidence mapping")

    @field_validator("evidence")
    @classmethod
    def validate_non_empty_strings(cls, v: Dict[str, str]) -> Dict[str, str]:
        for var, val in v.items():
            if not isinstance(var, str) or not isinstance(val, str):
                raise ValueError("All evidence keys and values must be non-empty strings")
            if not val.strip():
                raise ValueError(f"State for variable '{var}' cannot be empty")
        return v


class InferRequest(BaseModel):
    evidence: Dict[str, str] = Field(default_factory=dict)
    method: Literal["ve", "bp", "rejection", "lw", "gibbs"] = Field(default="ve")
    n_samples: int = Field(default=10000, ge=10, le=200000)
    targets: List[str] = Field(default_factory=lambda: ["Severity", "Condition"])


class MapRequest(BaseModel):
    evidence: Dict[str, str] = Field(default_factory=dict)
    target_vars: List[str] = Field(default_factory=lambda: ["Condition", "Severity"])
    top_k: int = Field(default=3, ge=1, le=10)


class DSepRequest(BaseModel):
    x: str
    y: str
    z: List[str] = Field(default_factory=list)


class DecideRequest(BaseModel):
    evidence: Dict[str, str] = Field(default_factory=dict)
    risk_aversion: float = Field(default=0.0, ge=0.0, le=2.0)


class VOIRequest(BaseModel):
    evidence: Dict[str, str] = Field(default_factory=dict)


class PatientQueueItem(BaseModel):
    patient_id: str
    evidence: Dict[str, str] = Field(default_factory=dict)
    arrival_time: float = 0.0
    wait_minutes: float = 0.0


class QueueRankRequest(BaseModel):
    patients: List[PatientQueueItem]
    clinicians: int = Field(default=3, ge=1, le=10)


class ReadingItem(BaseModel):
    minute: float
    vitals: Dict[str, str] = Field(default_factory=dict)


class FilterRequest(BaseModel):
    readings: List[ReadingItem]
    demographics: Optional[Dict[str, str]] = None
    alert_threshold: float = Field(default=0.35, ge=0.01, le=0.99)


class LearnRequest(BaseModel):
    method: Literal["mle", "bayes", "em"] = Field(default="mle")
    n: int = Field(default=5000, ge=10, le=5000)
    alpha: float = Field(default=1.0, ge=0.01, le=20.0)
    hide_latents: bool = Field(default=False)
    max_iters: int = Field(default=10, ge=1, le=50)


class StructureRequest(BaseModel):
    score: Literal["bic"] = Field(default="bic")
    n: int = Field(default=5000, ge=50, le=5000)
    max_parents: int = Field(default=3, ge=1, le=5)


class ConvergenceRequest(BaseModel):
    evidence: Dict[str, str] = Field(default_factory=dict)
    sample_sizes: Optional[List[int]] = None
    repeats: int = Field(default=10, ge=1, le=30)


class SimulateRequest(BaseModel):
    arrival_rate: float = Field(default=7.0, ge=1.0, le=20.0)
    clinicians: int = Field(default=3, ge=1, le=10)
    horizon_hours: float = Field(default=24.0, ge=1.0, le=48.0)
    replications: int = Field(default=20, ge=1, le=100)
    policies: Optional[List[str]] = None


class OverrideRequest(BaseModel):
    patient_id: str
    recommended_level: str
    chosen_level: str
    reason: str
    evidence: Optional[Dict[str, Any]] = None

    @field_validator("reason")
    @classmethod
    def validate_reason_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("A documented clinical reason is mandatory to override automated recommendation")
        return v.strip()


class UpdateUtilityRequest(BaseModel):
    utility_matrix: List[List[float]]
    harm_rate: Optional[List[float]] = None
    service_minutes: Optional[Dict[str, float]] = None
