"""
FastAPI API Routers: All endpoints for the ER Triage System.
Maps to the specification in MASTER_PROMPT Section 5.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import json

from fastapi import APIRouter, HTTPException, Query
import numpy as np
import pandas as pd

from backend.app.schemas import (
    InferRequest, MapRequest, DSepRequest, DecideRequest, VOIRequest,
    QueueRankRequest, FilterRequest, LearnRequest, StructureRequest,
    ConvergenceRequest, SimulateRequest, OverrideRequest, UpdateUtilityRequest,
)
from backend.app.pgm.variable_elimination import VariableElimination
from backend.app.pgm.clique_tree import CliqueTree
from backend.app.pgm.sampling import Sampler
from backend.app.pgm.map_inference import MAPInference
from backend.app.pgm.dseparation import DSeparationAnalyzer
from backend.app.pgm.learning import ParameterLearner
from backend.app.pgm.em import EMLearner
from backend.app.pgm.structure import StructureLearner
from backend.app.pgm.temporal import TemporalInference
from backend.app.decision.utility import DecisionConfig
from backend.app.decision.triage import TriageRecommender
from backend.app.decision.voi import VOIAnalyzer
from backend.app.decision.priority import PatientQueuePrioritizer
from backend.app.decision.sensitivity import SensitivityAnalyzer
from backend.app.sim.queue_sim import ERQueueSimulator
from backend.app.eval.calibration import CalibrationEvaluator
from backend.app.db import log_override, get_all_overrides

router = APIRouter()


def _get_state():
    """Import app_state lazily to avoid circular imports."""
    from backend.app.main import app_state
    return app_state


def _validate_evidence(evidence: Dict[str, str], net) -> Dict[str, str]:
    """Validate evidence keys and values against the network."""
    for var, val in evidence.items():
        if var not in net.nodes:
            raise HTTPException(400, f"Unknown variable: '{var}'. Valid: {net.nodes}")
        if val not in net.states[var]:
            raise HTTPException(400, f"Unknown state '{val}' for variable '{var}'. Valid: {net.states[var]}")
    return evidence


# ─── Model ───────────────────────────────────────────────────────────
@router.get("/model")
async def get_model():
    """Returns full model specification: nodes, states, edges, CPTs, decision config."""
    st = _get_state()
    nodes_info = {}
    for n in st.net.nodes:
        nodes_info[n] = {
            "states": st.net.states[n],
            "parents": st.net.parents[n],
            "children": st.net.children[n],
            "cpt": st.net.nodes_spec[n]["cpt"],
        }
    edges = [
        {"source": p, "target": n}
        for n in st.net.nodes for p in st.net.parents[n]
    ]
    return {
        "nodes": nodes_info,
        "edges": edges,
        "topological_order": st.net.topological_order,
        "decision": st.net.decision_spec,
        "vital_clinical_meaning": st.net.vital_clinical_meaning,
        "disclaimer": "Educational prototype. Not for real patient care.",
    }


@router.put("/model/utility")
async def update_utility(req: UpdateUtilityRequest):
    """Update utility matrix, harm rates, or service times."""
    st = _get_state()
    config = DecisionConfig.from_spec(st.net.decision_spec)
    config.U = np.asarray(req.utility_matrix, dtype=np.float64)
    if req.harm_rate:
        config.harm_rate = np.asarray(req.harm_rate, dtype=np.float64)
    if req.service_minutes:
        config.service_minutes = req.service_minutes
    warnings = config.validate_monotonicity()
    # Update in-memory spec
    st.net.decision_spec["utility_matrix_rows_levels_cols_severity"] = req.utility_matrix
    if req.harm_rate:
        st.net.decision_spec["harm_rate_per_min_by_severity"] = req.harm_rate
    if req.service_minutes:
        st.net.decision_spec["service_minutes_by_condition"] = req.service_minutes
    return {"status": "updated", "warnings": warnings}


# ─── Inference ───────────────────────────────────────────────────────
@router.post("/infer")
async def infer(req: InferRequest):
    """Posterior inference: VE, BP, rejection, LW, or Gibbs sampling."""
    st = _get_state()
    evidence = _validate_evidence(req.evidence, st.net)

    if req.method == "ve":
        ve = VariableElimination(st.net)
        posteriors = {}
        traces = {}
        for target in req.targets:
            factor, trace = ve.query([target], evidence)
            dist = factor.get_distribution({target: st.net.states[target]})
            posteriors[target] = {k: round(v, 6) for k, v in dist.items()}
            traces[target] = trace
        return {"method": "ve", "posteriors": posteriors, "traces": traces}

    elif req.method == "bp":
        ct = CliqueTree(st.net)
        ct.calibrate(evidence)
        posteriors = {}
        for target in req.targets:
            factor = ct.query(target, evidence)
            dist = factor.get_distribution({target: st.net.states[target]})
            posteriors[target] = {k: round(v, 6) for k, v in dist.items()}
        tree_structure = ct.get_tree_structure()
        return {"method": "bp", "posteriors": posteriors, "clique_tree": tree_structure}

    else:
        sampler = Sampler(st.net)
        if req.method == "rejection":
            result = sampler.rejection_sampling(req.targets, evidence, n_samples=req.n_samples)
        elif req.method == "lw":
            result = sampler.likelihood_weighting(req.targets, evidence, n_samples=req.n_samples)
        elif req.method == "gibbs":
            result = sampler.gibbs_sampling(req.targets, evidence, n_samples=req.n_samples)
        else:
            raise HTTPException(400, f"Unknown method: {req.method}")
        return result


# ─── MAP Inference ───────────────────────────────────────────────────
@router.post("/map")
async def map_inference(req: MapRequest):
    """Joint MAP with marginal-MAP comparison and top-k explanations."""
    st = _get_state()
    evidence = _validate_evidence(req.evidence, st.net)
    mapper = MAPInference(st.net)
    return mapper.joint_map(req.target_vars, evidence, top_k=req.top_k)


# ─── D-Separation ───────────────────────────────────────────────────
@router.post("/dsep")
async def dsep(req: DSepRequest):
    """D-separation test with active/blocked path analysis."""
    st = _get_state()
    analyzer = DSeparationAnalyzer(st.net)
    return analyzer.is_dseparated(req.x, req.y, req.z)


# ─── Decision ────────────────────────────────────────────────────────
@router.post("/decide")
async def decide(req: DecideRequest):
    """Single-patient triage: posterior, EU, best level, evidence contributions."""
    st = _get_state()
    evidence = _validate_evidence(req.evidence, st.net)
    config = DecisionConfig.from_spec(st.net.decision_spec)
    config.risk_aversion = req.risk_aversion
    triage = TriageRecommender(st.net, config)

    # Posteriors
    ve = VariableElimination(st.net)
    sev_factor, _ = ve.query(["Severity"], evidence)
    sev_dist = sev_factor.get_distribution({"Severity": st.net.states["Severity"]})

    cond_factor, _ = ve.query(["Condition"], evidence)
    cond_dist = cond_factor.get_distribution({"Condition": st.net.states["Condition"]})

    rec = triage.recommend(evidence, p_sev=sev_dist)
    contributions = triage.compute_evidence_contributions(evidence, baseline_p_crit=sev_dist.get("Critical", 0))

    # Sensitivity analysis
    sa = SensitivityAnalyzer(config)
    sensitivity = sa.sweep_p_critical()
    breakevens = sa.compute_pairwise_breakeven()

    return {
        "p_severity": {k: round(v, 4) for k, v in sev_dist.items()},
        "p_condition": {k: round(v, 4) for k, v in cond_dist.items()},
        "recommended_level": rec["best_level"],
        **rec,
        "evidence_contributions": contributions,
        "sensitivity": sensitivity,
        "breakevens": breakevens,
    }


# ─── Value of Information ────────────────────────────────────────────
@router.post("/voi")
async def voi_analysis(req: VOIRequest):
    """VOI and preposterior analysis for all unobserved tests."""
    st = _get_state()
    evidence = _validate_evidence(req.evidence, st.net)
    analyzer = VOIAnalyzer(st.net)
    return analyzer.analyze(evidence)


# ─── Queue Ranking ───────────────────────────────────────────────────
@router.post("/queue/rank")
async def queue_rank(req: QueueRankRequest):
    """Multi-patient prioritization with c*mu rule."""
    st = _get_state()
    patients = [p.model_dump() for p in req.patients]
    for p in patients:
        _validate_evidence(p["evidence"], st.net)
    prioritizer = PatientQueuePrioritizer(st.net)
    return prioritizer.rank_patients(patients, clinician_count=req.clinicians)


# ─── Temporal Filtering ─────────────────────────────────────────────
@router.post("/filter")
async def temporal_filter(req: FilterRequest):
    """Forward filtering / smoothing over time-series vital readings."""
    st = _get_state()
    temporal = TemporalInference(st.net)
    readings = [r.model_dump() for r in req.readings]
    filter_result = temporal.filter_trajectory(readings, req.demographics, req.alert_threshold)
    smoothed = temporal.smooth_trajectory(readings, req.demographics)
    viterbi_path = temporal.viterbi(readings, req.demographics)
    return {
        **filter_result,
        "smoothed": smoothed,
        "viterbi_path": viterbi_path,
    }


# ─── Parameter Learning ─────────────────────────────────────────────
@router.post("/learn")
async def learn(req: LearnRequest):
    """Parameter learning: MLE, Bayesian, or EM."""
    st = _get_state()
    learner = ParameterLearner(st.net)

    if req.method == "em":
        em = EMLearner(st.net)
        sub_df = st.df.head(min(req.n, len(st.df)))
        result = em.run_em(sub_df, max_iters=req.max_iters)
        return result

    # Run learning experiment (MLE vs Bayesian comparison)
    result = learner.run_learning_experiment(st.df, alpha=req.alpha)

    # Also return shared parameters demo
    shared = learner.shared_parameters_demo(st.df)

    return {**result, "shared_parameters": shared}


# ─── Structure Learning ─────────────────────────────────────────────
@router.post("/structure")
async def structure_learn(req: StructureRequest):
    """Hill-climbing BIC structure learning."""
    st = _get_state()
    sl = StructureLearner(st.net)
    sub_df = st.df.head(min(req.n, len(st.df)))
    return sl.hill_climbing(sub_df, max_parents=req.max_parents)


# ─── Convergence Experiment ──────────────────────────────────────────
@router.post("/convergence")
async def convergence(req: ConvergenceRequest):
    """Sampling convergence: KL vs N for rejection, LW, Gibbs."""
    st = _get_state()
    evidence = _validate_evidence(req.evidence, st.net)
    sampler = Sampler(st.net)
    return sampler.run_convergence_experiment(
        evidence,
        sample_sizes=req.sample_sizes,
        repeats=req.repeats,
    )


# ─── Policy Simulation ──────────────────────────────────────────────
@router.post("/simulate")
async def simulate(req: SimulateRequest):
    """ER queue simulation across policies."""
    st = _get_state()
    sim = ERQueueSimulator(st.net)
    return sim.compare_policies(
        arrival_rate_per_hour=req.arrival_rate,
        clinician_count=req.clinicians,
        replications=req.replications,
        horizon_hours=req.horizon_hours,
        policies=req.policies,
    )


# ─── Calibration ────────────────────────────────────────────────────
@router.get("/calibration")
async def calibration():
    """Held-out calibration evaluation: AUROC, Brier, ECE."""
    st = _get_state()
    evaluator = CalibrationEvaluator(st.net)
    # Use last 500 patients as held-out set
    holdout = st.df.tail(500)
    return evaluator.evaluate_model_on_heldout(holdout)


# ─── Audit / Override ────────────────────────────────────────────────
@router.post("/audit/override")
async def audit_override(req: OverrideRequest):
    """Log a human triage override."""
    override_id = log_override(
        patient_id=req.patient_id,
        recommended_level=req.recommended_level,
        chosen_level=req.chosen_level,
        reason=req.reason,
        evidence=req.evidence,
    )
    return {"status": "logged", "override_id": override_id}


@router.get("/audit/overrides")
async def get_overrides():
    """Get all logged overrides."""
    return get_all_overrides()


@router.get("/cases")
async def get_cases():
    """Returns the 6 demo cases with reference outputs."""
    st = _get_state()
    if isinstance(st.demo_cases, dict):
        return [{"id": k, **v} for k, v in st.demo_cases.items()]
    return st.demo_cases

