"""
Test suite covering BayesNet, Variable Elimination, Decision Recommender,
VOI, Queue Prioritization, and API endpoints against reference demo cases.
"""
import json
import pytest
import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import app, app_state
from backend.app.pgm.network import BayesNet
from backend.app.pgm.variable_elimination import VariableElimination
from backend.app.decision.utility import DecisionConfig
from backend.app.decision.triage import TriageRecommender
from backend.app.decision.voi import VOIAnalyzer
from backend.app.pgm.dseparation import DSeparationAnalyzer


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


def test_root(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "disclaimer" in res.json()


def test_get_model(client):
    res = client.get("/api/model")
    assert res.status_code == 200
    data = res.json()
    assert "Age" in data["nodes"]
    assert "Severity" in data["nodes"]
    assert len(data["edges"]) > 0


def test_get_cases(client):
    res = client.get("/api/cases")
    assert res.status_code == 200
    cases = res.json()
    assert len(cases) == 6


def test_dsep_rules(client):
    analyzer = DSeparationAnalyzer(app_state.net)
    
    # HeartRate ⊥ SpO2 | {} -> False
    res1 = analyzer.is_dseparated("HeartRate", "SpO2", [])
    assert res1["d_separated"] is False
    
    # HeartRate ⊥ SpO2 | {Condition, Severity} -> True
    res2 = analyzer.is_dseparated("HeartRate", "SpO2", ["Condition", "Severity"])
    assert res2["d_separated"] is True
    
    # HeartRate ⊥ SpO2 | {Severity} -> False
    res3 = analyzer.is_dseparated("HeartRate", "SpO2", ["Severity"])
    assert res3["d_separated"] is False
    
    # ECG_Abnormal ⊥ Troponin_High | {} -> False; | {Condition} -> True
    res4 = analyzer.is_dseparated("ECG_Abnormal", "Troponin_High", [])
    assert res4["d_separated"] is False
    res5 = analyzer.is_dseparated("ECG_Abnormal", "Troponin_High", ["Condition"])
    assert res5["d_separated"] is True
    
    # ArrivalMode ⊥ Condition | {Severity} -> True
    res6 = analyzer.is_dseparated("ArrivalMode", "Condition", ["Severity"])
    assert res6["d_separated"] is True


def test_decide_endpoint(client):
    # Test case E
    cases = client.get("/api/cases").json()
    case_e = next(c for c in cases if c["id"] == "E")
    payload = {"evidence": case_e["evidence"]}
    res = client.post("/api/decide", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "P1" in data["recommended_level"]
    assert data["p_severity"]["Critical"] > 0.8


def test_voi_endpoint(client):
    # Test case F (incomplete)
    cases = client.get("/api/cases").json()
    case_f = next(c for c in cases if c["id"] == "F_incomplete")
    payload = {"evidence": case_f["evidence"]}
    res = client.post("/api/voi", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "test_analyses" in data
    assert len(data["test_analyses"]) > 0
    assert data["recommended_test"] == "Lactate_High"


def test_queue_rank_endpoint(client):
    cases = client.get("/api/cases").json()
    patients = [
        {"patient_id": c["id"], "evidence": c["evidence"], "arrival_time": 0.0, "wait_minutes": 5.0}
        for c in cases[:3]
    ]
    res = client.post("/api/queue/rank", json={"patients": patients, "clinicians": 2})
    assert res.status_code == 200
    ranked = res.json()
    assert len(ranked["cmu_ranked"]) == 3
    assert len(ranked["fifo_ranked"]) == 3
    assert len(ranked["pcrit_ranked"]) == 3


def test_override_audit(client):
    payload = {
        "patient_id": "TEST_001",
        "recommended_level": "P2",
        "chosen_level": "P1",
        "reason": "Clinical intuition: patient looks unstable",
        "evidence": {"Age": "Elderly"}
    }
    res = client.post("/api/audit/override", json=payload)
    assert res.status_code == 200
    assert "override_id" in res.json()

    res_list = client.get("/api/audit/overrides")
    assert res_list.status_code == 200
    assert any(o["patient_id"] == "TEST_001" for o in res_list.json())
