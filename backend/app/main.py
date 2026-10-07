"""
ER Triage Decision-Support System: FastAPI Application.
Loads the Bayesian Network from model_spec.json and demo cases,
mounts all API routers, and serves the application.

DISCLAIMER: Educational prototype trained on synthetic data.
            Not clinically validated. Not for real patient care.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.pgm.network import BayesNet
from backend.app.db import init_db
from backend.app.api.routes import router as api_router

# Resolve data directory (support both /data and project-root files)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
if not DATA_DIR.exists():
    DATA_DIR = BASE_DIR  # fallback: files at project root

MODEL_SPEC_PATH = DATA_DIR / "model_spec.json"
DEMO_CASES_PATH = DATA_DIR / "demo_cases.json"
CSV_PATH = DATA_DIR / "patients_synthetic_5000.csv"


# Global shared state
class AppState:
    net: BayesNet = None  # type: ignore
    demo_cases: list = []
    df: pd.DataFrame = None  # type: ignore
    model_spec: dict = {}


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and data on startup."""
    # Load model spec
    with open(MODEL_SPEC_PATH, "r") as f:
        spec = json.load(f)
    app_state.model_spec = spec
    app_state.net = BayesNet(spec)

    # Load demo cases
    with open(DEMO_CASES_PATH, "r") as f:
        app_state.demo_cases = json.load(f)

    # Load CSV data
    app_state.df = pd.read_csv(CSV_PATH)

    # Initialize SQLite audit DB
    init_db()

    print(f"✓ Loaded BayesNet with {len(app_state.net.nodes)} nodes")
    print(f"✓ Loaded {len(app_state.demo_cases)} demo cases")
    print(f"✓ Loaded {len(app_state.df)} synthetic patients")

    yield


app = FastAPI(
    title="ER Triage Decision-Support System",
    description=(
        "Probabilistic Graphical Model-based ER triage and prioritization. "
        "Educational prototype – NOT for real clinical use."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")


@app.get("/")
async def root():
    return {
        "message": "ER Triage Decision-Support System",
        "disclaimer": "Educational prototype trained on synthetic data. Not clinically validated. Not for real patient care.",
        "docs": "/docs",
    }
