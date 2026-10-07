"""
Database & Audit Logging: SQLite schema and operations.
Implements clinical audit table for human triage overrides.

Non-negotiable Clinical Safety Requirement:
  Every clinical recommendation can be overridden by a medical professional;
  all overrides are immutably logged with timestamp, reasoning, and context.
"""
from __future__ import annotations
import sqlite3
import json
import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

DB_FILE = os.environ.get("DB_FILE", "triage_audit.db")


def init_db(db_path: str = DB_FILE) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS overrides (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        recommended_level TEXT NOT NULL,
        chosen_level TEXT NOT NULL,
        reason TEXT NOT NULL,
        evidence_json TEXT
    );
    """)
    conn.commit()
    conn.close()


def log_override(
    patient_id: str,
    recommended_level: str,
    chosen_level: str,
    reason: str,
    evidence: Optional[Dict[str, Any]] = None,
    db_path: str = DB_FILE
) -> int:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    ts = datetime.now(timezone.utc).isoformat()
    ev_str = json.dumps(evidence or {})
    cursor.execute(
        """
        INSERT INTO overrides (patient_id, timestamp, recommended_level, chosen_level, reason, evidence_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (patient_id, ts, recommended_level, chosen_level, reason, ev_str)
    )
    override_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return int(override_id or 0)


def get_all_overrides(db_path: str = DB_FILE) -> List[Dict[str, Any]]:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM overrides ORDER BY id DESC")
    rows = cursor.fetchall()
    results = []
    for r in rows:
        results.append({
            "id": r["id"],
            "patient_id": r["patient_id"],
            "timestamp": r["timestamp"],
            "recommended_level": r["recommended_level"],
            "chosen_level": r["chosen_level"],
            "reason": r["reason"],
            "evidence": json.loads(r["evidence_json"] or "{}")
        })
    conn.close()
    return results
