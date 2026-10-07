# UI assumptions (TriageIQ frontend)

- All probabilities, expected utilities, VOI values, and queue ranks are fetched from the FastAPI backend; the UI does not fabricate clinical numbers.
- Vital numeric inputs are discretized on the client using adult thresholds documented in `data/model_spec.json → vital_clinical_meaning` (same simplification as MASTER_PROMPT).
- Demo cases load representative numeric vitals for display; inference uses the discretized bands, which match the demo case categorical evidence.
- "Accept recommendation" is an acknowledgment only (no audit API); overrides require a reason and call `POST /api/audit/override`.
- Pages marked as minimal (Policy Simulation, Model Evaluation charts, full Learning tabs) expose existing endpoints first; richer Recharts/React Flow experiences are incremental.
