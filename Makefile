# TriageIQ – ER Triage Decision-Support System
# One-command developer setup

PYTHON := .venv/bin/python
UVICORN := .venv/bin/uvicorn
NPM := npm
PYTEST := .venv/bin/pytest

.PHONY: dev backend frontend test data

dev: ## Start backend (port 8000) and frontend (port 5173) concurrently
	@echo "Starting backend and frontend…"
	@(PYTHONPATH=. $(UVICORN) backend.app.main:app --host 0.0.0.0 --port 8000 --reload &) && \
	(cd frontend && $(NPM) run dev) || true

backend: ## Start backend only
	PYTHONPATH=. $(UVICORN) backend.app.main:app --host 0.0.0.0 --port 8000 --reload

frontend: ## Start frontend only
	cd frontend && $(NPM) run dev

test: ## Run backend pytest
	PYTHONPATH=. $(PYTEST) backend/tests/ -v

data: ## Generate synthetic data (if needed)
	$(PYTHON) data/generate_data.py
