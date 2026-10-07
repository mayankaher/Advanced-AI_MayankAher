# TriageIQ - ER Triage Decision-Support System

## Project Overview

TriageIQ is an advanced, AI-driven Clinical Decision Support System (CDSS) designed for Emergency Room (ER) environments. It leverages **Probabilistic Graphical Models (PGMs)**, specifically Bayesian Networks, and **Decision Theory** to assist triage nurses and medical professionals in accurately evaluating patient severity, recommending appropriate triage levels, and suggesting the most valuable next diagnostic tests.

Unlike traditional black-box machine learning models, TriageIQ provides complete transparency. Every probabilistic inference and clinical recommendation is mathematically rigorously derived and fully explainable, crucial for clinical environments where trust and auditability are paramount.

---

## Core Technologies

*   **Backend**: Python, FastAPI, `pgmpy` (for Bayesian Network modeling and inference), `pytest`.
*   **Frontend**: React 19, TypeScript, Vite, TailwindCSS, `react-router-dom`, `@xyflow/react` (for graph visualization), `recharts` (for data visualization).
*   **Architecture**: RESTful API communicating between a decoupled frontend and a modularized mathematical modeling backend.

---

## Key Functionalities & Modules

### 1. Probabilistic Graphical Model (PGM) Inference
At the core of the system is a custom-designed Bayesian Network that models the causal and correlational relationships between a patient's risk factors, symptoms, vitals, underlying conditions, and overall severity.
*   **Nodes**: Include evidence variables like Age, Medical History, Chest Pain, Heart Rate, and SpO2. Target variables include the patient's actual `Condition` (Cardiac, Respiratory, Trauma, etc.) and `Severity` (Low, Moderate, Critical).
*   **Inference Engine**: Using Variable Elimination algorithms, the system computes the exact posterior probabilities of `Severity` and `Condition` based on any combination of observed evidence.
*   **Explainability**: Users can see exactly *why* a patient is deemed Critical (e.g., "80% P(Critical) due to Abnormal Heart Rate and Yes to Chest Pain").

### 2. Maximum Expected Utility (MEU) Triage Decision
The system goes beyond simply predicting probabilities; it recommends actions based on clinical utilities.
*   **Triage Levels**: Maps to standard systems: P1 (Immediate), P2 (Urgent), P3 (Standard), P4 (Non-Urgent).
*   **Utility Matrix**: A predefined clinical utility matrix penalizes under-triaging (e.g., assigning a Critical patient to P4 is heavily penalized) and lightly penalizes over-triaging (wasting ER resources).
*   **Action Recommendation**: Calculates the Expected Utility (EU) for every possible triage level given the inferred probability distribution of the patient's severity. It recommends the triage level that maximizes the Expected Utility, mitigating clinical risk while optimizing ER resources.

### 3. Value of Information (VOI) Analysis
When a patient's presentation is ambiguous (e.g., moderate probabilities across different severities), the system dynamically recommends the most useful next step.
*   **Test Evaluation**: Evaluates potential subsequent tests (e.g., EKG, Lactate level, X-Ray).
*   **Net VOI Calculation**: Computes the expected increase in clinical utility if a specific test result were known, subtracting the inherent "cost" (time, resources, patient discomfort) of performing that test.
*   **Next Best Action**: If a test yields a positive Net VOI, the system surfaces it as a recommendation to clarify the clinical picture before making a final triage decision.

### 4. Dynamic Queue Management & Ranking
TriageIQ provides a dynamic, real-time dashboard for managing the ER waiting room.
*   **Algorithmic Sorting**: Patients are not just sorted by arrival time. They are ranked primarily by their recommended Triage Level (P1 > P2 > P3 > P4) and secondarily by their raw probability of being `Critical`.
*   **Continuous Updates**: As new vitals or test results are entered for a patient in the queue, their position dynamically shifts to reflect their updated clinical urgency.

### 5. Mathematical Transparency (d-Separation)
To build trust with clinicians, the system exposes the mathematical properties of the underlying Bayesian Network.
*   **d-Separation Queries**: Clinicians can query the system to understand conditional independencies. For example, the system can prove that "Chest Pain is conditionally independent of Medical History, *given* the underlying Condition." This confirms the model aligns with logical medical pathways.

### 6. Clinical Audit and Logging
Every decision made by the system is tracked for accountability.
*   **Decision Trails**: When a triage recommendation is generated, the system logs the exact evidence provided, the calculated probabilities, the expected utilities for each option, and the final recommendation.
*   **Override Tracking**: If a clinician chooses to override the AI's recommended triage level (e.g., AI suggests P2, clinician selects P1), the system records the override alongside a required justification, establishing a clear audit trail for quality assurance and model refinement.

---

## Project Structure

*   `backend/`
    *   `app/`: FastAPI application, route definitions (`api.py`), dependency injection.
    *   `modules/`: Core business logic.
        *   `pgm.py`: Bayesian Network definition, CPDs (Conditional Probability Distributions), and inference logic.
        *   `decision.py`: Expected utility calculations and VOI algorithms.
        *   `queue_manager.py`: Logic for ranking and sorting patient queues.
        *   `audit.py`: Logging and override tracking.
*   `frontend/`
    *   `src/`: React application source code.
        *   `components/`: Reusable UI components (Patient Cards, Charts, Forms).
        *   `pages/`: Main views (Dashboard/Queue, Triage Assessment, Case Studies).
        *   `lib/`: API clients, state management hooks, and narrative generation utilities.
*   `Makefile`: Developer orchestration for running the system locally and executing test suites.
