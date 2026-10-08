# TriageIQ — Triage Desk: In-Depth Technical Reference

> **Disclaimer:** This is an educational prototype. NOT clinically validated. Do NOT use for real patient care.

---

## Table of Contents

1. [What Is the Triage Desk?](#1-what-is-the-triage-desk)
2. [The Full Data Flow — End to End](#2-the-full-data-flow--end-to-end)
3. [The Bayesian Network Model](#3-the-bayesian-network-model)
4. [Evidence Collection & Discretization](#4-evidence-collection--discretization)
5. [Posterior Inference: Variable Elimination](#5-posterior-inference-variable-elimination)
6. [Decision Theory: Expected Utility](#6-decision-theory-expected-utility)
7. [Evidence Contributions (LOO Attribution)](#7-evidence-contributions-loo-attribution)
8. [Value of Information (VOI)](#8-value-of-information-voi)
9. [Risk Aversion — The What-If Slider](#9-risk-aversion--the-what-if-slider)
10. [The Recommendation Panel: What Every Number Means](#10-the-recommendation-panel-what-every-number-means)
11. [Override & Audit Logging](#11-override--audit-logging)
12. [The Pipeline Stepper & Explainability Drawer](#12-the-pipeline-stepper--explainability-drawer)
13. [Demo Cases: Walking Through Each Patient](#13-demo-cases-walking-through-each-patient)
14. [How Things Change: Sensitivity Examples](#14-how-things-change-sensitivity-examples)
15. [Architecture & Code Map](#15-architecture--code-map)

---

## 1. What Is the Triage Desk?

The **Triage Desk** is the primary clinical interface of TriageIQ. It is a real-time decision-support tool that helps an ED (Emergency Department) clinician assign a **triage priority level** to a patient arriving in the ER.

Rather than relying on a simple rule-based scorecard (e.g., "if HR > 100, escalate"), the Triage Desk uses a **Probabilistic Graphical Model (Bayesian Network)** to reason under uncertainty. Given whatever partial evidence is available about the patient — some vitals, symptoms, age, arrival mode — the system:

1. **Infers** a full posterior probability distribution over the patient's **Severity** (Low / Moderate / Critical) and **Condition** (Cardiac, Respiratory, Infection, Trauma, Minor).
2. **Decides** which triage level (P1 Immediate → P4 Non-Urgent) maximises **Expected Utility** given those beliefs.
3. **Explains** its reasoning in plain language, showing which evidence items most shifted the probability of a critical outcome.
4. **Advises** whether any additional diagnostic test (ECG, Troponin, Lactate, Chest X-ray) would change the decision enough to be worth its cost.
5. **Adjusts** in real-time as the clinician adds or removes evidence, and allows a **risk-aversion dial** to simulate how a more cautious approach changes the recommendation.

---

## 2. The Full Data Flow — End to End

```
Clinician input (React form)
        │
        ▼
 AssessmentState (12 possible fields)
        │   assessmentToEvidence()
        ▼
 Evidence dict  { "Age": "Elderly", "HeartRate": "Severe", … }
        │   debounced 250 ms → POST /api/decide + POST /api/voi
        ▼
 FastAPI backend
   ├─ BayesNet (loaded from model_spec.json at startup)
   ├─ VariableElimination.query(["Severity"], evidence)   → P(Severity | e)
   ├─ VariableElimination.query(["Condition"], evidence)  → P(Condition | e)
   ├─ TriageRecommender.recommend()   → EU by level, best level
   ├─ TriageRecommender.compute_evidence_contributions()  → LOO attribution
   └─ VOIAnalyzer.analyze()           → net VOI per test
        │
        ▼
  JSON response → React Query cache
        │
        ▼
 UI panels update:
   • Recommendation card  (best level, P(Critical), plain-language summary)
   • Severity Posterior   (probability bars: Low / Moderate / Critical)
   • Expected Utility     (bar chart per triage level)
   • Evidence Contributions (which vital moved the needle most)
   • VOI panel            (which test to order next, if any)
   • What-If slider       (re-runs with different risk-aversion λ)
```

---

## 3. The Bayesian Network Model

### 3.1 Structure

The model lives in `model_spec.json`. It is a **Directed Acyclic Graph (DAG)** with **18 nodes**:

| Node | Type | States |
|------|------|--------|
| `Age` | Demographics (root) | Child, Adult, Elderly |
| `History` | Demographics | None, Cardiac, Respiratory, Metabolic |
| `Condition` | Latent (hidden) | Cardiac, Respiratory, Infection, Trauma, Minor |
| `Severity` | Latent (hidden) | Low, Moderate, Critical |
| `ChestPain` | Symptom | No, Yes |
| `Dyspnea` | Symptom | No, Yes |
| `VisibleInjury` | Symptom | No, Yes |
| `HeartRate` | Vital | Normal, Abnormal, Severe |
| `SystolicBP` | Vital | Normal, Abnormal, Severe |
| `SpO2` | Vital | Normal, Abnormal, Severe |
| `RespRate` | Vital | Normal, Abnormal, Severe |
| `Temperature` | Vital | Normal, Abnormal |
| `Consciousness` | Clinical | Alert, Confused, Altered |
| `ArrivalMode` | Clinical | WalkIn, Ambulance |
| `ECG_Abnormal` | Diagnostic test | No, Yes |
| `Troponin_High` | Diagnostic test | No, Yes |
| `Lactate_High` | Diagnostic test | No, Yes |
| `ChestXray_Abnormal` | Diagnostic test | No, Yes |

### 3.2 Key Dependencies (Causal Structure)

```
Age ─────────────────► History
 │                        │
 └────────────────────────┤
                          ▼
                      Condition ──────────────────► Severity
                          │                             │
             ┌────────────┼──────────┐     ┌───────────┤
             ▼            ▼          ▼     ▼           ▼
         ChestPain    Dyspnea  VisibleInj HeartRate  SystolicBP
                                         SpO2  RespRate
                                         Temperature
                                         Consciousness
                                         ArrivalMode
                                         ECG_Abnormal
                                         Troponin_High
                                         Lactate_High
                                         ChestXray_Abnormal
```

The two key **latent (unobserved) nodes** are `Condition` and `Severity`. These are nodes that matter clinically but cannot be directly measured — they must be *inferred* from the observable evidence. Everything visible to the clinician is a "leaf" or "near-leaf" node acting as evidence.

### 3.3 Conditional Probability Tables (CPTs)

Every node has a **CPT** stored in `model_spec.json`. A CPT defines: "Given the state of my parents, what is the probability of each of my own states?"

**Example — HeartRate CPT (simplified):**
```json
{
  "Cardiac|Critical":  [0.05, 0.20, 0.75],
  "Cardiac|Moderate":  [0.20, 0.50, 0.30],
  "Cardiac|Low":       [0.70, 0.25, 0.05]
}
```

Reading this: if a patient has a Cardiac condition AND Critical severity, there is only a 5% chance their heart rate is Normal, and a **75% chance it is Severe**.

The model's joint distribution is the product of all CPTs:
```
P(X₁, X₂, …, X₁₈) = ∏ᵢ P(Xᵢ | Parents(Xᵢ))
```

### 3.4 Model Validation

On startup, `BayesNet.__init__()`:
- Runs **Kahn's algorithm** to verify the graph is a valid DAG (no directed cycles).
- Checks every CPT row sums to 1.0 (within tolerance 1e-4).
- Builds `Factor` objects (multi-dimensional NumPy arrays) for efficient computation.

---

## 4. Evidence Collection & Discretization

### 4.1 The Assessment Form

The React form (`EvidenceForm`) collects up to **12 evidence fields**. The clinician can provide any subset — missing fields remain **unobserved** and the model treats them as marginalized out.

| Field | UI Control | Network Variable |
|-------|-----------|-----------------|
| Age (years) | Number input | `Age` |
| Medical history | Dropdown | `History` |
| Chest pain | Yes/No toggle | `ChestPain` |
| Dyspnea | Yes/No toggle | `Dyspnea` |
| Visible injury | Yes/No toggle | `VisibleInjury` |
| Heart rate (bpm) | Number + checkbox | `HeartRate` |
| Systolic BP (mmHg) | Number + checkbox | `SystolicBP` |
| SpO₂ (%) | Number + checkbox | `SpO2` |
| Resp. rate (br/min) | Number + checkbox | `RespRate` |
| Temperature (°C) | Number + checkbox | `Temperature` |
| Consciousness | Dropdown | `Consciousness` |
| Arrival mode | Button group | `ArrivalMode` |

### 4.2 Vital Sign Discretization

The Bayesian network is **discrete** — it works with categories, not raw numbers. `vitals.ts` converts real measurements into discrete states:

```
Heart Rate (bpm):
  < 60 or > 120  →  "Severe"
  100–120        →  "Abnormal"
  60–100         →  "Normal"

Systolic BP (mmHg):
  < 90 or > 180  →  "Severe"
  120–180        →  "Abnormal"
  90–120         →  "Normal"

SpO₂ (%):
  < 90           →  "Severe"
  90–95          →  "Abnormal"
  ≥ 95           →  "Normal"

Respiratory Rate (br/min):
  < 12 or > 30   →  "Severe"
  20–30          →  "Abnormal"
  12–20          →  "Normal"

Temperature (°C):
  < 36 or > 38.5 →  "Abnormal"
  otherwise      →  "Normal"

Age (years):
  < 18           →  "Child"
  18–64          →  "Adult"
  ≥ 65           →  "Elderly"
```

> **Important:** An unchecked vital is treated as **unobserved** (not in evidence), which is fundamentally different from being Normal. The model has NO information about unobserved variables.

### 4.3 Evidence Transformation

`assessmentToEvidence()` converts the UI state into a flat dict matching the network's variable names:

```typescript
// Example output for Case E:
{
  "Age": "Elderly",
  "History": "Cardiac",
  "ChestPain": "Yes",
  "Dyspnea": "Yes",
  "HeartRate": "Severe",
  "SystolicBP": "Severe",
  "SpO2": "Severe",
  "RespRate": "Abnormal",
  "Consciousness": "Altered",
  "ArrivalMode": "Ambulance"
}
```

---

## 5. Posterior Inference: Variable Elimination

### 5.1 The Core Problem

Given evidence `e`, we want:
```
P(Severity | e)   and   P(Condition | e)
```

These are **posterior distributions** — updated beliefs after seeing evidence. Computing naively would require summing over all 18 variables' joint (billions of configurations). **Variable Elimination (VE)** does this efficiently.

### 5.2 How Variable Elimination Works

Implemented in `variable_elimination.py`:

**Step 1 — Barren Node Pruning**

Unobserved leaf nodes that are neither targets nor evidence are "barren" — their CPTs sum to 1 and cancel out. Pruning shrinks the computation. Example: if `ECG_Abnormal` is not observed and not queried, it is pruned before inference begins.

**Step 2 — Factor Reduction by Evidence**

Each CPT is "sliced" by fixing observed variables to their observed state. If `HeartRate = "Severe"` is observed, the HeartRate CPT is reduced to only the "Severe" row.

**Step 3 — Elimination Order (min-fill heuristic)**

Non-target, non-evidence variables must be summed out. The **min-fill** heuristic greedily selects the variable that would add the fewest edges to the interaction graph, minimizing intermediate factor sizes.

**Step 4 — Eliminate Variables One by One**

For each variable in the chosen order:
- Collect all factors that contain this variable.
- **Multiply** them (tensor product).
- **Marginalize** (sum out) that variable.
- Add the smaller resulting factor back to the pool.

**Step 5 — Multiply & Normalize**

Multiply all remaining factors. Normalize (divide by sum) to get a valid posterior.

### 5.3 Mathematical Formula

```
P(Severity = s | e) ∝ Σ_{non-target vars} ∏ᵢ P(Xᵢ | Parents(Xᵢ))
```

### 5.4 What You See

**Severity Posterior card** — three probability bars:
- **Low** — mild, self-resolving condition
- **Moderate** — needs prompt but not immediate care
- **Critical** — needs immediate life-saving intervention

**Example (Case E — Elderly cardiac arrest):**
```
Low:      0.0%  ░░░░░░░░░░░░░░
Moderate: 0.2%  ▓░░░░░░░░░░░░░
Critical: 99.8% ████████████████
```

The **Condition posterior** generates the plain-language summary: *"The model assigns the highest condition probability to cardiac (96.3%)."*

---

## 6. Decision Theory: Expected Utility

### 6.1 Four Triage Levels

| Level | Code | Meaning | Target Time |
|-------|------|---------|-------------|
| P1 | `P1_Immediate` | Life-threatening, immediate | Now |
| P2 | `P2_Urgent` | Serious, urgent | < 15 min |
| P3 | `P3_Standard` | Stable, standard | < 60 min |
| P4 | `P4_NonUrgent` | Minor, non-urgent | < 120 min |

### 6.2 The Utility Matrix

`U[level, severity]` captures the clinical consequence of assigning each triage level to a patient of each true severity. This is a **4 × 3 matrix**:

```
              Low      Moderate   Critical
P1_Immediate: -20.0    -6.0        0.0
P2_Urgent:     -8.0    -2.0      -25.0
P3_Standard:   -2.0   -15.0      -80.0
P4_NonUrgent:   0.0   -40.0     -100.0
```

**Reading this table:**
- `U[P1, Critical] = 0.0` — Correct! P1 for a Critical patient = zero loss.
- `U[P4, Critical] = -100.0` — Catastrophic under-triage. Penalty 100.
- `U[P1, Low] = -20.0` — Wasteful over-triage. Resources misused. Penalty 20.
- `U[P4, Low] = 0.0` — Correct! P4 for a Low-severity patient = zero loss.

**Clinical principle encoded:** under-triaging a critical patient is FAR worse than over-triaging a low-severity patient.

### 6.3 Computing Expected Utility

Given `P(Severity | e) = [P(Low), P(Moderate), P(Critical)]`:

```
EU(a) = P(Low) × U[a, Low]  +  P(Moderate) × U[a, Moderate]  +  P(Critical) × U[a, Critical]
```

This is a **weighted average** of outcomes — the model doesn't know the true severity, so it averages over all possibilities weighted by posterior probability.

**Full worked example — Case E (99.8% Critical):**
```
EU(P1_Immediate)  = 0.0×(-20) + 0.002×(-6)  + 0.998×(0.0)   = -0.012  ← BEST ✓
EU(P2_Urgent)     = 0.0×(-8)  + 0.002×(-2)  + 0.998×(-25)   = -24.954
EU(P3_Standard)   = 0.0×(-2)  + 0.002×(-15) + 0.998×(-80)   = -79.870
EU(P4_NonUrgent)  = 0.0×(0)   + 0.002×(-40) + 0.998×(-100)  = -99.880
```

P1_Immediate wins because -0.012 is the highest (least negative) EU.

### 6.4 The Decision Rule

```
best_level = argmax_a  EU(a | evidence)
```

This is **Expected Utility Maximization** — the gold-standard decision rule from von Neumann–Morgenstern decision theory. The system always picks the action that minimises expected harm.

### 6.5 What You See

The **Expected Utility by Triage Level** card shows all four levels as horizontal bars. The best level is highlighted with a blue ring. The formula `EU(level) = Σ P(severity | e) × U(level, severity)` is shown as a footer.

**Why some EUs are very negative and others barely negative:** The magnitude of EU depends entirely on the posterior. Near-certain cases (Case E: 99.8% Critical) have very high-confidence EUs. Ambiguous cases (Case D: 77% Low, 23% Moderate) have narrower spreads where the choice is much closer.

---

## 7. Evidence Contributions (LOO Attribution)

### 7.1 The Problem

The recommendation says "P1 Immediate." *Which piece of evidence is driving this?* The **evidence contributions** panel answers with a **Leave-One-Out (LOO) analysis**.

### 7.2 How It Works

For each observed variable `X = x`:

1. Compute `P(Critical | full evidence)` — the baseline (e.g., 99.77%).
2. Remove `X` from evidence, re-run VE: `P(Critical | evidence \ {X})`.
3. **Impact** = baseline − LOO posterior = *how much P(Critical) drops when we remove X*.

Additionally, a **Log Odds Ratio** is computed:
```
LOR = log( odds_baseline / odds_loo )
```
This is a statistically robust measure less distorted near 0% or 100%.

### 7.3 Interpretation

- **Impact > 0 (▲):** Observing X *raised* P(Critical). Example: `HeartRate=Severe` strongly suggests deterioration.
- **Impact < 0 (▼):** Observing X *lowered* P(Critical). Example: `Temperature=Normal` is a reassuring signal.

Contributions are sorted by absolute impact — most influential evidence shown first.

**What you see on screen (Case E):**
```
▲ SystolicBP = Severe      ΔP(Crit) +1.09%
▲ HeartRate = Severe       ΔP(Crit) +0.77%
▲ SpO2 = Severe            ΔP(Crit) +0.43%
▲ ArrivalMode = Ambulance  ΔP(Crit) +0.43%
```

> Note: The LOO impacts look small because P(Critical) is already near 100% — there isn't much room to move. The log odds ratios (shown in API response) are more meaningful in this regime.

---

## 8. Value of Information (VOI)

### 8.1 The Clinical Question

Should we order an additional test before triaging? The **VOI analysis** formalises this as: is the expected gain in decision quality worth the test's cost?

### 8.2 The Four Tests

| Test | Network Variable | States | Cost (utility units) |
|------|-----------------|--------|---------------------|
| ECG | `ECG_Abnormal` | No / Yes | 0.5 |
| Troponin | `Troponin_High` | No / Yes | 1.5 |
| Lactate | `Lactate_High` | No / Yes | 1.0 |
| Chest X-ray | `ChestXray_Abnormal` | No / Yes | 2.0 |

Only unobserved tests are analysed.

### 8.3 VOI Formula (Step by Step)

For test `T` with outcomes `o ∈ {No, Yes}`:

**1. Predict test outcome probabilities:**
```
P(T = o | current evidence)   ← Variable Elimination
```

**2. For each hypothetical outcome, find the new best EU:**
```
aug_evidence = current_evidence + {T: o}
best_EU_if_o = max_a  EU(a | aug_evidence)
```

**3. Expected posterior EU:**
```
E_T[max EU] = Σ_o  P(T=o | e) × best_EU_if_o
```

**4. Gross VOI:**
```
VOI(T) = max(0,  E_T[max EU]  −  current best EU)
```

**5. Net VOI (the decision):**
```
net_VOI(T) = VOI(T) − cost(T)
```
- `net_VOI > 0` → **Order the test. The expected gain outweighs cost.**
- `net_VOI ≤ 0` → **Skip the test. Decision is confident enough.**

### 8.4 EVPI — Expected Value of Perfect Information

The upper bound: if a magic oracle instantly revealed the true Severity, how much better could we do?
```
EVPI = Σ_s P(s | e) × max_a U[a, s]  −  current best EU
```
If EVPI ≈ 0, current evidence is sufficient. If EVPI is large, there is significant decision uncertainty.

### 8.5 What You See

**"Next Best Diagnostic Test" card:**
- A plain-language summary: *"No additional test has positive net VOI"* or *"Perform Lactate_High (net value +1.27)"*
- A ranked list of all tests with net VOI and cost.
- **Click any test** to expand its preposterior tree — showing "if the test came back Negative: P(outcome)=X% → Recommend Y with EU=Z" and vice versa for Positive.

**Demo case F example (partial evidence, 5 fields):**
- Lactate_High: net_VOI = **+1.27** → System recommends ordering Lactate!
- If Lactate Negative → P2_Urgent stays correct
- If Lactate Positive → switches to P1_Immediate

---

## 9. Risk Aversion — The What-If Slider

### 9.1 What It Does

The **What-If** card has a slider for **Risk Aversion λ** (0.00 to 2.00).

- **λ = 0 (default):** Risk-neutral. Uses the raw utility matrix.
- **λ > 0:** Risk-averse. Applies an **exponential utility transformation** that amplifies the penalty for catastrophic outcomes.

### 9.2 The Math: Arrow-Pratt Exponential Utility

```
U_eff(a, s) = −exp(−λ × U(a, s))
```

As λ increases:
- `U[P4, Critical] = -100` becomes `−exp(−1.5 × −100) = −exp(150) ≈ −10^65` — astronomically bad.
- Even a 1% chance of Critical severity now totally dominates the EU calculation.
- The system becomes more conservative, preferring P1 even in ambiguous cases.

**Practical example with Case D (77% Low, 23% Moderate, 0.03% Critical):**
- λ=0: Recommends P3_Standard (EU=-5.01 vs P2=-6.62)
- λ=1.5: The tiny 0.03% Critical probability, amplified exponentially, makes P3 look unacceptably risky → flips to P2_Urgent

### 9.3 Live Behaviour

Every slider change triggers a fresh `POST /api/decide` with the new `risk_aversion` value (250ms debounce). The entire right panel — recommendation, severity bars, EU chart — updates in real-time.

---

## 10. The Recommendation Panel: What Every Number Means

Complete annotation of the UI visible in the screenshot:

```
┌────────────────────────────────────────────────────────────────┐
│  RECOMMENDED                        P(Critical) = 99.78%  [B] │
│  ▲ P1 Immediate  [EU]                                          │
├────────────────────────────────────────────────────────────────┤
│  "The model assigns the highest condition probability to        │
│   cardiac (96.2%). Severity is very likely critical (99.8%     │
│   P(Critical)), with 0.2% moderate and 0.0% low."             │
│                                                                │
│  ▲ SystolicBP = Severe      ΔP(Crit) +1.09%                   │
│  ▲ HeartRate = Severe       ΔP(Crit) +0.77%                   │
│  ▲ ArrivalMode = Ambulance  ΔP(Crit) +0.43%                   │
│  ▲ SpO2 = Severe            ΔP(Crit) +0.43%                   │
│                                                                │
│  [Accept recommendation] [Override with reason] [Why this?]   │
└────────────────────────────────────────────────────────────────┘
```

| UI Element | What It Means |
|-----------|---------------|
| `▲ P1 Immediate` | The triage level with highest EU. Color-coded pills: red=P1, orange=P2, blue=P3, green=P4 |
| `[EU]` badge | Click for a tooltip explaining Expected Utility theory |
| `P(Critical) = 99.78%` | Posterior probability from Variable Elimination — how likely the patient is critically ill |
| `[B]` badge | Click for a tooltip explaining Bayesian inference |
| Plain-language summary | Generated by `summarizePosterior()` using thresholds: ≥85%="very likely", ≥60%="likely", ≥35%="possible", ≥10%="unlikely", else "very unlikely" |
| `▲ SystolicBP = Severe +1.09%` | LOO attribution: removing this observation would lower P(Critical) by 1.09 points |
| `Accept recommendation` | Records acknowledgment (UI state only, no backend call) |
| `Override with reason` | Opens override modal → logs to SQLite audit trail |
| `Why this?` | Expands inline explanation: evidence→beliefs→action, with EU margin over next-best option |

---

## 11. Override & Audit Logging

### 11.1 Why It Exists

Clinical AI must keep the human clinician in control. When the clinician disagrees with the AI, they must document why. This creates an **audit trail** for review and model improvement.

### 11.2 Override Flow

1. Click **"Override with reason"** → modal opens.
2. Select **chosen level** (what you're actually assigning).
3. Write **reason** (required free-text — cannot submit blank).
4. Click **"Log override"** → `POST /api/audit/override`:

```json
{
  "patient_id": "E",
  "recommended_level": "P1_Immediate",
  "chosen_level": "P2_Urgent",
  "reason": "Patient appears more stable than vitals suggest",
  "evidence": { "Age": "Elderly", "HeartRate": "Severe", "..." : "..." }
}
```

5. Persisted to **SQLite** (`triage_audit.db`) with timestamp.
6. Retrievable via `GET /api/audit/overrides`.

---

## 12. The Pipeline Stepper & Explainability Drawer

### 12.1 The Pipeline Stepper

At the bottom of the Triage Desk is the **"How this was computed"** pipeline showing 5 stages with live details:

```
① Evidence      → ② Bayesian Network → ③ Posterior (VE) → ④ Expected Utility → ⑤ Recommendation
  "9 of 12          "18 nodes"            "47 ms             "Best: P1_Immed."    "No test needed"
  fields"
```

- **Step 1:** Count of filled evidence fields out of 12 possible.
- **Step 2:** Node count from loaded model.
- **Step 3:** Actual round-trip inference time (end-to-end API latency).
- **Step 4:** Name of the winning triage level.
- **Step 5:** First 48 chars of the VOI summary.

### 12.2 The Drawer

Click any step → **slide-out drawer** appears with:
- Plain-English explanation of that stage.
- The exact mathematical formula.
- For Step 3: live values — current P(Critical) and P(Moderate).

| Step | Formula Shown |
|------|--------------|
| Evidence | `e = { observed variable → state }` |
| Bayesian Network | `P(X₁,…,Xₙ) = ∏ᵢ P(Xᵢ \| parents(Xᵢ))` |
| Posterior (VE) | `P(S \| e) = Σ_{C,…} P(S, C, …, e) / P(e)` |
| Expected Utility | `EU(a) = Σ_s P(s \| e) · U(a, s)` |
| Recommendation & VOI | `net_VOI = VOI(test) − cost(test)` |

---

## 13. Demo Cases: Walking Through Each Patient

Six pre-loaded cases in `demo_cases.json`, loaded via `GET /api/cases`. Case E loads by default.

### Case A — Adult Cardiac, Moderate-High Acuity
- **Profile:** Adult, cardiac history, chest pain + dyspnea, abnormal HR/BP/SpO2/RR, alert, ambulance
- **Posterior:** P(Critical)=29.1%, P(Moderate)=70.8%
- **Recommendation:** **P1_Immediate** (EU=-4.27 vs P2=-8.69)
- **Why P1 despite 70% Moderate?** The 29% chance of Critical with U[P2,Critical]=-25 vs U[P1,Critical]=0 tips the EU calculation firmly to P1. Expected harm from under-triage > over-triage cost.
- **VOI:** All tests negative net VOI — don't order anything.

### Case B — Child, Healthy
- **Profile:** Child, no history, no symptoms, all normal vitals, alert, walk-in
- **Posterior:** P(Low)=99.4%, P(Critical)≈0%
- **Recommendation:** **P4_NonUrgent** (EU=-0.22)
- **VOI:** No test recommended.

### Case C — Elderly Respiratory, Critical
- **Profile:** Elderly, respiratory history, dyspnea, severe SpO2 + RR, altered consciousness, ambulance
- **Posterior:** P(Critical)=93.3%
- **Recommendation:** **P1_Immediate** (EU=-0.40 vs P2=-23.5)
- **Key driver:** `Consciousness=Altered` is a very strong signal — heavily weights toward Critical in the network.

### Case D — Adult Metabolic, Low Acuity
- **Profile:** Adult, metabolic history, abnormal HR + temperature only, walk-in
- **Posterior:** P(Low)=77%, P(Moderate)=23%
- **Recommendation:** **P3_Standard** (EU=-5.01 vs P2=-6.62)
- **Margin is narrow:** Moving the risk aversion slider to λ>1 flips this to P2_Urgent.

### Case E — Elderly Critical Cardiac *(default)*
- **Profile:** Elderly, cardiac history, chest pain + dyspnea, ALL vitals severe, altered consciousness, ambulance
- **Posterior:** P(Critical)=99.77%
- **Recommendation:** **P1_Immediate** (EU=-0.014 — essentially zero expected loss!)
- **EVPI ≈ 0:** Case is so clear that no additional test could improve the decision. All tests have negative net VOI.

### Case F (Incomplete) — Elderly Cardiac, Partial Evidence
- **Profile:** Elderly, cardiac history, chest pain, abnormal HR only — just 5 of 12 fields filled
- **Posterior:** P(Critical)=16.2%, P(Moderate)=68.8%
- **Recommendation:** **P2_Urgent** (EU=-6.64 vs P1=-7.13 — very close!)
- **VOI:** Lactate_High net_VOI = **+1.27** → System recommends ordering Lactate before deciding!
  - If Lactate Normal (72% likely): P2_Urgent confirmed
  - If Lactate High (28% likely): switches to P1_Immediate

---

## 14. How Things Change: Sensitivity Examples

### 14.1 Adding Evidence to Case F

Start with Case F (5 fields). System recommends P2_Urgent, P(Critical)=16%.

Add SpO2=Severe:
- P(Critical) jumps to ~60%
- EU of P1 exceeds EU of P2
- System flips to **P1_Immediate**
- Lactate VOI drops (decision now more confident)

This demonstrates the real-time reactive inference: every field added immediately updates the posterior and recommendation.

### 14.2 Changing ArrivalMode

On any case, switching from Ambulance → WalkIn:
- `ArrivalMode` has a CPT conditioned on Condition.
- Walk-in patients are less likely to have severe Conditions (less ambulance-worthy presentations).
- P(Critical) decreases by a few percentage points.
- May or may not change the recommendation, depending on how close the EU margin is.

### 14.3 Risk Aversion Slider on Case D

Case D is currently P3_Standard with a narrow margin over P2.

Set λ = 1.5:
- `U_eff[P4, Critical]` = −exp(−1.5 × −100) ≈ −10^65
- Even the tiny 0.03% Critical probability becomes unbearably risky under this transform.
- System switches to **P2_Urgent** (more conservative).

Set λ = 0.5:
- Mild risk aversion — still P3_Standard.

This shows how the risk preference parameter models institutional policy (a more cautious hospital vs. a resource-constrained one).

### 14.4 Complete Evidence vs. Partial

The EU calculation becomes more confident (larger spread between levels) as more evidence is added. With no evidence, the system defaults to prior probabilities and the EU values cluster close together. As evidence accumulates, the posterior sharpens, and one level's EU clearly dominates.

---

## 15. Architecture & Code Map

```
files-2/
├── model_spec.json                  # BayesNet: 18 nodes, CPTs, decision config
├── demo_cases.json                  # 6 reference cases with ground-truth posteriors
├── patients_synthetic_5000.csv      # 5,000 synthetic patients for calibration/learning
│
├── backend/app/
│   ├── main.py                      # FastAPI app startup, loads BayesNet + data
│   ├── schemas.py                   # Pydantic request/response models
│   ├── db.py                        # SQLite audit trail (override logging)
│   │
│   ├── pgm/                         # Probabilistic Graphical Models
│   │   ├── factor.py                # Factor class: NumPy tensor multiply/marginalize/reduce/normalize
│   │   ├── network.py               # BayesNet: DAG validation, CPT building, joint probability
│   │   ├── variable_elimination.py  # CORE: exact VE inference (barren pruning, min-fill, traces)
│   │   ├── clique_tree.py           # Belief propagation (alternative exact inference)
│   │   ├── sampling.py              # Rejection / LW / Gibbs sampling
│   │   ├── map_inference.py         # MAP and MMAP inference
│   │   ├── dseparation.py           # D-separation tests and path analysis
│   │   ├── learning.py              # MLE + Bayesian parameter learning from data
│   │   ├── em.py                    # EM algorithm (hidden variables)
│   │   ├── structure.py             # Hill-climbing BIC structure learning
│   │   └── temporal.py             # Forward-backward filtering, Viterbi (time series)
│   │
│   ├── decision/                    # Decision Theory
│   │   ├── utility.py               # DecisionConfig: U matrix, test costs, risk aversion
│   │   ├── triage.py                # CORE: EU computation, argmax, LOO attribution
│   │   ├── voi.py                   # CORE: VOI, EVPI, preposterior trees
│   │   ├── priority.py              # Multi-patient c*mu queue ranking
│   │   └── sensitivity.py           # Breakeven P(Critical) curves
│   │
│   ├── eval/
│   │   └── calibration.py           # AUROC, Brier score, ECE on held-out data
│   │
│   ├── sim/
│   │   └── queue_sim.py             # M/M/c ER queue simulation across policies
│   │
│   └── api/
│       └── routes.py                # All FastAPI endpoints
│
└── frontend/src/
    ├── lib/
    │   ├── api.ts                   # Typed fetch wrappers for all endpoints
    │   ├── evidence-form.ts         # AssessmentState, assessmentToEvidence(), LOO helpers
    │   ├── vitals.ts                # Discretization: bpm/mmHg/% → Normal/Abnormal/Severe
    │   ├── triage-levels.ts         # Level constants, color mapping, pill classes
    │   └── narrate.ts               # Plain-language generators: summarizePosterior(), whyThisAction(), voiSummary()
    │
    ├── pages/
    │   └── TriageDeskPage.tsx       # MAIN: the entire Triage Desk page (this document's subject)
    │
    └── components/
        ├── domain/
        │   └── EvidenceForm.tsx     # Patient assessment input form
        └── ui/
            ├── Card.tsx             # Reusable card container with title/subtitle
            ├── Button.tsx           # Styled button (primary/secondary/ghost variants)
            ├── ProbBar.tsx          # Animated probability bar (severity posterior)
            ├── TriageLevelPill.tsx  # Color-coded level badge (P1=red, P2=orange, etc.)
            ├── Modal.tsx            # Override modal
            ├── Drawer.tsx           # Pipeline step detail drawer
            ├── Stepper.tsx          # 5-step pipeline progress component
            └── ConceptBadge.tsx     # Clickable theory tooltips ([EU], [B], [VOI])
```

### API Endpoints Used by the Triage Desk

| Endpoint | Method | When Called | Purpose |
|----------|--------|------------|---------|
| `GET /api/model` | GET | Once at startup | Load network topology and node count |
| `GET /api/cases` | GET | Once at startup | Load 6 demo patient cases |
| `POST /api/decide` | POST | Every 250ms debounce on evidence change | Posterior + EU + LOO attribution |
| `POST /api/voi` | POST | Every 250ms debounce on evidence change | VOI analysis for all unobserved tests |
| `POST /api/audit/override` | POST | On override submit | Log clinician override to SQLite |

### Real-Time Behaviour Details

- **250ms debounce:** Prevents flooding the backend while the clinician is still typing. The `useEffect` sets a timeout that clears on each change.
- **AbortController:** If new evidence arrives before the previous inference completes, the in-flight request is aborted. Prevents stale results overwriting fresh ones.
- **React Query caching:** `/api/model` and `/api/cases` are fetched once and cached indefinitely. Only `/api/decide` and `/api/voi` are re-fetched on evidence changes.
- **Optimistic loading state:** The `animate-pulse` text "Updating recommendation…" appears while inference is running, keeping the user informed.
- **Framer Motion pulse:** When a new recommendation arrives, the recommendation card pulses with a blue glow animation (respects `prefers-reduced-motion`).

---

## Summary: Three Foundational Principles

The Triage Desk is built on three ideas from **probabilistic AI and decision theory**:

### 1. Reasoning Under Uncertainty
Not "if HR > 100 then escalate" but "given all evidence, what is the full distribution over possible patient states?" This handles missing data gracefully, degrades smoothly with incomplete observations, and avoids brittle if-else rules.

### 2. Decision-Theoretic Optimality
Not "flag abnormal vitals" but "which action minimises expected harm given current beliefs?" The utility matrix encodes clinical consequences. Expected utility maximisation unifies beliefs (posterior) with consequences (utilities) into a principled action.

### 3. Explainability by Design
Not just "the answer is P1" but:
- **Which evidence drove this** (LOO attribution, ΔP(Critical))
- **How confident is the system** (P(Critical) in the header)
- **What the uncertainty is** (full posterior probability bars)
- **Whether more information would help** (VOI, net benefit of ordering a test)
- **Why this level beats alternatives** (EU comparison, "Why this?" panel)

These three together make TriageIQ a demonstration of how probabilistic AI can be applied transparently and responsibly in high-stakes clinical domains.
