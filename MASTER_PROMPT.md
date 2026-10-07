# MASTER PROMPT: Probabilistic ER Triage & Prioritization Decision-Support System

> Paste this whole file into Antigravity as the project brief. Place the other four files from this kit
> (`model_spec.json`, `demo_cases.json`, `patients_synthetic_5000.csv`, `generate_data.py`) in `/data` of the workspace first.

---

## 0. ROLE AND MISSION

You are a senior ML engineer and full-stack developer building a **university course project for "Advanced AI"
(Probabilistic Graphical Models, Inference, Learning, Decision Theory)**. Build a complete, runnable, well-tested web
application: **a probabilistic decision-support system for Emergency Room (ER) triage**.

The system:
1. Takes a patient's age, history, symptoms, vitals and arrival mode (possibly **incomplete**).
2. Uses a **Bayesian Network** to infer P(Severity | evidence) and P(Condition | evidence).
3. Uses **decision theory** (expected utility) to recommend a triage level for one patient.
4. Uses **value of information** to recommend the next most useful diagnostic test.
5. Ranks **multiple waiting patients** under limited clinicians using a decision-theoretic priority index.
6. Demonstrates, via simulation, that the probabilistic policy beats FIFO and a rule-based baseline.

**Primary goal: academic demonstration of the syllabus topics with a real-world framing.** Every concept must be
*visible and explainable in the UI*, not hidden in a library call. A professor must be able to open any tab and
see the concept working.

### Non-negotiable framing and safety
- Everything is **clinical decision SUPPORT, not an autonomous medical system.** A permanent banner on every page:
  "Educational prototype trained on synthetic data. Not clinically validated. Not for real patient care."
- A human can always **override** the recommendation; overrides are logged with a reason (SQLite audit table).
- All data is **synthetic**. Never claim clinical accuracy. Never include real patient data.

---

## 1. FILES PROVIDED (in `/data`)

| File | Content |
|---|---|
| `model_spec.json` | Complete ground-truth Bayesian network: all nodes, states, parents, CPTs (key format `"ParentState1\|ParentState2"`), utility matrix, harm rates, service times, transition matrix, test costs, clinical meaning of each discretized vital. **This is the source of truth. Load it; do not re-invent numbers.** |
| `demo_cases.json` | 6 demo patients with evidence and **reference outputs** (posteriors, EU, priority index, VOI) computed by an independent enumeration implementation. Use them as regression tests. |
| `patients_synthetic_5000.csv` | 5000 patients sampled from the ground-truth model, with the latent labels `Condition` and `Severity` included. Used for parameter learning, structure learning, calibration and EM. |
| `generate_data.py` | The generator and reference enumeration code. Reuse/adapt it for `backend/app/data/simulator.py`. Seeded (42). |

If any file is missing, **stop and tell me**; do not fabricate replacements.

---

## 2. TECH STACK (fixed)

- **Backend:** Python 3.11+, FastAPI, Pydantic v2, NumPy, pandas, SciPy, SQLite (via SQLModel or sqlite3), pytest.
  `pgmpy` is allowed **only** for (a) cross-checking your own code in tests and (b) structure learning. Check the installed pgmpy
  version first: class names changed across versions (`BayesianNetwork` vs `DiscreteBayesianNetwork`).
- **Frontend:** React + Vite + TypeScript + Tailwind CSS, Recharts for charts, React Flow for the network graph,
  TanStack Query for API calls.
- **One-command run:** `make dev` (or `docker compose up`) starts backend on :8000 and frontend on :5173. Provide `make test`, `make data`.
- **Important:** The core algorithms (factors, Variable Elimination, sampling, MAP, d-separation, MLE/Bayesian estimation, EM,
  filtering, expected utility, VOI) **must be implemented from scratch in NumPy** in `backend/app/pgm/`. This is a coursework
  requirement: the code itself is graded. pgmpy results are used only as a correctness oracle in tests.

### Repo layout
```
er-triage/
  Makefile  docker-compose.yml  README.md
  data/            (the 4 provided files)
  backend/
    app/
      main.py  api/  (routers per module)
      pgm/    factor.py  network.py  variable_elimination.py  sampling.py  map_inference.py
              dseparation.py  learning.py  em.py  structure.py  temporal.py
      decision/ utility.py  triage.py  voi.py  priority.py  sensitivity.py
      sim/    patient_generator.py  queue_sim.py  policies.py
      eval/   metrics.py  calibration.py
      db.py  schemas.py
    tests/
  frontend/  src/ pages/ components/ lib/ 
  docs/  architecture.md  syllabus_mapping.md  viva_script.md
```

---

## 3. DOMAIN MODEL (load from `model_spec.json`)

**Variables and states**

| Node | States | Parents |
|---|---|---|
| Age | Child, Adult, Elderly | none |
| History | None, Cardiac, Respiratory, Metabolic | Age |
| Condition (latent) | Cardiac, Respiratory, Infection, Trauma, Minor | Age, History |
| Severity (latent) | Low, Moderate, Critical | Age, Condition |
| ChestPain, Dyspnea, VisibleInjury | No, Yes | Condition |
| HeartRate, SystolicBP, SpO2, RespRate, Temperature | Normal, Abnormal, Severe | Condition, Severity |
| Consciousness | Alert, Altered, Unresponsive | Condition, Severity |
| ArrivalMode | WalkIn, Ambulance | Severity |
| ECG_Abnormal, Troponin_High, ChestXray_Abnormal | No, Yes | Condition (test nodes, **unobserved by default**) |
| Lactate_High | No, Yes | Severity (test node, unobserved by default) |

**Design rationale to document in the README:** *Condition* explains which symptoms appear; *Severity* explains how abnormal
the vitals are; vitals depend on both because the same severity looks different for a respiratory vs trauma patient. Test nodes
are modelled as noisy sensors of a latent variable so VOI can be computed.

**Reading the evidence form:** Users enter numeric vitals; the UI discretizes them with the thresholds in
`model_spec.json → vital_clinical_meaning` (adult thresholds; document the simplification for children). Every field can be
left as "Not measured": missing evidence simply means that node is not conditioned on. The form must show a live
"evidence completeness" indicator.

**Decision model** (all in `model_spec.json → decision`):
- 4 triage levels: P1 Immediate, P2 Urgent, P3 Standard, P4 NonUrgent.
- Utility matrix `U[level][true severity]` (higher = better, 0 = ideal; editable in the UI).
- `harm_rate_per_min_by_severity` = utility lost per minute of waiting.
- `service_minutes_by_condition` = clinician time required.
- `severity_transition_per_10min` and `adverse_event_hazard_per_10min_by_severity` = deterioration while waiting.
- Test costs in utility units.

---

## 4. MODULES: DETAILED SPECIFICATION

### M1. Factor and network engine (`pgm/factor.py`, `network.py`)
- `Factor(vars, cards, values)` with **multiply, marginalize (sum-out), reduce (evidence), normalize, maximize (max-out)**, all using
  `np.einsum` or broadcasting. Maintain variable-ordering correctness.
- `BayesNet` loaded from JSON: topological order, parents/children, CPT→Factor, joint probability of a full assignment, `ancestral_sample()`.
- Validate on load: every CPT row sums to 1 (±1e-6), DAG acyclic.

### M2. Exact inference: Variable Elimination (`variable_elimination.py`)
- `query(target_vars, evidence, elimination_order=None)` → posterior Factor.
- Implement elimination-order heuristics: min-neighbors, min-fill, and user-supplied order. Prune **barren nodes** (unobserved leaves)
  before elimination. Document why this is valid.
- Return a **trace**: at each step, the variable eliminated, factors multiplied, resulting factor scope and size. Expose to the UI.
- Also implement **clique tree / junction tree** construction (moralize, triangulate, maximal cliques, tree) and **sum-product belief propagation**
  (two-pass message passing) so all marginals come from one calibration. Show the clique tree in the UI and verify it matches VE.

### M3. Approximate inference (`sampling.py`)
- **Rejection sampling**, **likelihood weighting**, **Gibbs sampling** (with burn-in and thinning; Markov blanket sampling), all producing
  posterior estimates over Severity and Condition given evidence, with effective sample size.
- Convergence experiment endpoint: for N in {100, 300, 1k, 3k, 10k, 30k, 100k}, run each method ≥20 times; return mean and std of
  KL(exact || estimate) and max absolute error. UI plots KL vs N on log axes for all three methods.
- Explain in the UI why rejection sampling collapses with many observed variables (report acceptance rate).

### M4. MAP inference (`map_inference.py`)
- MAP over (Condition, Severity) jointly via max-product variable elimination with backtracking.
- Show the contrast: **joint MAP vs. the product of marginal MAPs** (they can differ). Provide an example from the demo cases or
  construct one. Also expose top-k most probable explanations.

### M5. Independence and I-maps (`dseparation.py`)
- Implement d-separation (Bayes-ball or reachability algorithm). UI panel: pick X, Y and a conditioning set Z; show True/False **and the
  active/blocked paths with the reason** (chain/fork/collider, observed or not).
- Verify empirically: for a few queries, estimate mutual information from the 5000-row dataset and show it is ≈0 when d-separated.
- Required test truths (verify against pgmpy too):
  - HeartRate ⊥ SpO2 | {} → **False** (common parent)
  - HeartRate ⊥ SpO2 | {Condition, Severity} → **True**
  - HeartRate ⊥ SpO2 | {Severity} → **False** (path via Condition remains open)
  - ECG_Abnormal ⊥ Troponin_High | {} → **False**; | {Condition} → **True**
  - ArrivalMode ⊥ Condition | {Severity} → **True**
  - Age ⊥ Condition | {} → **False**
- Also show I-map discussion: the graph is an I-map of the ground truth distribution; show that adding spurious edges keeps it an I-map but loses minimality.

### M6. Parameter learning (`learning.py`, `em.py`)
- **Maximum likelihood** estimation from counts. **Bayesian estimation** with Dirichlet priors (BDeu with equivalent sample size α, user-adjustable).
- Experiment 1 (fully observed): train on n ∈ {20, 50, 100, 300, 1000, 5000} rows; plot **average KL between learned and true CPTs** for MLE vs Bayesian;
  show Bayesian wins at small n (zero-count problem).
- Experiment 2 (**partially observed / hidden variables**): hide `Condition` and `Severity` from the data, initialise CPTs from a perturbed "expert" model,
  run **EM**; plot log-likelihood per iteration (must be monotone non-decreasing) and final CPT error. Explain label-switching and initialization sensitivity.
- **Expert-vs-learned** comparison: create the "expert model" = ground-truth with logit-noise σ=0.5 on every CPT row (seeded). Show that learning from data
  repairs expert errors, evaluated on held-out patients (log-likelihood, AUROC for Critical, calibration).
- **Shared parameters** demo: tie the five vital CPTs' *severity effect* with a shared "severity-shift" parameterization
  (as in template models), learn it, and compare parameter count vs untied model.

### M7. Structure learning (`structure.py`)
- Score-based search: hill-climbing with **BIC** (own implementation of the decomposable score with add/delete/reverse moves and acyclicity check),
  restarts, and tabu list. Run on the fully observed 5000 rows.
- Report **Structural Hamming Distance** vs true graph, edges recovered/missing/extra, and score vs n. Show the learned graph beside the true graph,
  coloring edges (correct/missing/extra). Optionally cross-check with pgmpy's `HillClimbSearch`.
- Note the real-world caveat: latent variables make structure learning harder; show what happens when Condition is hidden.

### M8. Temporal reasoning (`temporal.py`)
Patients wait. Model **severity as a hidden state evolving over time**:
- Hidden state S_t ∈ {Low, Moderate, Critical}, transition matrix per 10-minute step from `model_spec.json`, observation model = the vital/symptom CPTs.
- Implement **forward filtering** (belief update), **smoothing** (forward-backward), and **Viterbi**. Re-triage readings arrive every 15–30 minutes.
- UI: a timeline per patient showing belief P(Critical) over time with markers for new vitals; an "alert" when P(Critical) crosses a threshold. Include a
  button "simulate deterioration" that injects worsening vitals.

### M9. Decision theory for one patient (`decision/triage.py`)
- EU(level) = Σ_s P(s | evidence) · U[level][s]. Recommend argmax. Show the full EU table and a bar chart.
- **Sensitivity analysis:** vary P(Critical) 0→1 and show the regions where each level is optimal (decision thresholds); also vary one utility entry and show where
  the recommendation flips. Report the **break-even probabilities** analytically.
- **Utility editing:** the user edits the matrix; the API validates (monotonicity warnings) and recomputes.
- **Risk attitude extension:** add an optional risk-averse utility (e.g., exponential transform) and show how the decision changes.

### M10. Value of Information (`decision/voi.py`)
- For each unobserved test T: **VOI(T) = Σ_o P(o|e) · max_a EU(a | e, o) − max_a EU(a | e)**. Net value = VOI − cost(T).
- Recommend the test with highest **positive** net value, else "No additional test is worth it: proceed with recommended level".
- Display the preposterior analysis: for each outcome o, show P(o|e), updated posterior and updated best action, so the logic is transparent.
- Also compute **expected value of perfect information** about Severity as an upper bound on any test.
- Reference expectation: for case `F_incomplete`, Lactate has clearly positive net VOI; for fully observed critical patients, all tests have net VOI < 0.

### M11. Multi-patient prioritization (`decision/priority.py`)
Resource-constrained queueing with k clinicians. For each waiting patient i with posterior (Severity, Condition):
- `priority_i = E[harm_rate | e_i] / E[service_minutes | e_i]` (the **cμ rule**: serve highest expected delay-cost per unit service time first; optimal
  for a single server with linear waiting costs, and a strong heuristic for k servers).
- Return ranked list with rank, P(Critical), recommended level, EU, priority index, and an **explanation of why** (e.g. "ranks above Patient E despite lower
  P(Critical) because expected service time is shorter").
- Also provide a **P(Critical)-only ranking** for comparison and show where the two orderings differ. This is an important viva talking point.
- Support patient aging in queue: waiting time updates each patient's belief via the temporal model (M8) so priority changes dynamically.

### M12. Policy simulation (`sim/queue_sim.py`)
Discrete-event simulation of one ER:
- Patients generated from the ground-truth model with arrival rate λ (default 7/hour, Poisson), k=3 clinicians (configurable), service time = Condition-specific mean
  with exponential or lognormal noise. Simulate 24h × ≥200 replications with common random numbers across policies.
- Policies: **(a) FIFO**, **(b) Rule-based ESI-style baseline** (e.g. any Severe vital or Unresponsive → P1; ≥2 Abnormal vitals → P2; else by arrival), **(c) BN + cμ (ours)**,
  **(d) Oracle** (knows true Severity; upper bound).
- While waiting, true severity evolves by the transition matrix; adverse events occur with the hazard table. Metrics: mean and 95th-percentile wait for true-Critical patients, adverse
  events per 100 patients, total utility lost, and utilization. Report mean ± 95% CI and paired differences.
- UI: load sweep (λ from 3 to 12/hr): plot adverse events vs load for each policy. Expected pattern: policies converge at low load and diverge at high load.
  **Report what you actually observe**; do not tune parameters to force a win. If BN+cμ does not beat the baseline, report that honestly and investigate.

### M13. Explanation and calibration
- **Evidence contribution:** for the current patient, show per-evidence log-likelihood-ratio contribution to P(Critical) (leave-one-out posterior change), as a ranked bar chart.
- **Calibration:** on a held-out synthetic set of 2000 patients, compute AUROC, Brier score, reliability diagram, and expected calibration error for P(Critical).
  Compare the learned model vs the expert model vs the rule-based baseline.
- **Under-triage rate:** fraction of truly Critical patients assigned below P1; show per policy.

---

## 5. BACKEND API (FastAPI, JSON, OpenAPI docs at /docs)

| Method + path | Purpose |
|---|---|
| `GET /api/model` | Nodes, states, edges, CPTs, utility config |
| `PUT /api/model/utility` | Update utility matrix / harm rates / service times (validated) |
| `POST /api/infer` | body `{evidence, method: "ve"\|"bp"\|"rejection"\|"lw"\|"gibbs", n_samples, targets}` → posteriors (+VE trace, ESS) |
| `POST /api/map` | MAP / top-k explanations |
| `POST /api/dsep` | `{x, y, z}` → bool + path analysis |
| `POST /api/decide` | evidence → posterior, EU per level, best level, evidence contributions |
| `POST /api/voi` | evidence → VOI table, net VOI, preposterior details, recommendation |
| `POST /api/queue/rank` | patients[], clinicians → ranked list with explanations |
| `POST /api/filter` | time-stamped observations → belief trajectory (filter/smooth) |
| `POST /api/learn` | `{method: "mle"\|"bayes"\|"em", n, alpha, hide_latents}` → CPT errors, EM trace |
| `POST /api/structure` | `{score:"bic", n}` → learned DAG, SHD, edge diff |
| `POST /api/convergence` | sampler convergence experiment |
| `POST /api/simulate` | policy simulation config → metrics |
| `GET /api/calibration` | held-out evaluation metrics |
| `POST /api/audit/override` | log a human override (patient, recommended, chosen, reason) |
| `GET /api/cases` | demo cases |

Use Pydantic schemas with strict state validation (reject unknown states with a clear error). Cache expensive experiments (keyed by params+seed). Long experiments: run in
a background task and stream progress (SSE or polling).

---

## 6. FRONTEND: PAGES AND BEHAVIOUR

Global: dark/light theme, permanent disclaimer banner, responsive, loading/error states, accessible colors (never color-only meaning), keyboard-friendly forms.

1. **Triage Desk** (single patient)
   - Form: age, history, symptoms (toggles), vitals (numeric inputs auto-discretized, each with "Not measured"), arrival mode. "Load demo case" dropdown (A–F).
   - Results: severity and condition probability bars (animated), **recommended triage level** card with EU of all 4 levels, evidence-contribution chart,
     "What-if" sliders that re-query live (debounced), **Recommended next test** card from VOI (with net-value numbers and the "why"), and Override button (reason required).
2. **ER Queue Board**
   - Board of waiting patients (add/remove, import the 6 demo cases), clinician count control, simulated clock with play/pause/speed.
   - Live ranked list with priority index, P(Critical), level, wait time; belief trend sparkline; highlight when ranking changes and explain why.
   - Toggle to compare **BN+cμ ranking vs P(Critical)-only vs FIFO** side by side.
3. **Network & Learning**
   - React Flow graph of the BN (nodes colored by role: demographic / latent / symptom / vital / test); click a node to see its CPT as a heatmap table with parents.
   - Parameter learning lab: n slider, MLE vs Bayesian (α slider), error-vs-n plot; EM panel with log-likelihood curve.
   - Structure learning panel: true vs learned graph with edge diff and SHD.
4. **Inference Lab**
   - Method selector (VE / BP / rejection / LW / Gibbs), evidence picker, results table vs exact.
   - VE step-by-step trace viewer (stepper UI), clique tree visualization, convergence plot (KL vs N, log-log), MAP vs marginal-MAP comparison, d-separation explorer.
5. **Decision & VOI**
   - Editable utility matrix, decision-region plot over P(Critical), break-even table, risk-aversion slider.
   - VOI explorer: pick a patient, see preposterior tree per test.
6. **Policy Simulation**
   - Configure λ, clinicians, horizon, replications, policies; results table with CIs; load-sweep line chart; wait-time distributions for critical patients.
7. **Evaluation**
   - Calibration curve, AUROC, Brier, under-triage rates (learned vs expert vs rule-based).
8. **About & Syllabus Map**
   - Table mapping every syllabus topic → where in the app → file/function implementing it. Disclaimer, limitations, ethics notes.

---

## 7. REFERENCE VALUES (regression tests: from `demo_cases.json`; tolerance 1e-3 for exact inference)

| Case | Description | P(Critical) | P(Mod) | Best level | Priority index |
|---|---|---|---|---|---|
| A | Adult, cardiac hx, chest pain, dyspnea, mildly abnormal vitals, ambulance | 0.2906 | 0.7083 | P1_Immediate | 0.00942 |
| B | Child, healthy, all normal, walk-in | 0.0000 | 0.0056 | P4_NonUrgent | 0.00092 |
| C | Elderly, respiratory hx, SpO2 and RR severe, altered, ambulance | 0.9333 | 0.0667 | P1_Immediate | 0.02810 |
| D | Adult, metabolic hx, fever + tachycardia only, walk-in | 0.0003 | 0.2301 | P3_Standard | 0.00165 |
| E | Elderly, cardiac hx, chest pain, severe HR/BP/SpO2, altered, ambulance | 0.9977 | 0.0023 | P1_Immediate | 0.02413 |
| F_incomplete | Elderly, cardiac hx, chest pain, abnormal HR, nothing else measured | 0.1624 | 0.6875 | P2_Urgent | 0.00619 |

Expected **priority order by cμ index: C, E, A, F, D, B.** Note this differs from the P(Critical)-only order (E, C, A, F, D, B) because Respiratory cases have a
shorter expected service time than Cardiac cases. This is a feature to showcase, not a bug. Net VOI check: F_incomplete → Lactate_High has net VOI ≈ +1.27;
case B → all tests negative.

Prior (no evidence): P(Low, Moderate, Critical) ≈ (0.513, 0.334, 0.152).

**Rule for the agent:** if your implementation disagrees with these numbers, **your code is wrong until proven otherwise**. Cross-check with brute-force enumeration over the
(Age, History, Condition, Severity) core (valid because unobserved leaves are barren, see `generate_data.py`). Never edit the CPTs or reference values to make tests pass.

---

## 8. TESTING REQUIREMENTS (pytest, must all pass; run in CI via `make test`)

1. Factor algebra: multiply/marginalize/reduce against brute-force numpy on random factors.
2. **VE == brute-force joint enumeration** (tol 1e-9) on 50 random evidence sets; **VE == BP (clique tree)**; VE == pgmpy (tol 1e-6) if pgmpy is installed.
3. Demo-case reference values match (Section 7).
4. Sampling: LW and Gibbs with 200k samples within KL < 0.005 of exact on every demo case; rejection sampling acceptance rate reported; all seeded.
5. MAP: joint MAP equals brute-force argmax; includes a constructed case where joint MAP ≠ marginal MAP, or documented proof that none exists in this model.
6. d-separation truths from M5 hold; agree with pgmpy `is_dconnected`.
7. Learning: MLE equals empirical frequencies; Bayesian with α→0 converges to MLE; EM log-likelihood is non-decreasing (assert every iteration); learned CPT KL decreases with n (on average).
8. Structure learning: BIC score is decomposable (score difference of a move equals the local change); acyclicity is never violated.
9. Decision: EU argmax correct; decision thresholds satisfy U-linear break-even formula; VOI ≥ 0 before costs for every test (information never hurts in expectation); EVPI ≥ every test VOI.
10. Temporal: filtering matches brute-force HMM forward computation; smoothing probabilities sum to 1; Viterbi equals brute-force best path on short sequences.
11. API: schema validation, unknown-state errors, response latency < 200 ms for single-patient inference.
12. Simulation: with common random numbers and λ small, all policies give statistically equal results (sanity); with seed fixed, results are reproducible.

Frontend: component tests for evidence-form discretization and a Playwright smoke test that loads each page and runs one query.

---

## 9. DOCUMENTATION DELIVERABLES

- `README.md`: install, run, test, screenshots, architecture diagram, **how each module maps to a syllabus topic**, model assumptions and limitations.
- `docs/syllabus_mapping.md` with this exact table filled with file paths:

| Syllabus topic | Where implemented |
|---|---|
| Probability theory, Bayes nets | M1, `pgm/network.py` |
| Independence, I-maps | M5, `pgm/dseparation.py` |
| Local models / template (shared parameters) | M6 shared-parameter demo |
| Variable Elimination, clique trees, belief propagation | M2 |
| Approximate inference: sampling, Markov chains | M3 (LW, rejection, Gibbs) |
| MAP inference | M4 |
| Inference in temporal models | M8 |
| Parameter estimation, Bayesian estimation | M6 |
| Structure learning and search | M7 |
| Partially observed data, EM, hidden variables | M6 Experiment 2 |
| Causality | `docs/` + UI note: distinguish "observing" from "intervening" (e.g. P(Critical \| see ambulance) vs P(Critical \| do(ambulance))); implement a small `do()` operator by graph surgery |
| Utility functions, expected utility, decision problems | M9 |
| Value of information | M10 |
| Sequential decision making | M11/M12 (queue policy), M8 (re-triage over time) |

- `docs/viva_script.md`: a 10-minute demo script with exact clicks/inputs, expected outputs, and 12 likely viva questions with answers
  (e.g., "Why is the graph an I-map?", "Why does cμ rank C above E?", "Why does Bayesian estimation help at small n?", "What does EM do with the latent Severity?",
  "Why can VOI never be negative?", "What are the limitations of synthetic data?").
- A short "Limitations & ethics" section: synthetic data, no real validation, discretization loss, independence assumptions among vitals given (Condition, Severity),
  utility values are illustrative and political/ethical in real life, automation bias risk, need for human oversight.

---

## 10. BUILD ORDER AND ACCEPTANCE GATES

Work in phases. **After each phase, run the tests, show me the results, and commit.** Do not start the next phase while the current gate fails.

1. **Scaffold** (repo, Makefile, FastAPI hello, Vite app, lint, CI). Gate: `make dev` runs both.
2. **M1+M2 + tests** (factors, VE, trace, BP/clique tree). Gate: demo-case values match; VE==enumeration==BP.
3. **M9+M10** (EU decision, sensitivity, VOI) + `/api/decide`, `/api/voi`. Gate: Section 7 values incl. VOI checks.
4. **Triage Desk UI** end to end. Gate: load case E and F in the browser, correct numbers shown.
5. **M3, M4, M5** (sampling, MAP, d-sep) + Inference Lab UI. Gate: convergence plot and d-sep truths.
6. **M6, M7** (learning, EM, structure) + Network & Learning UI. Gate: EM monotone; error-vs-n plots.
7. **M8, M11** (temporal + queue ranking) + ER Queue Board. Gate: order C,E,A,F,D,B; belief timeline works.
8. **M12, M13** (simulation, calibration) + Policy/Evaluation UIs. Gate: reproducible results; honest reporting.
9. **Causality demo, docs, viva script, polish, full test run, Playwright smoke.**

## 11. RULES OF ENGAGEMENT

- **Verify by running.** Execute tests and the app yourself; paste real outputs. Never claim something works without running it.
- **No fake results.** Do not hardcode outputs, CPTs, plots or metrics. All numbers shown in the UI come from computation.
- **Report deviations honestly**, including if a policy simulation does not show the expected advantage.
- **Keep it explainable:** each module has a docstring stating the algorithm, its complexity, and the syllabus topic it implements; inline comments for non-obvious steps.
- **Reproducibility:** all randomness seeded via a single config; seeds shown in the UI for experiments.
- Prefer clarity over cleverness. Type hints everywhere in Python and strict TypeScript.
- If a requirement is ambiguous, make the most reasonable assumption, **write it down in `docs/assumptions.md`**, and continue. Ask me only for blocking issues.
- Final answer: summary of what was built, test results, how to run, known limitations, and a list of assumptions made.

## 12. DEFINITION OF DONE

All 8 pages work; `make test` is green; all Section 7 reference values reproduced; every syllabus topic in the mapping table is demonstrable in the UI;
README and viva script complete; the disclaimer is visible everywhere; the demo script runs end to end without errors.
