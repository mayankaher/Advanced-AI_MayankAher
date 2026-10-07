"""
ER Triage Decision-Support: ground-truth model, spec export, synthetic data, sanity checks.
Run:  python generate_data.py
Outputs: model_spec.json, patients_synthetic_5000.csv, demo_cases.json
Pure numpy (no pgmpy needed). Seeded -> reproducible.
"""
import json, csv, itertools
import numpy as np

SEED = 42
rng = np.random.default_rng(SEED)

def norm(v):
    v = np.asarray(v, float); return v / v.sum()

# ---------------- State spaces ----------------
AGE   = ["Child", "Adult", "Elderly"]
HIST  = ["None", "Cardiac", "Respiratory", "Metabolic"]
COND  = ["Cardiac", "Respiratory", "Infection", "Trauma", "Minor"]
SEV   = ["Low", "Moderate", "Critical"]
YN    = ["No", "Yes"]
V3    = ["Normal", "Abnormal", "Severe"]
CONS  = ["Alert", "Altered", "Unresponsive"]
ARR   = ["WalkIn", "Ambulance"]
VITALS = ["HeartRate", "SystolicBP", "SpO2", "RespRate", "Consciousness", "Temperature"]
SYMPT  = ["ChestPain", "Dyspnea", "VisibleInjury"]
TESTS  = ["ECG_Abnormal", "Troponin_High", "Lactate_High", "ChestXray_Abnormal"]

STATES = {"Age":AGE,"History":HIST,"Condition":COND,"Severity":SEV,"ArrivalMode":ARR}
for n in SYMPT+TESTS: STATES[n]=YN
for n in VITALS: STATES[n]= CONS if n=="Consciousness" else V3

# ---------------- CPTs ----------------
P_age = norm([0.15, 0.55, 0.30])
P_hist_given_age = np.array([[.80,.02,.15,.03],[.55,.12,.13,.20],[.25,.35,.15,.25]])

cond_base = np.array([.12,.15,.15,.18,.40])
age_mult  = np.array([[.1,1.2,1.3,1.2,1.2],[1,1,1,1.1,1],[2.5,1.5,1.6,.9,.6]])
hist_mult = np.array([[1,1,1,1,1],[4,1,1,1,.7],[1,4,1.5,1,.7],[1.8,1,2,1,.8]])
P_cond = np.zeros((3,4,5))
for a in range(3):
    for h in range(4):
        P_cond[a,h] = norm(cond_base*age_mult[a]*hist_mult[h])

sev_base = np.array([[.30,.45,.25],[.40,.45,.15],[.40,.45,.15],[.45,.42,.13],[.95,.049,.001]])  # by Condition
age_sev  = np.array([[1.1,1,.8],[1,1,1],[.7,1,1.6]])                                          # by Age
P_sev = np.zeros((3,5,3))   # [Age, Condition, Sev]
for a in range(3):
    for c in range(5):
        P_sev[a,c] = norm(sev_base[c]*age_sev[a])

sympt_yes = {  # P(Yes | Condition) in COND order
 "ChestPain":     [.85,.15,.05,.20,.05],
 "Dyspnea":       [.55,.90,.30,.15,.05],
 "VisibleInjury": [.01,.01,.02,.90,.06]}
P_sympt = {n: np.array([[1-p,p] for p in sympt_yes[n]]) for n in SYMPT}

vital_base = {  # by Severity: [Normal, Abnormal, Severe]
 "HeartRate":     [[.85,.14,.01],[.45,.45,.10],[.15,.40,.45]],
 "SystolicBP":    [[.88,.11,.01],[.55,.38,.07],[.20,.35,.45]],
 "SpO2":          [[.92,.07,.01],[.55,.35,.10],[.20,.35,.45]],
 "RespRate":      [[.88,.11,.01],[.50,.40,.10],[.15,.40,.45]],
 "Consciousness": [[.95,.045,.005],[.80,.17,.03],[.40,.35,.25]],
 "Temperature":   [[.88,.11,.01],[.70,.26,.04],[.60,.30,.10]]}
vital_boost = {  # multiplier on Abnormal/Severe states by Condition (COND order)
 "HeartRate":     [1.3,1.2,1.4,1.5,1.0],
 "SystolicBP":    [1.5,0.9,1.5,2.0,1.0],
 "SpO2":          [1.2,2.5,1.1,1.0,1.0],
 "RespRate":      [1.0,2.5,1.5,1.2,1.0],
 "Consciousness": [1.1,1.1,1.2,1.5,1.0],
 "Temperature":   [0.8,1.0,3.0,0.7,1.0]}
P_vital = {}
for n in VITALS:
    t = np.zeros((5,3,3))   # [Condition, Severity, state]
    for c in range(5):
        for s in range(3):
            w = np.array(vital_base[n][s], float)
            w[1:] *= vital_boost[n][c]
            t[c,s] = norm(w)
    P_vital[n] = t

arr_yes = [.05,.25,.75]   # P(Ambulance | Severity)
P_arr = np.array([[1-p,p] for p in arr_yes])

test_yes_cond = {"ECG_Abnormal":[.85,.15,.10,.10,.05],
                 "Troponin_High":[.75,.08,.10,.05,.02],
                 "ChestXray_Abnormal":[.30,.85,.45,.35,.05]}
test_yes_sev  = {"Lactate_High":[.05,.25,.75]}
P_test = {n: np.array([[1-p,p] for p in v]) for n,v in test_yes_cond.items()}
P_test["Lactate_High"] = np.array([[1-p,p] for p in test_yes_sev["Lactate_High"]])
TEST_PARENT = {"ECG_Abnormal":"Condition","Troponin_High":"Condition","ChestXray_Abnormal":"Condition","Lactate_High":"Severity"}
TEST_COST   = {"ECG_Abnormal":0.5,"Troponin_High":1.5,"Lactate_High":1.0,"ChestXray_Abnormal":2.0}  # utility units

# ---------------- Decision model ----------------
LEVELS = ["P1_Immediate","P2_Urgent","P3_Standard","P4_NonUrgent"]
U = np.array([[-20,-6,0],[-8,-2,-25],[-2,-15,-80],[0,-40,-100]], float)  # [level, true severity]
harm_rate = np.array([0.01,0.08,0.60])           # utility lost per minute of waiting, by Severity
service_min = np.array([25,20,20,30,10])         # minutes of clinician time, by Condition
# severity drift per 10-min step while waiting untreated
TRANS = np.array([[.97,.029,.001],[.02,.93,.05],[0.0,.04,.96]])
ADVERSE_HAZARD = np.array([.0002,.002,.03])      # per 10-min step

# ---------------- Inference by enumeration over core (valid: unobserved leaves are barren) ----------------
def likelihood_terms(ev, a, h, c, s):
    p = 1.0
    for n,val in ev.items():
        i = STATES[n].index(val)
        if n in SYMPT: p *= P_sympt[n][c,i]
        elif n in VITALS: p *= P_vital[n][c,s,i]
        elif n=="ArrivalMode": p *= P_arr[s,i]
        elif n in TESTS: p *= P_test[n][c if TEST_PARENT[n]=="Condition" else s, i]
    return p

def core_posterior(ev):
    ev = {k:v for k,v in ev.items() if k not in ("Age","History")} | {}
    return ev

def posterior(ev):
    """returns dict with joint over core and marginals for Severity and Condition"""
    J = np.zeros((3,4,5,3))
    for a,h,c,s in itertools.product(range(3),range(4),range(5),range(3)):
        if "Age" in ev and AGE[a]!=ev["Age"]: continue
        if "History" in ev and HIST[h]!=ev["History"]: continue
        pr = P_age[a]*P_hist_given_age[a,h]*P_cond[a,h,c]*P_sev[a,c,s]
        J[a,h,c,s] = pr*likelihood_terms({k:v for k,v in ev.items() if k not in("Age","History")},a,h,c,s)
    Z = J.sum()
    J /= Z
    return {"Z":Z,"sev":J.sum((0,1,2)),"cond":J.sum((0,1,3)),"joint_cs":J.sum((0,1))}

def expected_utility(psev): return U @ psev
def priority_index(post):
    return float(post["sev"]@harm_rate) / float(post["cond"]@service_min)

def voi(ev, test):
    base = posterior(ev); eu0 = expected_utility(base["sev"]).max()
    tot = 0.0
    for o in YN:
        e2 = dict(ev); e2[test]=o
        pe = posterior(e2)
        p_o = pe["Z"]/base["Z"]
        tot += p_o*expected_utility(pe["sev"]).max()
    return tot-eu0

# ---------------- Sampling ----------------
def sample_patient(rng):
    a = rng.choice(3,p=P_age); h = rng.choice(4,p=P_hist_given_age[a])
    c = rng.choice(5,p=P_cond[a,h]); s = rng.choice(3,p=P_sev[a,c])
    row = {"Age":AGE[a],"History":HIST[h],"Condition":COND[c],"Severity":SEV[s]}
    for n in SYMPT: row[n]=YN[rng.choice(2,p=P_sympt[n][c])]
    for n in VITALS: row[n]=STATES[n][rng.choice(3,p=P_vital[n][c,s])]
    row["ArrivalMode"]=ARR[rng.choice(2,p=P_arr[s])]
    return row

# ---------------- Export spec ----------------
def table_to_dict(arr, parents):
    out = {}
    dims = arr.shape[:-1]
    for idx in itertools.product(*[range(d) for d in dims]):
        key = "|".join(STATES[p][i] for p,i in zip(parents, idx))
        out[key] = [round(float(x),6) for x in arr[idx]]
    return out

nodes = {}
nodes["Age"]      = {"states":AGE,"parents":[],"cpt":{"":[float(x) for x in P_age]}}
nodes["History"]  = {"states":HIST,"parents":["Age"],"cpt":table_to_dict(P_hist_given_age,["Age"])}
nodes["Condition"]= {"states":COND,"parents":["Age","History"],"cpt":table_to_dict(P_cond,["Age","History"])}
nodes["Severity"] = {"states":SEV,"parents":["Age","Condition"],"cpt":table_to_dict(P_sev,["Age","Condition"])}
for n in SYMPT:  nodes[n]={"states":YN,"parents":["Condition"],"cpt":table_to_dict(P_sympt[n],["Condition"])}
for n in VITALS: nodes[n]={"states":STATES[n],"parents":["Condition","Severity"],"cpt":table_to_dict(P_vital[n],["Condition","Severity"])}
nodes["ArrivalMode"]={"states":ARR,"parents":["Severity"],"cpt":table_to_dict(P_arr,["Severity"])}
for n in TESTS:
    par = TEST_PARENT[n]
    nodes[n]={"states":YN,"parents":[par],"cpt":table_to_dict(P_test[n],[par]),"observed_by_default":False,"cost":TEST_COST[n]}

spec = {
 "version":1,"seed":SEED,
 "note":"Ground-truth/expert-reference model. Synthetic, for education only - NOT clinically validated.",
 "nodes":nodes,
 "decision":{"levels":LEVELS,"utility_matrix_rows_levels_cols_severity":U.tolist(),"severity_order":SEV,
             "harm_rate_per_min_by_severity":harm_rate.tolist(),
             "service_minutes_by_condition":dict(zip(COND,service_min.tolist())),
             "severity_transition_per_10min":TRANS.tolist(),
             "adverse_event_hazard_per_10min_by_severity":ADVERSE_HAZARD.tolist(),
             "test_cost_utility_units":TEST_COST},
 "vital_clinical_meaning":{
   "HeartRate":"Normal 60-100 | Abnormal 40-59 or 101-130 | Severe <40 or >130 (adult; scale for age)",
   "SystolicBP":"Normal 100-140 | Abnormal 90-99 or 141-180 | Severe <90 or >180",
   "SpO2":"Normal >=95 | Abnormal 90-94 | Severe <90",
   "RespRate":"Normal 12-20 | Abnormal 21-29 or 8-11 | Severe >=30 or <8",
   "Consciousness":"Alert | Altered (verbal/confused) | Unresponsive (pain/unresponsive)",
   "Temperature":"Normal 36.0-37.9C | Abnormal 38.0-39.4C or <36 | Severe >=39.5C or <35"}}
json.dump(spec, open("model_spec.json","w"), indent=1)

# ---------------- Synthetic dataset ----------------
rows=[sample_patient(rng) for _ in range(5000)]
with open("patients_synthetic_5000.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# ---------------- Demo cases ----------------
cases = {
 "A":{"Age":"Adult","History":"Cardiac","ChestPain":"Yes","Dyspnea":"Yes","VisibleInjury":"No","HeartRate":"Abnormal","SystolicBP":"Abnormal","SpO2":"Abnormal","RespRate":"Abnormal","Consciousness":"Alert","Temperature":"Normal","ArrivalMode":"Ambulance"},
 "B":{"Age":"Child","History":"None","ChestPain":"No","Dyspnea":"No","VisibleInjury":"No","HeartRate":"Normal","SystolicBP":"Normal","SpO2":"Normal","RespRate":"Normal","Consciousness":"Alert","Temperature":"Normal","ArrivalMode":"WalkIn"},
 "C":{"Age":"Elderly","History":"Respiratory","ChestPain":"No","Dyspnea":"Yes","VisibleInjury":"No","HeartRate":"Abnormal","SystolicBP":"Normal","SpO2":"Severe","RespRate":"Severe","Consciousness":"Altered","Temperature":"Abnormal","ArrivalMode":"Ambulance"},
 "D":{"Age":"Adult","History":"Metabolic","ChestPain":"No","Dyspnea":"No","VisibleInjury":"No","HeartRate":"Abnormal","SystolicBP":"Normal","SpO2":"Normal","RespRate":"Normal","Consciousness":"Alert","Temperature":"Abnormal","ArrivalMode":"WalkIn"},
 "E":{"Age":"Elderly","History":"Cardiac","ChestPain":"Yes","Dyspnea":"Yes","VisibleInjury":"No","HeartRate":"Severe","SystolicBP":"Severe","SpO2":"Severe","RespRate":"Abnormal","Consciousness":"Altered","Temperature":"Normal","ArrivalMode":"Ambulance"},
 "F_incomplete":{"Age":"Elderly","History":"Cardiac","ChestPain":"Yes","HeartRate":"Abnormal","ArrivalMode":"WalkIn"},
}
out = {}
for k,ev in cases.items():
    p = posterior(ev); eu = expected_utility(p["sev"])
    out[k] = {"evidence":ev,
      "reference_P_severity":dict(zip(SEV,[round(float(x),4) for x in p["sev"]])),
      "reference_P_condition":dict(zip(COND,[round(float(x),4) for x in p["cond"]])),
      "reference_EU_by_level":dict(zip(LEVELS,[round(float(x),3) for x in eu])),
      "reference_best_level":LEVELS[int(eu.argmax())],
      "reference_priority_index":round(priority_index(p),5),
      "reference_VOI_by_test":{t:round(voi(ev,t),4) for t in TESTS},
      "reference_net_VOI_by_test":{t:round(voi(ev,t)-TEST_COST[t],4) for t in TESTS}}
json.dump(out, open("demo_cases.json","w"), indent=1)

# ---------------- Sanity checks ----------------
for n,d in nodes.items():
    for k,v in d["cpt"].items(): assert abs(sum(v)-1)<1e-4,(n,k)
prior_sev = sum(P_age[a]*P_hist_given_age[a,h]*P_cond[a,h,c]*P_sev[a,c] for a,h,c in itertools.product(range(3),range(4),range(5)))
print("Prior P(Severity):", dict(zip(SEV,np.round(prior_sev,3))))
for k,v in out.items():
    print(k, v["reference_P_severity"], v["reference_best_level"], "prio", v["reference_priority_index"])
    print("   net VOI:", v["reference_net_VOI_by_test"])
order = sorted(out, key=lambda k:-out[k]["reference_priority_index"])
print("Priority order:", order)
