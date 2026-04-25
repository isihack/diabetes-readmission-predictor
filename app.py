import pickle
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Readmission Predictor",
    page_icon="🏥",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 2rem; }

    .risk-card {
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .risk-high   { background: rgba(239,68,68,0.12);  border: 1px solid rgba(239,68,68,0.4); }
    .risk-mod    { background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.4); }
    .risk-low    { background: rgba(34,197,94,0.12);  border: 1px solid rgba(34,197,94,0.4); }

    .risk-high  .risk-val { color: #ef4444; }
    .risk-mod   .risk-val { color: #f59e0b; }
    .risk-low   .risk-val { color: #22c55e; }

    .risk-val   { font-size: 3rem; font-weight: 800; }
    .risk-label { font-size: 1.1rem; color: #94a3b8; margin-top: 4px; }

    .metric-row {
        display: flex; gap: 12px; margin-top: 1rem;
    }
    .metric-box {
        flex: 1; background: #1e2240; border: 1px solid #2e3250;
        border-radius: 12px; padding: 1rem; text-align: center;
    }
    .metric-box .val { font-size: 1.5rem; font-weight: 700; }
    .metric-box .name { font-size: 0.72rem; color: #94a3b8;
                        text-transform: uppercase; letter-spacing: 0.05em; }

    .info-box {
        background: #1a1d27; border: 1px solid #2e3250;
        border-radius: 12px; padding: 1rem 1.2rem;
        font-size: 0.82rem; color: #94a3b8; line-height: 1.8;
    }
    .info-box b { color: #e2e8f0; }

    .section-header {
        font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.08em; color: #4f8ef7;
        border-bottom: 1px solid #2e3250; padding-bottom: 6px;
        margin-bottom: 14px; margin-top: 8px;
    }

    div[data-testid="stMetric"] {
        background: #1a1d27;
        border: 1px solid #2e3250;
        border-radius: 10px;
        padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model artifacts ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = Path(__file__).parent / "pickle-files"
    try:
        scaler = pickle.load(open(base / "scaler.sav",          "rb"))
        model  = pickle.load(open(base / "best_classifier.pkl", "rb"))
        return scaler, model
    except FileNotFoundError:
        return None, None

scaler, model = load_artifacts()

# ── Feature constants ─────────────────────────────────────────────────────────
TOP_10 = ["UNK","InternalMedicine","Emergency/Trauma","Family/GeneralPractice",
          "Cardiology","Surgery-General","Nephrology","Orthopedics",
          "Orthopedics-Reconstructive","Radiologist"]

COLS_NUM = ["time_in_hospital","num_lab_procedures","num_procedures","num_medications",
            "number_outpatient","number_emergency","number_inpatient","number_diagnoses"]

COLS_CAT = ["race","gender","max_glu_serum","A1Cresult","metformin","repaglinide",
            "nateglinide","chlorpropamide","glimepiride","acetohexamide","glipizide",
            "glyburide","tolbutamide","pioglitazone","rosiglitazone","acarbose","miglitol",
            "troglitazone","tolazamide","insulin","glyburide-metformin","glipizide-metformin",
            "glimepiride-pioglitazone","metformin-rosiglitazone","metformin-pioglitazone",
            "change","diabetesMed","payer_code"]

COLS_CAT_NUM = ["admission_type_id","discharge_disposition_id","admission_source_id"]

AGE_MAP = {"[0-10)":0,"[10-20)":10,"[20-30)":20,"[30-40)":30,"[40-50)":40,
           "[50-60)":50,"[60-70)":60,"[70-80)":70,"[80-90)":80,"[90-100)":90}

MED_OPTIONS = ["No","Steady","Up","Down"]

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(inputs: dict) -> np.ndarray:
    row = inputs.copy()
    row["admission_type_id"]        = str(row["admission_type_id"])
    row["discharge_disposition_id"] = str(row["discharge_disposition_id"])
    row["admission_source_id"]      = str(row["admission_source_id"])
    row["med_spec"] = row["medical_specialty"] if row["medical_specialty"] in TOP_10 else "Other"

    df = pd.DataFrame([row])
    df_cat = pd.get_dummies(df[COLS_CAT + COLS_CAT_NUM + ["med_spec"]], drop_first=True)
    df["age_group"]  = df["age"].map(AGE_MAP).fillna(50)
    df["has_weight"] = 0  # weight not collected in UI

    full = pd.concat([df[COLS_NUM], df_cat, df[["age_group","has_weight"]]], axis=1)

    if hasattr(scaler, "feature_names_in_"):
        full = full.reindex(columns=scaler.feature_names_in_, fill_value=0)

    return scaler.transform(full.values)

# ═════════════════════════════════════════════════════════════════════════════
# UI
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("# 🏥 Diabetes Readmission Predictor")
st.markdown(
    "Predicts the probability of a diabetic patient being readmitted within **30 days**. "
    "Best model: **Gradient Boosting** · Test AUC **0.666** · Recall **57.6%**"
)

if model is None:
    st.error(
        "⚠️ Model files not found. "
        "Place `scaler.sav` and `best_classifier.pkl` in the `pickle-files/` folder.",
        icon="🚨"
    )
    st.stop()

st.divider()

# ── Sidebar — input form ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 Patient Details")

    st.markdown('<div class="section-header">Demographics</div>', unsafe_allow_html=True)
    race   = st.selectbox("Race", ["Caucasian","AfricanAmerican","Hispanic","Asian","Other","UNK"])
    gender = st.selectbox("Gender", ["Female","Male"])
    age    = st.selectbox("Age Group", list(AGE_MAP.keys()), index=5)

    st.markdown('<div class="section-header">Admission</div>', unsafe_allow_html=True)
    admission_type_id        = st.selectbox("Admission Type",
        [1,2,3,4,5,6,7,8],
        format_func=lambda x: {1:"Emergency",2:"Urgent",3:"Elective",4:"Newborn",
                                5:"Not Available",6:"NULL",7:"Trauma",8:"Not Mapped"}[x])
    discharge_disposition_id = st.selectbox("Discharge Disposition",
        [1,2,3,6,7,18,22,25],
        format_func=lambda x: {1:"Home",2:"Care/Nursing Fac.",3:"SNF",6:"Home Health",
                                7:"AMA",18:"Unknown",22:"Rehab Facility",25:"Not Mapped"}[x])
    admission_source_id      = st.selectbox("Admission Source",
        [1,2,4,5,7,9,17],
        format_func=lambda x: {1:"Physician Referral",2:"Clinic Referral",
                                4:"Transfer from Hospital",5:"Transfer from SNF",
                                7:"Emergency Room",9:"Not Available",17:"NULL"}[x],
        index=4)
    medical_specialty = st.selectbox("Medical Specialty",
        ["UNK","InternalMedicine","Emergency/Trauma","Family/GeneralPractice",
         "Cardiology","Surgery-General","Nephrology","Orthopedics",
         "Orthopedics-Reconstructive","Radiologist","Other"])

    st.markdown('<div class="section-header">Visit Counts</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        time_in_hospital   = st.number_input("Days in Hospital",   1, 14, 3)
        num_lab_procedures = st.number_input("Lab Procedures",      0, 200, 41)
        num_procedures     = st.number_input("Procedures",          0, 6, 1)
        number_diagnoses   = st.number_input("Diagnoses",           1, 20, 7)
    with col2:
        num_medications    = st.number_input("Medications",         0, 100, 15)
        number_outpatient  = st.number_input("Outpatient Visits",   0, 50, 0)
        number_emergency   = st.number_input("Emergency Visits",    0, 50, 0)
        number_inpatient   = st.number_input("⭐ Inpatient Visits", 0, 50, 0,
                                              help="Strongest predictor of readmission")

    st.markdown('<div class="section-header">Lab Results</div>', unsafe_allow_html=True)
    max_glu_serum = st.selectbox("Max Glucose Serum", ["None","Norm",">200",">300"])
    A1Cresult     = st.selectbox("HbA1c Result",      ["None","Norm",">7",">8"])

    st.markdown('<div class="section-header">Medications</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        metformin = st.selectbox("Metformin",  MED_OPTIONS)
        insulin   = st.selectbox("Insulin",    MED_OPTIONS)
        glipizide = st.selectbox("Glipizide",  MED_OPTIONS)
    with col4:
        glyburide   = st.selectbox("Glyburide",    MED_OPTIONS)
        change      = st.selectbox("Med Change",   ["No","Ch"])
        diabetesMed = st.selectbox("On Diabetes Medication", ["Yes","No"])

    predict_btn = st.button("🔍 Predict Readmission Risk", use_container_width=True, type="primary")

# ── Main area ─────────────────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="large")

with right:
    st.markdown("### 📊 Model Performance")
    m1, m2 = st.columns(2)
    m1.metric("Test AUC",    "0.666")
    m2.metric("Recall",      "57.6%")
    m3, m4 = st.columns(2)
    m3.metric("Specificity", "66.4%")
    m4.metric("Prevalence",  "11.4%")

    st.markdown("""
    <div class="info-box" style="margin-top:1rem">
        <b>Algorithm:</b> Gradient Boosting Classifier<br>
        <b>Hyperparams:</b> n_estimators=100, max_depth=3, lr=0.1<br>
        <b>Tuning:</b> RandomizedSearchCV (20 iterations, 2-fold CV)<br>
        <b>Dataset:</b> 99,343 diabetic patient encounters<br>
        <b>Top feature:</b> number_inpatient ⭐
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔑 Key Risk Factors")
    factors = {
        "Prior inpatient visits":   95,
        "Discharge disposition":    78,
        "Emergency visits":         65,
        "Number of diagnoses":      58,
        "Time in hospital":         52,
        "Insulin dosage change":    44,
        "Number of medications":    38,
    }
    for name, importance in factors.items():
        st.progress(importance / 100, text=f"{name}  ({importance}%)")

with left:
    st.markdown("### 🎯 Prediction Result")

    if predict_btn:
        inputs = {
            "time_in_hospital":         time_in_hospital,
            "num_lab_procedures":       num_lab_procedures,
            "num_procedures":           num_procedures,
            "num_medications":          num_medications,
            "number_outpatient":        number_outpatient,
            "number_emergency":         number_emergency,
            "number_inpatient":         number_inpatient,
            "number_diagnoses":         number_diagnoses,
            "race":                     race,
            "gender":                   gender,
            "age":                      age,
            "admission_type_id":        admission_type_id,
            "discharge_disposition_id": discharge_disposition_id,
            "admission_source_id":      admission_source_id,
            "max_glu_serum":            max_glu_serum,
            "A1Cresult":                A1Cresult,
            "metformin":                metformin,
            "repaglinide":              "No",
            "nateglinide":              "No",
            "chlorpropamide":           "No",
            "glimepiride":              "No",
            "acetohexamide":            "No",
            "glipizide":                glipizide,
            "glyburide":                glyburide,
            "tolbutamide":              "No",
            "pioglitazone":             "No",
            "rosiglitazone":            "No",
            "acarbose":                 "No",
            "miglitol":                 "No",
            "troglitazone":             "No",
            "tolazamide":               "No",
            "insulin":                  insulin,
            "glyburide-metformin":      "No",
            "glipizide-metformin":      "No",
            "glimepiride-pioglitazone": "No",
            "metformin-rosiglitazone":  "No",
            "metformin-pioglitazone":   "No",
            "change":                   change,
            "diabetesMed":              diabetesMed,
            "payer_code":               "UNK",
            "medical_specialty":        medical_specialty,
        }

        with st.spinner("Running prediction…"):
            X = preprocess(inputs)
            prob = float(model.predict_proba(X)[0, 1])
            pred = int(prob > 0.5)
            pct  = round(prob * 100, 1)

        if prob < 0.15:
            risk, css_class, emoji = "Low",      "risk-low", "🟢"
        elif prob < 0.35:
            risk, css_class, emoji = "Moderate", "risk-mod", "🟡"
        else:
            risk, css_class, emoji = "High",     "risk-high","🔴"

        st.markdown(f"""
        <div class="risk-card {css_class}">
            <div style="font-size:3.5rem">{emoji}</div>
            <div class="risk-val">{pct}%</div>
            <div class="risk-label">30-day readmission probability</div>
            <div style="font-size:1.4rem; font-weight:700; margin-top:12px">
                {risk} Risk
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Probability",  f"{pct}%")
        c2.metric("Prediction",   "Readmitted" if pred else "Not Readmitted")
        c3.metric("Threshold",    "0.50")

        st.divider()

        # Contextual advice
        st.markdown("#### 💡 Clinical Notes")
        if number_inpatient >= 2:
            st.warning(f"⚠️ **{number_inpatient} prior inpatient visits** — this is the strongest readmission signal. Consider intensive follow-up care.")
        if number_emergency >= 1:
            st.warning(f"⚠️ **{number_emergency} prior emergency visits** — indicative of unstable condition management.")
        if insulin in ["Up","Down"]:
            st.info("💊 Insulin dosage was recently adjusted — monitor glucose control closely.")
        if discharge_disposition_id == 1 and risk == "High":
            st.error("🏠 Patient discharged home but flagged as **High Risk** — consider transitional care program.")
        if risk == "Low":
            st.success("✅ Low readmission risk — standard follow-up protocol recommended.")

    else:
        st.info("👈 Fill in patient details in the sidebar and click **Predict Readmission Risk**.")

        st.markdown("""
        <div class="info-box" style="margin-top:1.5rem">
            <b>How it works:</b><br>
            This tool uses a Gradient Boosting model trained on 99,343 diabetic patient encounters
            from 130 US hospitals (1999–2008). It predicts the probability that a patient will be
            readmitted within 30 days of discharge.<br><br>
            <b>Risk levels:</b><br>
            🟢 <b>Low</b> — probability &lt; 15%<br>
            🟡 <b>Moderate</b> — probability 15–35%<br>
            🔴 <b>High</b> — probability &gt; 35%<br><br>
            <b>⚠️ Disclaimer:</b> For research use only. Not a substitute for clinical judgment.
        </div>
        """, unsafe_allow_html=True)

