"""
Diabetes Readmission Prediction API
Best model: Optimized Gradient Boosting Classifier (Test AUC: 0.666)
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Diabetes Readmission Predictor",
    description="Predicts 30-day hospital readmission risk for diabetic patients.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load artifacts ────────────────────────────────────────────────────────────
ARTIFACTS_DIR = Path("pickle-files")

try:
    scaler = pickle.load(open(ARTIFACTS_DIR / "scaler.sav", "rb"))
    model  = pickle.load(open(ARTIFACTS_DIR / "best_classifier.pkl", "rb"))
    print("✅ Model and scaler loaded successfully.")
except FileNotFoundError as e:
    print(f"⚠️  Artifact not found: {e}. Run the training notebook first.")
    scaler = None
    model  = None

# ── Feature definitions (must match training) ─────────────────────────────────
COLS_NUM = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]

COLS_CAT = [
    "race", "gender", "max_glu_serum", "A1Cresult",
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide",
    "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone",
    "tolazamide", "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone", "change", "diabetesMed", "payer_code",
]

COLS_CAT_NUM = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]

TOP_10_SPECIALTIES = [
    "UNK", "InternalMedicine", "Emergency/Trauma",
    "Family/GeneralPractice", "Cardiology", "Surgery-General",
    "Nephrology", "Orthopedics", "Orthopedics-Reconstructive", "Radiologist",
]

AGE_MAP = {
    "[0-10)": 0, "[10-20)": 10, "[20-30)": 20, "[30-40)": 30,
    "[40-50)": 40, "[50-60)": 50, "[60-70)": 60, "[70-80)": 70,
    "[80-90)": 80, "[90-100)": 90,
}

# ── Request / Response schemas ────────────────────────────────────────────────
class PatientInput(BaseModel):
    # Numeric features
    time_in_hospital:    int   = Field(..., ge=1, le=14,  example=3)
    num_lab_procedures:  int   = Field(..., ge=0,          example=41)
    num_procedures:      int   = Field(..., ge=0, le=6,    example=1)
    num_medications:     int   = Field(..., ge=0,          example=15)
    number_outpatient:   int   = Field(..., ge=0,          example=0)
    number_emergency:    int   = Field(..., ge=0,          example=0)
    number_inpatient:    int   = Field(..., ge=0,          example=0)
    number_diagnoses:    int   = Field(..., ge=1,          example=7)

    # Demographics
    race:   str = Field("Caucasian", example="Caucasian")
    gender: str = Field("Female",    example="Female")
    age:    str = Field("[50-60)",   example="[50-60)")

    # IDs (categorical)
    admission_type_id:        int = Field(1, example=1)
    discharge_disposition_id: int = Field(1, example=1)
    admission_source_id:      int = Field(7, example=7)

    # Lab results
    max_glu_serum: str = Field("None", example="None")
    A1Cresult:     str = Field("None", example="None")

    # Medications (No / Steady / Up / Down)
    metformin:                str = Field("No", example="No")
    repaglinide:              str = Field("No", example="No")
    nateglinide:              str = Field("No", example="No")
    chlorpropamide:           str = Field("No", example="No")
    glimepiride:              str = Field("No", example="No")
    acetohexamide:            str = Field("No", example="No")
    glipizide:                str = Field("No", example="No")
    glyburide:                str = Field("No", example="No")
    tolbutamide:              str = Field("No", example="No")
    pioglitazone:             str = Field("No", example="No")
    rosiglitazone:            str = Field("No", example="No")
    acarbose:                 str = Field("No", example="No")
    miglitol:                 str = Field("No", example="No")
    troglitazone:             str = Field("No", example="No")
    tolazamide:               str = Field("No", example="No")
    insulin:                  str = Field("No", example="No")
    glyburide_metformin:      str = Field("No", example="No")
    glipizide_metformin:      str = Field("No", example="No")
    glimepiride_pioglitazone: str = Field("No", example="No")
    metformin_rosiglitazone:  str = Field("No", example="No")
    metformin_pioglitazone:   str = Field("No", example="No")

    # Other
    change:             str = Field("No",  example="No")
    diabetesMed:        str = Field("Yes", example="Yes")
    payer_code:         str = Field("MC",  example="MC")
    medical_specialty:  str = Field("UNK", example="InternalMedicine")
    weight:             Optional[str] = Field(None, example=None)


class PredictionResponse(BaseModel):
    readmission_probability: float
    readmission_risk:        str
    prediction:              int
    model:                   str
    threshold:               float


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(patient: PatientInput) -> np.ndarray:
    """Reproduce the exact feature engineering from the training notebook."""

    # Build a single-row DataFrame
    row = {
        "time_in_hospital":        patient.time_in_hospital,
        "num_lab_procedures":      patient.num_lab_procedures,
        "num_procedures":          patient.num_procedures,
        "num_medications":         patient.num_medications,
        "number_outpatient":       patient.number_outpatient,
        "number_emergency":        patient.number_emergency,
        "number_inpatient":        patient.number_inpatient,
        "number_diagnoses":        patient.number_diagnoses,
        "race":                    patient.race or "UNK",
        "gender":                  patient.gender,
        "age":                     patient.age,
        "admission_type_id":       str(patient.admission_type_id),
        "discharge_disposition_id":str(patient.discharge_disposition_id),
        "admission_source_id":     str(patient.admission_source_id),
        "max_glu_serum":           patient.max_glu_serum,
        "A1Cresult":               patient.A1Cresult,
        "metformin":               patient.metformin,
        "repaglinide":             patient.repaglinide,
        "nateglinide":             patient.nateglinide,
        "chlorpropamide":          patient.chlorpropamide,
        "glimepiride":             patient.glimepiride,
        "acetohexamide":           patient.acetohexamide,
        "glipizide":               patient.glipizide,
        "glyburide":               patient.glyburide,
        "tolbutamide":             patient.tolbutamide,
        "pioglitazone":            patient.pioglitazone,
        "rosiglitazone":           patient.rosiglitazone,
        "acarbose":                patient.acarbose,
        "miglitol":                patient.miglitol,
        "troglitazone":            patient.troglitazone,
        "tolazamide":              patient.tolazamide,
        "insulin":                 patient.insulin,
        "glyburide-metformin":     patient.glyburide_metformin,
        "glipizide-metformin":     patient.glipizide_metformin,
        "glimepiride-pioglitazone":patient.glimepiride_pioglitazone,
        "metformin-rosiglitazone": patient.metformin_rosiglitazone,
        "metformin-pioglitazone":  patient.metformin_pioglitazone,
        "change":                  patient.change,
        "diabetesMed":             patient.diabetesMed,
        "payer_code":              patient.payer_code or "UNK",
        "medical_specialty":       patient.medical_specialty or "UNK",
        "weight":                  patient.weight,
    }
    df = pd.DataFrame([row])

    # med_spec bucketing
    df["med_spec"] = df["medical_specialty"].where(
        df["medical_specialty"].isin(TOP_10_SPECIALTIES), other="Other"
    )

    # One-hot encode
    df_cat = pd.get_dummies(
        df[COLS_CAT + COLS_CAT_NUM + ["med_spec"]], drop_first=True
    )

    # Age → numeric, weight → binary flag
    df["age_group"] = df["age"].map(AGE_MAP).fillna(50)
    df["has_weight"] = df["weight"].notnull().astype(int)

    # Reconstruct full feature vector using the scaler's feature names
    # (fill any column the model expects but this patient doesn't have)
    expected_cols = scaler.feature_names_in_ if hasattr(scaler, "feature_names_in_") else None

    full = pd.concat([df[COLS_NUM], df_cat, df[["age_group", "has_weight"]]], axis=1)

    if expected_cols is not None:
        full = full.reindex(columns=expected_cols, fill_value=0)

    return scaler.transform(full.values)


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientInput):
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not found. Copy pickle-files/ from your Colab notebook.",
        )

    X = preprocess(patient)
    prob = float(model.predict_proba(X)[0, 1])
    thresh = 0.5
    pred = int(prob > thresh)

    if prob < 0.15:
        risk = "Low"
    elif prob < 0.35:
        risk = "Moderate"
    else:
        risk = "High"

    return PredictionResponse(
        readmission_probability=round(prob, 4),
        readmission_risk=risk,
        prediction=pred,
        model="GradientBoostingClassifier (n_estimators=100, max_depth=3, lr=0.1)",
        threshold=thresh,
    )


# ── Serve frontend ────────────────────────────────────────────────────────────
app.mount("/", StaticFiles(directory="static", html=True), name="static")
