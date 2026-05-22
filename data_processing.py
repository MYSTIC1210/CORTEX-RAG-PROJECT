"""
data_processing.py
==================
Cortex-RAG: Data Acquisition and Preprocessing Module
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from typing import Dict, List, Optional, Tuple
import re
import json
import logging
from datetime import datetime, timedelta
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DIAGNOSES = [
    "Type 2 Diabetes", "Hypertension", "Coronary Artery Disease",
    "Heart Failure", "COPD", "Chronic Kidney Disease",
    "Atrial Fibrillation", "Hypothyroidism", "Osteoarthritis",
]

HIGH_RISK_DIAGNOSES = [
    "Heart Failure",
    "Chronic Kidney Disease",
    "Coronary Artery Disease",
    "Atrial Fibrillation",
]

LOW_RISK_DIAGNOSES = [
    "Hypertension",
    "Hypothyroidism",
    "Osteoarthritis",
    "COPD",
    "Type 2 Diabetes",
]

MEDICATIONS = [
    "Metformin", "Lisinopril", "Atorvastatin", "Aspirin",
    "Amlodipine", "Metoprolol", "Furosemide", "Levothyroxine",
    "Omeprazole", "Warfarin", "Insulin Glargine",
]

CLINICAL_TEMPLATES = [
    "Patient presents with {symptom}. BP {bp}. HR {hr} bpm. "
    "Labs notable for {lab_finding}. Assessment: {diagnosis}. "
    "Plan: continue {medication}, follow up in {weeks} weeks.",
    "Follow-up visit for {diagnosis}. Patient reports {symptom}. "
    "Vitals stable. {lab_finding}. Medication adjusted: {medication}.",
    "Urgent visit. Chief complaint: {symptom}. "
    "History of {diagnosis}. EKG performed. {lab_finding}. "
    "Prescribed {medication}.",
]

SYMPTOMS = [
    "fatigue", "shortness of breath", "chest pain", "palpitations",
    "leg swelling", "dizziness", "headache", "polyuria", "polydipsia",
]

LAB_FINDINGS = [
    "HbA1c 8.2%", "Creatinine 1.4 mg/dL", "K+ 4.1 mEq/L",
    "WBC 11.2 K/uL", "Hgb 11.8 g/dL", "BNP 450 pg/mL",
    "TSH 0.3 mIU/L", "LDL 142 mg/dL", "eGFR 55 mL/min",
]


def simulate_patient_demographics(n_patients: int = 200) -> pd.DataFrame:
    np.random.seed(42)
    ages = np.random.normal(loc=62, scale=14, size=n_patients).clip(18, 95)
    genders = np.random.choice(["M", "F"], size=n_patients)
    ethnicities = np.random.choice(
        ["White", "Black", "Hispanic", "Asian", "Other"],
        size=n_patients, p=[0.40, 0.22, 0.20, 0.12, 0.06],
    )
    bmi = np.random.normal(loc=28.5, scale=5.5, size=n_patients).clip(16, 50)
    smokers = np.random.choice([0, 1], size=n_patients, p=[0.75, 0.25])
    raw_risk = (
        0.03 * (ages - 50)
        + 0.05 * (bmi - 25)
        + 0.25 * smokers
        + np.random.normal(0, 1.0, size=n_patients)
    )
    synthetic_risk = np.clip(1.0 / (1.0 + np.exp(-raw_risk / 6.0)), 0.0, 1.0)
    return pd.DataFrame({
        "patient_id": [f"P{str(i).zfill(4)}" for i in range(1, n_patients + 1)],
        "age": ages.astype(int),
        "gender": genders,
        "ethnicity": ethnicities,
        "bmi": bmi.round(1),
        "smoker": smokers,
        "synthetic_risk": synthetic_risk.round(3),
    })


def simulate_lab_results(demographics: pd.DataFrame, visits_per_patient: int = 5) -> pd.DataFrame:
    records = []
    base_date = datetime(2020, 1, 1)
    for _, row in demographics.iterrows():
        pid = row["patient_id"]
        age_factor = float(row["age"]) / 60.0
        bmi_factor = float(row["bmi"]) / 25.0
        risk_factor = float(row.get("synthetic_risk", 0.5))
        for visit in range(visits_per_patient):
            visit_date = base_date + timedelta(days=int(visit * 90 + np.random.randint(-10, 10)))
            bnp_val = (
                round(float(np.clip(np.random.lognormal(4.5, 0.8), 10, 2000)), 1)
                if np.random.rand() < 0.3 else None
            )
            records.append({
                "patient_id": pid,
                "visit_date": visit_date.strftime("%Y-%m-%d"),
                "visit_number": visit + 1,
                "hba1c":         round(float(np.clip(np.random.normal(5.5 + 5.5 * risk_factor + 0.4 * bmi_factor, 0.45), 4.5, 14.0)), 1),
                "creatinine":    round(float(np.clip(np.random.normal(0.7 + 1.7 * risk_factor + 0.15 * age_factor, 0.12), 0.5, 5.0)), 2),
                "glucose":       round(float(np.clip(np.random.normal(85 + 180 * risk_factor, 10), 60, 400)), 1),
                "cholesterol_ldl": round(float(np.clip(np.random.normal(95 + 90 * risk_factor, 12), 50, 250)), 1),
                "systolic_bp":   round(float(np.clip(np.random.normal(110 + 45 * risk_factor + 8 * age_factor, 6), 80, 200)), 0),
                "diastolic_bp":  round(float(np.clip(np.random.normal(68 + 18 * risk_factor, 4), 50, 130)), 0),
                "heart_rate":    round(float(np.clip(np.random.normal(68 + 12 * risk_factor, 4), 40, 140)), 0),
                "egfr":          round(float(np.clip(np.random.normal(105 - 60 * risk_factor - 5 * age_factor, 4), 5, 120)), 1),
                "bnp":           round(float(np.clip(np.random.normal(40 + 850 * risk_factor, 40), 10, 2000)), 1) if risk_factor > 0.25 else bnp_val,
            })
    return pd.DataFrame(records)


def simulate_clinical_notes(demographics: pd.DataFrame, n_notes_per_patient: int = 3) -> pd.DataFrame:
    random.seed(42)
    records = []
    base_date = datetime(2020, 1, 1)
    for _, row in demographics.iterrows():
        pid = row["patient_id"]
        risk = float(row.get("synthetic_risk", 0.5))
        primary_dx_pool = HIGH_RISK_DIAGNOSES if risk >= 0.6 else LOW_RISK_DIAGNOSES
        primary_dx = random.choice(primary_dx_pool)
        for i in range(n_notes_per_patient):
            template = random.choice(CLINICAL_TEMPLATES)
            note_date = base_date + timedelta(days=i * 120 + random.randint(-15, 15))
            note_text = template.format(
                symptom=random.choice(SYMPTOMS),
                bp=f"{random.randint(110,170)}/{random.randint(65,100)}",
                hr=random.randint(55, 110),
                lab_finding=random.choice(LAB_FINDINGS),
                diagnosis=primary_dx,
                medication=random.choice(MEDICATIONS),
                weeks=random.choice([4, 6, 8, 12]),
            )
            records.append({
                "patient_id": pid,
                "note_date": note_date.strftime("%Y-%m-%d"),
                "note_type": random.choice(["Progress Note", "Consult", "Discharge Summary"]),
                "note_text": note_text,
                "provider_specialty": random.choice(
                    ["Internal Medicine", "Cardiology", "Endocrinology", "Nephrology"]
                ),
            })
    return pd.DataFrame(records)


def simulate_imaging_features(demographics: pd.DataFrame) -> pd.DataFrame:
    np.random.seed(42)
    records = []
    for _, row in demographics.iterrows():
        pid = row["patient_id"]
        if np.random.rand() < 0.6:
            embedding = np.random.normal(0, 1, 128).round(4).tolist()
            records.append({
                "patient_id": pid,
                "imaging_date": (datetime(2020, 6, 1) + timedelta(days=int(np.random.randint(0, 365)))).strftime("%Y-%m-%d"),
                "modality": np.random.choice(["Chest X-Ray", "Echo", "CT Chest", "MRI Brain"]),
                "impression": np.random.choice([
                    "No acute cardiopulmonary process.",
                    "Mild cardiomegaly. No pleural effusion.",
                    "Reduced EF 35%. Severe mitral regurgitation.",
                    "Pulmonary hyperinflation consistent with COPD.",
                ]),
                "embedding": json.dumps(embedding),
            })
    return pd.DataFrame(records)


def simulate_outcomes(demographics: pd.DataFrame) -> pd.DataFrame:
    np.random.seed(99)
    records = []
    for _, row in demographics.iterrows():
        risk = float(row.get("synthetic_risk", 0.5))
        age_risk = float(row["age"]) / 100.0
        bmi_risk = (float(row["bmi"]) - 18) / 60.0
        linear_risk = 0.7 * risk + 0.2 * age_risk + 0.1 * bmi_risk
        p_readmit = float(np.clip(0.10 + 0.80 * linear_risk, 0, 0.98))
        p_adverse = float(np.clip(0.06 + 0.70 * linear_risk, 0, 0.95))
        p_mortality = float(np.clip(0.03 + 0.55 * linear_risk, 0, 0.90))
        records.append({
            "patient_id": row["patient_id"],
            "readmission_30d":   int(np.random.rand() < p_readmit),
            "adverse_event_90d": int(np.random.rand() < p_adverse),
            "mortality_1yr":     int(np.random.rand() < p_mortality),
        })
    return pd.DataFrame(records)


class ClinicalDataPreprocessor:
    """End-to-end preprocessing pipeline for multimodal clinical data."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.imputer = SimpleImputer(strategy="median")
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self._fitted = False

    def preprocess_labs(self, df: pd.DataFrame, fit: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
        df = df.copy()
        df["visit_date"] = pd.to_datetime(df["visit_date"])
        df = df.sort_values(["patient_id", "visit_date"])

        df["days_since_first_visit"] = df.groupby("patient_id")["visit_date"].transform(
            lambda x: (x - x.min()).dt.days
        )

        numeric_labs = ["hba1c", "creatinine", "glucose", "cholesterol_ldl",
                        "systolic_bp", "diastolic_bp", "heart_rate", "egfr", "bnp"]
        for col in numeric_labs:
            df[f"{col}_delta"] = df.groupby("patient_id")[col].diff().fillna(0)

        df["cardiometabolic_risk"] = (
            (df["systolic_bp"].fillna(130) - 120) / 40
            + (df["hba1c"].fillna(6.5) - 5.7) / 3
            + (df["cholesterol_ldl"].fillna(100) - 70) / 100
        ).clip(-2, 4)

        df["renal_risk"] = np.where(df["egfr"].fillna(60) < 60, 1, 0)

        self.numeric_cols = [
            c for c in df.columns
            if df[c].dtype in [np.float64, np.int64] and c not in ["visit_number"]
        ]

        X = df[self.numeric_cols].values
        X_imputed = self.imputer.fit_transform(X) if fit else self.imputer.transform(X)
        df[self.numeric_cols] = X_imputed

        X_scaled = self.scaler.fit_transform(df[self.numeric_cols]) if fit else self.scaler.transform(df[self.numeric_cols])
        return df, X_scaled

    def preprocess_demographics(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in ["gender", "ethnicity"]:
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        df["age_group"] = pd.cut(df["age"], bins=[0, 40, 55, 65, 75, 100],
                                  labels=["<40", "40-55", "55-65", "65-75", "75+"])
        df["age_group_enc"] = LabelEncoder().fit_transform(df["age_group"].astype(str))
        df["obesity"] = (df["bmi"] >= 30).astype(int)
        return df

    def extract_note_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["note_length"] = df["note_text"].str.len()
        df["mention_diabetes"] = df["note_text"].str.contains(r"diabet|HbA1c|glucose|insulin", case=False, regex=True).astype(int)
        df["mention_cardiac"]  = df["note_text"].str.contains(r"cardiac|heart|EKG|BNP|arrhythmia|atrial", case=False, regex=True).astype(int)
        df["mention_renal"]    = df["note_text"].str.contains(r"renal|kidney|creatinine|eGFR|dialysis", case=False, regex=True).astype(int)
        df["mention_urgent"]   = df["note_text"].str.contains(r"urgent|emergent|acute|critical|severe", case=False, regex=True).astype(int)
        med_pattern = "|".join(re.escape(m) for m in MEDICATIONS)
        df["medication_mentions"] = df["note_text"].str.findall(med_pattern, flags=re.IGNORECASE).apply(len)
        for symptom in ["fatigue", "pain", "swelling", "dyspnea", "shortness"]:
            df[f"symptom_{symptom}"] = df["note_text"].str.contains(symptom, case=False).astype(int)
        return df

    def aggregate_patient_features(
        self,
        demographics: pd.DataFrame,
        labs: pd.DataFrame,
        notes: pd.DataFrame,
        imaging: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        lab_num_cols = ["hba1c", "creatinine", "glucose", "cholesterol_ldl",
                        "systolic_bp", "diastolic_bp", "heart_rate", "egfr",
                        "cardiometabolic_risk", "renal_risk"]
        labs_agg = labs.groupby("patient_id")[lab_num_cols].agg(["mean", "last"]).round(3)
        labs_agg.columns = ["_".join(c) for c in labs_agg.columns]
        labs_agg = labs_agg.reset_index()

        note_feat_cols = ["mention_diabetes", "mention_cardiac", "mention_renal",
                          "mention_urgent", "medication_mentions"]
        notes_agg = notes.groupby("patient_id")[note_feat_cols].sum().reset_index()
        notes_agg["n_notes"] = notes.groupby("patient_id").size().values

        df = demographics.merge(labs_agg, on="patient_id", how="left")
        df = df.merge(notes_agg, on="patient_id", how="left")

        if imaging is not None and not imaging.empty:
            imaging_emb = imaging.copy()
            imaging_emb["embedding_arr"] = imaging_emb["embedding"].apply(json.loads)
            emb_df = pd.DataFrame(
                imaging_emb.groupby("patient_id")["embedding_arr"]
                .apply(lambda x: np.mean(np.stack(x.values), axis=0))
                .tolist(),
                columns=[f"img_emb_{i}" for i in range(128)],
            )
            emb_df["patient_id"] = imaging_emb.groupby("patient_id").size().index.tolist()
            df = df.merge(emb_df, on="patient_id", how="left")

        df = df.fillna(0)
        logger.info(f"Patient feature matrix shape: {df.shape}")
        return df


def load_and_preprocess(n_patients: int = 200) -> Dict:
    logger.info("Simulating clinical datasets ...")
    demographics = simulate_patient_demographics(n_patients)
    labs_raw     = simulate_lab_results(demographics)
    notes_raw    = simulate_clinical_notes(demographics)
    imaging_raw  = simulate_imaging_features(demographics)
    outcomes     = simulate_outcomes(demographics)

    preprocessor       = ClinicalDataPreprocessor()
    demographics_clean = preprocessor.preprocess_demographics(demographics)
    labs_clean, labs_scaled = preprocessor.preprocess_labs(labs_raw, fit=True)
    notes_clean        = preprocessor.extract_note_features(notes_raw)
    patient_features   = preprocessor.aggregate_patient_features(
        demographics_clean, labs_clean, notes_clean, imaging_raw
    )

    logger.info("Preprocessing complete.")
    return {
        "demographics":    demographics_clean,
        "labs":            labs_clean,
        "labs_scaled":     labs_scaled,
        "notes":           notes_clean,
        "imaging":         imaging_raw,
        "patient_features": patient_features,
        "outcomes":        outcomes,
        "preprocessor":    preprocessor,
    }