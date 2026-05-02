# Diabetes Patients Early Readmission Prediction

> A complete end-to-end machine learning project to predict whether a diabetic patient will be **readmitted to hospital within 30 days** of discharge  using EDA, preprocessing pipelines, model selection, and hyperparameter tuning.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-latest-green.svg)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

##  1. Business Problem / Motivation

Hospital readmissions are one of the most expensive and preventable problems in healthcare. Every wrong treatment choice for a diabetic patient can harm their health and lead to early readmission  driving up costs for both the patient and the healthcare system.

- 🇺🇸 US hospitals pay **billions annually** in avoidable readmission penalties
- Standard methods (e.g. LACE index) for identifying high-risk patients perform poorly
- With a growing number of diabetic patients, there is an urgent need for **data-driven, automated risk identification**

**The goal:** build a binary classifier that predicts whether a patient will be readmitted within 30 days, enabling hospitals to intervene *before* discharge.

---

##  2. Project Overview

| | |
|---|---|
| **Task** | Binary classification — readmitted within 30 days? |
| **Dataset** | UCI Diabetes 130-US Hospitals (1999–2008) |
| **Records** | 101,766 patient encounters |
| **Features** | 50 raw → 53 after engineering |
| **Best Model** | Random Forest (tuned) |
| **Test AUC** | **0.651** |
| **Test Recall** | **0.62** |

**Project flow:**
1. Exploratory Data Analysis (EDA)
2. Data Preprocessing Pipeline
3. Model Selection (RF, XGBoost, MLP + ensemble methods)
4. Final Model Evaluation with RFECV + hyperparameter tuning

---

##  3. Data

**Source:** [UCI ML Repository — Diabetes 130-US Hospitals (1999–2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

| Attribute | Value |
|---|---|
| Records | 101,766 patient encounters |
| Hospitals | 130 US hospitals |
| Period | 1999–2008 |
| Raw features | 50 (numeric + categorical) |
| Target | `readmitted`: NO / >30 / <30 → binary |
| Positive class | ~11.1% (readmitted within 30 days) |

**Feature types:**
- **Demographics:** race, gender, age (10-year intervals)
- **Admission:** admission type, discharge disposition, admission source
- **Clinical:** time in hospital, number of diagnoses, lab procedures, procedures
- **History:** number of prior inpatient / outpatient / emergency visits
- **Medications:** 23 diabetes drug features (metformin, insulin, etc.)
- **Lab results:** HbA1c (A1Cresult), max glucose serum

---

##  4. Exploratory Data Analysis

### Target Class Distribution

The original dataset has 3 classes. We convert to binary (readmitted `<30` days = 1, everything else = 0):

<img width="357" height="356" alt="download" src="https://github.com/user-attachments/assets/c0cffca6-795e-4f50-b0d9-da93b023f16a" />


The classes are **highly imbalanced**  only ~11% of patients are readmitted within 30 days. This drives the need for class balancing strategies.

---

### Missing Values

<img width="1483" height="780" alt="download" src="https://github.com/user-attachments/assets/1252bf9e-8679-448b-a453-3588ab1782b2" />

Only 7 attributes have missing values. Notably:
- `weight` is **96.9% missing** → converted to binary flag (`has_weight`)
- `payer_code` is **39.6% missing** → dropped
- `medical_specialty` is **49.1% missing** → filled with most-frequent value
- `race` is **2.2% missing** → filled with most-frequent value
- Only 1 record has all 3 diagnoses missing → dropped

---

### Demographics


<img width="612" height="377" alt="download" src="https://github.com/user-attachments/assets/815ac991-fc12-4a1f-9c63-2bb99bb1ba5c" />
<img width="612" height="376" alt="download" src="https://github.com/user-attachments/assets/310ad85d-1ef3-4762-993a-9ab490e7a368" />
<img width="612" height="376" alt="download" src="https://github.com/user-attachments/assets/4a22ec73-8241-44b9-a7ce-f73b75a128ef" />

- **Race:** Caucasian patients dominate (75.8%). Readmission rates are fairly consistent across races.
- **Gender:** Near-equal split. No strong gender-based readmission pattern.

- **Age:** Readmission risk increases with age, peaking in the 70–80 range. Younger patients are significantly harder to predict.

---

### Clinical Features


Patients readmitted within 30 days tend to spend slightly **longer in hospital** during their current visit — a weak but consistent signal.

- **Lab procedures:** readmitted patients undergo slightly more lab tests
- **Medications:** readmitted patients receive more distinct medications on average

---

### Prior Visit History

**`number_inpatient`** (prior inpatient visits in the year preceding the encounter) is the strongest single predictor. Patients with 2+ prior admissions are significantly more likely to be readmitted.

---

### Discharge Disposition

Discharge to SNF (Skilled Nursing Facility), rehab, or short-term hospitals is strongly associated with readmission — these patients have ongoing medical needs that are not fully resolved at discharge.

---

### Diagnosis Categories


ICD-9 codes were mapped to 18 diagnosis categories. Circulatory and respiratory conditions show the highest readmission rates.

---

### Medication Patterns

Most diabetes medications show a "No" (not prescribed) majority. Insulin usage and dosage changes (`change`) have the strongest signal among medication features.

---

### Feature Engineering

Three new features were created from existing data:

**1. `visits_sum`** total prior healthcare utilization (emergency + outpatient + inpatient):

**2. `number_medicaments_changes`** total medication dosage changes across 23 drug features

**3. `number_medicaments`** total number of medications actively prescribed

---

### Pairplot & Correlation


Key correlations:
- `num_medications` ↔ `time_in_hospital`: moderate positive (0.45)
- `num_medications` ↔ `num_lab_procedures`: moderate positive (0.44)
- `num_procedures` ↔ `time_in_hospital`: moderate positive (0.41)
- Most features show **weak correlation with target** — confirming the difficulty of this classification task

---

##  5. Data Preprocessing

A full **scikit-learn Pipeline** was built to ensure reproducibility:

| Step | Action |
|---|---|
| **Row filtering** | Drop rows with all 3 diagnoses missing; drop invalid gender values |
| **Column filtering** | Drop `encounter_id`, `patient_nbr`, `payer_code`; drop high-NaN columns |
| **Low diversity columns** | Drop columns where one value dominates >90% of records |
| **Small category merging** | Merge categories with <5% frequency into `Other` |
| **Missing value imputation** | Categorical → most frequent; Numeric → median |
| **Ordinal mapping** | Age intervals → 0–9 numeric scale |
| **Feature engineering** | `visits_sum`, `number_medicaments_changes`, `number_medicaments`, diagnosis categories |
| **One-hot encoding** | All remaining categorical features |

**Result:** 50 raw features → **53 features** after preprocessing (80/20 train/test split)

Train: **81,409 samples** | Test: **20,353 samples**

---

##  6. Handling Class Imbalance

Three strategies were tested on all models:

| Strategy | Description |
|---|---|
| **Original** | Use `class_weight='balanced'` in model |
| **Random Undersampling** | Randomly remove majority class samples to match minority |
| **SMOTE Oversampling** | Generate synthetic minority samples via interpolation |

> **Key finding:** SMOTE consistently *hurt* performance on this dataset. Undersampling matched or beat the original balanced weights approach for most models. This aligns with Mačinec & Šefčík (2021) — SMOTE is inappropriate here because the data structure is not well-suited to synthetic interpolation.

---

##  7. Modeling Approach

### Baseline Models

Three model families were evaluated, each on 3 data setups (original / undersampled / oversampled):

#### Random Forest

<img width="783" height="296" alt="download" src="https://github.com/user-attachments/assets/95b16fb7-ae3c-4ba9-a840-3b9e7b0a04d8" />


| Model | Accuracy | F1 (macro) | Precision | Recall | AUC ROC |
|---|---|---|---|---|---|
| RF | 0.67 | 0.52 | 0.17 | 0.53 | **0.65** |
| RF undersampled | 0.60 | 0.49 | 0.16 | 0.63 | **0.65** |
| RF oversampled | 0.42 | 0.38 | 0.13 | 0.75 | 0.60 |

#### XGBoost

<img width="783" height="296" alt="download" src="https://github.com/user-attachments/assets/5d0c41c1-898e-4d8c-9e44-135e641459f0" />


| Model | Accuracy | F1 (macro) | Precision | Recall | AUC ROC |
|---|---|---|---|---|---|
| XGBoost | 0.52 | 0.45 | 0.15 | 0.69 | 0.64 |
| XGBoost undersampled | 0.58 | 0.48 | 0.16 | 0.63 | 0.64 |
| XGBoost oversampled | 0.42 | 0.38 | 0.12 | 0.69 | 0.57 |

#### MLP (Multilayer Perceptron)

<img width="779" height="296" alt="download" src="https://github.com/user-attachments/assets/a21d3efd-bb24-44ea-9b48-9ff084997bdc" />

| Model | Accuracy | F1 (macro) | Precision | Recall | AUC ROC |
|---|---|---|---|---|---|
| MLP | 0.89 | 0.49 | 0.41 | 0.02 | 0.64 |
| MLP undersampled | 0.63 | 0.50 | 0.16 | 0.56 | 0.64 |
| MLP oversampled | 0.89 | 0.49 | 0.35 | 0.02 | 0.64 |

> MLP without balancing collapses to near-zero recall — it predicts the majority class almost exclusively.


**Winner: Random Forest on undersampled data (AUC 0.65)** — selected for simplicity, speed, and interpretability.

---

##  8. Final Model Training

### Data Visualization (t-SNE)

Before final training, t-SNE was used to visualize the decision boundary in 2D and 3D:


The classes are **not clearly separable** — the two classes overlap heavily, confirming this is an inherently difficult classification problem and setting realistic expectations for AUC.

---

### Baseline Evaluation (Random Forest, default params)


**Baseline RF results:**
```
              precision    recall  f1-score   support
           0       0.93      0.59      0.72     18,090
           1       0.16      0.63      0.26      2,263
    accuracy                           0.60     20,353
   macro avg       0.54      0.61      0.49     20,353
AUC ROC: 0.651
```

---

### Feature Selection (RFECV)

Recursive Feature Elimination with Cross-Validation (RFECV) was applied to identify the optimal feature subset:


- **Optimal features:** 49 out of 53 selected
- AUC remained stable — the 4 removed features added noise without signal
- Learning curve shows model converges well and is not severely overfitting

---

### Hyperparameter Tuning (RandomizedSearchCV)

**Search space:**
```python
{
    'n_estimators':      [10, 30, 50, ..., 1000],
    'max_depth':         [10, 20, 30, 40, 50, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4, 8],
    'criterion':         ['gini', 'entropy'],
    'bootstrap':         [True, False]
}
```

**Best hyperparameters:**
```python
{
    'n_estimators':      230,
    'min_samples_split': 2,
    'min_samples_leaf':  8,
    'max_depth':         49,
    'criterion':         'entropy',
    'bootstrap':         True
}
```
Best CV AUC score: **0.652**

---

##  9. Results

### Final Model Evaluation


**Final classification report (test set):**
```
              precision    recall  f1-score   support

           0       0.93      0.60      0.73     18,090
           1       0.16      0.62      0.26      2,263

    accuracy                           0.60     20,353
   macro avg       0.54      0.61      0.49     20,353
weighted avg       0.84      0.60      0.68     20,353

ROC AUC score: 0.651
```

### Summary Table

| Model | AUC ROC | Recall (class 1) | F1 (macro) | Notes |
|---|---|---|---|---|
| Gaussian NB | ~0.50 | — | — | Near-random |
| MLP (no balance) | 0.64 | 0.02 | 0.49 | Collapses recall |
| XGBoost | 0.64 | 0.69 | 0.45 | High recall, low precision |
| RF oversampled | 0.60 | 0.75 | 0.38 | SMOTE hurts AUC |
| RF / XGBoost undersampled | 0.65 | 0.63 | 0.49 | Good balance |
| RF (original + class_weight) | 0.65 | 0.53 | 0.52 | Clean, interpretable |
| Ensemble (Bagging/Voting/Stacking) | 0.65 | 0.60–0.64 | 0.49–0.50 | No gain over single model |
| ** RF Tuned (final)** | **0.651** | **0.62** | **0.49** | **Best overall** |

> **Why AUC over accuracy?** With 11% class prevalence, a model predicting "never readmitted" achieves 89% accuracy. AUC-ROC is the correct metric — it measures the model's ability to rank patients by risk across all thresholds.

---

##  10. Model Interpretation

<img width="1513" height="1155" alt="image" src="https://github.com/user-attachments/assets/74445347-e2ab-43c6-b683-11d89fa695b7" />

### Feature Importance (Final Tuned Random Forest)

Top features driving readmission predictions:

| Rank | Feature | Interpretation |
|---|---|---|
| **1** | `number_inpatient` | Prior inpatient visits — strongest signal by far |
| **2** | `discharge_disposition_id` | Where patient is sent after discharge |
| **3** | `number_diagnoses` | Number of diagnoses — reflects clinical complexity |
| **4** | `time_in_hospital` | Longer stays indicate more serious conditions |
| **5** | `number_emergency` | Prior ER visits |
| **6** | `visits_sum` | Engineered feature: total prior healthcare utilization |
| **7** | `num_medications` | Number of medications prescribed |
| **8** | `num_lab_procedures` | Number of lab tests — clinical workup intensity |
| **9** | `age` | Older patients are higher risk |
| **10** | `number_medicaments` | Engineered: count of active diabetes medications |

**Cross-model consistency:** Both Random Forest and XGBoost independently ranked `number_inpatient` as the dominant feature, validating its clinical significance.

> **Clinical interpretation:** A patient's *history of hospitalization* is more predictive than anything that happens during the current visit. This suggests systemic factors — disease progression, care continuity gaps, or social determinants — that a single encounter cannot fully capture.

---

##  11. Key Insights

### What Worked Best
- **Random Forest on undersampled data** delivered the best balance of AUC and recall — simple, interpretable, and robust
- **Feature engineering** (`visits_sum`, `number_medicaments`) added predictive signal beyond raw features
- **Diagnosis category mapping** (ICD-9 → 18 categories) reduced noise from 800+ raw codes
- **RFECV feature selection** confirmed 49/53 features are useful — no major redundancy

### What Did Not Work
- **SMOTE oversampling** consistently hurt AUC (drops from 0.65 → 0.60). Prior literature claiming 90%+ AUC with SMOTE applied it to the whole dataset before splitting — this is data leakage, not genuine performance
- **MLP without balancing** collapses to near-zero recall — predicts no readmissions at all
- **Ensemble methods** (Bagging, Voting, Stacking) provided no improvement over the single best model

### Business Impact
The AUC ceiling of ~0.65 is not a failure — it reflects the inherent difficulty of predicting rare medical events from administrative data alone. Even at 0.651 AUC:
- A hospital using this model to prioritize follow-up for flagged patients would catch **~62% of actual readmissions**
- Without any model, catch rate for systematic intervention is **0%**

### Why the Ceiling is ~0.65
- Administrative data misses social determinants of health (housing, income, social support)
- ICD-9 codes are coarse — within-category disease severity is lost
- The data is 1999–2008; treatment protocols and drug landscapes have changed substantially

---

##  12. Conclusion

This project built a complete ML pipeline to predict 30-day readmissions for diabetic patients. A tuned Random Forest classifier trained on undersampled data achieves **AUC 0.651** and **62% recall** on the test set — matching the validated ceiling for this problem as reported in the academic literature.

The model is:
- **Explainable** via feature importance (tree-based, no black box)
- **Reproducible** via a scikit-learn preprocessing pipeline
- **Validated** against multiple baselines (RF, XGBoost, MLP, ensembles)
- **Honest** about its limitations — AUC 0.65 with proper train/test splitting, not inflated by data leakage

---

##  13. Future Work

- [ ] Retrain on current EHR data (post-2008 protocols differ significantly)
- [ ] Add SHAP values for per-patient explanations — critical for clinical adoption
- [ ] Investigate LightGBM and CatBoost with more extensive hyperparameter search
- [ ] Explore age-stratified models — performance degrades for older patients
- [ ] Incorporate social determinants of health (ZIP code, income proxies) as additional features
- [ ] Build a clinical decision support UI for care coordinators
- [ ] Explore threshold calibration to allow departments to adjust precision/recall trade-off

---

##  14. How to Run

### Install dependencies
```bash
git clone https://github.com/YOUR_USERNAME/diabetes-readmission-predictor.git
cd diabetes-readmission-predictor
pip install -r requirements.txt
```

### Download data
Place `data.csv` from [UCI Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) into `data/data.csv`

### Run preprocessing
```bash
jupyter notebook notebooks/diabetes_readmission_project.ipynb
```
Run **Part 2: Data Preprocessing** cells — this saves `X_train.csv` and `X_test.csv` to `data/`

### Train model
Run **Part 3: Model Selection** cells — trains RF, XGBoost, MLP and all ensemble variants

### Evaluate results
Run **Part 4: Final Model Evaluation** cells — RFECV, hyperparameter tuning, final metrics

---

##  15. Repository Structure

```
diabetes-readmission-predictor/
│
├── README.md                            ← This file
├── requirements.txt                     ← Python dependencies
│
├── notebooks/
│   └── diabetes_readmission_project.ipynb   ← Complete project notebook (4 parts)
│
└── images/                              ← All figures referenced in this README
    ├── fig01_class_distribution_original.png
    ├── fig02_class_distribution_binary.png
    ├── fig03_missing_values.png
    ├── fig04_race_by_class.png
    ├── fig05_gender_by_class.png
    ├── fig06_age_by_class.png
    ├── fig07_age_boxplot.png
    ├── fig08_discharge_disposition.png
    ├── fig09_time_in_hospital_kde.png
    ├── fig10_time_in_hospital_boxplot.png
    ├── fig11_lab_procedures_kde.png
    ├── fig12_num_medications_kde.png
    ├── fig13_number_inpatient_boxplot.png
    ├── fig14_diagnosis_primary.png
    ├── fig15_medications_overview.png
    ├── fig16_visits_sum_kde.png
    ├── fig17_pairplot_numerical.png
    ├── fig18_correlation_heatmap.png
    ├── fig20_rf_feature_importance.png
    ├── fig22_xgboost_feature_importance.png
    ├── fig25_tsne_2d.png
    ├── fig26_tsne_3d.png
    ├── fig27_baseline_rf_evaluation.png
    ├── fig28_baseline_learning_curve.png
    ├── fig29_rfecv_evaluation.png
    ├── fig30_rfecv_learning_curve.png
    ├── fig31_final_model_evaluation.png
    ├── fig32_final_learning_curve.png
    └── fig33_final_feature_importance.png
```

| File / Folder | Purpose |
|---|---|
| `notebooks/diabetes_readmission_project.ipynb` | Full ML pipeline — EDA, preprocessing, modelling, evaluation |
| `data/data.csv` | Raw UCI dataset (not committed — download from link above) |
| `images/` | All 29 figures generated during EDA and model evaluation |

---

##  16. Requirements

```bash
pip install -r requirements.txt
```

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
matplotlib>=3.6.0
seaborn>=0.12.0
imbalanced-learn>=0.10.0
jupyter>=1.0.0
```

---

##  References

1. Strack, B. et al. (2014). *Impact of HbA1c measurement on hospital readmission rates: Analysis of 70,000 clinical database patient records.* BioMed Research International. [Dataset source]
3. Chawla, N. et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR. [SMOTE methodology]

---
