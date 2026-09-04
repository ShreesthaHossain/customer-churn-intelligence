# Customer Churn Intelligence

## Problem Statement

Predict the probability that an active telecom customer will churn, then convert that probability into a business retention decision (who to contact, when, and at what cost trade-off).

## Dataset

**Source:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

**Expected file:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`

**Location:** Place the downloaded CSV in `data/raw/` (do not modify the original file in place).

Raw data files are excluded from version control. Clone this repository, download the dataset from the link above, and save it to `data/raw/` before running any analysis.

## Planned Project Architecture

```
customer-churn-intelligence/
├── README.md              # Project overview and documentation
├── requirements.txt       # Python dependencies
├── AGENTS.md              # Permanent ML development rules
├── data/
│   ├── raw/               # Original dataset (not tracked in Git)
│   └── processed/         # Cleaned / feature-engineered outputs
├── notebooks/             # Exploration, EDA, and reporting
├── src/                   # Reusable production Python code
├── models/                # Saved trained pipelines
├── reports/
│   └── figures/           # Generated plots and report assets
├── app/                   # Streamlit dashboard (later stage)
├── api/                   # FastAPI inference endpoint (later stage)
└── tests/                 # Unit and integration tests
```

**Planned flow:** data understanding → EDA → leakage audit → train/validation/test split → leakage-safe preprocessing → baseline and advanced models → calibration and threshold optimization → SHAP explainability → saved pipeline → Streamlit + FastAPI → tests and monitoring.

## Leakage Prevention

Churn predictions are made for **active customers at a snapshot in time**. Features must reflect information available **before** the churn outcome — never post-cancellation data (e.g., cancellation date, termination reason, final invoice).

| Column role | Column(s) | Rule |
|-------------|-----------|------|
| **Identifier** | `customerID` | Retained for linking predictions; **never** used as a model feature |
| **Target** | `Churn` | Outcome label only; **never** included in the feature matrix |
| **Features** | 19 approved columns (see `src/data_separation.py`) | Allowed at prediction time; strong correlation with churn is **not** grounds for removal |

**Audit result (Step 7):** No post-outcome leakage columns were found in the Telco dataset. Future process safeguards include: fit preprocessing on training data only (sklearn `Pipeline`), keep the final test set untouched until the model and threshold policy are frozen, apply SMOTE only after splitting and inside cross-validation on training folds, and tune thresholds/calibration on validation data only.

See `notebooks/05_leakage_audit.ipynb` for the full column-by-column audit.

## Validation Strategy

| Partition | Share | Rows (approx.) | Purpose |
|-----------|-------|----------------|---------|
| **Training** | 70% | 4,930 | Fit models and **all learned preprocessing** (scaling, encoding, imputation) |
| **Validation** | 15% | 1,056 | Model comparison, hyperparameter tuning, calibration, threshold optimization |
| **Test** | 15% | 1,057 | **Final evaluation only** — untouched until model + decision policy are frozen |

- **Split method:** Stratified two-stage split (`train_test_split` → 70% train vs 30% holdout, then holdout split 50/50 into validation and test)
- **`random_state=42`** for reproducibility
- **Stratification:** `Churn` class proportions preserved across all three sets (~26.5% Yes)
- **Artifacts:** `data/processed/split_manifest.json` (indices + metadata only; no preprocessed test CSV)
- **Preprocessing policy:** All transformers fitted on **training data only**, applied via sklearn `Pipeline`

See `notebooks/06_data_split.ipynb` for split verification.

## Current Project Status

**Stage 8: Train/Validation/Test Split complete**

Steps 1–8 complete through stratified data splitting. No preprocessing or modeling yet.
