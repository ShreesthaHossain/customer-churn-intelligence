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

## Current Project Status

**Stage 7: Leakage Audit complete**

Steps 1–7 complete: project setup, data understanding, cleaning, feature/target/ID separation, focused EDA, and leakage audit. No train/test split or modeling yet.
