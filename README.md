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

**Planned flow:** data understanding → EDA → leakage-safe preprocessing → baseline and advanced models → calibration and threshold optimization → SHAP explainability → saved pipeline → Streamlit + FastAPI → tests and monitoring.

## Current Project Status

**Stage 1: Data Understanding**

Initial repository structure is in place. No exploratory analysis or modeling has been run yet.
