# Customer Churn Intelligence

Portfolio-quality ML engineering project that predicts telecom customer churn probability and converts it into a business retention decision.

## Problem Statement

Predict the probability that an **active** telecom customer will churn, then apply a **frozen decision policy** to recommend retention outreach. The goal is not accuracy alone — ranking quality (PR-AUC), calibration, and cost-aware thresholding matter for imbalanced churn (~26.5% positive class).

## Learning Guide

New to the project or ML engineering? Start with [docs/00_INDEX.md](docs/00_INDEX.md) — a beginner-friendly, chapter-by-chapter walkthrough of every decision, notebook, and module (what was built, how, and why).

---

## Final Model

| Item | Value |
|------|-------|
| **Model** | XGBoost + class weighting (`scale_pos_weight ≈ 2.77`) |
| **Calibration** | Sigmoid (Platt) — retained |
| **Decision threshold** | **0.10** (validation-tuned on calibrated probabilities) |
| **Saved pipeline** | `models/churn_pipeline.joblib` |
| **Config** | `models/model_config.json` |
| **Probabilities** | Calibrated churn probability |

Hyperparameters (frozen): `n_estimators=200`, `max_depth=3`, `learning_rate=0.1`, `subsample=0.8`.

---

## Final Test Results (Single Evaluation)

Evaluated **once** on the untouched 15% holdout (`n=1,057`) using the frozen pipeline and threshold. See `notebooks/17_final_test_evaluation.ipynb`.

| Metric | Test | Validation (reference) |
|--------|------|------------------------|
| **Accuracy** | 0.641 | 0.633 |
| **Precision** | 0.419 | 0.415 |
| **Recall** | 0.908 | 0.946 |
| **F1** | 0.574 | 0.577 |
| **ROC-AUC** | 0.833 | 0.845 |
| **PR-AUC** | 0.659 | 0.645 |
| **Brier score** | 0.140 | 0.136 |
| **Predicted churners** | 608 | 638 |
| **Threshold** | 0.10 | 0.10 |

**Test confusion matrix** (threshold = 0.10):

|  | Predicted No | Predicted Churn |
|--|--------------|-----------------|
| **Actual No** | 423 (TN) | 353 (FP) |
| **Actual Yes** | 26 (FN) | 255 (TP) |

**Validation vs test:** Metrics are broadly aligned. Test recall is slightly lower (0.908 vs 0.946) with more false negatives (26 vs 15), while PR-AUC is marginally higher on test. Ranking and calibration remain stable (ROC-AUC ~0.83–0.85, Brier ~0.14). Gaps are modest and consistent with sampling variation on ~1k-row holdouts — **not** used to retune the model.

Artifacts: `reports/final_test_metrics.json`, `reports/final_validation_test_comparison.csv`, `reports/figures/15_final_test_confusion_matrix.png`

---

## Architecture

```
customer-churn-intelligence/
├── README.md
├── requirements.txt
├── AGENTS.md
├── data/
│   ├── raw/                  # Original CSV (not in Git)
│   └── processed/            # Cleaned data, split manifest, column roles
├── notebooks/                # EDA → modeling → final test evaluation
├── src/                      # Reusable Python modules
│   ├── data_cleaning.py
│   ├── data_separation.py
│   ├── data_split.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── inference.py
│   ├── policy.py
│   ├── evaluation.py
│   └── config.py
├── models/
│   ├── churn_pipeline.joblib
│   └── model_config.json
├── reports/                  # Metrics, SHAP, threshold analysis, figures
├── app/
│   └── app.py                # Streamlit dashboard
├── api/
│   └── main.py               # FastAPI service
└── tests/                    # Unit + consistency + API tests
```

**Inference flow:** raw customer features → `src/inference.prepare_inference_features()` → saved preprocessor → calibrated XGBoost → probability → frozen threshold → retention decision + risk band (`src/policy.py`).

---

## Dataset

**Source:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

**File:** `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` (7,043 rows × 21 columns)

**Cleaning:** `TotalCharges` coerced to numeric; 11 blank values (all `tenure=0`) imputed to `0.0`. Output: `data/processed/cleaned_churn.csv`

**Features:** 19 predictive columns (4 numeric, 15 categorical). `customerID` is linkage only; `Churn` is the target.

---

## Validation Strategy

| Partition | Share | Rows | Purpose |
|-----------|-------|------|---------|
| Train | 70% | 4,930 | Fit model, preprocessing, calibration |
| Validation | 15% | 1,056 | Model selection, calibration choice, threshold optimization |
| Test | 15% | 1,057 | **Final evaluation only** (this README's test metrics) |

- Stratified two-stage split, `random_state=42`
- Reproducibility: `data/processed/split_manifest.json` (indices only)

---

## Leakage Prevention

- No post-churn columns in features (audit in `notebooks/05_leakage_audit.ipynb`)
- `customerID` never in `X`; `Churn` never in features
- Preprocessing, calibration, and SMOTE (where used) fit on **train only**
- Validation used for tuning; test untouched until policy frozen

---

## Threshold & Business-Cost Logic

Simulated costs (validation optimization): **$50** per unnecessary retention contact (FP), **$500** per missed churner (FN).

- Uncalibrated optimum was 0.26 — **invalid** after sigmoid calibration
- Recalibrated validation optimum: **0.10** (`reports/chosen_threshold.json`)
- At threshold 0.10 on validation: Precision 0.415, Recall 0.946, F1 0.577

**Risk bands** (display/API policy in `src/policy.py` — not model thresholds):

| Band | Rule |
|------|------|
| Low | probability `< 0.10` |
| Elevated | `0.10` ≤ probability `< 0.50` |
| High | probability `≥ 0.50` |

---

## SHAP Explainability

`notebooks/16_shap_explainability.ipynb` — TreeExplainer on XGBoost; calibrated probabilities for local cases.

**Top global influences (validation):** Contract Month-to-month, tenure, MonthlyCharges, OnlineSecurity No, TotalCharges.

SHAP describes **model influence**, not causal churn drivers.

---

## Streamlit Dashboard

```bash
streamlit run app/app.py
```

Interactive scoring for all 19 raw features, model summary, calibrated probability, risk level, and retention recommendation.

### Batch upload (Streamlit)

Use the **Batch upload** tab to:

1. Upload a CSV and select a **primary key** column
2. Run a **compatibility check** against the saved Telco model schema
3. If compatible, score all customers and **download ranked predictions** (probability, risk level, retention flag, customer details)
4. If incompatible, review fix suggestions, download the schema template from `data/templates/telco_scoring_template.csv`, or use the optional **session-only retrain fallback** when your file includes a churn label column

The frozen production model in `models/` is never overwritten by upload scoring or session training.

---

## FastAPI Endpoint

```bash
uvicorn api.main:app --reload
```

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service and model status |
| POST | `/predict_churn` | Score one customer (Pydantic-validated JSON) |

Example response fields: `churn_probability`, `threshold`, `prediction`, `risk_level`, `recommended_action`, `model_version`.

---

## Tests

```bash
pytest tests/ -v
```

Coverage includes data cleaning, splits, preprocessing, inference, policy consistency, saved artifact loading, API parity with Python inference, and unseen-category handling. **63 tests** (validation/test artifacts; no test-set tuning in tests).

---

## Limitations

- Single static snapshot dataset; no temporal production logs
- Simulated business costs — not validated against real campaign ROI
- Class-weighted XGBoost prioritizes recall; low threshold increases false-positive contacts
- SHAP explains tree scores, not calibrated layer or causality
- Risk bands above 0.10 are UI/demo escalation bands, not re-tuned model thresholds

---

## Monitoring Plan

| Signal | What to monitor | Action |
|--------|-----------------|--------|
| **Feature drift** | Distribution shift vs training/validation baselines (PSI, KS) on key fields (`tenure`, `Contract`, `MonthlyCharges`, service flags) | Alert if PSI > 0.2; review data pipeline and retrain if sustained |
| **Prediction-score drift** | Mean/median calibrated `P(churn)`, score histograms, fraction above threshold 0.10 | Alert on >2σ shift week-over-week; check upstream feature changes |
| **Churn-rate drift** | Realized churn rate vs ~26.5% training prior | Investigate market/regulatory changes; may require policy review (not automatic retuning) |
| **Calibration / performance drift** | Rolling Brier score, PR-AUC on labeled holdout; precision/recall at threshold 0.10 | Quarterly recalibration evaluation on fresh validation slice; retrain only through governed ML lifecycle |

Log per prediction: timestamp, `customerID`, features hash, probability, threshold, decision, model version.

---

## How to Run the Project

### 1. Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Download the dataset to `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

### 2. Reproduce processed data (optional)

```bash
python -c "from src.data_cleaning import run_cleaning_pipeline; run_cleaning_pipeline()"
python -c "from src.data_separation import run_separation_pipeline; run_separation_pipeline()"
python -c "from src.data_split import run_split_pipeline; run_split_pipeline()"
```

### 3. Train/save pipeline (if artifacts missing)

```bash
python -c "from src.model import run_training_pipeline; run_training_pipeline()"
```

Pre-built artifacts are in `models/` for inference apps.

### 4. Run apps

```bash
streamlit run app/app.py
uvicorn api.main:app --reload
```

### 5. Tests

```bash
pytest tests/ -v
```

### 6. Notebooks

Execute in order (`01_eda` … `17_final_test_evaluation`) for the full analytical narrative.

---

## Project Lifecycle (Completed)

Problem definition → EDA → leakage audit → stratified split → preprocessing → baselines → Logistic Regression → Random Forest → XGBoost → imbalance experiments → model comparison → calibration → threshold optimization → SHAP → reusable `src/` → saved pipeline → Streamlit → FastAPI → tests → **final test evaluation**.

See `AGENTS.md` for permanent development rules.
