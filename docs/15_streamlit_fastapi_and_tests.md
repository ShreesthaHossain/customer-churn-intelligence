# Chapter 15 — Streamlit, FastAPI & Tests

**Previous:** [Chapter 14](14_production_engineering.md) | **Next:** [Chapter 16 — Monitoring & Limitations](16_monitoring_and_limitations.md)

---

## What You Will Learn

- How the **same saved pipeline** powers Streamlit and FastAPI
- API design and input validation
- What the **63 tests** guarantee
- How to run a project health audit

---

## Single Pipeline, Two Surfaces

Both apps call:

```python
from src.model import load_churn_pipeline
from src.config import load_model_config
```

| Surface | Path | Run command |
|---------|------|-------------|
| **Streamlit** | `app/app.py` | `streamlit run app/app.py` |
| **FastAPI** | `api/main.py` | `uvicorn api.main:app --reload` |

No duplicate preprocessing — both delegate to `src/inference.py`.

---

## Beginner Box: Inference Endpoint

An **inference endpoint** is a program entry point that accepts customer features and returns a prediction. FastAPI exposes this over HTTP; Streamlit exposes it through a web form.

---

## Streamlit Dashboard

**Features:**

- Form for all **19 raw features** with valid categorical options
- Sidebar: model summary, threshold 0.10, validation reference metrics
- `@st.cache_resource` loads pipeline once per session
- On submit: `predict_single_customer(pipeline, record)`
- Displays: probability, risk band, binary decision, retention recommendation
- Default form values from a **validation** row (not test)

**Service normalization:** `normalize_service_fields()` — if `InternetService=No`, dependent fields auto-set to `"No internet service"`.

---

## FastAPI Service

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service status, `model_loaded`, `model_version` |
| POST | `/predict_churn` | Score one customer |

### Lifespan

Pipeline + config loaded **once at startup** into `app.state` — not per request.

### Pydantic Validation

`CustomerFeatures` model enforces:

- Literal enums for categoricals (exact spelling)
- `tenure` 0–100, `SeniorCitizen` 0/1
- `TotalCharges` rules (blank only if tenure=0)
- Internet/phone service consistency (422 on invalid combos)

### Example Request

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 89.5,
  "TotalCharges": 1074.0,
  "customerID": "CUST-001"
}
```

### Example Response Fields

```json
{
  "churn_probability": 0.42,
  "threshold": 0.1,
  "prediction": "Churn",
  "risk_level": "Elevated",
  "recommended_action": "Recommend retention outreach — offer proactive save campaign.",
  "model_version": "XGBoost + class weighting|sigmoid (Platt)|threshold=0.1",
  "customerID": "CUST-001"
}
```

Built by `build_churn_prediction_response()` in `src/inference.py`.

---

## Test Suite (63 Tests)

Run: `pytest tests/ -v`

| Module | What it guards |
|--------|----------------|
| `test_data_cleaning.py` | Cleaning pipeline, audit |
| `test_data_separation.py` | 19 features, no ID in X |
| `test_data_split.py` | Stratified sizes, no overlap |
| `test_preprocessing.py` | Fit train only, unseen categories |
| `test_model.py` | Train on train, save/load bundle |
| `test_inference.py` | Feature prep, batch predict |
| `test_policy.py` | Risk bands, threshold alignment |
| `test_evaluation.py` | Metrics, cost sweep |
| `test_consistency.py` | Artifact threshold alignment |
| `test_api.py` | **API output == direct Python inference** |
| `test_config.py` | Threshold config loading |

**Fixtures** (`tests/conftest.py`): use validation data — never tune on test in tests.

Tests skip API/consistency checks if `models/churn_pipeline.joblib` is missing.

---

## Project Health Audit

Optional script: `scripts/audit_project.py`

```bash
python scripts/audit_project.py
```

Checks:

- Required artifacts exist (pipeline, config, threshold, metrics, data)
- Threshold aligned across bundle, config, chosen JSON (= 0.10)
- Split sizes 4930 / 1056 / 1057
- Validation & test probabilities in [0, 1]
- Inference response keys present
- FastAPI import smoke test

Exits with code 1 on any failure — useful after reproducing artifacts.

---

## Why Test ML Code?

ML bugs are silent: wrong preprocessing still "runs." Tests enforce:

- Same probability from API and Python path
- Threshold consistent in all artifacts
- Unseen categories don't crash encoder

---

## Hands-On

1. Start Streamlit — score a month-to-month, low-tenure customer.
2. Start FastAPI — POST to `/predict_churn` with the JSON above.
3. Run `pytest tests/test_api.py tests/test_consistency.py -v`.
4. Run `python scripts/audit_project.py`.

---

## Check Your Understanding

1. Where is the pipeline loaded in FastAPI — per request or at startup?
2. What test ensures API and Python inference match?
3. Why does Pydantic validate service field combinations?
4. What does the audit script check about threshold 0.10?

---

**Next:** [Chapter 16 — Monitoring & Limitations](16_monitoring_and_limitations.md)
