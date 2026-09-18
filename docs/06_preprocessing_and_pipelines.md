# Chapter 06 — Preprocessing & Pipelines

**Previous:** [Chapter 05](05_train_val_test_split.md) | **Next:** [Chapter 07 — Baselines & Model Progression](07_baselines_and_model_progression.md)

---

## What You Will Learn

- How raw features become numeric matrices for ML
- What **imputation**, **scaling**, and **one-hot encoding** do
- Why we use sklearn `Pipeline` and `ColumnTransformer`
- How inference uses the **exact same** preprocessing as training

---

## Preprocessing Overview

Models cannot consume raw strings like `"Fiber optic"` or `"Month-to-month"` directly. Preprocessing converts the 19-column DataFrame into a numeric matrix.

| Column type | Steps | sklearn components |
|-------------|-------|-------------------|
| **Numeric** (4 cols) | Impute missing → scale | `SimpleImputer(median)` → `StandardScaler` |
| **Categorical** (15 cols) | Impute missing → one-hot encode | `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown="ignore")` |

Built in `src/preprocessing.py`, function `build_preprocessor()`.

---

## Beginner Box: Key Preprocessing Terms

| Term | Meaning | Example |
|------|---------|---------|
| **Imputation** | Fill missing values | Median tenure if blank |
| **Scaling** | Rescale numbers to similar ranges | MonthlyCharges 20–120 → z-scores |
| **One-hot encoding** | Convert categories to 0/1 columns | `Contract_Month-to-month = 1` |
| **Fit** | Learn parameters from training data | Compute median, category list |
| **Transform** | Apply learned parameters | Scale validation rows |

**Critical rule:** `fit` on **train only**; `transform` on val/test.

---

## ColumnTransformer Design

```python
ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURE_COLS),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURE_COLS),
    ],
    remainder="drop",
)
```

Each branch is its own mini-`Pipeline`. After fit, ~19 raw columns expand to **more columns** (one per category level in one-hot encoding).

`get_transformed_feature_names()` returns names like `cat__Contract_Month-to-month` for interpretability.

---

## Why StandardScaler?

Logistic Regression is **scale-sensitive** — features with larger numeric ranges dominate. Tree models (Random Forest, XGBoost) are less sensitive, but scaling:

- Keeps one preprocessor compatible with **all** model types in notebooks
- Does not hurt tree models materially

---

## Why OneHotEncoder(handle_unknown="ignore")?

At inference, a customer might have a category level **not seen in training** (rare in this dataset). `handle_unknown="ignore"` sets all dummy columns for that feature to 0 instead of crashing.

Tests in `tests/test_preprocessing.py` verify unseen category handling.

---

## TransformedSplits Container

`fit_transform_train()` returns a `TransformedSplits` dataclass:

- `X_train`, `X_val`, `X_test` as numpy arrays
- `feature_names` list
- Fitted `preprocessor` object (saved in model bundle)

---

## Why sklearn Pipeline?

| Benefit | Explanation |
|---------|-------------|
| **No duplication** | Same code in notebooks, training script, Streamlit, FastAPI |
| **Leakage prevention** | Fit/transform boundary is explicit |
| **Persistence** | Save preprocessor inside `ChurnModelBundle` |

---

## Why Not Other Approaches?

| Alternative | Why not |
|-------------|---------|
| Manual pandas `get_dummies` in each notebook | Duplicated logic; easy to mismatch train/infer |
| LabelEncoder for categoricals | Implies false order (Contract One year < Two year) |
| Target encoding | Uses target labels — leakage risk without careful CV |
| Fit scaler on full dataset | Test statistics leak (Chapter 04) |
| Drop categoricals, numeric only | Loses contract type, services — strongest signals |

---

## Inference Path

At prediction time (`src/inference.py`):

1. `prepare_inference_features()` — validate raw record
2. `transform_features(preprocessor, X)` — **transform only**, no refit
3. Model `predict_proba()`

The saved `preprocessor` in `models/churn_pipeline.joblib` is the training-time object.

---

## Hands-On

1. Run `notebooks/07_preprocessing.ipynb` — compare raw vs transformed shapes.
2. Read `src/preprocessing.py` — trace `build_preprocessor()` and `fit_transform_train()`.
3. Run `pytest tests/test_preprocessing.py -v`.

---

## Check Your Understanding

1. Why must StandardScaler learn the mean/std from training data only?
2. What does one-hot encoding produce for the `Contract` column?
3. What happens at inference if an unknown payment method appears?
4. Why is ColumnTransformer preferred over preprocessing each column manually?

---

**Next:** [Chapter 07 — Baselines & Model Progression](07_baselines_and_model_progression.md)
