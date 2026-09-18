# Chapter 09 — Class Imbalance

**Previous:** [Chapter 08](08_hyperparameter_tuning.md) | **Next:** [Chapter 10 — Model Selection](10_model_selection.md)

---

## What You Will Learn

- What **class imbalance** means for churn modeling
- Four strategies compared in this project
- Why **class weighting** won for production
- Leakage-safe rules for SMOTE and undersampling

---

## The Imbalance Problem

~**26.5%** of customers churn (Yes); ~**73.5%** do not (No). Models can achieve high accuracy by ignoring the minority class.

**Goal:** Improve detection of churners (recall / PR-AUC) without destroying ranking quality.

---

## Beginner Box: Class Imbalance Terms

| Term | Meaning |
|------|---------|
| **Majority class** | No Churn (~73.5%) |
| **Minority class** | Yes Churn (~26.5%) |
| **Class weighting** | Penalize mistakes on minority class more heavily |
| **Undersampling** | Remove majority-class rows from training |
| **Oversampling / SMOTE** | Create synthetic minority-class examples |
| **scale_pos_weight** | XGBoost parameter: weight for positive class |

---

## Four Strategies Tested

Notebook: `12_imbalance_experiments.ipynb`  
Report: `reports/imbalance_comparison.csv`

### XGBoost results (validation)

| Strategy | Precision | Recall | F1 | PR-AUC |
|----------|-----------|--------|-----|--------|
| Original (no weight) | 0.648 | 0.546 | 0.593 | 0.634 |
| **Class weighting** | 0.533 | **0.829** | **0.649** | **0.640** |
| RandomUnderSampler | 0.510 | 0.846 | 0.636 | 0.623 |
| SMOTENC | 0.545 | 0.689 | 0.609 | 0.605 |

### Logistic Regression results (validation)

| Strategy | PR-AUC |
|----------|--------|
| Class weighting | **0.631** |
| Original | 0.631 |
| RandomUnderSampler | 0.631 |
| SMOTENC | 0.575 |

---

## Winner: Class Weighting

**Production choice:** XGBoost with `scale_pos_weight = neg_count / pos_count ≈ 2.77`

**Why it won:**

- Highest XGBoost PR-AUC (0.640)
- Strong recall (0.829) — aligns with business cost asymmetry (Chapter 01)
- No synthetic data — simpler, less leakage risk
- Implemented natively in XGBoost and sklearn

Code: `src/model.py` → `compute_scale_pos_weight()`, `build_xgb_classifier()`.

---

## Strategy Details

### 1. Original distribution

Baseline — no adjustment. High precision, lower recall.

### 2. Class weighting

- **LR:** `class_weight="balanced"`
- **XGB:** `scale_pos_weight = n_negative / n_positive`

Tells the model: "Missing a churner is worse than a false alarm."

### 3. RandomUnderSampler

Randomly drops majority-class training rows until classes are balanced.

**Downside:** Throws away real No-Churn examples → lost information → PR-AUC dropped to 0.623 for XGBoost.

### 4. SMOTENC

Synthetic Minority Over-sampling for **N**ominal and **C**ontinuous features — creates synthetic churners in feature space.

**Rules followed:**

- Applied **after** train/val/test split
- Applied **only** to training data inside `imblearn.pipeline.Pipeline`
- Never on validation or test

**Downside:** Hurt PR-AUC (0.605 for XGBoost) — synthetic points did not help ranking on this dataset.

---

## Why SMOTENC Over Vanilla SMOTE?

Vanilla SMOTE assumes continuous features. This dataset has 15 categorical columns. **SMOTENC** handles mixed types correctly.

Still, it underperformed class weighting here — a common outcome on small tabular datasets where weighting suffices.

---

## Why Not Other Approaches?

| Alternative | Why not |
|-------------|---------|
| SMOTE before split | **Leakage** — synthetic points influenced by full distribution |
| SMOTE on val/test | Invalid evaluation — val/test must reflect reality |
| Oversample validation | Validation must stay natural for honest metrics |
| Ignore imbalance | Misses churners — bad for $500 FN cost |
| Only use accuracy to compare | Misleading under imbalance |

---

## Production Implementation

```python
# src/model.py (conceptual flow)
scale_pos_weight = compute_scale_pos_weight(split.y_train)
model = XGBClassifier(**XGB_FIXED_PARAMS, scale_pos_weight=scale_pos_weight)
```

Class weighting is part of the **final saved pipeline** in `models/churn_pipeline.joblib`.

---

## Hands-On

1. Run `notebooks/12_imbalance_experiments.ipynb`.
2. Compare `reports/imbalance_comparison.csv` XGBoost rows.
3. Verify SMOTE cells apply only to training pipeline branches.

---

## Check Your Understanding

1. What is the churn rate (approximate) in this dataset?
2. Why did class weighting beat SMOTENC on PR-AUC?
3. Why must SMOTE never be applied before the train/test split?
4. What does `scale_pos_weight ≈ 2.77` tell XGBoost?

---

**Next:** [Chapter 10 — Model Selection](10_model_selection.md)
