# Chapter 07 — Baselines & Model Progression

**Previous:** [Chapter 06](06_preprocessing_and_pipelines.md) | **Next:** [Chapter 08 — Hyperparameter Tuning](08_hyperparameter_tuning.md)

---

## What You Will Learn

- Why you always start with a **dummy baseline**
- The required model progression for this project
- Validation metrics for each model family
- Why XGBoost became the final candidate (before imbalance/calibration refinements)

---

## Why Baselines Matter

A **baseline** is the simplest reasonable model. It answers: *Does our fancy model beat trivial strategies?*

If XGBoost cannot beat "always predict No Churn," something is fundamentally wrong.

---

## Beginner Box: What Is a Baseline?

The **DummyClassifier** with `strategy="most_frequent"` always predicts the majority class (`No`). It establishes the **floor** for metrics.

---

## Required Model Progression

Per `AGENTS.md`, models are tried **in order**:

1. **DummyClassifier** — `08_dummy_baseline.ipynb`
2. **Logistic Regression** — `09_logistic_regression.ipynb`
3. **Random Forest** — `10_random_forest.ipynb`
4. **XGBoost** — `11_xgboost.ipynb`

Not included (by design):

- LightGBM (XGBoost sufficient)
- Neural networks (no demonstrated need)
- Extra model families "for count"

---

## Validation Results (`reports/model_comparison.csv`)

All metrics on the **validation set** (1,056 rows), default or tuned configs from notebooks:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|----------|-----------|--------|-----|---------|--------|
| DummyClassifier | 0.735 | 0.0 | 0.0 | 0.0 | 0.5 | 0.265 |
| LogisticRegression | 0.806 | 0.646 | 0.593 | 0.618 | 0.845 | 0.631 |
| LogisticRegression (balanced) | 0.740 | 0.506 | **0.814** | 0.624 | 0.845 | 0.631 |
| RandomForest | 0.773 | 0.548 | 0.811 | **0.654** | 0.843 | 0.627 |
| XGBoost | 0.762 | 0.533 | 0.829 | 0.649 | 0.845 | **0.640** |

---

## Model Roles

### DummyClassifier

- **Role:** Floor
- **Lesson:** 73.5% accuracy with **zero recall** — never use accuracy alone

### Logistic Regression

- **Role:** Interpretable linear baseline
- **Strength:** Coefficients show direction of effect; strong PR-AUC (~0.631)
- **Balanced variant:** `class_weight="balanced"` boosts recall to 0.814
- **Kept as backup** for explainability if stakeholders need linear stories

### Random Forest

- **Role:** Nonlinear ensemble comparison
- **Strength:** Highest F1 (0.654) among defaults; good recall (0.811)
- **Weakness:** Slightly lower PR-AUC than XGBoost

### XGBoost

- **Role:** Primary boosted-tree candidate
- **Strength:** Best PR-AUC (0.640) and strong recall (0.829) before dedicated imbalance tuning
- **Selected** for production path (Chapters 09–10)

---

## Beginner Box: PR-AUC vs ROC-AUC (Imbalanced Data)

- **ROC-AUC** — True Positive Rate vs False Positive Rate across thresholds
- **PR-AUC** — Precision vs Recall across thresholds

When positives are minority (~26.5% churn), PR-AUC better reflects how well the model finds churners without flooding false alarms.

Both are reported; **PR-AUC** is the primary ranking metric for model selection.

---

## Why Not Other Models?

| Alternative | Reason |
|-------------|--------|
| LightGBM | Similar to XGBoost; one gradient boosting library enough |
| CatBoost | Not required; adds dependency without portfolio mandate |
| Neural network (MLP) | ~7k tabular rows; trees win with less tuning |
| SVM | Poor scalability / probability calibration path for this workflow |
| Stacking many models | Over-engineering for dataset size |

---

## Pipeline Pattern in Notebooks

Each notebook wraps:

```
Preprocessor (fit train) → Classifier → predict_proba on validation
```

Same `ColumnTransformer` from Chapter 06 ensures fair comparison.

---

## Decision at This Stage

After notebook 11, **XGBoost** leads on PR-AUC. Chapter 09 refines imbalance handling; Chapter 10 formalizes selection.

Logistic Regression (balanced) remains the **interpretable alternative** documented in the comparison notebook.

---

## Hands-On

1. Run notebooks `08` through `11` in order.
2. Open `reports/model_comparison.csv` — verify DummyClassifier recall = 0.
3. Plot validation PR curves if cells exist in notebook 13 preview.

---

## Check Your Understanding

1. Why must every project beat a dummy baseline?
2. Which model has the highest PR-AUC in the comparison table?
3. Why might Random Forest have higher accuracy than XGBoost but lower PR-AUC?
4. Why did we not add a neural network?

---

**Next:** [Chapter 08 — Hyperparameter Tuning](08_hyperparameter_tuning.md)
