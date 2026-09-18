# Chapter 08 — Hyperparameter Tuning

**Previous:** [Chapter 07](07_baselines_and_model_progression.md) | **Next:** [Chapter 09 — Class Imbalance](09_class_imbalance.md)

---

## What You Will Learn

- The difference between **parameters** and **hyperparameters**
- How grid search worked in this project
- Frozen XGBoost settings in production
- Why tuning uses validation F1 — and never the test set

---

## Parameters vs Hyperparameters

| Type | Set by | Examples |
|------|--------|----------|
| **Parameters** | Training (learning from data) | Tree split thresholds, logistic coefficients |
| **Hyperparameters** | Human / search **before** training | `max_depth`, `learning_rate`, `n_estimators` |

You cannot learn `max_depth` from data directly — you choose it, train, evaluate, and iterate.

---

## Tuning Approach in This Project

**Method:** Manual grid search with `itertools.product` in notebooks — **not** `GridSearchCV`.

**Selection metric:** Validation **F1** (notebooks 10 and 11)

**Data used:** Training partition only for fitting; validation for scoring each combination

> **Sacred test set:** Hyperparameters were **not** chosen using test metrics.

---

## Random Forest Grid (`10_random_forest.ipynb`)

```python
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 5],
    "class_weight": [None, "balanced"],
}
```

**Best validation F1 configuration:**

- `n_estimators=200`
- `max_depth=10`
- `min_samples_leaf=5`
- `class_weight="balanced"`

---

## XGBoost Grid (`11_xgboost.ipynb`)

```python
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "scale_pos_weight": [1.0, neg/pos, neg/pos * 1.2],
}
```

**Frozen production hyperparameters** (`src/config.py` → `XGB_FIXED_PARAMS`):

| Hyperparameter | Value | Plain-English meaning |
|----------------|-------|----------------------|
| `n_estimators` | 200 | Number of boosting trees |
| `max_depth` | 3 | Max depth per tree (shallow → less overfit) |
| `learning_rate` | 0.1 | Step size each tree contributes |
| `subsample` | 0.8 | Each tree uses 80% of rows |
| `colsample_bytree` | 1.0 | Each tree uses all feature columns |
| `scale_pos_weight` | ~2.77 | Computed at train time: negatives/positives |

`scale_pos_weight` is set dynamically in `src/model.py` via `compute_scale_pos_weight(y_train)`.

---

## Hyperparameter Intuition

### max_depth = 3 (shallow)

With ~4,930 training rows, deep trees memorize noise. Depth 3 balances expressiveness and generalization.

### learning_rate = 0.1

Higher rate trains faster; paired with 200 trees gives sufficient capacity.

### subsample = 0.8

Row subsampling adds randomness — mild regularization.

### n_estimators = 200

More trees than 100 improved validation F1 in grid; diminishing returns beyond grid range not explored (scope control).

---

## Why Not Other Tuning Methods?

| Alternative | Why not |
|-------------|---------|
| Tune on test set | Overfits threshold to test noise — forbidden |
| Optuna / Hyperopt | Valid tools; manual grid sufficient for portfolio scope |
| GridSearchCV | Could be used; notebooks use explicit loops for transparency |
| Bayesian optimization | Overkill for ~7k rows and small grid |
| Deeper trees (max_depth 10+) | Overfit risk on small tabular data |
| Retune after every experiment | Hyperparams frozen after notebook 11; imbalance handled separately (Ch 09) |

---

## From Notebook to Production

After tuning in notebooks, values are **copied to** `src/config.py`. `src/model.py` does **not** re-run grid search — it trains once with frozen params.

This separation means:

- Reproducible training: `run_training_pipeline()`
- Clear audit trail: config file matches notebook winner

---

## Hands-On

1. Open `notebooks/11_xgboost.ipynb` — find the grid results table.
2. Compare `src/config.py` `XGB_FIXED_PARAMS` to the notebook winner.
3. Run `pytest tests/test_model.py -v` — verifies training uses train only.

---

## Check Your Understanding

1. What is the difference between a learned tree split and `max_depth`?
2. Why was validation F1 used to pick hyperparameters?
3. What does `scale_pos_weight ≈ 2.77` represent?
4. Why are hyperparameters frozen in `config.py` instead of searching every training run?

---

**Next:** [Chapter 09 — Class Imbalance](09_class_imbalance.md)
