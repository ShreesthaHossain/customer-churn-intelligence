# Chapter 10 — Model Selection

**Previous:** [Chapter 09](09_class_imbalance.md) | **Next:** [Chapter 11 — Probability Calibration](11_probability_calibration.md)

---

## What You Will Learn

- How to compare models **fairly** on the same validation set
- The selection criteria used to pick the final model
- Why XGBoost + class weighting became production model
- Why highest accuracy is not the selection rule

---

## Model Comparison Notebook

`notebooks/13_model_comparison.ipynb` consolidates results after:

- Hyperparameter tuning (Chapter 08)
- Imbalance experiments (Chapter 09)

Report artifacts:

- `reports/model_comparison.csv` — broader comparison
- `reports/final_model_comparison.csv` — final contenders with class weighting

---

## Final Contenders (Validation, Class-Weighted Where Applicable)

| Model | PR-AUC | Recall | F1 | Notes |
|-------|--------|--------|-----|-------|
| **XGBoost + class weighting** | **0.640** | **0.829** | 0.649 | **Selected** |
| Logistic Regression (balanced) | 0.631 | 0.814 | 0.624 | Interpretable backup |
| Random Forest (tuned, balanced) | 0.627 | 0.811 | 0.654 | Strong F1, lower PR-AUC |

---

## Selection Criteria (In Order)

1. **PR-AUC** — primary ranking quality for minority class
2. **Recall at reasonable precision** — catch churners without unbounded false alarms
3. **Stability & deployability** — sklearn-compatible pipeline, reasonable training time
4. **Interpretability backup** — LR available if stakeholders need linear explanations

> **Sacred test set:** Model selection used **validation only**. Test evaluated once in Chapter 16.

---

## Why XGBoost Won

- Best validation **PR-AUC** among compared models
- Strong **recall** (0.829) with class weighting — aligns with $500 FN vs $50 FP costs
- Handles nonlinear interactions (contract × services × tenure) without manual feature engineering
- Mature serialization path with sklearn calibration wrapper

---

## Why Not Highest Accuracy?

Unweighted Logistic Regression has **80.6% accuracy** vs XGBoost weighted **76.2%**:

| Model | Accuracy | Recall | PR-AUC |
|-------|----------|--------|--------|
| LR (default) | **0.806** | 0.593 | 0.631 |
| XGBoost (weighted) | 0.762 | **0.829** | **0.640** |

For retention, **missing churners** is costlier than extra contacts. Accuracy rewards "predict No" — wrong objective.

---

## Backup Model: Logistic Regression (Balanced)

Near-identical PR-AUC to RF; coefficients easier to explain to non-technical stakeholders.

Not deployed as primary because XGBoost edges PR-AUC and recall. Documented as alternative in comparison materials.

---

## Why Not Other Selection Rules?

| Alternative | Why rejected |
|-------------|--------------|
| Pick highest accuracy | Ignores churn capture |
| Pick highest precision only | Misses too many churners |
| Ensemble everything | Complexity without clear PR-AUC gain |
| Select on test set | Overfits selection to test |
| Use ROC-AUC alone | Less informative than PR-AUC here |

---

## What Gets Frozen After Selection

After this step, the project commits to:

- **Model family:** XGBoost
- **Imbalance handling:** Class weighting (`scale_pos_weight`)
- **Hyperparameters:** From `src/config.py`

Next steps refine **probability calibration** (Chapter 11) and **decision threshold** (Chapter 12) — not the core tree model family.

---

## Hands-On

1. Run `notebooks/13_model_comparison.ipynb`.
2. Open `reports/final_model_comparison.csv`.
3. Articulate in one sentence why XGBoost beat LR on the business objective.

---

## Check Your Understanding

1. What was the primary metric for final model selection?
2. Why does Logistic Regression have higher accuracy but lower recall than weighted XGBoost?
3. When is the test set used in the selection process?
4. What model serves as the interpretable backup?

---

**Next:** [Chapter 11 — Probability Calibration](11_probability_calibration.md)
