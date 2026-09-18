# Chapter 13 — SHAP Explainability

**Previous:** [Chapter 12](12_threshold_and_business_cost.md) | **Next:** [Chapter 14 — Production Engineering](14_production_engineering.md)

---

## What You Will Learn

- What **SHAP values** measure
- Top feature influences in the final XGBoost model
- The difference between **model influence** and **causality**
- Why SHAP is reporting-only (not in the API)

---

## Why Explainability?

Stakeholders ask: *"Why did the model flag this customer?"*

**SHAP (SHapley Additive exPlanations)** assigns each feature a contribution to a prediction relative to a baseline. It helps audit whether the model uses sensible signals.

---

## Beginner Box: What SHAP Is (and Is Not)

| SHAP is… | SHAP is not… |
|----------|--------------|
| A measure of **feature influence on model output** | Proof that changing a feature **causes** churn |
| Useful for debugging and trust | A replacement for A/B tests or causal studies |
| Local (one customer) and global (dataset average) | Applied to the calibrated layer in this project |

---

## Implementation in This Project

Notebook: `16_shap_explainability.ipynb`

| Choice | Detail |
|--------|--------|
| Explainer | `shap.TreeExplainer` on **uncalibrated XGBoost** |
| Data | Validation set (transformed features) |
| Outputs | `reports/shap_global_importance.csv`, `reports/shap_local_explanations.json` |
| Figures | `reports/figures/12_*` – `14_*` (when notebooks run) |

**Why uncalibrated trees?** TreeExplainer works natively on XGBoost base model. The calibration layer is a monotonic transform — global influence patterns remain similar, but local SHAP on calibrated probabilities is not computed here.

---

## Top Global Features (Mean |SHAP|)

From `reports/shap_global_importance.csv`:

| Rank | Feature | Mean \|SHAP\| |
|------|---------|---------------|
| 1 | Contract Month-to-month | 0.707 |
| 2 | tenure | 0.498 |
| 3 | MonthlyCharges | 0.297 |
| 4 | OnlineSecurity No | 0.251 |
| 5 | TotalCharges | 0.210 |
| 6 | InternetService Fiber optic | 0.202 |
| 7 | TechSupport No | 0.175 |
| 8 | PaymentMethod Electronic check | 0.168 |

**Story aligns with EDA (Chapter 03):** month-to-month contracts, low tenure, higher charges, missing security/support → higher model scores.

---

## Local Explanations

The notebook generates case studies:

- **High risk** customer — features pushing probability up
- **Low risk** customer — features pushing probability down
- **Near-threshold** customer — balanced contributions

Saved in `reports/shap_local_explanations.json` for portfolio review.

---

## Safe Stakeholder Language

**Say:** "The model weighted month-to-month contract heavily for this prediction."

**Do not say:** "Month-to-month contract **causes** churn — force them to two-year plans."

Interventions require business experiments, not SHAP alone.

---

## Why Not in Production API?

| Reason | Explanation |
|--------|-------------|
| Latency | SHAP adds compute per request |
| Scope | Portfolio focuses on scoring + policy |
| Calibration gap | Explainer on uncalibrated base model |
| Complexity | TreeExplainer requires transformed feature matrix |

SHAP is a **notebook/reporting** deliverable. Production serves probability + policy from `src/inference.py`.

---

## Why Not Other Explainability Tools?

| Tool | Why not primary here |
|------|---------------------|
| LIME | Less consistent for tree ensembles at scale |
| Feature importance (gain) | Less rigorous than SHAP for local cases |
| Partial dependence only | Shows marginal effect, not full allocation |
| Causal inference (DoWhy) | Out of scope; needs experimental design |

---

## Hands-On

1. Run `notebooks/16_shap_explainability.ipynb`.
2. Open `reports/shap_global_importance.csv`.
3. Inspect one local explanation JSON entry — map SHAP signs to feature values.

---

## Check Your Understanding

1. What does a large positive SHAP value for `Contract Month-to-month` mean?
2. Why must SHAP not be described as causal?
3. Why is TreeExplainer applied to the uncalibrated XGBoost?
4. Name two EDA findings that SHAP confirms at the model level.

---

**Next:** [Chapter 14 — Production Engineering](14_production_engineering.md)
