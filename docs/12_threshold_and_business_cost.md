# Chapter 12 — Threshold & Business Cost

**Previous:** [Chapter 11](11_probability_calibration.md) | **Next:** [Chapter 13 — SHAP Explainability](13_shap_explainability.md)

---

## What You Will Learn

- How a **decision threshold** converts probability to action
- **Precision, recall, FP, FN** in telecom terms
- The **0.26 → 0.10** threshold story
- **Risk bands** vs model thresholds
- Simulated business cost optimization

---

## From Probability to Decision

The model outputs P(churn) ∈ [0, 1]. The business needs a yes/no:

```
If P(churn) >= threshold → recommend retention outreach
Else → routine monitoring
```

Default ML threshold is often **0.50** — **not optimal** when false negatives cost 10× false positives.

---

## Beginner Box: Confusion Matrix Terms

Assume **positive = Churn (Yes)**:

| Term | Telecom meaning | Cost in this project |
|------|-----------------|----------------------|
| **True Positive (TP)** | Correctly flagged churner | Good — save attempt |
| **False Positive (FP)** | Contacted customer who would stay | $50 wasted offer |
| **False Negative (FN)** | Missed customer who churns | $500 lost value |
| **True Negative (TN)** | Correctly left happy customer alone | No cost |

**Precision** = TP / (TP + FP) — "When we contact, how often were we right?"  
**Recall** = TP / (TP + FN) — "What fraction of churners did we catch?"

---

## The 0.26 → 0.10 Story

### Step 1: Uncalibrated threshold (`14_threshold_optimization.ipynb`)

On **uncalibrated** XGBoost validation probabilities:

- Cost-optimal threshold ≈ **0.26**
- Minimizes: `FP × $50 + FN × $500`

### Step 2: Calibration (Chapter 11)

Sigmoid calibration shifts probabilities downward (mean ~0.41 → ~0.27).

The 0.26 threshold was tuned for a **different scale** — **invalid** after calibration.

### Step 3: Recalibrated threshold (`14b_calibrated_threshold_optimization.ipynb`)

On **calibrated** validation probabilities:

- New cost-optimal threshold: **0.10**
- Saved to `reports/chosen_threshold.json`

**Validation metrics at 0.10:**

| Metric | Value |
|--------|-------|
| Precision | 0.415 |
| Recall | 0.946 |
| F1 | 0.577 |
| Predicted churners | 638 |
| Simulated total cost | $26,150 |

> **Sacred test set:** Threshold tuned on validation only.

---

## Cost Optimization Logic

`src/evaluation.py`:

```python
def compute_simulated_business_cost(fp, fn, retention_offer_cost=50, lost_customer_cost=500):
    return fp * 50 + fn * 500
```

`sweep_thresholds()` tests thresholds from 0.05 to 0.95 (step 0.01).  
`select_min_cost_threshold()` picks the minimum-cost point on **validation**.

Reports: `reports/threshold_analysis.csv`, `reports/threshold_analysis_calibrated.csv`

---

## Why Threshold 0.10 Is So Low

Because **FN costs 10× FP**, the optimizer prefers contacting more customers (higher recall) even at lower precision (~0.42).

At 0.10 on validation:

- Catches **94.6%** of churners
- ~60% of contacts are unnecessary (FP)

That trade-off is **by design** given simulated costs — real businesses might adjust costs or cap contact volume.

---

## Risk Bands (UI / API Only)

`src/policy.py` — **not** separate model thresholds:

| Band | Rule | Purpose |
|------|------|---------|
| **Low** | prob < 0.10 | Below action threshold |
| **Elevated** | 0.10 ≤ prob < 0.50 | Flagged for outreach |
| **High** | prob ≥ 0.50 | Priority escalation display |

`HIGH_RISK_UI_BAND = 0.50` is for **display prioritization** within contacted customers — not a second trained model.

---

## Why Not Other Threshold Strategies?

| Alternative | Why not |
|-------------|---------|
| Fixed 0.50 | Ignores cost asymmetry; misses most churners |
| Maximize accuracy | Rewards predicting No |
| Tune on test set | Overfits to test noise |
| Use uncalibrated 0.26 after calibration | Wrong probability scale |
| Maximize F1 alone | Ign explicit dollar costs (cost sweep used instead) |

---

## Test Set Performance at Frozen 0.10

Single evaluation (`17_final_test_evaluation.ipynb`, `reports/final_test_metrics.json`):

| Metric | Test | Validation |
|--------|------|------------|
| Precision | 0.419 | 0.415 |
| Recall | 0.908 | 0.946 |
| F1 | 0.574 | 0.577 |
| Predicted churners | 608 | 638 |

Test aligns with validation — policy not retuned.

**Test confusion matrix:**

|  | Pred No | Pred Churn |
|--|---------|------------|
| Actual No | 423 (TN) | 353 (FP) |
| Actual Yes | 26 (FN) | 255 (TP) |

---

## Hands-On

1. Run `notebooks/14b_calibrated_threshold_optimization.ipynb`.
2. Read `reports/chosen_threshold.json` and `reports/threshold_analysis_calibrated.csv`.
3. Trace `src/policy.py` → `classify_risk_level()` and `churn_prediction_label()`.
4. Run `pytest tests/test_evaluation.py tests/test_policy.py -v`.

---

## Check Your Understanding

1. Why did the optimal threshold drop from 0.26 to 0.10?
2. What is the simulated cost of one false negative vs one false positive?
3. What is the difference between the decision threshold (0.10) and the High risk band (0.50)?
4. Why should you not use 0.50 as the default threshold here?

---

**Next:** [Chapter 13 — SHAP Explainability](13_shap_explainability.md)
