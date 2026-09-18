# Chapter 11 — Probability Calibration

**Previous:** [Chapter 10](10_model_selection.md) | **Next:** [Chapter 12 — Threshold & Business Cost](12_threshold_and_business_cost.md)

---

## What You Will Learn

- The difference between a **ranking score** and a **calibrated probability**
- How **Platt scaling (sigmoid)** calibration works in this project
- Why calibration **invalidates** previously tuned thresholds
- Why sigmoid beat isotonic (marginally) here

---

## Why Calibrate?

Tree models like XGBoost output scores that **rank** customers well but are not always **true probabilities**. Example: if the model says P(churn)=0.40 for many customers but only 27% actually churn, probabilities are **overestimated**.

**Calibration** adjusts outputs so that among customers scored around 0.30, roughly 30% actually churn.

Business stakeholders need trustworthy probabilities for:

- Budget planning ("how many high-risk customers?")
- Consistent threshold policies
- Comparing campaigns over time

---

## Beginner Box: Probability vs Score

| Concept | Meaning |
|---------|---------|
| **Score** | Model output used for ranking — may not match real frequencies |
| **Calibrated probability** | Adjusted so predicted rates match observed rates |
| **Brier score** | Measures probability accuracy (lower = better); 0 is perfect, 0.25 is uninformative for balanced binary |

---

## Methods Compared

Notebook: `15_probability_calibration.ipynb`  
Report: `reports/calibration_comparison.csv`

| Method | Brier Score | PR-AUC | Mean Cal. Error | Mean Prob |
|--------|-------------|--------|-----------------|-----------|
| Uncalibrated | 0.1631 | 0.6399 | 0.168 | 0.405 |
| **Sigmoid (Platt)** | **0.1360** | 0.6446 | **0.0249** | 0.273 |
| Isotonic | 0.1362 | 0.6469 | 0.0376 | 0.274 |

**Production choice:** **Sigmoid (Platt)** — lowest Brier score and lowest mean calibration error.

---

## Implementation

`src/model.py`:

```python
CalibratedClassifierCV(
    base_estimator=XGBClassifier(...),
    method="sigmoid",  # Platt scaling
    cv=5,
)
```

- Wraps the trained XGBoost
- Uses **5-fold cross-validation on training data** to learn calibration mapping
- Validation/test used only for **evaluation**, not fitting calibration curves

Constants in `src/config.py`:

```python
CALIBRATION_METHOD = "sigmoid"
CALIBRATION_CV = 5
```

---

## Critical Lesson: Thresholds Must Be Re-Tuned

Calibration **changes the probability scale**:

- Uncalibrated mean probability ≈ **0.41**
- Calibrated mean probability ≈ **0.27**

Notebook `14_threshold_optimization.ipynb` found optimal threshold **0.26** on **uncalibrated** scores. After calibration, that threshold is **invalid**.

`reports/chosen_threshold.json` documents:

```json
"step16_uncalibrated_threshold": 0.26,
"step16_threshold_valid_after_calibration": false
```

Chapter 12 explains re-tuning on calibrated validation probabilities → new optimum **0.10**.

> **Sacred test set:** Calibration method chosen on validation metrics; test not used.

---

## Sigmoid vs Isotonic

| Method | Pros | Cons |
|--------|------|------|
| **Sigmoid (Platt)** | Stable with limited data; smooth mapping | Assumes sigmoid shape |
| **Isotonic** | Flexible step-wise mapping | Can overfit small samples; hit prob=1.0 extremes |

With ~1,300 positive training examples, sigmoid's stability won on Brier score. Isotonic was close — either could be defended; sigmoid retained for lower calibration error.

---

## Why Not Skip Calibration?

| Alternative | Downside |
|-------------|------------|
| Use raw XGBoost probabilities | Misleading frequency statements to business |
| Calibrate on validation | Leaks validation into calibration fit |
| Calibrate on test | Forbidden |
| Only use rankings, ignore calibration | Threshold cost optimization needs comparable probabilities |

---

## Ranking vs Calibration

Important nuance: calibration improves **probability interpretation** more than **ranking**. PR-AUC moved slightly (0.640 → 0.645 on validation) — ranking largely preserved.

---

## Hands-On

1. Run `notebooks/15_probability_calibration.ipynb`.
2. Plot reliability diagrams if available in notebook cells.
3. Read `reports/calibration_comparison.csv` — compare Brier scores.
4. Run `pytest tests/test_model.py -v` — checks `CalibratedClassifierCV` type.

---

## Check Your Understanding

1. Why might XGBoost scores not equal true churn rates?
2. What does a Brier score of 0.136 indicate compared to 0.163?
3. Why did threshold 0.26 become invalid after calibration?
4. Why was sigmoid preferred over isotonic?

---

**Next:** [Chapter 12 — Threshold & Business Cost](12_threshold_and_business_cost.md)
