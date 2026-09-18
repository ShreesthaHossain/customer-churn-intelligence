# Chapter 01 — Problem & Business Context

**Previous:** [Index](00_INDEX.md) | **Next:** [Chapter 02 — Data & Cleaning](02_data_understanding_and_cleaning.md)

---

## What You Will Learn

- What **customer churn** means in a telecom business
- Why this project predicts **probabilities**, not just yes/no labels
- Why **accuracy** is a misleading metric here
- How **business costs** shape the retention decision
- Why we use classical ML (XGBoost) instead of deep learning

---

## The Business Problem

A telecom company wants to **keep customers who might leave**. Churn means a customer cancels their service. If we identify at-risk customers **before** they leave, the company can offer retention incentives (discounts, plan changes, support outreach).

This project answers one question:

> For each **currently active** customer, what is the probability they will churn?

That probability then drives a **decision policy**: should we contact this customer for retention?

---

## Beginner Box: Key Terms

| Term | Plain-English meaning |
|------|----------------------|
| **Churn** | Customer leaves / cancels service |
| **Active customer** | Still subscribed — we predict while they are active |
| **Probability P(churn)** | A number from 0 to 1: model's estimate of churn likelihood |
| **Retention outreach** | Phone call, email, or offer to keep the customer |
| **Classification threshold** | Cutoff on probability: above = "contact for retention" |

---

## What We Built (Summary)

| Component | Choice |
|-----------|--------|
| Problem type | Binary classification with **probability output** |
| Target | `Churn`: Yes / No (~26.5% Yes in training data) |
| Primary metrics | PR-AUC, recall, precision, business cost — **not accuracy alone** |
| Final model | XGBoost + class weighting, sigmoid-calibrated |
| Decision rule | Contact if P(churn) ≥ **0.10** (validation-tuned) |
| Deployment | Saved pipeline → Streamlit dashboard + FastAPI |

---

## Why Probabilities, Not Just Labels?

A model could output "Churn" or "No Churn." That is simpler but **less useful**:

1. **Ranking:** Probabilities let you sort customers by risk and contact the highest-risk first when budget is limited.
2. **Threshold flexibility:** Different campaigns need different aggressiveness — probabilities support that without retraining.
3. **Calibration:** Stakeholders can interpret "27% chance of churn" more meaningfully than a raw score.

We still produce a binary decision at the end — but only **after** choosing a threshold based on business costs.

---

## Why Not Accuracy?

The dataset is **imbalanced**: about 73% of customers do **not** churn. A naive model that always predicts "No Churn" achieves ~73% **accuracy** but catches **zero** churners.

Our DummyClassifier baseline in `reports/model_comparison.csv`:

| Model | Accuracy | Recall | PR-AUC |
|-------|----------|--------|--------|
| DummyClassifier | 0.735 | **0.0** | 0.265 |

High accuracy, useless for retention. That is why we prioritize **PR-AUC** (Precision-Recall Area Under Curve) and **recall** (how many actual churners we catch).

---

## Beginner Box: PR-AUC vs ROC-AUC

- **ROC-AUC** measures ranking across all thresholds; can look optimistic on imbalanced data.
- **PR-AUC** focuses on the **positive (churn) class** — more informative when churn is the minority event (~26.5%).

This project treats PR-AUC as the primary ranking metric for model comparison.

---

## Business Cost Story

Retention has trade-offs:

| Error type | Meaning | Simulated cost |
|------------|---------|----------------|
| **False Positive (FP)** | We contact a customer who would **not** have churned | $50 (wasted retention offer) |
| **False Negative (FN)** | We **miss** a customer who **does** churn | $500 (lost customer value) |

Missing a churner costs **10×** more than an unnecessary contact. That asymmetry pushes us toward **lower thresholds** (contact more customers) and **higher recall**.

These costs live in `src/config.py`:

```python
DEFAULT_RETENTION_OFFER_COST = 50
DEFAULT_LOST_CUSTOMER_COST = 500
```

They are **simulated** for portfolio demonstration — real companies would estimate from finance and marketing data.

---

## Decision Summary

| Topic | Project choice | Why not alternatives |
|-------|----------------|----------------------|
| Problem framing | Predict P(churn) for **active** customers | Not post-mortem analysis of already-churned accounts |
| Success metrics | PR-AUC + recall + business cost | Not accuracy alone |
| Output | Calibrated probability + threshold policy | Not a fixed 0.50 cutoff |
| Model family | Tabular ML (XGBoost) | Not neural networks — small tabular dataset, no clear deep-learning advantage |
| Boosting library | XGBoost | Not LightGBM — one strong boosted-tree model is enough; avoid model sprawl |

---

## Why Not Deep Learning?

Neural networks shine on large unstructured data (images, text, huge tabular sets). Here:

- ~7,000 rows, 19 structured features
- Strong baselines (Logistic Regression) already perform well
- Tree models (XGBoost) handle mixed numeric/categorical data naturally
- Interpretability and deployment simplicity matter for a portfolio project

Notebook `11_xgboost.ipynb` documents this scope decision explicitly.

---

## How This Chapter Connects to the Rest

```mermaid
flowchart LR
    ch01[Chapter01_Business] --> ch02[Chapter02_Data]
    ch02 --> ch05[Chapter05_Split]
    ch05 --> ch07[Chapter07_Models]
    ch07 --> ch12[Chapter12_Threshold]
    ch12 --> ch15[Chapter15_Deploy]
```

Everything downstream serves the business goal: **rank customers by churn risk and contact the right ones at the right cost**.

---

## Hands-On

1. Open [README.md](../README.md) — read "Problem Statement" and "Final Test Results."
2. Skim `reports/model_comparison.csv` — notice DummyClassifier's zero recall.
3. Open `src/config.py` — find cost constants and frozen XGBoost parameters.

---

## Check Your Understanding

1. Why does a 73%-accurate "always No Churn" model fail the business goal?
2. What is the difference between a probability and a final yes/no decision?
3. Why is a false negative more expensive than a false positive in this project's cost model?
4. Why is PR-AUC preferred over accuracy for churn ranking?

---

**Next:** [Chapter 02 — Data Understanding & Cleaning](02_data_understanding_and_cleaning.md)
