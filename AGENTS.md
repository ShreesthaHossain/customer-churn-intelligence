# Customer Churn Intelligence — Agent Development Rules

Permanent development rules for this project. Follow these in every session unless the user explicitly overrides them.

---

## Project Objective

Predict the probability that an active telecom customer will churn and convert that probability into a business retention decision.

---

## Dataset

- The dataset is located in `data/raw/`.
- **Do not modify the original raw CSV.**

---

## General Project Rules

### 1. Portfolio-Quality Engineering

This is a portfolio-quality ML engineering project, not just a notebook assignment.

### 2. Project Lifecycle

Follow this overall lifecycle in order:

1. Problem definition
2. Data understanding
3. EDA
4. Leakage audit
5. Train/validation/test split
6. Preprocessing
7. Dummy baseline
8. Logistic Regression
9. Random Forest
10. XGBoost or LightGBM
11. Imbalance experiments
12. Model comparison
13. Probability calibration
14. Business threshold optimization
15. SHAP explainability
16. Reusable `src/` code
17. Saved inference pipeline
18. Streamlit dashboard
19. FastAPI endpoint
20. Tests
21. Monitoring plan
22. Professional README

### 3. Final Test Set Is Sacred

Never use the final test set for:

- Hyperparameter tuning
- Feature selection
- Threshold selection
- Model selection
- Calibration decisions

The final test set must remain untouched until the model and decision policy are frozen.

### 4. Prevent Data Leakage

Any preprocessing, scaling, encoding, imputation, SMOTE, feature selection, or other learned transformations must be **fitted using training data only**.

### 5. Customer ID

Do not use `customerID` as a predictive feature. Keep it only for linking predictions back to customers.

### 6. No Future / Post-Churn Information

Do not use future or post-churn information as features. Examples of forbidden features if encountered:

- Cancellation date
- Termination reason
- Final invoice
- Post-churn account status
- Information created after the prediction time

### 7. sklearn Pipelines

Use `sklearn.Pipeline` and `ColumnTransformer` whenever appropriate so training and inference use exactly the same preprocessing.

### 8. Categorical Variables

Handle categorical variables with a safe approach such as:

```python
OneHotEncoder(handle_unknown="ignore")
```

### 9. Evaluation Metrics

Do not evaluate this project using accuracy alone.

Important metrics include:

- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- Confusion Matrix
- Probability calibration / Brier score where appropriate

### 10. Baselines Before Advanced Models

Always establish simple baselines before advanced models.

Required model progression:

1. `DummyClassifier`
2. Logistic Regression
3. Random Forest
4. XGBoost **or** LightGBM

Do not add unnecessary models simply to increase the model count.

### 11. Model Roles

- **Logistic Regression** — interpretable baseline
- **Random Forest** — nonlinear comparison
- **XGBoost or LightGBM** — main boosted-tree candidate

### 12. No Neural Networks by Default

Do not use a neural network unless later evidence demonstrates a clear reason for it.

### 13. Class Imbalance Experiments

Compare:

- Original distribution
- Class weighting
- Under-sampling
- SMOTE where appropriate

**SMOTE must never be applied before the data split or to validation/test data.**

### 14. Probability Outputs

Predictions should be probabilities, not only Churn/No Churn labels.

### 15. Threshold Optimization

Do not assume a 0.50 classification threshold is optimal.

Later, optimize the threshold using:

- False positives
- False negatives
- Precision
- Recall
- Number of customers contacted
- Simulated retention cost
- Simulated lost-customer cost

### 16. SHAP Interpretation

SHAP explanations must describe **model influence**, not causality.

### 17. Notebooks vs. Source Code

Keep notebooks for exploration and reporting. Move reusable production logic into `src/`.

### 18. Single Inference Pipeline

Eventually the same saved pipeline must be used by:

- Python inference
- Streamlit
- FastAPI

### 19. No Duplicated Preprocessing

Do not duplicate preprocessing logic between notebooks, Streamlit, FastAPI, and `src/`.

### 20. Code Quality

Write clean, readable Python. Use functions where appropriate. Add concise comments explaining **why** important ML decisions are made.

### 21. No Fabricated Results

Do not fabricate results, metrics, dataset characteristics, or business impact. Calculate them from the actual dataset.

### 22. Step Completion Protocol

When a step is completed:

1. Run the relevant code/tests
2. Check for errors
3. Summarize what was changed
4. Report important findings
5. **Do not automatically continue into the next major project phase unless requested**
