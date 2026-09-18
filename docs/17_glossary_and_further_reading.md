# Chapter 17 — Glossary & Further Reading

**Previous:** [Chapter 16](16_monitoring_and_limitations.md) | **Index:** [00_INDEX.md](00_INDEX.md)

---

Alphabetical reference for terms used across this learning guide. Each entry includes where it appears in the project.

---

## A

**Accuracy** — Fraction of all predictions that are correct; misleading when classes are imbalanced.  
→ [Chapter 01](01_problem_and_business_context.md), [Chapter 07](07_baselines_and_model_progression.md)

**Active customer** — Customer still subscribed at prediction time; target is future churn while active.  
→ [Chapter 01](01_problem_and_business_context.md)

**AUC (Area Under Curve)** — Summary of ranking quality across thresholds; see ROC-AUC and PR-AUC.  
→ [Chapter 07](07_baselines_and_model_progression.md)

---

## B

**Baseline model** — Simple reference model (e.g., DummyClassifier) that advanced models must beat.  
→ [Chapter 07](07_baselines_and_model_progression.md)

**Brier score** — Mean squared error between predicted probabilities and actual outcomes; lower is better.  
→ [Chapter 11](11_probability_calibration.md)

**Business cost** — Simulated dollar cost of FP ($50) and FN ($500) used in threshold optimization.  
→ [Chapter 01](01_problem_and_business_context.md), [Chapter 12](12_threshold_and_business_cost.md)

---

## C

**Calibration** — Adjusting model scores so predicted probabilities match observed event rates.  
→ [Chapter 11](11_probability_calibration.md)

**CalibratedClassifierCV** — sklearn wrapper that calibrates a base classifier using cross-validation.  
→ [Chapter 11](11_probability_calibration.md), [Chapter 14](14_production_engineering.md)

**Categorical feature** — Non-numeric column with discrete levels (e.g., `Contract`).  
→ [Chapter 02](02_data_understanding_and_cleaning.md), [Chapter 06](06_preprocessing_and_pipelines.md)

**Churn** — Customer cancels or leaves the service.  
→ [Chapter 01](01_problem_and_business_context.md)

**ChurnModelBundle** — Saved dataclass: preprocessor + calibrated model + threshold + metadata.  
→ [Chapter 14](14_production_engineering.md)

**Class imbalance** — Unequal frequency of target classes (~26.5% churn vs ~73.5% no churn).  
→ [Chapter 09](09_class_imbalance.md)

**Class weighting** — Penalizing errors on the minority class more (`scale_pos_weight`, `class_weight="balanced"`).  
→ [Chapter 09](09_class_imbalance.md)

**ColumnTransformer** — sklearn tool applying different pipelines to numeric vs categorical columns.  
→ [Chapter 06](06_preprocessing_and_pipelines.md)

**Confusion matrix** — Table of TN, FP, FN, TP at a chosen threshold.  
→ [Chapter 12](12_threshold_and_business_cost.md)

---

## D

**Data leakage** — Using information outside training in a way that inflates offline performance.  
→ [Chapter 04](04_leakage_and_data_hygiene.md)

**Decision threshold** — Probability cutoff for binary action; production value **0.10**.  
→ [Chapter 12](12_threshold_and_business_cost.md)

**Drift (feature / score / label)** — Distribution or behavior change over time in production.  
→ [Chapter 16](16_monitoring_and_limitations.md)

**DummyClassifier** — sklearn baseline that predicts most frequent class.  
→ [Chapter 07](07_baselines_and_model_progression.md)

---

## E

**EDA (Exploratory Data Analysis)** — Visual and summary analysis before modeling.  
→ [Chapter 03](03_eda_and_insights.md)

---

## F

**False Negative (FN)** — Actual churner predicted as no churn; costly ($500).  
→ [Chapter 12](12_threshold_and_business_cost.md)

**False Positive (FP)** — Non-churner flagged for retention; $50 simulated cost.  
→ [Chapter 12](12_threshold_and_business_cost.md)

**F1 score** — Harmonic mean of precision and recall.  
→ [Chapter 08](08_hyperparameter_tuning.md), [Chapter 12](12_threshold_and_business_cost.md)

**Feature** — Input column used for prediction (19 in this project).  
→ [Chapter 02](02_data_understanding_and_cleaning.md)

**Fit / Transform** — Fit learns parameters on train; transform applies them to val/test/inference.  
→ [Chapter 06](06_preprocessing_and_pipelines.md)

---

## G

**Gradient boosting (XGBoost)** — Ensemble of shallow trees trained sequentially to correct errors.  
→ [Chapter 07](07_baselines_and_model_progression.md), [Chapter 08](08_hyperparameter_tuning.md)

---

## H

**Holdout set** — Data not used for training; here split into validation (15%) and test (15%).  
→ [Chapter 05](05_train_val_test_split.md)

**Hyperparameter** — Model setting chosen before training (e.g., `max_depth`).  
→ [Chapter 08](08_hyperparameter_tuning.md)

---

## I

**Identifier column** — `customerID`; for linkage only, never a feature.  
→ [Chapter 02](02_data_understanding_and_cleaning.md)

**Imputation** — Filling missing values (median/mode in preprocessing).  
→ [Chapter 06](06_preprocessing_and_pipelines.md)

**Inference** — Running a saved model on new data to get predictions.  
→ [Chapter 14](14_production_engineering.md), [Chapter 15](15_streamlit_fastapi_and_tests.md)

**Isotonic regression** — Non-parametric calibration method (compared but not selected).  
→ [Chapter 11](11_probability_calibration.md)

---

## J

**joblib** — Library used to persist the `ChurnModelBundle` to disk.  
→ [Chapter 14](14_production_engineering.md)

---

## K

**KS statistic (Kolmogorov–Smirnov)** — Distance between two distributions; used in drift monitoring plans.  
→ [Chapter 16](16_monitoring_and_limitations.md)

---

## L

**Leakage-safe resampling** — SMOTE/undersampling applied only after split, on train only.  
→ [Chapter 04](04_leakage_and_data_hygiene.md), [Chapter 09](09_class_imbalance.md)

**Logistic Regression** — Linear probabilistic classifier; interpretable baseline.  
→ [Chapter 07](07_baselines_and_model_progression.md)

---

## M

**Manifest (`split_manifest.json`)** — JSON file storing train/val/test row indices.  
→ [Chapter 05](05_train_val_test_split.md)

---

## O

**One-hot encoding** — Representing each category level as its own binary column.  
→ [Chapter 06](06_preprocessing_and_pipelines.md)

**Overfitting** — Model memorizes training noise; hurts generalization.  
→ [Chapter 08](08_hyperparameter_tuning.md)

---

## P

**Pipeline (sklearn)** — Chains preprocessing + model so steps cannot be skipped or duplicated.  
→ [Chapter 06](06_preprocessing_and_pipelines.md)

**Platt scaling** — Sigmoid calibration method (`method="sigmoid"`).  
→ [Chapter 11](11_probability_calibration.md)

**Precision** — TP / (TP + FP); quality of positive predictions.  
→ [Chapter 12](12_threshold_and_business_cost.md)

**PR-AUC** — Area under precision-recall curve; primary ranking metric here.  
→ [Chapter 01](01_problem_and_business_context.md), [Chapter 10](10_model_selection.md)

**Probability calibration** — See **Calibration**.  
→ [Chapter 11](11_probability_calibration.md)

**PSI (Population Stability Index)** — Drift metric comparing score/feature distributions.  
→ [Chapter 16](16_monitoring_and_limitations.md)

---

## R

**Random Forest** — Bagged ensemble of decision trees; compared in model progression.  
→ [Chapter 07](07_baselines_and_model_progression.md)

**Random state** — Seed (`42`) for reproducible splits and training.  
→ [Chapter 05](05_train_val_test_split.md)

**Recall** — TP / (TP + FN); fraction of churners caught.  
→ [Chapter 12](12_threshold_and_business_cost.md)

**Retention outreach** — Business action triggered when P(churn) ≥ threshold.  
→ [Chapter 01](01_problem_and_business_context.md)

**Risk band** — Low / Elevated / High UI labels from `src/policy.py`.  
→ [Chapter 12](12_threshold_and_business_cost.md)

**ROC-AUC** — Area under ROC curve; reported but secondary to PR-AUC.  
→ [Chapter 07](07_baselines_and_model_progression.md)

---

## S

**Sacred test set** — Final 15% holdout used once for evaluation — never for tuning.  
→ [Chapter 04](04_leakage_and_data_hygiene.md), [Chapter 16](16_monitoring_and_limitations.md)

**scale_pos_weight** — XGBoost class imbalance parameter ≈ 2.77 in production.  
→ [Chapter 08](08_hyperparameter_tuning.md), [Chapter 09](09_class_imbalance.md)

**Sentinel category** — Values like `"No internet service"` encoding absence of a parent service.  
→ [Chapter 03](03_eda_and_insights.md)

**SHAP** — Shapley-based feature attribution for model explanations.  
→ [Chapter 13](13_shap_explainability.md)

**Sigmoid calibration** — See **Platt scaling**.  
→ [Chapter 11](11_probability_calibration.md)

**SMOTE** — Synthetic oversampling of minority class.  
→ [Chapter 09](09_class_imbalance.md)

**SMOTENC** — SMOTE for mixed numeric and categorical features; tested, not selected.  
→ [Chapter 09](09_class_imbalance.md)

**StandardScaler** — Scales numeric features to zero mean, unit variance (fit on train).  
→ [Chapter 06](06_preprocessing_and_pipelines.md)

**Stratified split** — Split preserving class proportions in each partition.  
→ [Chapter 05](05_train_val_test_split.md)

---

## T

**Target variable** — `Churn` (Yes/No); what the model predicts.  
→ [Chapter 02](02_data_understanding_and_cleaning.md)

**Test set** — 1,057 rows (15%); final unbiased evaluation.  
→ [Chapter 05](05_train_val_test_split.md), [Chapter 16](16_monitoring_and_limitations.md)

**Threshold optimization** — Choosing cutoff minimizing simulated business cost on validation.  
→ [Chapter 12](12_threshold_and_business_cost.md)

**Train set** — 4,930 rows (70%); used to fit model and preprocessing.  
→ [Chapter 05](05_train_val_test_split.md)

**True Negative / True Positive** — Correct no-churn / churn predictions.  
→ [Chapter 12](12_threshold_and_business_cost.md)

---

## U

**Undersampling** — Removing majority-class training rows; tested via RandomUnderSampler.  
→ [Chapter 09](09_class_imbalance.md)

---

## V

**Validation set** — 1,056 rows (15%); model selection, calibration, threshold tuning.  
→ [Chapter 04](04_leakage_and_data_hygiene.md), [Chapter 05](05_train_val_test_split.md)

---

## X

**XGBoost** — Extreme Gradient Boosting; final production model family.  
→ [Chapter 07](07_baselines_and_model_progression.md) through [Chapter 14](14_production_engineering.md)

---

## Further Reading

### Official documentation

- [scikit-learn: User Guide](https://scikit-learn.org/stable/user_guide.html) — Pipelines, metrics, calibration
- [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html)
- [imbalanced-learn](https://imbalanced-learn.org/stable/) — SMOTE, RandomUnderSampler
- [SHAP documentation](https://shap.readthedocs.io/en/latest/)
- [FastAPI tutorial](https://fastapi.tiangolo.com/tutorial/)
- [Streamlit docs](https://docs.streamlit.io/)

### Concepts

- [Precision and Recall (Google ML Crash Course)](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall)
- [Calibration of probabilities (scikit-learn)](https://scikit-learn.org/stable/modules/calibration.html)
- [Telco Customer Churn dataset (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

### This repository

- [README.md](../README.md) — Executive summary and runbook
- [AGENTS.md](../AGENTS.md) — Permanent development rules
- [00_INDEX.md](00_INDEX.md) — Guide navigation

---

**Back to:** [Index](00_INDEX.md)
