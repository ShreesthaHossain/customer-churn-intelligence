# Customer Churn Intelligence — Learning Guide Index

Welcome. This guide teaches the **entire project** from business problem to deployed API — not just the final numbers, but the reasoning behind every step.

---

## What This Project Does (One Paragraph)

We predict the **probability** that an active telecom customer will churn, then convert that probability into a **retention decision** using a business-cost-aware threshold. The final system is a saved scikit-learn/XGBoost pipeline served by Streamlit and FastAPI, with 63 automated tests guarding correctness.

---

## How to Use This Guide

1. **Read a chapter** in order (or jump via the map below).
2. **Run the matching notebook** to see the analysis in code.
3. **Inspect the report artifact** (CSV/JSON in `reports/`) for frozen numbers.
4. **Answer the "Check your understanding" questions** at the end of each chapter.

### Recommended reading order vs notebook order

Notebooks are numbered for the repo workflow. Two notebooks (`14`, `14b`) optimize thresholds **before** notebook `15` calibrates probabilities — that is intentional exploration order. The **logical learning order** for thresholding is:

1. Calibrate probabilities ([Chapter 11](11_probability_calibration.md))
2. Re-tune threshold on calibrated scores ([Chapter 12](12_threshold_and_business_cost.md))

When reading notebooks, follow numeric order; when learning concepts, read chapters 11 then 12.

---

## Project Flow (High Level)

```mermaid
flowchart LR
    problem[BusinessProblem] --> data[DataCleaning]
    data --> eda[EDA]
    eda --> leakage[LeakageAudit]
    leakage --> split[TrainValTestSplit]
    split --> prep[Preprocessing]
    prep --> models[ModelProgression]
    models --> imbalance[ImbalanceExperiments]
    imbalance --> select[ModelSelection]
    select --> calibrate[Calibration]
    calibrate --> threshold[ThresholdOptimization]
    threshold --> shap[SHAP]
    shap --> src[ProductionCode]
    src --> deploy[StreamlitAndFastAPI]
    deploy --> monitor[MonitoringAndLimits]
```

---

## Chapter Map

| Ch | Title | Read when you want to understand… |
|----|-------|-----------------------------------|
| [01](01_problem_and_business_context.md) | Problem & Business Context | Why churn ML, metrics, costs |
| [02](02_data_understanding_and_cleaning.md) | Data & Cleaning | Raw CSV, cleaning rules, schema |
| [03](03_eda_and_insights.md) | EDA & Insights | Patterns before modeling |
| [04](04_leakage_and_data_hygiene.md) | Leakage & Hygiene | Train/val/test roles, leakage traps |
| [05](05_train_val_test_split.md) | Train/Val/Test Split | Stratification, manifest, sacred test |
| [06](06_preprocessing_and_pipelines.md) | Preprocessing & Pipelines | Scaling, encoding, sklearn Pipeline |
| [07](07_baselines_and_model_progression.md) | Baselines & Models | Dummy → LR → RF → XGBoost |
| [08](08_hyperparameter_tuning.md) | Hyperparameter Tuning | Grids, frozen params |
| [09](09_class_imbalance.md) | Class Imbalance | Weights vs SMOTE vs undersampling |
| [10](10_model_selection.md) | Model Selection | Why XGBoost won |
| [11](11_probability_calibration.md) | Probability Calibration | Platt scaling, Brier score |
| [12](12_threshold_and_business_cost.md) | Threshold & Business Cost | 0.26 → 0.10 story, risk bands |
| [13](13_shap_explainability.md) | SHAP Explainability | Feature influence vs causality |
| [14](14_production_engineering.md) | Production Engineering | `src/` modules, saved bundle |
| [15](15_streamlit_fastapi_and_tests.md) | Streamlit, FastAPI & Tests | Serving and testing ML |
| [16](16_monitoring_and_limitations.md) | Monitoring & Limitations | Drift plan, honest limits |
| [17](17_glossary_and_further_reading.md) | Glossary | All key terms defined |

---

## Quick Reference: Chapters ↔ Notebooks ↔ Code ↔ Reports

| Chapter | Notebook(s) | `src/` module(s) | Key report(s) |
|---------|-------------|------------------|---------------|
| 01 | — | `config.py` | — |
| 02 | `02_data_cleaning`, `03_feature_target_id_separation` | `data_cleaning.py`, `data_separation.py` | — |
| 03 | `01_eda`, `04_focused_eda` | `data_loading.py` | `reports/figures/01_*`–`06_*` |
| 04 | `05_leakage_audit` | — | — |
| 05 | `06_data_split` | `data_split.py` | `data/processed/split_manifest.json` |
| 06 | `07_preprocessing` | `preprocessing.py` | — |
| 07 | `08`–`11` | — | `reports/model_comparison.csv` |
| 08 | `10_random_forest`, `11_xgboost` | `config.py` | — |
| 09 | `12_imbalance_experiments` | `model.py` | `reports/imbalance_comparison.csv` |
| 10 | `13_model_comparison` | — | `reports/final_model_comparison.csv` |
| 11 | `15_probability_calibration` | `model.py` | `reports/calibration_comparison.csv` |
| 12 | `14`, `14b` | `evaluation.py`, `policy.py` | `reports/chosen_threshold.json` |
| 13 | `16_shap_explainability` | — | `reports/shap_global_importance.csv` |
| 14 | — | all `src/` | `models/churn_pipeline.joblib` |
| 15 | — | `inference.py`, `app/app.py`, `api/main.py` | — |
| 16 | `17_final_test_evaluation` | — | `reports/final_test_metrics.json` |
| 17 | — | — | — |

---

## Final Model Snapshot (Reference)

| Item | Value |
|------|-------|
| Model | XGBoost + class weighting (`scale_pos_weight ≈ 2.77`) |
| Calibration | Sigmoid (Platt), 5-fold CV |
| Threshold | **0.10** (validation, min business cost) |
| Test PR-AUC | 0.659 |
| Test recall @ 0.10 | 0.908 |

Details: [Chapter 16](16_monitoring_and_limitations.md), [README.md](../README.md).

---

## Prerequisites

- Basic Python
- Familiarity with pandas helps
- No prior ML required — terms are explained in each chapter and in the [Glossary](17_glossary_and_further_reading.md)

---

## Next Step

Begin with [Chapter 01: Problem & Business Context](01_problem_and_business_context.md).
