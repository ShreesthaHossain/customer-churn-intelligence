"""Evaluation helpers for probability-based churn models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def encode_churn_labels(y: pd.Series | np.ndarray | list[str]) -> np.ndarray:
    """Convert Yes/No churn labels to binary integers."""
    if isinstance(y, pd.Series):
        values = y.to_numpy()
    else:
        values = np.asarray(y)

    if not np.issubdtype(values.dtype, np.number):
        return (values == "Yes").astype(int)
    return values.astype(int)


def predict_labels(y_proba: np.ndarray, threshold: float) -> np.ndarray:
    """Apply a decision threshold to predicted probabilities."""
    return (np.asarray(y_proba) >= threshold).astype(int)


def compute_confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """Return TN, FP, FN, TP counts."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def compute_simulated_business_cost(
    false_positives: int,
    false_negatives: int,
    *,
    retention_offer_cost: float = 50.0,
    lost_customer_cost: float = 500.0,
) -> int:
    """Simulate total retention cost used during threshold optimization."""
    return int(false_positives * retention_offer_cost + false_negatives * lost_customer_cost)


def compute_classification_metrics(
    y_true: pd.Series | np.ndarray | list[str],
    y_proba: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Compute standard classification metrics at a given threshold."""
    y_true_bin = encode_churn_labels(y_true)
    y_pred = predict_labels(y_proba, threshold)
    counts = compute_confusion_counts(y_true_bin, y_pred)

    return {
        "threshold": round(float(threshold), 4),
        "precision": round(float(precision_score(y_true_bin, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true_bin, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true_bin, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true_bin, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true_bin, y_proba)), 4),
        "brier_score": round(float(brier_score_loss(y_true_bin, y_proba)), 4),
        "predicted_churners": int(y_pred.sum()),
        **counts,
    }


def evaluate_at_threshold(
    y_true: pd.Series | np.ndarray | list[str],
    y_proba: np.ndarray,
    threshold: float,
    *,
    retention_offer_cost: float = 50.0,
    lost_customer_cost: float = 500.0,
) -> dict[str, Any]:
    """Return metrics plus simulated business cost at one threshold."""
    metrics = compute_classification_metrics(y_true, y_proba, threshold)
    metrics["total_cost_usd_simulated"] = compute_simulated_business_cost(
        metrics["false_positives"],
        metrics["false_negatives"],
        retention_offer_cost=retention_offer_cost,
        lost_customer_cost=lost_customer_cost,
    )
    return metrics


def sweep_thresholds(
    y_true: pd.Series | np.ndarray | list[str],
    y_proba: np.ndarray,
    *,
    start: float = 0.05,
    stop: float = 0.95,
    step: float = 0.01,
    retention_offer_cost: float = 50.0,
    lost_customer_cost: float = 500.0,
) -> pd.DataFrame:
    """Evaluate a grid of thresholds for business-cost optimization."""
    rows: list[dict[str, Any]] = []
    thresholds = np.arange(start, stop + step / 2, step)

    for threshold in thresholds:
        metrics = evaluate_at_threshold(
            y_true,
            y_proba,
            float(threshold),
            retention_offer_cost=retention_offer_cost,
            lost_customer_cost=lost_customer_cost,
        )
        rows.append(
            {
                "Threshold": round(float(threshold), 2),
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1": metrics["f1"],
                "False_Positives": metrics["false_positives"],
                "False_Negatives": metrics["false_negatives"],
                "Predicted_Churners": metrics["predicted_churners"],
                "Total_Cost_USD": metrics["total_cost_usd_simulated"],
            }
        )

    return pd.DataFrame(rows)


def select_min_cost_threshold(threshold_df: pd.DataFrame) -> pd.Series:
    """Select the threshold row with minimum simulated business cost."""
    if threshold_df.empty:
        raise ValueError("Threshold sweep produced no rows.")
    return threshold_df.loc[threshold_df["Total_Cost_USD"].idxmin()]
