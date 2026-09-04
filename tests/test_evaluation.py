"""Tests for evaluation helpers."""

import numpy as np
import pandas as pd

from src.evaluation import (
    compute_classification_metrics,
    encode_churn_labels,
    evaluate_at_threshold,
    select_min_cost_threshold,
    sweep_thresholds,
)


def test_encode_churn_labels() -> None:
    y = pd.Series(["No", "Yes", "No"])
    encoded = encode_churn_labels(y)
    np.testing.assert_array_equal(encoded, np.array([0, 1, 0]))


def test_compute_classification_metrics_perfect_scores() -> None:
    y_true = np.array([0, 1, 1, 0])
    y_proba = np.array([0.1, 0.9, 0.8, 0.2])

    metrics = compute_classification_metrics(y_true, y_proba, threshold=0.5)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0


def test_evaluate_at_threshold_includes_business_cost() -> None:
    y_true = ["No", "Yes", "Yes", "No"]
    y_proba = np.array([0.9, 0.2, 0.8, 0.1])

    metrics = evaluate_at_threshold(y_true, y_proba, threshold=0.5)

    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["total_cost_usd_simulated"] == 550


def test_sweep_thresholds_and_select_min_cost() -> None:
    y_true = ["No", "Yes", "Yes", "No", "Yes"]
    y_proba = np.array([0.05, 0.95, 0.85, 0.10, 0.40])

    sweep = sweep_thresholds(y_true, y_proba, start=0.1, stop=0.9, step=0.1)
    best = select_min_cost_threshold(sweep)

    assert "Threshold" in best
    assert "Total_Cost_USD" in best
    assert best["Total_Cost_USD"] == sweep["Total_Cost_USD"].min()
