"""Tests for shared retention policy and risk-band logic."""

import pytest

from src.policy import (
    HIGH_RISK_UI_BAND,
    churn_prediction_label,
    classify_risk_level,
    recommended_action,
)


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.0, "Low"),
        (0.09, "Low"),
        (0.10, "Elevated"),
        (0.49, "Elevated"),
        (0.50, "High"),
        (0.99, "High"),
    ],
)
def test_classify_risk_level_bands(probability: float, expected: str) -> None:
    assert classify_risk_level(probability, decision_threshold=0.10) == expected


def test_risk_band_threshold_alignment() -> None:
    threshold = 0.10
    assert classify_risk_level(threshold - 0.001, threshold) == "Low"
    assert classify_risk_level(threshold, threshold) == "Elevated"
    assert classify_risk_level(HIGH_RISK_UI_BAND, threshold) == "High"


@pytest.mark.parametrize(
    ("probability", "threshold", "expected"),
    [
        (0.09, 0.10, "No Churn"),
        (0.10, 0.10, "Churn"),
        (0.75, 0.10, "Churn"),
    ],
)
def test_churn_prediction_label(probability: float, threshold: float, expected: str) -> None:
    assert churn_prediction_label(probability, threshold) == expected


def test_recommended_action_matches_retention_flag() -> None:
    outreach = recommended_action(True)
    monitor = recommended_action(False)
    assert "retention outreach" in outreach.lower()
    assert "monitoring" in monitor.lower()
