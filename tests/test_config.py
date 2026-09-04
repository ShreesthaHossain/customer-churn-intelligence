"""Tests for project configuration helpers."""

from pathlib import Path

import pytest

from src.config import get_decision_threshold, load_chosen_threshold_config, project_root

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THRESHOLD_PATH = PROJECT_ROOT / "reports" / "chosen_threshold.json"


@pytest.mark.skipif(not THRESHOLD_PATH.exists(), reason="chosen_threshold.json not generated yet")
def test_load_chosen_threshold_config() -> None:
    config = load_chosen_threshold_config(THRESHOLD_PATH)

    assert config["calibration_retained"] is True
    assert config["threshold"] == 0.1


@pytest.mark.skipif(not THRESHOLD_PATH.exists(), reason="chosen_threshold.json not generated yet")
def test_get_decision_threshold() -> None:
    assert get_decision_threshold(THRESHOLD_PATH) == 0.1


def test_project_root_points_to_repo() -> None:
    assert project_root().name == "customer-churn-intelligence"
