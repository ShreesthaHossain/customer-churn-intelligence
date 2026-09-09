"""Tests for batch upload scoring."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data_cleaning import IDENTIFIER_COL, TARGET_COL
from src.data_split import load_split_from_manifest
from src.inference import score_uploaded_batch
from src.model import run_training_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "split_manifest.json"
THRESHOLD_PATH = PROJECT_ROOT / "reports" / "chosen_threshold.json"


@pytest.fixture(scope="module")
def trained_bundle(tmp_path_factory):
    if not SPLIT_MANIFEST.exists() or not THRESHOLD_PATH.exists():
        pytest.skip("project artifacts missing")

    split = load_split_from_manifest()
    model_path = tmp_path_factory.mktemp("models") / "bundle.joblib"
    bundle, _ = run_training_pipeline(split, save_path=model_path)
    return bundle, split


def test_score_uploaded_batch_returns_ranked_results(trained_bundle) -> None:
    bundle, split = trained_bundle
    sample = split.X_val.iloc[:5].copy()
    sample[IDENTIFIER_COL] = split.id_val.iloc[:5].astype(str).tolist()
    sample[TARGET_COL] = split.y_val.iloc[:5].tolist()

    results = score_uploaded_batch(bundle, sample, IDENTIFIER_COL)

    assert len(results) == 5
    assert "primary_key" in results.columns
    assert "churn_probability" in results.columns
    assert "prediction" in results.columns
    assert "risk_level" in results.columns
    assert "retention_recommended" in results.columns
    assert "actual_churn" in results.columns
    assert results["churn_probability"].is_monotonic_decreasing
