"""Tests for model training and persistence helpers."""

from pathlib import Path

import joblib
import numpy as np
import pytest

from src.config import get_decision_threshold
from src.data_split import load_split_from_manifest
from src.evaluation import compute_classification_metrics
from src.model import (
    ChurnModelBundle,
    build_calibrated_model,
    compute_scale_pos_weight,
    load_model_bundle,
    run_training_pipeline,
    save_model_bundle,
    train_final_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "split_manifest.json"
THRESHOLD_PATH = PROJECT_ROOT / "reports" / "chosen_threshold.json"


@pytest.fixture(scope="module")
def split_data():
    if not SPLIT_MANIFEST.exists():
        pytest.skip("split manifest missing")
    return load_split_from_manifest()


@pytest.mark.skipif(not THRESHOLD_PATH.exists(), reason="threshold config missing")
def test_compute_scale_pos_weight(split_data) -> None:
    weight = compute_scale_pos_weight(split_data.y_train)
    assert round(weight, 4) == 2.7691


def test_build_calibrated_model_type() -> None:
    model = build_calibrated_model(scale_pos_weight=2.7691)
    assert model.method == "sigmoid"
    assert model.cv == 5


@pytest.mark.skipif(not THRESHOLD_PATH.exists(), reason="threshold config missing")
def test_train_final_model_uses_train_only(split_data) -> None:
    bundle = train_final_model(split_data)

    assert isinstance(bundle, ChurnModelBundle)
    assert bundle.threshold == get_decision_threshold()
    assert bundle.metadata["trained_on"] == "train_only"
    assert hasattr(bundle.preprocessor, "transformers_")
    assert hasattr(bundle.model, "calibrated_classifiers_")


@pytest.mark.skipif(not THRESHOLD_PATH.exists(), reason="threshold config missing")
def test_save_and_load_model_bundle(split_data, tmp_path: Path) -> None:
    bundle = train_final_model(split_data)
    output_path = save_model_bundle(bundle, tmp_path / "bundle.joblib")
    loaded = load_model_bundle(output_path)

    assert isinstance(loaded, ChurnModelBundle)
    assert loaded.threshold == bundle.threshold
    assert loaded.metadata["scale_pos_weight"] == bundle.metadata["scale_pos_weight"]


@pytest.mark.skipif(not THRESHOLD_PATH.exists(), reason="threshold config missing")
def test_validation_metrics_match_frozen_threshold(split_data) -> None:
    bundle = train_final_model(split_data)
    from src.inference import predict_churn_probability

    val_proba = predict_churn_probability(bundle, split_data.X_val)
    metrics = compute_classification_metrics(split_data.y_val, val_proba, bundle.threshold)

    assert metrics["precision"] == 0.4154
    assert metrics["recall"] == 0.9464
    assert metrics["f1"] == 0.5773


@pytest.mark.skipif(not THRESHOLD_PATH.exists(), reason="threshold config missing")
def test_run_training_pipeline_writes_bundle(split_data, tmp_path: Path) -> None:
    bundle, output_path = run_training_pipeline(split_data, save_path=tmp_path / "trained.joblib")

    assert output_path.exists()
    reloaded = joblib.load(output_path)
    assert isinstance(reloaded, ChurnModelBundle)
    assert np.isclose(reloaded.threshold, bundle.threshold)
