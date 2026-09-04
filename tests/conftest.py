"""Shared pytest fixtures for deployment and consistency tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"
MODEL_CONFIG_PATH = PROJECT_ROOT / "models" / "model_config.json"
CHOSEN_THRESHOLD_PATH = PROJECT_ROOT / "reports" / "chosen_threshold.json"
SPLIT_MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "split_manifest.json"


def artifacts_available() -> bool:
    return (
        PIPELINE_PATH.exists()
        and MODEL_CONFIG_PATH.exists()
        and CHOSEN_THRESHOLD_PATH.exists()
        and SPLIT_MANIFEST_PATH.exists()
    )


@pytest.fixture(scope="session")
def saved_pipeline():
    if not PIPELINE_PATH.exists():
        pytest.skip("Saved pipeline artifact missing.")
    from src.model import load_churn_pipeline

    return load_churn_pipeline()


@pytest.fixture(scope="session")
def saved_model_config() -> dict[str, Any]:
    if not MODEL_CONFIG_PATH.exists():
        pytest.skip("Saved model config missing.")
    from src.config import load_model_config

    return load_model_config()


@pytest.fixture(scope="session")
def validation_customer_record() -> dict[str, Any]:
    """One validation-set customer for end-to-end checks (test set not used)."""
    if not SPLIT_MANIFEST_PATH.exists():
        pytest.skip("Split manifest missing.")
    from src.data_split import load_split_from_manifest

    split = load_split_from_manifest()
    record = split.X_val.iloc[5].to_dict()
    record["customerID"] = str(split.id_val.iloc[5])
    return record


@pytest.fixture(scope="session")
def sample_api_payload(validation_customer_record) -> dict[str, Any]:
    """API-ready payload derived from a validation customer."""
    payload = dict(validation_customer_record)
    payload.pop("Churn", None)
    return payload
