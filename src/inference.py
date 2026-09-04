"""Inference helpers for single-customer and batch churn prediction."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.data_cleaning import IDENTIFIER_COL, TOTAL_CHARGES_COL, TENURE_COL
from src.data_separation import FEATURE_COLS
from src.model import ChurnModelBundle, load_model_bundle
from src.preprocessing import transform_features


def _normalize_total_charges(record: dict[str, Any]) -> float:
    """
    Apply the same TotalCharges rule used in cleaning for inference-time records.

    Blank charges are allowed only for new customers with tenure=0.
    """
    raw_value = record[TOTAL_CHARGES_COL]
    if pd.isna(raw_value):
        raise ValueError("TotalCharges cannot be missing.")

    text = str(raw_value).strip()
    if text == "":
        if record[TENURE_COL] != 0:
            raise ValueError("Blank TotalCharges is only valid when tenure is 0.")
        return 0.0

    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"Invalid TotalCharges value: {raw_value!r}") from exc


def prepare_inference_features(record: Mapping[str, Any]) -> pd.DataFrame:
    """
    Validate and normalize one raw customer record into a model-ready feature row.

    The record must contain all predictive feature columns. customerID and Churn
    are optional and ignored if present.
    """
    missing = [col for col in FEATURE_COLS if col not in record]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    normalized = {col: record[col] for col in FEATURE_COLS}
    normalized[TOTAL_CHARGES_COL] = _normalize_total_charges(normalized)
    normalized[TENURE_COL] = int(normalized[TENURE_COL])
    normalized["SeniorCitizen"] = int(normalized["SeniorCitizen"])
    normalized["MonthlyCharges"] = float(normalized["MonthlyCharges"])

    return pd.DataFrame([normalized], columns=FEATURE_COLS)


def predict_churn_probability(
    bundle: ChurnModelBundle,
    features: pd.DataFrame,
) -> np.ndarray:
    """Return calibrated churn probabilities for one or more feature rows."""
    if features.empty:
        raise ValueError("features must contain at least one row.")
    transformed = transform_features(bundle.preprocessor, features)
    return bundle.model.predict_proba(transformed)[:, 1]


def apply_retention_decision(
    probabilities: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Convert probabilities into retention-outreach flags using the frozen threshold."""
    return (np.asarray(probabilities) >= float(threshold)).astype(int)


def predict_batch(
    bundle: ChurnModelBundle,
    features: pd.DataFrame,
    *,
    customer_ids: pd.Series | None = None,
) -> pd.DataFrame:
    """Predict probabilities and retention decisions for multiple customers."""
    probabilities = predict_churn_probability(bundle, features)
    decisions = (probabilities >= bundle.threshold).astype(int)

    output = pd.DataFrame(
        {
            "churn_probability": probabilities,
            "retention_recommended": decisions,
            "decision_threshold": bundle.threshold,
        }
    )
    if customer_ids is not None:
        if len(customer_ids) != len(output):
            raise ValueError("customer_ids length must match number of feature rows.")
        output.insert(0, IDENTIFIER_COL, customer_ids.reset_index(drop=True))
    return output


def predict_single_customer(
    bundle: ChurnModelBundle,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Score one raw customer record end-to-end without notebook dependencies.

    Returns calibrated probability and a retention recommendation based on the
    frozen validation threshold.
    """
    customer_id = record.get(IDENTIFIER_COL)
    features = prepare_inference_features(record)
    probability = float(predict_churn_probability(bundle, features)[0])
    retention_recommended = bool(probability >= bundle.threshold)

    return {
        IDENTIFIER_COL: customer_id,
        "churn_probability": round(probability, 4),
        "retention_recommended": retention_recommended,
        "decision_threshold": bundle.threshold,
        "interpretation_note": (
            "Probability and recommendation reflect model output; they do not "
            "establish causal reasons for churn."
        ),
    }


def load_bundle_and_predict(
    record: Mapping[str, Any],
    bundle_path: str | None = None,
) -> dict[str, Any]:
    """Convenience wrapper: load saved bundle and score one customer record."""
    bundle = load_model_bundle(bundle_path)
    return predict_single_customer(bundle, record)
