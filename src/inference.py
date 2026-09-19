"""Inference helpers for single-customer and batch churn prediction."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.data_cleaning import IDENTIFIER_COL, TARGET_COL, TOTAL_CHARGES_COL, TENURE_COL
from src.data_separation import FEATURE_COLS
from src.model import ChurnModelBundle, load_model_bundle
from src.policy import (
    churn_prediction_label,
    classify_risk_level,
    model_version_from_config,
    normalize_service_fields,
    recommended_action,
)
from src.preprocessing import transform_features
from src.upload_compatibility import apply_column_mapping


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
    *,
    model_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Score one raw customer record end-to-end without notebook dependencies.

    Returns calibrated probability and a retention recommendation based on the
    frozen validation threshold.
    """
    normalized_record, _ = normalize_service_fields(dict(record))
    customer_id = normalized_record.get(IDENTIFIER_COL)
    features = prepare_inference_features(normalized_record)
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


def build_churn_prediction_response(
    bundle: ChurnModelBundle,
    record: Mapping[str, Any],
    *,
    model_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Full prediction payload for API and dashboard consumers."""
    base = predict_single_customer(bundle, record, model_config=model_config)
    probability = base["churn_probability"]
    threshold = float(base["decision_threshold"])

    return {
        "churn_probability": probability,
        "threshold": threshold,
        "prediction": churn_prediction_label(probability, threshold),
        "risk_level": classify_risk_level(probability, threshold),
        "recommended_action": recommended_action(base["retention_recommended"]),
        "model_version": model_version_from_config(model_config),
        "customerID": base.get(IDENTIFIER_COL),
    }


def load_bundle_and_predict(
    record: Mapping[str, Any],
    bundle_path: str | None = None,
) -> dict[str, Any]:
    """Convenience wrapper: load saved bundle and score one customer record."""
    bundle = load_model_bundle(bundle_path)
    return predict_single_customer(bundle, record)


def _normalize_total_charges_value(raw_value: Any, tenure: int) -> float:
    if pd.isna(raw_value):
        raise ValueError("TotalCharges cannot be missing.")

    text = str(raw_value).strip()
    if text == "":
        if tenure != 0:
            raise ValueError("Blank TotalCharges is only valid when tenure is 0.")
        return 0.0

    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"Invalid TotalCharges value: {raw_value!r}") from exc


def _normalize_upload_row(row: pd.Series) -> pd.Series:
    record = row.to_dict()
    normalized, _ = normalize_service_fields(record)
    normalized[TOTAL_CHARGES_COL] = _normalize_total_charges_value(
        normalized[TOTAL_CHARGES_COL],
        int(normalized[TENURE_COL]),
    )
    normalized[TENURE_COL] = int(normalized[TENURE_COL])
    normalized["SeniorCitizen"] = int(normalized["SeniorCitizen"])
    normalized["MonthlyCharges"] = float(normalized["MonthlyCharges"])
    return pd.Series({col: normalized[col] for col in FEATURE_COLS})


def prepare_batch_upload_features(
    df: pd.DataFrame,
    id_col: str,
    *,
    column_mapping: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Normalize an uploaded Telco-compatible dataframe for batch scoring.

    Returns model-ready features, primary keys, and customer detail columns.
    """
    working = apply_column_mapping(df, column_mapping or {})
    resolved_id_col = (column_mapping or {}).get(id_col, id_col)

    if resolved_id_col not in working.columns:
        raise ValueError(f"Primary key column '{id_col}' not found after column mapping.")

    missing = [col for col in FEATURE_COLS if col not in working.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    normalized_rows = [
        _normalize_upload_row(row) for _, row in working[FEATURE_COLS].iterrows()
    ]
    features_df = pd.DataFrame(normalized_rows, columns=FEATURE_COLS)
    customer_ids = working[resolved_id_col].astype(str).reset_index(drop=True)
    details_df = working[FEATURE_COLS].reset_index(drop=True)
    return features_df, customer_ids, details_df


def score_uploaded_batch(
    bundle: ChurnModelBundle,
    df: pd.DataFrame,
    id_col: str,
    *,
    column_mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Score an uploaded Telco-compatible CSV with the frozen or session bundle."""
    features_df, customer_ids, details_df = prepare_batch_upload_features(
        df,
        id_col,
        column_mapping=column_mapping,
    )
    threshold = float(bundle.threshold)
    probabilities = predict_churn_probability(bundle, features_df)
    retention_flags = probabilities >= threshold

    output = pd.DataFrame(
        {
            "primary_key": customer_ids,
            "churn_probability": np.round(probabilities, 4),
            "prediction": [churn_prediction_label(float(p), threshold) for p in probabilities],
            "risk_level": [classify_risk_level(float(p), threshold) for p in probabilities],
            "retention_recommended": retention_flags.astype(int),
            "recommended_action": [recommended_action(bool(flag)) for flag in retention_flags],
            "decision_threshold": threshold,
        }
    )

    for col in FEATURE_COLS:
        output[col] = details_df[col]

    working = apply_column_mapping(df, column_mapping or {})
    if TARGET_COL in working.columns:
        output["actual_churn"] = working[TARGET_COL].reset_index(drop=True)

    return output.sort_values("churn_probability", ascending=False).reset_index(drop=True)


BATCH_API_RESULT_COLUMNS = [
    "primary_key",
    "churn_probability",
    "prediction",
    "risk_level",
    "retention_recommended",
    "recommended_action",
    "decision_threshold",
]


def build_batch_api_response(
    results: pd.DataFrame,
    *,
    model_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Slim batch payload for API consumers (decision columns only)."""
    missing = [col for col in BATCH_API_RESULT_COLUMNS if col not in results.columns]
    if missing:
        raise ValueError(f"Batch results missing columns: {missing}")

    slim = results[BATCH_API_RESULT_COLUMNS].copy()
    flagged = int(slim["retention_recommended"].sum())
    return {
        "model_version": model_version_from_config(model_config),
        "customers_scored": len(slim),
        "retention_outreach_flagged": flagged,
        "decision_threshold": float(slim["decision_threshold"].iloc[0]),
        "results": slim.to_dict(orient="records"),
    }
