"""Shared retention policy and risk-band logic for apps and APIs."""

from __future__ import annotations

from typing import Any

# UI/API display band — not a model threshold.
HIGH_RISK_UI_BAND = 0.50

INTERNET_DEPENDENT_FIELDS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
NO_INTERNET_SERVICE = "No internet service"
NO_PHONE_SERVICE = "No phone service"


def classify_risk_level(probability: float, decision_threshold: float) -> str:
    """
    Map probability to risk bands anchored on the frozen decision threshold.

    - Low: below decision threshold
    - Elevated: at/above threshold but below HIGH_RISK_UI_BAND
    - High: at/above HIGH_RISK_UI_BAND (display band for priority escalation)
    """
    if probability < decision_threshold:
        return "Low"
    if probability < HIGH_RISK_UI_BAND:
        return "Elevated"
    return "High"


def churn_prediction_label(probability: float, threshold: float) -> str:
    """Binary churn class label using the frozen decision threshold."""
    return "Churn" if probability >= threshold else "No Churn"


def recommended_action(retention_recommended: bool) -> str:
    if retention_recommended:
        return "Recommend retention outreach — offer proactive save campaign."
    return "Routine monitoring — no immediate retention outreach."


def normalize_service_fields(record: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Align internet/phone-dependent fields with parent service selections."""
    normalized = dict(record)
    notes: list[str] = []

    if normalized.get("InternetService") == "No":
        for field in INTERNET_DEPENDENT_FIELDS:
            if normalized.get(field) != NO_INTERNET_SERVICE:
                normalized[field] = NO_INTERNET_SERVICE
                notes.append(
                    f"{field} was set to '{NO_INTERNET_SERVICE}' because Internet Service is No."
                )

    if normalized.get("PhoneService") == "No" and normalized.get("MultipleLines") != NO_PHONE_SERVICE:
        normalized["MultipleLines"] = NO_PHONE_SERVICE
        notes.append(
            f"Multiple Lines was set to '{NO_PHONE_SERVICE}' because Phone Service is No."
        )

    return normalized, notes


def model_version_from_config(config: dict[str, Any] | None) -> str | None:
    """Build a readable model version string from saved deployment config."""
    if not config:
        return None
    model_name = config.get("selected_model", "unknown")
    calibration = config.get("calibration_method")
    threshold = config.get("decision_threshold")
    if calibration:
        return f"{model_name}|{calibration}|threshold={threshold}"
    return f"{model_name}|threshold={threshold}"
