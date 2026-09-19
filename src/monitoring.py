"""Runtime service metrics and lightweight drift monitoring helpers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_loading import load_cleaned_churn_data
from src.data_separation import FEATURE_COLS, NUMERIC_FEATURE_COLS

PSI_ALERT_THRESHOLD = 0.2
_BASELINE_CACHE: dict[str, Any] | None = None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ServiceMetrics:
    """In-memory counters updated by the API during scoring."""

    predictions_total: int = 0
    batch_requests_total: int = 0
    errors_total: int = 0
    last_prediction_at: datetime | None = None
    recent_batch_sizes: list[int] = field(default_factory=list)

    def record_predictions(self, count: int, *, batch: bool = False) -> None:
        if count <= 0:
            return
        self.predictions_total += count
        if batch:
            self.batch_requests_total += 1
            self.recent_batch_sizes.append(count)
            self.recent_batch_sizes = self.recent_batch_sizes[-100:]
        self.last_prediction_at = _utc_now()

    def record_error(self) -> None:
        self.errors_total += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "predictions_total": self.predictions_total,
            "batch_requests_total": self.batch_requests_total,
            "errors_total": self.errors_total,
            "last_prediction_at": (
                self.last_prediction_at.isoformat() if self.last_prediction_at else None
            ),
            "avg_recent_batch_size": (
                round(float(np.mean(self.recent_batch_sizes)), 2)
                if self.recent_batch_sizes
                else None
            ),
        }


def prediction_log_path() -> Path | None:
    raw = os.getenv("PREDICTION_LOG_PATH")
    return Path(raw) if raw else None


def append_prediction_log(entry: dict[str, Any]) -> None:
    """Append one JSON line when PREDICTION_LOG_PATH is configured."""
    path = prediction_log_path()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": _utc_now().isoformat(), **entry}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def _population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Simple PSI for numeric features using quantile bins from the baseline."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    if len(edges) < 3:
        return 0.0

    expected_counts = np.histogram(expected, bins=edges)[0].astype(float)
    actual_counts = np.histogram(actual, bins=edges)[0].astype(float)
    expected_pct = expected_counts / max(expected_counts.sum(), 1.0)
    actual_pct = actual_counts / max(actual_counts.sum(), 1.0)

    psi = 0.0
    for exp, act in zip(expected_pct, actual_pct):
        if exp <= 0 and act <= 0:
            continue
        exp = max(exp, 1e-6)
        act = max(act, 1e-6)
        psi += (act - exp) * np.log(act / exp)
    return float(max(psi, 0.0))


def build_training_feature_baselines() -> dict[str, Any]:
    """Summarize training-like reference distributions from cleaned Telco data."""
    global _BASELINE_CACHE
    if _BASELINE_CACHE is not None:
        return _BASELINE_CACHE

    df = load_cleaned_churn_data()
    baselines: dict[str, Any] = {
        "source": "data/processed/cleaned_churn.csv",
        "row_count": len(df),
        "numeric": {},
        "categorical": {},
    }

    for col in NUMERIC_FEATURE_COLS:
        baselines["numeric"][col] = {
            "mean": round(float(df[col].mean()), 4),
            "median": round(float(df[col].median()), 4),
            "values": df[col].to_numpy(),
        }

    for col in FEATURE_COLS:
        if col in NUMERIC_FEATURE_COLS:
            continue
        proportions = df[col].value_counts(normalize=True).round(4).to_dict()
        baselines["categorical"][col] = proportions

    _BASELINE_CACHE = baselines
    return baselines


def compute_drift_report(scored_df: pd.DataFrame) -> dict[str, Any]:
    """
    Compare a scored batch to cleaned-data baselines.

    Returns PSI for numeric features and max category share shift for key fields.
    """
    baselines = build_training_feature_baselines()
    report: dict[str, Any] = {
        "rows_compared": len(scored_df),
        "psi_alert_threshold": PSI_ALERT_THRESHOLD,
        "numeric_drift": {},
        "categorical_drift": {},
        "alerts": [],
    }

    for col, stats in baselines["numeric"].items():
        if col not in scored_df.columns:
            continue
        psi = _population_stability_index(stats["values"], scored_df[col].to_numpy())
        baseline_mean = stats["mean"]
        batch_mean = round(float(scored_df[col].mean()), 4)
        entry = {"psi": round(psi, 4), "baseline_mean": baseline_mean, "batch_mean": batch_mean}
        report["numeric_drift"][col] = entry
        if psi >= PSI_ALERT_THRESHOLD:
            report["alerts"].append(f"Numeric drift on {col}: PSI={psi:.3f}")

    for col, baseline_props in baselines["categorical"].items():
        if col not in scored_df.columns:
            continue
        batch_props = scored_df[col].value_counts(normalize=True)
        max_shift = 0.0
        worst_category = None
        for category, baseline_share in baseline_props.items():
            batch_share = float(batch_props.get(category, 0.0))
            shift = abs(batch_share - float(baseline_share))
            if shift > max_shift:
                max_shift = shift
                worst_category = category
        entry = {
            "max_share_shift": round(max_shift, 4),
            "worst_category": worst_category,
        }
        report["categorical_drift"][col] = entry
        if max_shift >= 0.15:
            report["alerts"].append(
                f"Categorical shift on {col}: max share delta={max_shift:.3f}"
            )

    report["status"] = "alert" if report["alerts"] else "ok"
    return report
