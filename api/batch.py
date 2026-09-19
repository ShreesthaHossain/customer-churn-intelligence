"""Batch churn prediction API routes."""

from __future__ import annotations

import io
import logging
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from src.data_cleaning import IDENTIFIER_COL
from src.data_separation import FEATURE_COLS
from src.deployment import get_settings, require_api_key
from src.inference import build_batch_api_response, score_uploaded_batch
from src.monitoring import append_prediction_log, compute_drift_report
from src.upload_compatibility import check_upload_compatibility, normalize_upload_headers

from api.schemas import BatchPredictRequest, BatchPredictResponse

logger = logging.getLogger("churn.api")
router = APIRouter(tags=["batch"])


def _enforce_batch_limit(row_count: int) -> None:
    limit = get_settings().batch_max_rows
    if row_count > limit:
        raise HTTPException(
            status_code=413,
            detail=f"Batch size {row_count} exceeds limit of {limit} rows.",
        )


def _score_dataframe(
    request: Request,
    df: pd.DataFrame,
    id_col: str,
    *,
    request_id: str | None,
) -> BatchPredictResponse:
    pipeline = request.app.state.pipeline
    config: dict[str, Any] = request.app.state.model_config
    metrics = request.app.state.metrics

    _enforce_batch_limit(len(df))
    try:
        results = score_uploaded_batch(pipeline, df, id_col)
        payload = build_batch_api_response(results, model_config=config)
    except ValueError as exc:
        metrics.record_error()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        metrics.record_error()
        logger.exception(
            "batch_prediction_failed",
            extra={"event": "batch_prediction_failed", "request_id": request_id},
        )
        raise HTTPException(status_code=500, detail=f"Batch scoring failed: {exc}") from exc

    metrics.record_predictions(payload["customers_scored"], batch=True)

    feature_cols = [col for col in FEATURE_COLS if col in results.columns]
    drift = compute_drift_report(results[feature_cols]) if feature_cols else {"status": "ok", "alerts": []}
    request.app.state.last_drift_report = drift

    append_prediction_log(
        {
            "event": "batch_prediction_completed",
            "request_id": request_id,
            "customers_scored": payload["customers_scored"],
            "retention_outreach_flagged": payload["retention_outreach_flagged"],
            "model_version": payload["model_version"],
            "drift_status": drift.get("status"),
            "drift_alerts": drift.get("alerts", []),
        }
    )

    logger.info(
        "batch_prediction_completed",
        extra={
            "event": "batch_prediction_completed",
            "request_id": request_id,
            "batch_size": payload["customers_scored"],
            "retention_outreach_flagged": payload["retention_outreach_flagged"],
            "model_version": payload["model_version"],
        },
    )
    return BatchPredictResponse(**payload)


@router.post("/predict_churn_batch", response_model=BatchPredictResponse)
def predict_churn_batch_json(
    body: BatchPredictRequest,
    request: Request,
    _: None = Depends(require_api_key),
) -> BatchPredictResponse:
    """Score many customers from a JSON list using the frozen Telco schema."""
    records = [customer.model_dump() for customer in body.customers]
    df = pd.DataFrame(records)
    if body.primary_key_column not in df.columns:
        if IDENTIFIER_COL in df.columns:
            df[body.primary_key_column] = df[IDENTIFIER_COL]
        else:
            df[body.primary_key_column] = [f"BATCH-{i + 1:05d}" for i in range(len(df))]

    request_id = getattr(request.state, "request_id", None)
    return _score_dataframe(request, df, body.primary_key_column, request_id=request_id)


@router.post("/predict_churn_batch/file", response_model=BatchPredictResponse)
async def predict_churn_batch_file(
    request: Request,
    file: UploadFile = File(...),
    primary_key_column: str = IDENTIFIER_COL,
    _: None = Depends(require_api_key),
) -> BatchPredictResponse:
    """Score a Telco-compatible CSV upload (same schema as Streamlit batch upload)."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="Upload must be a CSV file.")

    raw = await file.read()
    try:
        df = normalize_upload_headers(pd.read_csv(io.BytesIO(raw)))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read CSV: {exc}") from exc

    if primary_key_column not in df.columns:
        raise HTTPException(
            status_code=422,
            detail=f"Primary key column '{primary_key_column}' not found in upload.",
        )

    report = check_upload_compatibility(df, primary_key_column)
    if not report.is_compatible:
        detail = "; ".join(report.user_friendly_messages or report.blocking_errors)
        raise HTTPException(status_code=422, detail=detail or "Upload is not compatible.")

    request_id = getattr(request.state, "request_id", None)
    return _score_dataframe(request, df, primary_key_column, request_id=request_id)
