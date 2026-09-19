"""FastAPI service for churn prediction."""

from __future__ import annotations

import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.batch import router as batch_router
from api.schemas import (
    CustomerFeatures,
    HealthResponse,
    MonitoringSummaryResponse,
    PredictChurnResponse,
)
from src.config import load_model_config
from src.deployment import configure_logging, get_settings, require_api_key
from src.inference import build_churn_prediction_response
from src.model import ChurnModelBundle, load_churn_pipeline
from src.monitoring import ServiceMetrics, append_prediction_log
from src.policy import normalize_service_fields, model_version_from_config

logger = logging.getLogger("churn.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.deployment import load_env_file

    load_env_file()
    get_settings.cache_clear()
    configure_logging()
    settings = get_settings()
    logger.info(
        "api_startup",
        extra={
            "event": "api_startup",
            "environment": settings.environment,
            "auth_enabled": settings.auth_enabled,
        },
    )
    app.state.pipeline = load_churn_pipeline()
    app.state.model_config = load_model_config()
    app.state.metrics = ServiceMetrics()
    app.state.last_drift_report = None
    yield


app = FastAPI(
    title="Customer Churn Intelligence API",
    description="Score telecom customers for churn risk using the saved calibrated pipeline.",
    version="1.1.0",
    lifespan=lifespan,
)
app.include_router(batch_router)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        "request_completed",
        extra={
            "event": "request_completed",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    pipeline: ChurnModelBundle | None = getattr(request.app.state, "pipeline", None)
    config: dict[str, Any] | None = getattr(request.app.state, "model_config", None)
    metrics: ServiceMetrics = request.app.state.metrics
    settings = get_settings()
    model_version = model_version_from_config(config) if config else None
    metrics_dict = metrics.as_dict()
    return HealthResponse(
        status="ok",
        model_loaded=pipeline is not None,
        model_version=model_version,
        environment=settings.environment,
        auth_enabled=settings.auth_enabled,
        predictions_total=metrics_dict["predictions_total"],
        batch_requests_total=metrics_dict["batch_requests_total"],
        last_prediction_at=metrics_dict["last_prediction_at"],
    )


@app.get("/monitoring/summary", response_model=MonitoringSummaryResponse)
def monitoring_summary(request: Request) -> MonitoringSummaryResponse:
    """Lightweight runtime monitoring snapshot for operators."""
    metrics: ServiceMetrics = request.app.state.metrics
    drift_report = getattr(request.app.state, "last_drift_report", None) or {
        "status": "no_data",
        "alerts": [],
    }
    settings = get_settings()
    return MonitoringSummaryResponse(
        service_metrics=metrics.as_dict(),
        drift_status=drift_report.get("status", "no_data"),
        drift_alerts=drift_report.get("alerts", []),
        batch_max_rows=settings.batch_max_rows,
    )


@app.post("/predict_churn", response_model=PredictChurnResponse)
def predict_churn(
    customer: CustomerFeatures,
    request: Request,
    _: None = Depends(require_api_key),
) -> PredictChurnResponse:
    pipeline: ChurnModelBundle = request.app.state.pipeline
    config: dict[str, Any] = request.app.state.model_config
    metrics: ServiceMetrics = request.app.state.metrics
    request_id = getattr(request.state, "request_id", None)

    record = customer.model_dump()
    record, _ = normalize_service_fields(record)

    try:
        result = build_churn_prediction_response(pipeline, record, model_config=config)
    except ValueError as exc:
        metrics.record_error()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        metrics.record_error()
        logger.exception(
            "prediction_failed",
            extra={"event": "prediction_failed", "request_id": request_id},
        )
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    metrics.record_predictions(1, batch=False)
    append_prediction_log(
        {
            "event": "prediction_completed",
            "request_id": request_id,
            "customer_id": result.get("customerID"),
            "churn_probability": result["churn_probability"],
            "prediction": result["prediction"],
            "model_version": result.get("model_version"),
        }
    )
    logger.info(
        "prediction_completed",
        extra={
            "event": "prediction_completed",
            "request_id": request_id,
            "customer_id": result.get("customerID"),
            "churn_probability": result["churn_probability"],
            "prediction": result["prediction"],
            "model_version": result.get("model_version"),
        },
    )
    return PredictChurnResponse(**result)
