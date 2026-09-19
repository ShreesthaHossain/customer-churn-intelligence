"""Production deployment settings, logging, and API security helpers."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


@dataclass(frozen=True)
class DeploymentSettings:
    """Runtime settings loaded from environment variables."""

    api_key: str | None
    api_base_url: str
    batch_max_rows: int
    log_level: str
    log_json: bool
    environment: str

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_key)

    @property
    def api_docs_url(self) -> str:
        return f"{self.api_base_url.rstrip('/')}/docs"

    @property
    def api_health_url(self) -> str:
        return f"{self.api_base_url.rstrip('/')}/health"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_env_file() -> None:
    """Load project-root `.env` when python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        load_dotenv(env_path)


@lru_cache(maxsize=1)
def get_settings() -> DeploymentSettings:
    load_env_file()
    return DeploymentSettings(
        api_key=os.getenv("CHURN_API_KEY") or None,
        api_base_url=os.getenv("CHURN_API_BASE_URL", "http://127.0.0.1:8000"),
        batch_max_rows=int(os.getenv("BATCH_MAX_ROWS", "10000")),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        log_json=_env_bool("LOG_JSON", False),
        environment=os.getenv("ENVIRONMENT", "development"),
    )


class JsonLogFormatter(logging.Formatter):
    """Emit one JSON object per log line for production log aggregators."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in (
            "request_id",
            "method",
            "path",
            "status_code",
            "duration_ms",
            "customer_id",
            "churn_probability",
            "prediction",
            "model_version",
            "event",
            "batch_size",
            "retention_outreach_flagged",
        ):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    """Configure root logging once at API startup."""
    settings = get_settings()
    handler = logging.StreamHandler()
    if settings.log_json:
        handler.setFormatter(JsonLogFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(settings.log_level)


async def require_api_key(api_key: str | None = Security(API_KEY_HEADER)) -> None:
    """Optional API-key guard — enabled when CHURN_API_KEY is set."""
    settings = get_settings()
    if not settings.auth_enabled:
        return
    if not api_key or api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
