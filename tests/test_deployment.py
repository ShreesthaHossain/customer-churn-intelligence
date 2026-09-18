"""Tests for production deployment helpers."""

import os

from src.deployment import get_settings


def test_api_auth_disabled_without_env_key(monkeypatch) -> None:
    monkeypatch.delenv("CHURN_API_KEY", raising=False)
    monkeypatch.setattr("src.deployment.load_env_file", lambda: None)
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.auth_enabled is False
    assert settings.api_docs_url == "http://127.0.0.1:8000/docs"


def test_api_auth_enabled_when_env_key_set(monkeypatch) -> None:
    monkeypatch.setenv("CHURN_API_KEY", "secret")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.auth_enabled is True
    assert settings.api_key == "secret"
    get_settings.cache_clear()
    monkeypatch.delenv("CHURN_API_KEY", raising=False)
