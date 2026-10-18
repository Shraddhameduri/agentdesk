"""Central configuration loaded from environment variables."""
from __future__ import annotations

import os


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


class Settings:
    APP_NAME = _get("APP_NAME", "agentdesk")
    API_KEY = _get("API_KEY", "demo-key")
    LLM_PROVIDER = _get("LLM_PROVIDER", "fake").lower()  # fake | openai | ollama
    FALLBACK_PROVIDER = "fake"

    OPENAI_API_KEY = _get("OPENAI_API_KEY", "")
    OPENAI_MODEL = _get("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_BASE_URL = _get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OLLAMA_BASE_URL = _get("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = _get("OLLAMA_MODEL", "llama3.1")

    DATABASE_URL = _get("DATABASE_URL", "sqlite:///./data/app.db")
    CACHE_TTL_SECONDS = _get_int("CACHE_TTL_SECONDS", 300)
    REQUEST_TIMEOUT_SECONDS = _get_int("REQUEST_TIMEOUT_SECONDS", 30)
    LOG_LEVEL = _get("LOG_LEVEL", "INFO")

    PROVIDER_MAX_RETRIES = _get_int("PROVIDER_MAX_RETRIES", 3)
    PROVIDER_BACKOFF_BASE = 0.5  # seconds, exponential

    # Public demo mode: force fake provider even if configured otherwise
    DEMO_MODE = _get("DEMO_MODE", "false").lower() == "true"


settings = Settings()
