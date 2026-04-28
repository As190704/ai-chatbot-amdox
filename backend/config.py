"""
config.py — Centralised settings loader using pydantic-settings.
All values can be overridden via a .env file or real environment variables.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration."""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── OpenAI ────────────────────────────────────────────────
    openai_api_key: str = ""
    openai_model: str = "gpt-3.5-turbo"

    # ── Local LLM ─────────────────────────────────────────────
    local_model_name: str = "facebook/blenderbot-400M-distill"
    preload_local_model: bool = False

    # ── Database ──────────────────────────────────────────────
    database_url: str = "sqlite:///./chatbot.db"

    # ── Server ────────────────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = True

    # ── NLP ───────────────────────────────────────────────────
    context_window: int = 10
    intent_confidence_threshold: float = 0.35

    # ── Derived helpers ───────────────────────────────────────
    @property
    def has_openai(self) -> bool:
        """True when a non-empty OpenAI API key is configured."""
        return bool(self.openai_api_key and self.openai_api_key.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()


# Convenient module-level alias
settings: Settings = get_settings()
