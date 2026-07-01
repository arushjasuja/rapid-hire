"""Runtime configuration.

Everything tunable lives here so the rest of the code never reads os.environ
directly. Values come from environment variables prefixed with ``RAPIDHIRE_`` or
from a local ``.env`` file. Secrets are not read at import time -- ``get_settings``
is what triggers the read, and it's cached, so tests can set env vars first.
"""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAPIDHIRE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- credentials ---------------------------------------------------------
    # Accepts RAPIDHIRE_OPENAI_API_KEY. We also fall back to a bare OPENAI_API_KEY
    # in get_settings(), because that's what most people already have exported and
    # what LiteLLM / AutoGen read on their own.
    openai_api_key: str | None = Field(default=None)

    # --- models --------------------------------------------------------------
    reasoning_model: str = "gpt-4o"  # scoring + summary; the expensive, careful work
    intake_model: str = "gpt-4o-mini"  # extraction + light formatting; cheap and fine
    embedding_model: str = "all-MiniLM-L6-v2"  # 384-dim, English, truncates at 256 tokens

    # --- vector store --------------------------------------------------------
    chroma_mode: str = "persistent"  # "persistent" (single process) or "http" (server)
    chroma_path: str = "./data/chroma"  # used when chroma_mode == "persistent"
    chroma_host: str = "localhost"  # used when chroma_mode == "http"
    chroma_port: int = 8000
    criteria_dir: str = "./data/criteria"  # seeded into the vector store on first run
    top_k: int = 5  # nearest chunks pulled during matching/retrieval
    relevance_threshold: float = 0.3  # cosine floor; weaker retrieved chunks are dropped

    # --- scoring / routing ---------------------------------------------------
    # Overall scores are 0-100. Anything in [borderline_low, borderline_high) is a
    # "hold" and, if the panel is on, gets a second opinion from the AutoGen debate.
    borderline_low: float = 50.0
    borderline_high: float = 70.0
    enable_panel: bool = False  # off by default; the pipeline is fully functional without it
    panel_max_messages: int = 8  # hard stop so a stuck debate can't run forever

    # --- misc ----------------------------------------------------------------
    log_level: str = "INFO"
    request_timeout: float = 60.0  # seconds, per LLM call
    max_retries: int = 3  # tenacity attempts on transient LLM/network errors

    @property
    def is_borderline_band(self) -> tuple[float, float]:
        return (self.borderline_low, self.borderline_high)


class MissingAPIKeyError(RuntimeError):
    """Raised when an LLM call is attempted without a configured key."""


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    # Bare OPENAI_API_KEY is the common case; adopt it if the prefixed one is unset.
    if not settings.openai_api_key:
        settings.openai_api_key = os.environ.get("OPENAI_API_KEY")
    return settings


def require_api_key(settings: Settings | None = None) -> str:
    """Return the OpenAI key or fail with an actionable message.

    Call this at the edge of anything that hits the LLM, not at import time --
    the vector-store and parsing paths run fine without a key, and tests rely on that.
    Pass the ``settings`` you've already resolved so this checks the same object rather
    than re-reading the global cache (which can differ under a custom Settings).
    """
    key = (settings or get_settings()).openai_api_key
    if not key:
        raise MissingAPIKeyError(
            "No OpenAI API key found. Set RAPIDHIRE_OPENAI_API_KEY (or OPENAI_API_KEY) "
            "in your environment or a .env file. See .env.example."
        )
    # Downstream libraries (LiteLLM via CrewAI, autogen-ext) read OPENAI_API_KEY from
    # the environment rather than from our Settings object, so mirror it there once.
    os.environ.setdefault("OPENAI_API_KEY", key)
    return key
