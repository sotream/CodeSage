"""Application settings.

Centralizing configuration here — instead of scattering `os.environ.get(...)`
calls through the pipeline modules — means every setting is typed, validated
once at startup, and documented in one place.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, overridable via environment variables or `.env`."""

    model_config = SettingsConfigDict(env_prefix="CODESAGE_", env_file=".env", extra="ignore")

    anthropic_api_key: str = Field(default="")
    anthropic_model: str = Field(default="claude-sonnet-4-6")
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    index_path: str = Field(default=".codesage/index.json")
    top_k: int = Field(default=5, ge=1, le=50)
    max_function_lines: int = Field(
        default=200,
        ge=1,
        description="Functions longer than this are still kept as one chunk; "
        "documented here as a known limitation rather than silently truncated.",
    )


def get_settings() -> Settings:
    """Load settings once per call site. Callers should hold the result, not re-call per chunk."""
    return Settings()
