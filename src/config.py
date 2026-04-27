"""Configuration management for LiteLLM MCP server."""

import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_FILE_PATH)


class LiteLLMSettings(BaseSettings):
    """LiteLLM MCP server settings."""

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    proxy_url: str = Field(
        default="http://localhost:4000",
        alias="LITELLM_PROXY_URL",
        description="LiteLLM proxy base URL",
    )

    api_key: Optional[SecretStr] = Field(
        default=None,
        alias="LITELLM_API_KEY",
        description="LiteLLM API key (master key or virtual key, sent as Bearer token)",
    )

    transport: str = Field(
        default="stdio",
        alias="LITELLM_TRANSPORT",
        description="MCP transport mode: stdio or streamable-http",
    )

    timeout_seconds: float = Field(
        default=30.0,
        alias="LITELLM_TIMEOUT_SECONDS",
        description="HTTP request timeout in seconds",
    )

    max_retries: int = Field(
        default=2,
        alias="LITELLM_MAX_RETRIES",
        description="Number of retries on transient HTTP errors",
    )

    @property
    def has_api_key(self) -> bool:
        return self.api_key is not None

    def get_api_key_value(self) -> str:
        if self.api_key is None:
            raise ValueError("LITELLM_API_KEY is required but not set")
        return self.api_key.get_secret_value()


def mask_credential(value: str, visible_chars: int = 2) -> str:
    """Mask a credential for safe logging."""
    if not value:
        return "<empty>"
    if len(value) <= visible_chars * 2:
        return "*" * len(value)
    return (
        f"{value[:visible_chars]}{'*' * (len(value) - visible_chars * 2)}{value[-visible_chars:]}"
    )


settings = LiteLLMSettings()
