"""LiteLLM MCP server — proxy admin and execution tools."""

import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from src.config import mask_credential, settings
from src.litellm_client import LiteLLMClient

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# ── Response field filtering ──

RESPONSE_FIELDS: dict[str, dict[str, Optional[list[str]]]] = {
    "model": {
        "minimal": ["id"],
        "standard": ["id", "object", "owned_by", "created"],
        "full": None,
    },
}

VALID_VERBOSITY_LEVELS = {"minimal", "standard", "full"}


def _filter_response(response: Any, resource_type: str, verbosity: str = "standard") -> Any:
    """Filter response fields based on verbosity level."""
    if response is None:
        return None

    if verbosity not in VALID_VERBOSITY_LEVELS:
        logger.warning("Invalid verbosity '%s', using 'standard'", verbosity)
        verbosity = "standard"

    if verbosity == "full" or resource_type not in RESPONSE_FIELDS:
        return response

    fields = RESPONSE_FIELDS[resource_type].get(verbosity)
    if fields is None:
        return response

    field_set = set(fields)

    def filter_dict(d: dict) -> dict:
        return {k: v for k, v in d.items() if k in field_set}

    if isinstance(response, list):
        return [filter_dict(item) for item in response if isinstance(item, dict)]
    if isinstance(response, dict):
        return filter_dict(response)
    return response


# ── Client accessor ──

_client: Optional[LiteLLMClient] = None


def get_client() -> LiteLLMClient:
    if _client is None:
        raise RuntimeError("LiteLLM client not initialised; server lifespan did not run")
    return _client


@asynccontextmanager
async def lifespan(_server: FastMCP) -> AsyncIterator[None]:
    """Initialise the shared LiteLLM client for the server lifetime."""
    global _client
    if not settings.has_api_key:
        raise RuntimeError("LITELLM_API_KEY is required but not set")

    api_key = settings.get_api_key_value()
    logger.info(
        "Connecting to LiteLLM proxy at %s (key=%s)",
        settings.proxy_url,
        mask_credential(api_key, visible_chars=4),
    )

    _client = LiteLLMClient(
        base_url=settings.proxy_url,
        api_key=api_key,
        timeout=settings.timeout_seconds,
        max_retries=settings.max_retries,
    )
    try:
        yield
    finally:
        if _client is not None:
            await _client.close()
            _client = None


mcp = FastMCP("litellm-mcp", lifespan=lifespan)


# ── Tools ──


@mcp.tool()
async def list_models(verbosity: str = "standard") -> list[dict]:
    """List models exposed by the LiteLLM proxy (`GET /v1/models`).

    Args:
        verbosity: 'minimal' (id only), 'standard' (id/object/owned_by/created),
            or 'full' (all fields returned by the proxy).
    """
    models = await get_client().list_models()
    return _filter_response(models, "model", verbosity)


@mcp.tool()
async def get_model(model_id: str, verbosity: str = "standard") -> dict:
    """Get a single model entry by id (`GET /v1/models/{model_id}`).

    Args:
        model_id: The OpenAI-style model id (e.g. `gpt-4o`).
        verbosity: 'minimal' / 'standard' / 'full'.
    """
    model = await get_client().get_model(model_id)
    return _filter_response(model, "model", verbosity)


@mcp.tool()
async def add_model(
    model_name: str,
    litellm_params: dict,
    model_info: dict,
) -> dict:
    """Register a new model deployment (`POST /model/new`).

    Args:
        model_name: client-facing alias (e.g. `gpt-4o`).
        litellm_params: provider routing dict (e.g.
            `{"model": "openai/gpt-4o", "api_key": "sk-..."}`).
        model_info: deployment metadata (e.g. `{"id": "...", "db_model": false}`).
    """
    return await get_client().add_model(model_name, litellm_params, model_info)


@mcp.tool()
async def update_model(
    model_id: str,
    model_name: Optional[str] = None,
    litellm_params: Optional[dict] = None,
    model_info: Optional[dict] = None,
) -> dict:
    """Patch an existing model deployment (`PATCH /model/{model_id}/update`).

    Only the provided fields are sent. The deployment id (`model_id`) goes in
    the path, not the body.
    """
    return await get_client().update_model(model_id, model_name, litellm_params, model_info)


@mcp.tool()
async def delete_model(model_id: str) -> dict:
    """Delete a model deployment (`POST /model/delete`).

    Args:
        model_id: deployment id (sent in body as `{"id": model_id}`).
    """
    return await get_client().delete_model(model_id)


@mcp.tool()
async def list_public_models(verbosity: str = "standard") -> Any:
    """List models published to the public Model Hub (`GET /public/model_hub`).

    Args:
        verbosity: 'minimal' / 'standard' / 'full'.
    """
    payload = await get_client().list_public_models()
    return _filter_response(payload, "public_hub", verbosity)


@mcp.tool()
async def get_public_hub_info() -> dict:
    """Get Model Hub metadata — title, description, useful links (`GET /public/model_hub/info`)."""
    return await get_client().get_public_hub_info()


@mcp.tool()
async def get_model_cost_map() -> dict:
    """Get the LiteLLM static model cost / capability map (`GET /public/litellm_model_cost_map`).

    Large response (~1MB). Useful for the admin agent to look up token pricing
    and context windows without an outbound call.
    """
    return await get_client().get_model_cost_map()


@mcp.tool()
async def make_model_group_public(model_groups: list[str]) -> dict:
    """Publish model groups to the public Model Hub (`POST /model_group/make_public`).

    Replaces (not appends) the current set of published groups.

    Args:
        model_groups: list of model_name strings to publish.
    """
    return await get_client().make_model_group_public(model_groups)


@mcp.tool()
async def get_model_info(litellm_model_id: Optional[str] = None) -> dict:
    """Get admin-side model info — full deployment details (`GET /model/info`).

    Returns the upstream payload as-is (no verbosity filtering — this endpoint
    surfaces operational details like litellm_params and model_info that
    callers usually want to see in full).

    Args:
        litellm_model_id: Optional litellm internal model id to filter to a
            single deployment.
    """
    return await get_client().get_model_info(litellm_model_id)


# ── Entrypoint ──


def main() -> None:
    transport = settings.transport.lower()
    if transport == "stdio":
        mcp.run(transport="stdio")
    elif transport in {"streamable-http", "http"}:
        mcp.run(transport="streamable-http")
    else:
        print(f"Unknown LITELLM_TRANSPORT: {settings.transport}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
