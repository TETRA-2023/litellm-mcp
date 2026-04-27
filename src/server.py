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
    "access_group": {
        "minimal": ["access_group"],
        "standard": ["access_group", "model_names", "model_ids"],
        "full": None,
    },
    "credential": {
        # 'standard' deliberately omits credential_values to avoid leaking
        # secrets into agent transcripts. Use verbosity='full' to inspect them.
        "minimal": ["credential_name"],
        "standard": ["credential_name", "credential_info"],
        "full": None,
    },
    "key": {
        "minimal": ["token", "key_name", "key_alias"],
        "standard": [
            "token",
            "key_name",
            "key_alias",
            "spend",
            "max_budget",
            "models",
            "user_id",
            "team_id",
            "expires",
            "blocked",
        ],
        "full": None,
    },
    "public_hub": {
        "minimal": ["model_group"],
        "standard": ["model_group", "providers", "max_input_tokens", "max_output_tokens"],
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
async def update_model_hub_links(useful_links: dict) -> dict:
    """Update the Model Hub useful-links section (`POST /model_hub/update_useful_links`).

    Args:
        useful_links: free-form mapping of label → URL
            (e.g. `{"Documentation": "https://..."}`).
    """
    return await get_client().update_model_hub_links(useful_links)


@mcp.tool()
async def list_model_access_groups(verbosity: str = "standard") -> Any:
    """List all model access groups (`GET /access_group/list`).

    Args:
        verbosity: 'minimal' / 'standard' / 'full'.
    """
    payload = await get_client().list_model_access_groups()
    return _filter_response(payload, "access_group", verbosity)


@mcp.tool()
async def get_model_access_group(access_group: str, verbosity: str = "standard") -> dict:
    """Get a single model access group by name (`GET /access_group/{access_group}/info`).

    Args:
        access_group: access group name (e.g. `engineering`).
        verbosity: 'minimal' / 'standard' / 'full'.
    """
    payload = await get_client().get_model_access_group(access_group)
    return _filter_response(payload, "access_group", verbosity)


@mcp.tool()
async def create_model_access_group(
    access_group: str,
    model_names: Optional[list[str]] = None,
    model_ids: Optional[list[str]] = None,
) -> dict:
    """Create a new model access group (`POST /access_group/new`).

    Args:
        access_group: name of the new group.
        model_names: optional list of model_name aliases to include.
        model_ids: optional list of deployment ids to include.
    """
    return await get_client().create_model_access_group(access_group, model_names, model_ids)


@mcp.tool()
async def update_model_access_group(
    access_group: str,
    model_names: Optional[list[str]] = None,
    model_ids: Optional[list[str]] = None,
) -> dict:
    """Update membership of a model access group (`PUT /access_group/{access_group}/update`).

    Replaces (not appends) the current membership.

    Args:
        access_group: target access group name.
        model_names: optional new list of model_name aliases.
        model_ids: optional new list of deployment ids.
    """
    return await get_client().update_model_access_group(access_group, model_names, model_ids)


@mcp.tool()
async def delete_model_access_group(access_group: str) -> dict:
    """Delete a model access group by name (`DELETE /access_group/{access_group}/delete`)."""
    return await get_client().delete_model_access_group(access_group)


@mcp.tool()
async def list_credentials(verbosity: str = "standard") -> Any:
    """List provider credentials (`GET /credentials`).

    Credential values are not included; use `get_credential` for details.

    Args:
        verbosity: 'minimal' / 'standard' / 'full'.
    """
    payload = await get_client().list_credentials()
    return _filter_response(payload, "credential", verbosity)


@mcp.tool()
async def get_credential(credential_name: str, verbosity: str = "standard") -> dict:
    """Get a credential by name (`GET /credentials/by_name/{credential_name}`).

    Args:
        credential_name: name of the stored credential.
        verbosity: 'minimal' / 'standard' / 'full'.
    """
    payload = await get_client().get_credential(credential_name)
    return _filter_response(payload, "credential", verbosity)


@mcp.tool()
async def get_credential_by_model(model_id: str, verbosity: str = "standard") -> dict:
    """Get the credential bound to a deployment (`GET /credentials/by_model/{model_id}`).

    Args:
        model_id: deployment id.
        verbosity: 'minimal' / 'standard' / 'full'.
    """
    payload = await get_client().get_credential_by_model(model_id)
    return _filter_response(payload, "credential", verbosity)


@mcp.tool()
async def create_credential(
    credential_name: str,
    credential_info: dict,
    credential_values: Optional[dict] = None,
    model_id: Optional[str] = None,
) -> dict:
    """Create a credential (`POST /credentials`).

    Args:
        credential_name: identifier for the new credential.
        credential_info: metadata dict (e.g. `{"custom_llm_provider": "openai"}`).
        credential_values: raw credential values (e.g. `{"api_key": "sk-..."}`).
        model_id: bind to existing deployment instead of providing values.
    """
    return await get_client().create_credential(
        credential_name, credential_info, credential_values, model_id
    )


@mcp.tool()
async def update_credential(
    credential_name: str,
    credential_info: dict,
    credential_values: dict,
) -> dict:
    """Update a credential (`PATCH /credentials/{credential_name}`).

    The upstream CredentialItem schema requires all three fields.

    Args:
        credential_name: name of credential to update.
        credential_info: full metadata dict.
        credential_values: full credential values dict.
    """
    return await get_client().update_credential(
        credential_name, credential_info, credential_values
    )


@mcp.tool()
async def delete_credential(credential_name: str) -> dict:
    """Delete a credential (`DELETE /credentials/{credential_name}`)."""
    return await get_client().delete_credential(credential_name)


@mcp.tool()
async def list_keys(
    page: Optional[int] = None,
    size: Optional[int] = None,
    user_id: Optional[str] = None,
    team_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    key_alias: Optional[str] = None,
    return_full_object: Optional[bool] = None,
    include_team_keys: Optional[bool] = None,
    sort_by: Optional[str] = None,
    sort_order: Optional[str] = None,
    verbosity: str = "standard",
) -> dict:
    """List virtual keys (`GET /key/list`).

    All filter args are optional. The response is the full paginated object;
    only `keys` items are filtered by verbosity if it's a list.

    Args:
        verbosity: 'minimal' / 'standard' / 'full'.
    """
    payload = await get_client().list_keys(
        page, size, user_id, team_id, organization_id, key_alias,
        return_full_object, include_team_keys, sort_by, sort_order,
    )
    if isinstance(payload, dict) and "keys" in payload:
        payload = dict(payload)
        payload["keys"] = _filter_response(payload["keys"], "key", verbosity)
    return payload


@mcp.tool()
async def list_key_aliases(
    page: Optional[int] = None,
    size: Optional[int] = None,
    search: Optional[str] = None,
    team_id: Optional[str] = None,
) -> dict:
    """List key aliases (`GET /key/aliases`).

    Lighter-weight than `list_keys`; returns only alias metadata.
    """
    return await get_client().list_key_aliases(page, size, search, team_id)


@mcp.tool()
async def get_key_info(key: Optional[str] = None, verbosity: str = "standard") -> dict:
    """Get info about a key (`GET /key/info`).

    Args:
        key: Optional key value. If omitted, returns info about the caller's own key.
        verbosity: 'minimal' / 'standard' / 'full'.
    """
    payload = await get_client().get_key_info(key)
    return _filter_response(payload, "key", verbosity)


@mcp.tool()
async def generate_key(
    key_alias: Optional[str] = None,
    duration: Optional[str] = None,
    models: Optional[list[str]] = None,
    max_budget: Optional[float] = None,
    budget_duration: Optional[str] = None,
    user_id: Optional[str] = None,
    team_id: Optional[str] = None,
    tpm_limit: Optional[int] = None,
    rpm_limit: Optional[int] = None,
    metadata: Optional[dict] = None,
    guardrails: Optional[list[str]] = None,
    blocked: Optional[bool] = None,
    extras: Optional[dict] = None,
) -> dict:
    """Generate a new virtual key (`POST /key/generate`).

    Common fields are explicit args. Use `extras` for the ~30 less-common
    fields in the upstream `GenerateKeyRequest` (see Swagger).

    Args:
        key_alias: human-readable alias.
        duration: TTL like `30d`, `1h`.
        models: model_name allowlist (or empty for all).
        max_budget: spend cap (USD).
        budget_duration: budget reset window (`30d`).
        user_id: bind to user.
        team_id: bind to team.
        tpm_limit: tokens-per-minute cap.
        rpm_limit: requests-per-minute cap.
        metadata: free-form metadata.
        guardrails: guardrail names to enforce.
        blocked: create as blocked.
        extras: any other GenerateKeyRequest field (merged last).
    """
    return await get_client().generate_key(
        key_alias, duration, models, max_budget, budget_duration,
        user_id, team_id, tpm_limit, rpm_limit, metadata, guardrails,
        blocked, extras,
    )


@mcp.tool()
async def generate_service_account_key(
    key_alias: Optional[str] = None,
    duration: Optional[str] = None,
    models: Optional[list[str]] = None,
    max_budget: Optional[float] = None,
    budget_duration: Optional[str] = None,
    user_id: Optional[str] = None,
    team_id: Optional[str] = None,
    tpm_limit: Optional[int] = None,
    rpm_limit: Optional[int] = None,
    metadata: Optional[dict] = None,
    guardrails: Optional[list[str]] = None,
    blocked: Optional[bool] = None,
    extras: Optional[dict] = None,
) -> dict:
    """Generate a service-account virtual key (`POST /key/service-account/generate`).

    Same body shape as `generate_key`. Upstream tags the resulting key as a
    service account (no human user binding required).
    """
    return await get_client().generate_service_account_key(
        key_alias, duration, models, max_budget, budget_duration,
        user_id, team_id, tpm_limit, rpm_limit, metadata, guardrails,
        blocked, extras,
    )


@mcp.tool()
async def update_key(
    key: str,
    key_alias: Optional[str] = None,
    duration: Optional[str] = None,
    models: Optional[list[str]] = None,
    max_budget: Optional[float] = None,
    budget_duration: Optional[str] = None,
    tpm_limit: Optional[int] = None,
    rpm_limit: Optional[int] = None,
    metadata: Optional[dict] = None,
    guardrails: Optional[list[str]] = None,
    blocked: Optional[bool] = None,
    extras: Optional[dict] = None,
) -> dict:
    """Update a virtual key (`POST /key/update`).

    Only `key` is required. All other fields are merged sparsely. Use `extras`
    for less-common UpdateKeyRequest fields not surfaced as named args.
    """
    return await get_client().update_key(
        key, key_alias, duration, models, max_budget, budget_duration,
        tpm_limit, rpm_limit, metadata, guardrails, blocked, extras,
    )


@mcp.tool()
async def regenerate_key(
    key: str,
    new_master_key: Optional[str] = None,
    duration: Optional[str] = None,
    extras: Optional[dict] = None,
) -> dict:
    """Regenerate a virtual key (`POST /key/{key}/regenerate`).

    Returns the new key value. Body is optional; pass `extras` to atomically
    update RegenerateKeyRequest fields on the new key.
    """
    return await get_client().regenerate_key(key, new_master_key, duration, extras)


@mcp.tool()
async def set_key_blocked(key: str, blocked: bool) -> dict:
    """Block or unblock a virtual key.

    Routes to `POST /key/block` when `blocked=True`, else `POST /key/unblock`.

    Args:
        key: virtual key value.
        blocked: True to block, False to unblock.
    """
    return await get_client().set_key_blocked(key, blocked)


@mcp.tool()
async def delete_keys(
    keys: Optional[list[str]] = None,
    key_aliases: Optional[list[str]] = None,
) -> dict:
    """Batch-delete virtual keys (`POST /key/delete`).

    Provide either `keys` (raw values) or `key_aliases` (alias names); at
    least one must be non-empty.
    """
    return await get_client().delete_keys(keys, key_aliases)


@mcp.tool()
async def reset_key_spend(key: str, reset_to: float = 0.0) -> dict:
    """Reset a key's accumulated spend (`POST /key/{key}/reset_spend`).

    Args:
        key: virtual key value.
        reset_to: target spend (defaults to 0.0).
    """
    return await get_client().reset_key_spend(key, reset_to)


@mcp.tool()
async def key_health() -> dict:
    """Probe the health of the caller's key (`POST /key/health`).

    Returns connectivity / permission status for the bearer key.
    """
    return await get_client().key_health()


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
