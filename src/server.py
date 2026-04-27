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
    "user": {
        "minimal": ["user_id", "user_email"],
        "standard": [
            "user_id",
            "user_email",
            "user_alias",
            "user_role",
            "teams",
            "organizations",
            "models",
            "max_budget",
            "spend",
            "blocked",
        ],
        "full": None,
    },
    "customer": {
        "minimal": ["user_id", "alias"],
        "standard": [
            "user_id",
            "alias",
            "max_budget",
            "spend",
            "blocked",
            "default_model",
            "allowed_model_region",
            "budget_duration",
        ],
        "full": None,
    },
    "organization": {
        "minimal": ["organization_id", "organization_alias"],
        "standard": [
            "organization_id",
            "organization_alias",
            "models",
            "max_budget",
            "spend",
            "budget_duration",
            "metadata",
        ],
        "full": None,
    },
    "project": {
        "minimal": ["project_id", "project_alias"],
        "standard": [
            "project_id",
            "project_alias",
            "team_id",
            "description",
            "models",
            "max_budget",
            "spend",
            "blocked",
            "tags",
        ],
        "full": None,
    },
    "user_access_group": {
        "minimal": ["access_group_id", "access_group_name"],
        "standard": [
            "access_group_id",
            "access_group_name",
            "description",
            "access_model_names",
            "access_mcp_server_ids",
            "access_agent_ids",
            "assigned_team_ids",
            "assigned_key_ids",
        ],
        "full": None,
    },
    "budget": {
        "minimal": ["budget_id"],
        "standard": [
            "budget_id",
            "max_budget",
            "soft_budget",
            "budget_duration",
            "tpm_limit",
            "rpm_limit",
            "model_max_budget",
            "budget_reset_at",
        ],
        "full": None,
    },
    "spend_record": {
        "minimal": ["request_id", "model"],
        "standard": [
            "request_id",
            "call_type",
            "api_key",
            "user",
            "team_id",
            "model",
            "total_tokens",
            "spend",
            "startTime",
            "endTime",
        ],
        "full": None,
    },
    "chat_completion": {
        "minimal": ["id", "model"],
        "standard": ["id", "object", "created", "model", "choices", "usage"],
        "full": None,
    },
    "embedding": {
        "minimal": ["model"],
        "standard": ["object", "model", "data", "usage"],
        "full": None,
    },
    "health": {
        "minimal": ["healthy_count", "unhealthy_count"],
        "standard": [
            "healthy_count",
            "unhealthy_count",
            "healthy_endpoints",
            "unhealthy_endpoints",
        ],
        "full": None,
    },
    "mcp_server": {
        "minimal": ["server_id", "alias"],
        "standard": [
            "server_id",
            "server_name",
            "alias",
            "description",
            "transport",
            "url",
            "auth_type",
            "mcp_access_groups",
            "allowed_tools",
            "approval_status",
        ],
        "full": None,
    },
    "mcp_tool": {
        "minimal": ["name"],
        "standard": ["name", "description", "server_id", "inputSchema"],
        "full": None,
    },
    "mcp_submission": {
        "minimal": ["server_id", "approval_status"],
        "standard": [
            "server_id",
            "server_name",
            "alias",
            "transport",
            "url",
            "approval_status",
            "submitted_by",
            "submitted_at",
        ],
        "full": None,
    },
    "mcp_credential": {
        # Standard never includes the raw credential value — listing endpoints
        # don't return it either, but this guards against future upstream
        # additions.
        "minimal": ["server_id"],
        "standard": ["server_id", "has_credential", "oauth_enabled", "expires_at"],
        "full": None,
    },
    "mcp_health": {
        "minimal": ["server_id", "status"],
        "standard": [
            "server_id",
            "server_name",
            "status",
            "last_checked",
            "latency_ms",
            "error_message",
        ],
        "full": None,
    },
    "mcp_toolset": {
        "minimal": ["toolset_id", "toolset_name"],
        "standard": [
            "toolset_id",
            "toolset_name",
            "description",
            "tools",
            "created_at",
            "updated_at",
        ],
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

    Note: this endpoint only resolves router-registered deployments. Passthrough
    ids returned by `list_models()` will 404 — for admin-side info on router
    deployments, use `get_model_info()` instead.

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

    Despite the PATCH verb, the upstream CredentialItem schema requires all
    three fields — this is a full replace, not a partial update. Fetch the
    current state with `get_credential(..., verbosity='full')` first if you
    only want to change one field.

    Args:
        credential_name: name of credential to update.
        credential_info: full metadata dict (will replace existing).
        credential_values: full credential values dict (will replace existing).
    """
    return await get_client().update_credential(credential_name, credential_info, credential_values)


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
        page,
        size,
        user_id,
        team_id,
        organization_id,
        key_alias,
        return_full_object,
        include_team_keys,
        sort_by,
        sort_order,
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
        key_alias,
        duration,
        models,
        max_budget,
        budget_duration,
        user_id,
        team_id,
        tpm_limit,
        rpm_limit,
        metadata,
        guardrails,
        blocked,
        extras,
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
        key_alias,
        duration,
        models,
        max_budget,
        budget_duration,
        user_id,
        team_id,
        tpm_limit,
        rpm_limit,
        metadata,
        guardrails,
        blocked,
        extras,
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
        key,
        key_alias,
        duration,
        models,
        max_budget,
        budget_duration,
        tpm_limit,
        rpm_limit,
        metadata,
        guardrails,
        blocked,
        extras,
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


# ── Internal User tools ──


@mcp.tool()
async def list_users(
    role: Optional[str] = None,
    user_ids: Optional[str] = None,
    sso_user_ids: Optional[str] = None,
    user_email: Optional[str] = None,
    team: Optional[str] = None,
    page: Optional[int] = None,
    page_size: Optional[int] = None,
    sort_by: Optional[str] = None,
    sort_order: Optional[str] = None,
    organization_ids: Optional[str] = None,
    verbosity: str = "standard",
) -> dict:
    """List internal LiteLLM users (`GET /user/list`).

    `user_ids`, `sso_user_ids`, `organization_ids` are upstream-comma-separated
    strings (not arrays). Verbosity filters the items inside the `users` array
    if present.

    Args:
        verbosity: 'minimal' / 'standard' / 'full'.
    """
    payload = await get_client().list_users(
        role,
        user_ids,
        sso_user_ids,
        user_email,
        team,
        page,
        page_size,
        sort_by,
        sort_order,
        organization_ids,
    )
    if isinstance(payload, dict) and "users" in payload:
        payload = dict(payload)
        payload["users"] = _filter_response(payload["users"], "user", verbosity)
    elif isinstance(payload, list):
        payload = _filter_response(payload, "user", verbosity)
    return payload


@mcp.tool()
async def get_user_info(user_id: Optional[str] = None, verbosity: str = "standard") -> dict:
    """Get info about an internal user (`GET /user/info`).

    Args:
        user_id: Optional user id. If omitted, returns info about the caller.
        verbosity: 'minimal' / 'standard' / 'full'.
    """
    payload = await get_client().get_user_info(user_id)
    return _filter_response(payload, "user", verbosity)


@mcp.tool()
async def create_user(
    user_email: Optional[str] = None,
    user_id: Optional[str] = None,
    user_alias: Optional[str] = None,
    user_role: Optional[str] = None,
    teams: Optional[list[str]] = None,
    organizations: Optional[list[str]] = None,
    models: Optional[list[str]] = None,
    max_budget: Optional[float] = None,
    budget_duration: Optional[str] = None,
    tpm_limit: Optional[int] = None,
    rpm_limit: Optional[int] = None,
    metadata: Optional[dict] = None,
    guardrails: Optional[list[str]] = None,
    blocked: Optional[bool] = None,
    auto_create_key: Optional[bool] = None,
    send_invite_email: Optional[bool] = None,
    extras: Optional[dict] = None,
) -> dict:
    """Create an internal user (`POST /user/new`).

    All fields are optional. Use `extras` for any NewUserRequest field not
    surfaced as a named arg (~20 less common fields like `permissions`,
    `model_max_budget`, `aliases`, `object_permission`).
    """
    return await get_client().create_user(
        user_email,
        user_id,
        user_alias,
        user_role,
        teams,
        organizations,
        models,
        max_budget,
        budget_duration,
        tpm_limit,
        rpm_limit,
        metadata,
        guardrails,
        blocked,
        auto_create_key,
        send_invite_email,
        extras,
    )


@mcp.tool()
async def update_user(
    user_id: str,
    user_email: Optional[str] = None,
    user_alias: Optional[str] = None,
    user_role: Optional[str] = None,
    password: Optional[str] = None,
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
    """Update an internal user (`POST /user/update`).

    Only `user_id` is required. Use `extras` for less-common UpdateUserRequest
    fields (e.g. `permissions`, `aliases`, `object_permission`).
    """
    return await get_client().update_user(
        user_id,
        user_email,
        user_alias,
        user_role,
        password,
        models,
        max_budget,
        budget_duration,
        tpm_limit,
        rpm_limit,
        metadata,
        guardrails,
        blocked,
        extras,
    )


@mcp.tool()
async def delete_user(user_ids: list[str]) -> dict:
    """Batch-delete internal users (`POST /user/delete`).

    Args:
        user_ids: list of user ids to delete (at least one).
    """
    return await get_client().delete_user(user_ids)


# ── Customer tools ──


@mcp.tool()
async def list_customers(verbosity: str = "standard") -> Any:
    """List end-user customers (`GET /customer/list`).

    Args:
        verbosity: 'minimal' / 'standard' / 'full'.
    """
    payload = await get_client().list_customers()
    return _filter_response(payload, "customer", verbosity)


@mcp.tool()
async def get_customer_info(end_user_id: str, verbosity: str = "standard") -> dict:
    """Get a customer by end_user_id (`GET /customer/info`).

    Args:
        end_user_id: customer's stable end-user id.
        verbosity: 'minimal' / 'standard' / 'full'.
    """
    payload = await get_client().get_customer_info(end_user_id)
    return _filter_response(payload, "customer", verbosity)


@mcp.tool()
async def create_customer(
    user_id: str,
    alias: Optional[str] = None,
    max_budget: Optional[float] = None,
    soft_budget: Optional[float] = None,
    budget_id: Optional[str] = None,
    budget_duration: Optional[str] = None,
    tpm_limit: Optional[int] = None,
    rpm_limit: Optional[int] = None,
    max_parallel_requests: Optional[int] = None,
    blocked: Optional[bool] = None,
    allowed_model_region: Optional[str] = None,
    default_model: Optional[str] = None,
    extras: Optional[dict] = None,
) -> dict:
    """Create a customer (`POST /customer/new`).

    Args:
        user_id: customer's stable end-user id (required).
        extras: any NewCustomerRequest field not surfaced as a named arg.
    """
    return await get_client().create_customer(
        user_id,
        alias,
        max_budget,
        soft_budget,
        budget_id,
        budget_duration,
        tpm_limit,
        rpm_limit,
        max_parallel_requests,
        blocked,
        allowed_model_region,
        default_model,
        extras,
    )


@mcp.tool()
async def update_customer(
    user_id: str,
    alias: Optional[str] = None,
    max_budget: Optional[float] = None,
    budget_id: Optional[str] = None,
    blocked: Optional[bool] = None,
    allowed_model_region: Optional[str] = None,
    default_model: Optional[str] = None,
    extras: Optional[dict] = None,
) -> dict:
    """Update a customer (`POST /customer/update`).

    UpdateCustomerRequest is intentionally narrower than NewCustomerRequest
    upstream (no `tpm_limit` / `rpm_limit` / `max_parallel_requests`). Pass
    less-common fields via `extras`.
    """
    return await get_client().update_customer(
        user_id, alias, max_budget, budget_id, blocked, allowed_model_region, default_model, extras
    )


@mcp.tool()
async def delete_customer(user_ids: list[str]) -> dict:
    """Batch-delete customers (`POST /customer/delete`).

    Args:
        user_ids: list of customer ids to delete.
    """
    return await get_client().delete_customer(user_ids)


@mcp.tool()
async def set_customer_blocked(user_ids: list[str], blocked: bool) -> dict:
    """Block or unblock customers.

    Routes to `POST /customer/block` when `blocked=True`, else
    `POST /customer/unblock`. Body in both cases is `{"user_ids": [...]}`.

    Args:
        user_ids: list of customer ids.
        blocked: True to block, False to unblock.
    """
    return await get_client().set_customer_blocked(user_ids, blocked)


@mcp.tool()
async def get_customer_daily_activity(
    end_user_ids: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    page: Optional[int] = None,
    page_size: Optional[int] = None,
    exclude_end_user_ids: Optional[str] = None,
) -> dict:
    """Per-customer daily activity (`GET /customer/daily/activity`).

    Args:
        start_date / end_date: ISO `YYYY-MM-DD`.
        end_user_ids / exclude_end_user_ids: comma-separated customer ids.
    """
    return await get_client().get_customer_daily_activity(
        end_user_ids,
        start_date,
        end_date,
        model,
        api_key,
        page,
        page_size,
        exclude_end_user_ids,
    )


# ── Organization tools ──


@mcp.tool()
async def list_organizations(
    org_id: Optional[str] = None,
    org_alias: Optional[str] = None,
    verbosity: str = "standard",
) -> Any:
    """List organizations (`GET /organization/list`).

    Args:
        verbosity: 'minimal' / 'standard' / 'full'.
    """
    payload = await get_client().list_organizations(org_id, org_alias)
    return _filter_response(payload, "organization", verbosity)


@mcp.tool()
async def get_organization_info(organization_id: str, verbosity: str = "standard") -> dict:
    """Get a single organization by id (`GET /organization/info`).

    Args:
        organization_id: organization id.
        verbosity: 'minimal' / 'standard' / 'full'.
    """
    payload = await get_client().get_organization_info(organization_id)
    return _filter_response(payload, "organization", verbosity)


@mcp.tool()
async def create_organization(
    organization_alias: str,
    organization_id: Optional[str] = None,
    models: Optional[list[str]] = None,
    max_budget: Optional[float] = None,
    soft_budget: Optional[float] = None,
    budget_id: Optional[str] = None,
    budget_duration: Optional[str] = None,
    tpm_limit: Optional[int] = None,
    rpm_limit: Optional[int] = None,
    max_parallel_requests: Optional[int] = None,
    metadata: Optional[dict] = None,
    extras: Optional[dict] = None,
) -> dict:
    """Create an organization (`POST /organization/new`).

    Args:
        organization_alias: human-readable name (required).
        extras: less-common NewOrganizationRequest fields.
    """
    return await get_client().create_organization(
        organization_alias,
        organization_id,
        models,
        max_budget,
        soft_budget,
        budget_id,
        budget_duration,
        tpm_limit,
        rpm_limit,
        max_parallel_requests,
        metadata,
        extras,
    )


@mcp.tool()
async def update_organization(
    organization_id: str,
    organization_alias: Optional[str] = None,
    models: Optional[list[str]] = None,
    max_budget: Optional[float] = None,
    budget_duration: Optional[str] = None,
    tpm_limit: Optional[int] = None,
    rpm_limit: Optional[int] = None,
    metadata: Optional[dict] = None,
    extras: Optional[dict] = None,
) -> dict:
    """Patch an organization (`PATCH /organization/update`).

    Upstream OpenAPI omits the request body schema for this endpoint; the
    wrapper builds an UpdateOrganization payload mirroring NewOrganizationRequest.
    Use `extras` for any field not surfaced as a named arg.
    """
    return await get_client().update_organization(
        organization_id,
        organization_alias,
        models,
        max_budget,
        budget_duration,
        tpm_limit,
        rpm_limit,
        metadata,
        extras,
    )


@mcp.tool()
async def delete_organization(organization_ids: list[str]) -> dict:
    """Batch-delete organizations (`DELETE /organization/delete`).

    DELETE with JSON body — `{"organization_ids": [...]}`.
    """
    return await get_client().delete_organization(organization_ids)


@mcp.tool()
async def add_org_member(
    organization_id: str,
    member: dict,
    max_budget_in_organization: Optional[float] = None,
) -> dict:
    """Add a member to an organization (`POST /organization/member_add`).

    Args:
        organization_id: target org id.
        member: upstream Member shape, e.g.
            `{"user_id": "u-1", "role": "internal_user"}` or
            `{"user_email": "alice@x.com", "role": "org_admin"}`.
        max_budget_in_organization: per-member budget cap inside the org.
    """
    return await get_client().add_org_member(organization_id, member, max_budget_in_organization)


@mcp.tool()
async def update_org_member(
    organization_id: str,
    user_id: Optional[str] = None,
    user_email: Optional[str] = None,
    role: Optional[str] = None,
    max_budget_in_organization: Optional[float] = None,
) -> dict:
    """Update an org member (`PATCH /organization/member_update`).

    Identify the member by `user_id` or `user_email`. Only the fields you
    pass are sent.
    """
    return await get_client().update_org_member(
        organization_id, user_id, user_email, role, max_budget_in_organization
    )


@mcp.tool()
async def delete_org_member(
    organization_id: str,
    user_id: Optional[str] = None,
    user_email: Optional[str] = None,
) -> dict:
    """Remove an org member (`DELETE /organization/member_delete`).

    Identify the member by `user_id` or `user_email`. DELETE with JSON body.
    """
    return await get_client().delete_org_member(organization_id, user_id, user_email)


@mcp.tool()
async def get_org_daily_activity(
    organization_ids: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    page: Optional[int] = None,
    page_size: Optional[int] = None,
    exclude_organization_ids: Optional[str] = None,
) -> dict:
    """Per-org daily activity (`GET /organization/daily/activity`).

    Args:
        start_date / end_date: ISO `YYYY-MM-DD`.
        organization_ids / exclude_organization_ids: comma-separated org ids.
    """
    return await get_client().get_org_daily_activity(
        organization_ids,
        start_date,
        end_date,
        model,
        api_key,
        page,
        page_size,
        exclude_organization_ids,
    )


# ── Project tools ──


@mcp.tool()
async def list_projects(verbosity: str = "standard") -> Any:
    """List projects (`GET /project/list`).

    Args:
        verbosity: 'minimal' / 'standard' / 'full'.
    """
    payload = await get_client().list_projects()
    return _filter_response(payload, "project", verbosity)


@mcp.tool()
async def get_project_info(project_id: str, verbosity: str = "standard") -> dict:
    """Get a project by id (`GET /project/info`).

    Args:
        project_id: project id.
        verbosity: 'minimal' / 'standard' / 'full'.
    """
    payload = await get_client().get_project_info(project_id)
    return _filter_response(payload, "project", verbosity)


@mcp.tool()
async def create_project(
    team_id: str,
    project_id: Optional[str] = None,
    project_alias: Optional[str] = None,
    description: Optional[str] = None,
    models: Optional[list[str]] = None,
    max_budget: Optional[float] = None,
    soft_budget: Optional[float] = None,
    budget_duration: Optional[str] = None,
    tpm_limit: Optional[int] = None,
    rpm_limit: Optional[int] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[dict] = None,
    guardrails: Optional[list[str]] = None,
    blocked: Optional[bool] = None,
    extras: Optional[dict] = None,
) -> dict:
    """Create a project (`POST /project/new`).

    Args:
        team_id: owning team (required — projects belong to teams).
        extras: less-common NewProjectRequest fields.
    """
    return await get_client().create_project(
        team_id,
        project_id,
        project_alias,
        description,
        models,
        max_budget,
        soft_budget,
        budget_duration,
        tpm_limit,
        rpm_limit,
        tags,
        metadata,
        guardrails,
        blocked,
        extras,
    )


@mcp.tool()
async def update_project(
    project_id: str,
    project_alias: Optional[str] = None,
    description: Optional[str] = None,
    models: Optional[list[str]] = None,
    max_budget: Optional[float] = None,
    budget_duration: Optional[str] = None,
    tpm_limit: Optional[int] = None,
    rpm_limit: Optional[int] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[dict] = None,
    guardrails: Optional[list[str]] = None,
    blocked: Optional[bool] = None,
    extras: Optional[dict] = None,
) -> dict:
    """Update a project (`POST /project/update`).

    Only `project_id` is required.
    """
    return await get_client().update_project(
        project_id,
        project_alias,
        description,
        models,
        max_budget,
        budget_duration,
        tpm_limit,
        rpm_limit,
        tags,
        metadata,
        guardrails,
        blocked,
        extras,
    )


@mcp.tool()
async def delete_project(project_ids: list[str]) -> dict:
    """Batch-delete projects (`DELETE /project/delete`).

    DELETE with JSON body — `{"project_ids": [...]}`.
    """
    return await get_client().delete_project(project_ids)


# ── Unified User Access Group tools ──


@mcp.tool()
async def list_user_access_groups(verbosity: str = "standard") -> Any:
    """List unified user access groups (`GET /v1/unified_access_group`).

    Distinct from `list_model_access_groups` — unified access groups gate
    users/teams against models, MCP servers, and agents in one shape.

    Args:
        verbosity: 'minimal' / 'standard' / 'full'.
    """
    payload = await get_client().list_user_access_groups()
    return _filter_response(payload, "user_access_group", verbosity)


@mcp.tool()
async def get_user_access_group(access_group_id: str, verbosity: str = "standard") -> dict:
    """Get a unified user access group by id (`GET /v1/unified_access_group/{id}`).

    Args:
        access_group_id: the group's id.
        verbosity: 'minimal' / 'standard' / 'full'.
    """
    payload = await get_client().get_user_access_group(access_group_id)
    return _filter_response(payload, "user_access_group", verbosity)


@mcp.tool()
async def create_user_access_group(
    access_group_name: str,
    description: Optional[str] = None,
    access_model_names: Optional[list[str]] = None,
    access_mcp_server_ids: Optional[list[str]] = None,
    access_agent_ids: Optional[list[str]] = None,
    assigned_team_ids: Optional[list[str]] = None,
    assigned_key_ids: Optional[list[str]] = None,
) -> dict:
    """Create a unified user access group (`POST /v1/unified_access_group`).

    Args:
        access_group_name: human-readable name (required).
        access_*: lists of resources this group grants access to (models / MCP
            servers / agents).
        assigned_*: lists of teams / keys this group is assigned to.
    """
    return await get_client().create_user_access_group(
        access_group_name,
        description,
        access_model_names,
        access_mcp_server_ids,
        access_agent_ids,
        assigned_team_ids,
        assigned_key_ids,
    )


@mcp.tool()
async def update_user_access_group(
    access_group_id: str,
    access_group_name: Optional[str] = None,
    description: Optional[str] = None,
    access_model_names: Optional[list[str]] = None,
    access_mcp_server_ids: Optional[list[str]] = None,
    access_agent_ids: Optional[list[str]] = None,
    assigned_team_ids: Optional[list[str]] = None,
    assigned_key_ids: Optional[list[str]] = None,
) -> dict:
    """Update a unified user access group (`PUT /v1/unified_access_group/{id}`).

    All fields are optional; only the ones passed are sent. Note: this is
    `PUT`, not `PATCH`, but the upstream merges sparsely.
    """
    return await get_client().update_user_access_group(
        access_group_id,
        access_group_name,
        description,
        access_model_names,
        access_mcp_server_ids,
        access_agent_ids,
        assigned_team_ids,
        assigned_key_ids,
    )


@mcp.tool()
async def delete_user_access_group(access_group_id: str) -> dict:
    """Delete a unified user access group (`DELETE /v1/unified_access_group/{id}`)."""
    return await get_client().delete_user_access_group(access_group_id)


# ── Budget tools ──


@mcp.tool()
async def list_budgets(verbosity: str = "standard") -> Any:
    """List configured budgets (`GET /budget/list`).

    Args:
        verbosity: 'minimal' / 'standard' / 'full'.
    """
    payload = await get_client().list_budgets()
    return _filter_response(payload, "budget", verbosity)


@mcp.tool()
async def get_budget_info(budgets: list[str], verbosity: str = "standard") -> Any:
    """Get info about one or more budgets (`POST /budget/info`).

    Despite the GET-shaped name, the upstream uses POST with a body
    `{"budgets": [...]}` — this is a batch lookup, not a per-id GET.

    Args:
        budgets: list of budget ids/names to fetch.
        verbosity: 'minimal' / 'standard' / 'full'.
    """
    payload = await get_client().get_budget_info(budgets)
    return _filter_response(payload, "budget", verbosity)


@mcp.tool()
async def create_budget(
    budget_id: Optional[str] = None,
    max_budget: Optional[float] = None,
    soft_budget: Optional[float] = None,
    max_parallel_requests: Optional[int] = None,
    tpm_limit: Optional[int] = None,
    rpm_limit: Optional[int] = None,
    budget_duration: Optional[str] = None,
    model_max_budget: Optional[dict] = None,
    budget_reset_at: Optional[str] = None,
    extras: Optional[dict] = None,
) -> dict:
    """Create a budget (`POST /budget/new`).

    All fields optional per upstream `BudgetNewRequest`. Use `extras` for any
    field not surfaced as a named arg.

    Args:
        budget_id: optional explicit id (upstream auto-generates if omitted).
        max_budget: hard spend cap (USD).
        soft_budget: warning threshold below the hard cap.
        budget_duration: reset window (e.g. `30d`).
        model_max_budget: per-model caps, e.g. `{"gpt-4o": {"budget_limit": 10.0}}`.
    """
    return await get_client().create_budget(
        budget_id,
        max_budget,
        soft_budget,
        max_parallel_requests,
        tpm_limit,
        rpm_limit,
        budget_duration,
        model_max_budget,
        budget_reset_at,
        extras,
    )


@mcp.tool()
async def update_budget(
    budget_id: str,
    max_budget: Optional[float] = None,
    soft_budget: Optional[float] = None,
    max_parallel_requests: Optional[int] = None,
    tpm_limit: Optional[int] = None,
    rpm_limit: Optional[int] = None,
    budget_duration: Optional[str] = None,
    model_max_budget: Optional[dict] = None,
    budget_reset_at: Optional[str] = None,
    extras: Optional[dict] = None,
) -> dict:
    """Update a budget (`POST /budget/update`).

    Same `BudgetNewRequest` shape as create — `budget_id` identifies the row.
    """
    return await get_client().update_budget(
        budget_id,
        max_budget,
        soft_budget,
        max_parallel_requests,
        tpm_limit,
        rpm_limit,
        budget_duration,
        model_max_budget,
        budget_reset_at,
        extras,
    )


@mcp.tool()
async def delete_budget(budget_id: str) -> dict:
    """Delete a budget (`POST /budget/delete`).

    Body is `{"id": <budget_id>}` per upstream `BudgetDeleteRequest`.
    """
    return await get_client().delete_budget(budget_id)


@mcp.tool()
async def get_budget_settings(budget_id: str) -> dict:
    """Get effective budget settings (`GET /budget/settings`).

    Args:
        budget_id: required query param identifying the budget row.
    """
    return await get_client().get_budget_settings(budget_id)


# ── Spend tools ──


@mcp.tool()
async def get_global_spend_report(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    group_by: Optional[str] = None,
    api_key: Optional[str] = None,
    internal_user_id: Optional[str] = None,
    team_id: Optional[str] = None,
    customer_id: Optional[str] = None,
) -> Any:
    """Aggregated global spend report (`GET /global/spend/report`).

    All filters are optional query params.

    Args:
        start_date / end_date: ISO `YYYY-MM-DD`.
        group_by: upstream-defined dimension (`team`, `customer`, `api_key`, `model`).
    """
    return await get_client().get_global_spend_report(
        start_date, end_date, group_by, api_key, internal_user_id, team_id, customer_id
    )


@mcp.tool()
async def list_spend_logs(
    api_key: Optional[str] = None,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    summarize: Optional[bool] = None,
    verbosity: str = "standard",
) -> Any:
    """List per-request spend logs (`GET /spend/logs`).

    Args:
        summarize: True to return aggregated rows.
        verbosity: 'minimal' / 'standard' / 'full' — applied to the spend_record items.
    """
    payload = await get_client().list_spend_logs(
        api_key, user_id, request_id, start_date, end_date, summarize
    )
    return _filter_response(payload, "spend_record", verbosity)


@mcp.tool()
async def list_spend_tags(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Any:
    """List distinct spend tags within a date window (`GET /spend/tags`)."""
    return await get_client().list_spend_tags(start_date, end_date)


@mcp.tool()
async def calculate_spend(
    model: Optional[str] = None,
    messages: Optional[list[dict]] = None,
    completion_response: Optional[dict] = None,
) -> dict:
    """Estimate spend for a request (`POST /spend/calculate`).

    Either pass `model + messages` for a prospective estimate, or
    `model + completion_response` for a retrospective re-cost.
    """
    return await get_client().calculate_spend(model, messages, completion_response)


@mcp.tool()
async def get_user_daily_activity(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    user_id: Optional[str] = None,
    page: Optional[int] = None,
    page_size: Optional[int] = None,
    timezone: Optional[str] = None,
) -> dict:
    """Per-user daily activity (`GET /user/daily/activity`).

    Args:
        start_date / end_date: ISO `YYYY-MM-DD` recommended.
        timezone: IANA tz string (e.g. `Europe/Paris`) for daily-bucket alignment.
    """
    return await get_client().get_user_daily_activity(
        start_date, end_date, model, api_key, user_id, page, page_size, timezone
    )


# ── Execution tools ──


@mcp.tool()
async def chat_completion(
    model: str,
    messages: list[dict],
    body: Optional[dict] = None,
    verbosity: str = "standard",
) -> dict:
    """Chat completion (`POST /v1/chat/completions`).

    Synchronous only — `stream` is stripped from the body if passed. Use `body`
    for any extra OpenAI / LiteLLM fields (temperature, max_tokens, tools, etc.).

    Args:
        model: model_name alias (e.g. `gpt-4o`, `claude-opus-4-7`).
        messages: OpenAI-shaped message list.
        body: optional dict of additional request fields.
        verbosity: 'minimal' / 'standard' / 'full'.
    """
    payload = await get_client().chat_completion(model, messages, body)
    return _filter_response(payload, "chat_completion", verbosity)


@mcp.tool()
async def completion(
    model: str,
    prompt: str,
    body: Optional[dict] = None,
    verbosity: str = "standard",
) -> dict:
    """Legacy text completion (`POST /v1/completions`).

    Synchronous only — `stream` is stripped from the body if passed. The
    upstream OpenAPI spec only documents a `model` query param for this
    endpoint, but the implementation accepts the full OpenAI `/v1/completions`
    body — we send `model` in both query and body.

    Args:
        model: model_name alias.
        prompt: text prompt.
        body: optional dict of additional fields (max_tokens, temperature, etc.).
        verbosity: 'minimal' / 'standard' / 'full'.
    """
    payload = await get_client().completion(model, prompt, body)
    return _filter_response(payload, "chat_completion", verbosity)


@mcp.tool()
async def embed(
    model: str,
    input: list[str],
    body: Optional[dict] = None,
    verbosity: str = "standard",
) -> dict:
    """Generate embeddings (`POST /v1/embeddings`).

    Args:
        model: embedding model_name alias.
        input: list of input strings (wrap a single string as `[text]`).
        body: optional dict of additional fields (`dimensions`, `encoding_format`, ...).
        verbosity: 'minimal' / 'standard' / 'full'.
    """
    payload = await get_client().embed(model, input, body)
    return _filter_response(payload, "embedding", verbosity)


# ── Health tools ──


@mcp.tool()
async def check_health(
    model: Optional[str] = None,
    model_id: Optional[str] = None,
    verbosity: str = "standard",
) -> dict:
    """Run upstream health checks against deployments (`GET /health`).

    Args:
        model: optional model_name alias to narrow the probe.
        model_id: optional litellm internal id to narrow the probe.
        verbosity: 'minimal' / 'standard' / 'full'.
    """
    payload = await get_client().check_health(model, model_id)
    return _filter_response(payload, "health", verbosity)


@mcp.tool()
async def check_health_backlog() -> dict:
    """Get health-check queue backlog (`GET /health/backlog`)."""
    return await get_client().check_health_backlog()


@mcp.tool()
async def get_health_history(
    model: Optional[str] = None,
    status_filter: Optional[str] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
) -> dict:
    """Historical health check results (`GET /health/history`).

    Args:
        status_filter: upstream status (`healthy` / `unhealthy`).
    """
    return await get_client().get_health_history(model, status_filter, limit, offset)


@mcp.tool()
async def get_health_latest(verbosity: str = "standard") -> dict:
    """Latest health check snapshot (`GET /health/latest`).

    Args:
        verbosity: 'minimal' / 'standard' / 'full'.
    """
    payload = await get_client().get_health_latest()
    return _filter_response(payload, "health", verbosity)


@mcp.tool()
async def test_model_connection(
    litellm_params: dict,
    mode: Optional[str] = None,
    model_info: Optional[dict] = None,
) -> dict:
    """Test a candidate deployment connection (`POST /health/test_connection`).

    Useful for validating provider credentials before calling `add_model`.

    Args:
        litellm_params: provider routing dict (same shape as `add_model`).
        mode: probe mode (`chat`, `embedding`, `completion`).
        model_info: optional deployment metadata.
    """
    return await get_client().test_model_connection(litellm_params, mode, model_info)


# ── MCP Gateway: server CRUD ──


@mcp.tool()
async def list_mcp_servers(team_id: Optional[str] = None, verbosity: str = "standard") -> Any:
    """List registered upstream MCP servers (`GET /v1/mcp/server`).

    Args:
        team_id: optional team filter.
        verbosity: 'minimal' / 'standard' / 'full'.
    """
    payload = await get_client().list_mcp_servers(team_id)
    return _filter_response(payload, "mcp_server", verbosity)


@mcp.tool()
async def get_mcp_server(server_id: str, verbosity: str = "standard") -> dict:
    """Get a single MCP server by id (`GET /v1/mcp/server/{server_id}`).

    Args:
        server_id: upstream server id (UUID-shaped).
        verbosity: 'minimal' / 'standard' / 'full'.
    """
    payload = await get_client().get_mcp_server(server_id)
    return _filter_response(payload, "mcp_server", verbosity)


@mcp.tool()
async def add_mcp_server(
    transport: str,
    server_name: Optional[str] = None,
    alias: Optional[str] = None,
    description: Optional[str] = None,
    url: Optional[str] = None,
    auth_type: Optional[str] = None,
    spec_path: Optional[str] = None,
    mcp_info: Optional[dict] = None,
    mcp_access_groups: Optional[list[str]] = None,
    allowed_tools: Optional[list[str]] = None,
    credentials: Optional[dict] = None,
    server_id: Optional[str] = None,
    extras: Optional[dict] = None,
) -> dict:
    """Admin-register an upstream MCP server (`POST /v1/mcp/server`).

    Distinct from `register_mcp_server`: admin path creates servers in
    approved state directly, no review workflow.

    Args:
        transport: required (`http`, `sse`, `stdio`). Gateway brokers HTTP/SSE.
        url: HTTP/SSE endpoint URL (e.g. `https://mcp.context7.com/mcp`).
        auth_type: `none`, `api_key`, `oauth2`.
        mcp_access_groups: gate this server behind named access groups.
        allowed_tools: optional whitelist of tool names from this upstream.
        extras: any other NewMCPServerRequest field (static_headers, oauth2_flow,
            tool_name_to_display_name, allow_all_keys, etc.).
    """
    return await get_client().add_mcp_server(
        transport,
        server_name,
        alias,
        description,
        url,
        auth_type,
        spec_path,
        mcp_info,
        mcp_access_groups,
        allowed_tools,
        credentials,
        server_id,
        extras,
    )


@mcp.tool()
async def update_mcp_server(
    server_id: str,
    server_name: Optional[str] = None,
    alias: Optional[str] = None,
    description: Optional[str] = None,
    transport: Optional[str] = None,
    url: Optional[str] = None,
    auth_type: Optional[str] = None,
    spec_path: Optional[str] = None,
    mcp_info: Optional[dict] = None,
    mcp_access_groups: Optional[list[str]] = None,
    allowed_tools: Optional[list[str]] = None,
    credentials: Optional[dict] = None,
    extras: Optional[dict] = None,
) -> dict:
    """Admin-update an MCP server (`PUT /v1/mcp/server`).

    Note: `server_id` goes in the body, not the path — UpdateMCPServerRequest
    treats it as the row identifier.
    """
    return await get_client().update_mcp_server(
        server_id,
        server_name,
        alias,
        description,
        transport,
        url,
        auth_type,
        spec_path,
        mcp_info,
        mcp_access_groups,
        allowed_tools,
        credentials,
        extras,
    )


@mcp.tool()
async def delete_mcp_server(server_id: str) -> dict:
    """Delete an MCP server (`DELETE /v1/mcp/server/{server_id}`)."""
    return await get_client().delete_mcp_server(server_id)


@mcp.tool()
async def register_mcp_server(
    transport: str,
    server_name: Optional[str] = None,
    alias: Optional[str] = None,
    description: Optional[str] = None,
    url: Optional[str] = None,
    auth_type: Optional[str] = None,
    spec_path: Optional[str] = None,
    mcp_info: Optional[dict] = None,
    mcp_access_groups: Optional[list[str]] = None,
    allowed_tools: Optional[list[str]] = None,
    credentials: Optional[dict] = None,
    extras: Optional[dict] = None,
) -> dict:
    """Self-register an MCP server (`POST /v1/mcp/server/register`).

    User-facing submission path: creates a server in `pending` approval state.
    An admin then approves / rejects via the submissions workflow tools.
    Use `add_mcp_server` for the direct admin-register path.
    """
    return await get_client().register_mcp_server(
        transport,
        server_name,
        alias,
        description,
        url,
        auth_type,
        spec_path,
        mcp_info,
        mcp_access_groups,
        allowed_tools,
        credentials,
        extras,
    )


# ── MCP Gateway: submissions workflow ──


@mcp.tool()
async def list_mcp_server_submissions(verbosity: str = "standard") -> Any:
    """List pending MCP server submissions (`GET /v1/mcp/server/submissions`).

    Args:
        verbosity: 'minimal' / 'standard' / 'full' (filters to mcp_submission shape).
    """
    payload = await get_client().list_mcp_server_submissions()
    return _filter_response(payload, "mcp_submission", verbosity)


@mcp.tool()
async def approve_mcp_server_submission(server_id: str) -> dict:
    """Approve a pending MCP submission (`PUT /v1/mcp/server/{id}/approve`).

    Promotes the submission to approved/active state.
    """
    return await get_client().approve_mcp_server_submission(server_id)


@mcp.tool()
async def reject_mcp_server_submission(server_id: str, review_notes: Optional[str] = None) -> dict:
    """Reject a pending MCP submission (`PUT /v1/mcp/server/{id}/reject`).

    Args:
        server_id: pending submission id.
        review_notes: optional rejection reason captured upstream.
    """
    return await get_client().reject_mcp_server_submission(server_id, review_notes)


# ── MCP Gateway: health ──


@mcp.tool()
async def check_mcp_servers_health(
    server_ids: Optional[str] = None, verbosity: str = "standard"
) -> dict:
    """Probe upstream MCP server health (`GET /v1/mcp/server/health`).

    Args:
        server_ids: optional upstream-comma-separated string of server ids.
        verbosity: 'minimal' / 'standard' / 'full' (filters to mcp_health shape).
    """
    payload = await get_client().check_mcp_servers_health(server_ids)
    return _filter_response(payload, "mcp_health", verbosity)


# ── MCP Gateway: tool discovery & invocation ──


@mcp.tool()
async def list_mcp_tools(verbosity: str = "standard") -> Any:
    """List tools across all registered MCP servers (`GET /v1/mcp/tools`).

    Args:
        verbosity: 'minimal' / 'standard' / 'full' (filters to mcp_tool shape).
    """
    payload = await get_client().list_mcp_tools()
    return _filter_response(payload, "mcp_tool", verbosity)


@mcp.tool()
async def list_mcp_tools_rest(server_id: Optional[str] = None, verbosity: str = "standard") -> Any:
    """List tools (REST shape) for a registered MCP server (`GET /mcp-rest/tools/list`).

    Lighter-weight per-server view than `list_mcp_tools`. Upstream returns
    `{"tools": [...]}` — items are filtered by verbosity.
    """
    payload = await get_client().list_mcp_tools_rest(server_id)
    if isinstance(payload, dict) and "tools" in payload:
        payload = dict(payload)
        payload["tools"] = _filter_response(payload["tools"], "mcp_tool", verbosity)
        return payload
    return _filter_response(payload, "mcp_tool", verbosity)


@mcp.tool()
async def call_mcp_tool(
    server_id: str,
    name: str,
    arguments: Optional[dict] = None,
) -> Any:
    """Invoke a tool on a registered MCP server (`POST /mcp-rest/tools/call`).

    Returns the provider-shaped tool response unmodified (typically the MCP
    `tools/call` envelope `{"content": [...], "isError": bool}`). No verbosity
    filtering — callers usually want the full provider output.

    Args:
        server_id: UUID of a registered MCP server.
        name: tool name as listed by `list_mcp_tools_rest(server_id)`.
        arguments: tool arguments dict.
    """
    return await get_client().call_mcp_tool(server_id, name, arguments)


@mcp.tool()
async def test_mcp_connection(
    transport: str,
    url: Optional[str] = None,
    auth_type: Optional[str] = None,
    spec_path: Optional[str] = None,
    credentials: Optional[dict] = None,
    extras: Optional[dict] = None,
) -> dict:
    """Test an MCP server connection without registering it (`POST /mcp-rest/test/connection`).

    Body is a NewMCPServerRequest — pass the same fields you'd give to
    `add_mcp_server`. Useful for validating URL + auth before persisting.
    """
    return await get_client().test_mcp_connection(
        transport, url, auth_type, spec_path, credentials, extras
    )


# ── MCP Gateway: discovery / registry / hub ──


@mcp.tool()
async def discover_mcp_servers(query: Optional[str] = None, category: Optional[str] = None) -> Any:
    """Search the public MCP server registry (`GET /v1/mcp/discover`).

    Args:
        query: free-text search.
        category: registry-defined category filter.
    """
    return await get_client().discover_mcp_servers(query, category)


@mcp.tool()
async def get_mcp_openapi_registry() -> Any:
    """Get the OpenAPI registry of MCP servers (`GET /v1/mcp/openapi-registry`).

    Returns descriptors useful for tooling that prefers OpenAPI over MCP.
    """
    return await get_client().get_mcp_openapi_registry()


@mcp.tool()
async def get_mcp_registry() -> Any:
    """Get the MCP servers registry JSON (`GET /v1/mcp/registry.json`)."""
    return await get_client().get_mcp_registry()


@mcp.tool()
async def list_mcp_access_groups() -> Any:
    """List MCP access groups (`GET /v1/mcp/access_groups`).

    Distinct from the model-access-groups family and the unified user
    access groups — these gate keys/teams against MCP servers specifically.
    """
    return await get_client().list_mcp_access_groups()


@mcp.tool()
async def make_mcp_servers_public(mcp_server_ids: list[str]) -> dict:
    """Publish MCP servers to the public hub (`POST /v1/mcp/make_public`).

    Replaces (not appends) the published set with `mcp_server_ids`.
    """
    return await get_client().make_mcp_servers_public(mcp_server_ids)


@mcp.tool()
async def get_public_mcp_hub() -> Any:
    """Get the public MCP hub catalog (`GET /public/mcp_hub`)."""
    return await get_client().get_public_mcp_hub()


# ── MCP Gateway: user credentials ──


@mcp.tool()
async def list_mcp_user_credentials(verbosity: str = "standard") -> Any:
    """List the caller's stored MCP user credentials (`GET /v1/mcp/user-credentials`).

    Returns one entry per server the caller has credentials for. Credential
    values are not included.

    Args:
        verbosity: 'minimal' / 'standard' / 'full' (filters to mcp_credential shape).
    """
    payload = await get_client().list_mcp_user_credentials()
    return _filter_response(payload, "mcp_credential", verbosity)


@mcp.tool()
async def set_mcp_user_credential(
    server_id: str,
    oauth: bool = False,
    credential: Optional[str] = None,
    save: Optional[bool] = None,
    access_token: Optional[str] = None,
    refresh_token: Optional[str] = None,
    expires_in: Optional[int] = None,
    scopes: Optional[list[str]] = None,
) -> dict:
    """Set the caller's user credential for an MCP server.

    The `oauth` boolean discriminator routes between the two upstream
    endpoints, compressing what would otherwise be two near-identical tools:
    - `oauth=False` → `POST /v1/mcp/server/{id}/user-credential` with body
      `{"credential": ..., "save": ...}` (MCPUserCredentialRequest).
    - `oauth=True` → `POST /v1/mcp/server/{id}/oauth-user-credential` with body
      `{"access_token": ..., "refresh_token": ..., "expires_in": ..., "scopes": [...]}`
      (MCPOAuthUserCredentialRequest).

    Args:
        server_id: target MCP server id.
        oauth: route discriminator (default False).
        credential / save: non-oauth fields.
        access_token / refresh_token / expires_in / scopes: oauth fields.
    """
    return await get_client().set_mcp_user_credential(
        server_id,
        oauth,
        credential,
        save,
        access_token,
        refresh_token,
        expires_in,
        scopes,
    )


@mcp.tool()
async def delete_mcp_user_credential(server_id: str, oauth: bool = False) -> dict:
    """Delete the caller's user credential for an MCP server.

    The `oauth` boolean routes between
    `DELETE /v1/mcp/server/{id}/oauth-user-credential` and
    `DELETE /v1/mcp/server/{id}/user-credential`.
    """
    return await get_client().delete_mcp_user_credential(server_id, oauth)


@mcp.tool()
async def get_mcp_oauth_user_credential_status(server_id: str) -> dict:
    """Get OAuth credential status for an MCP server (`GET /v1/mcp/server/{id}/oauth-user-credential/status`).

    Returns whether the caller has a valid OAuth credential and (optionally)
    when it expires.
    """
    return await get_client().get_mcp_oauth_user_credential_status(server_id)


# ── MCP Gateway: utility ──


@mcp.tool()
async def get_mcp_client_ip() -> dict:
    """Get the caller's resolved client IP (`GET /v1/mcp/network/client-ip`).

    Useful for diagnosing IP-allowlist / proxy-header issues without bouncing
    through an external service.
    """
    return await get_client().get_mcp_client_ip()


# ── MCP Toolset operations ──


@mcp.tool()
async def list_mcp_toolsets(verbosity: str = "standard") -> Any:
    """List defined MCP toolsets (`GET /v1/mcp/toolset`).

    Toolsets are named bundles of tools sourced from one or more registered
    MCP servers. Once defined, the proxy exposes each toolset as a brokered
    MCP endpoint at `/toolset/{name}/mcp` (transport-level — not wrapped here).

    Args:
        verbosity: 'minimal' / 'standard' / 'full'.
    """
    payload = await get_client().list_mcp_toolsets()
    return _filter_response(payload, "mcp_toolset", verbosity)


@mcp.tool()
async def get_mcp_toolset(toolset_id: str, verbosity: str = "standard") -> dict:
    """Get a single MCP toolset by id (`GET /v1/mcp/toolset/{toolset_id}`).

    Args:
        toolset_id: upstream toolset id (UUID-shaped).
        verbosity: 'minimal' / 'standard' / 'full'.
    """
    payload = await get_client().get_mcp_toolset(toolset_id)
    return _filter_response(payload, "mcp_toolset", verbosity)


@mcp.tool()
async def add_mcp_toolset(
    toolset_name: str,
    tools: list[str],
    description: Optional[str] = None,
) -> dict:
    """Create an MCP toolset (`POST /v1/mcp/toolset`).

    Args:
        toolset_name: human-readable name (must satisfy upstream tool-name
            constraints: A–Z, a–z, 0–9, underscore, dash, dot — no spaces).
        tools: list of tool identifiers, typically `<server_alias>/<tool_name>`
            strings sourced from `list_mcp_tools()`.
        description: optional human-readable description.
    """
    return await get_client().add_mcp_toolset(toolset_name, tools, description)


@mcp.tool()
async def update_mcp_toolset(
    toolset_id: str,
    toolset_name: Optional[str] = None,
    description: Optional[str] = None,
    tools: Optional[list[str]] = None,
) -> dict:
    """Update an MCP toolset (`PUT /v1/mcp/toolset`).

    Note: like `update_mcp_server`, the id goes in the body (per
    `UpdateMCPToolsetRequest`), not the path. Only the fields you pass are
    sent.
    """
    return await get_client().update_mcp_toolset(toolset_id, toolset_name, description, tools)


@mcp.tool()
async def delete_mcp_toolset(toolset_id: str) -> dict:
    """Delete an MCP toolset (`DELETE /v1/mcp/toolset/{toolset_id}`)."""
    return await get_client().delete_mcp_toolset(toolset_id)


# ── Provider passthrough ──


@mcp.tool()
async def passthrough(
    provider: str,
    endpoint: str,
    method: str = "GET",
    body: Optional[dict] = None,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
) -> Any:
    """Proxy a request to a provider's native API via LiteLLM (`<METHOD> /<provider>/<endpoint>`).

    LiteLLM exposes 15+ providers' native APIs at `/<provider>/{endpoint:path}`
    with all 5 HTTP methods (GET, POST, PUT, DELETE, PATCH). One generic tool
    covers ~85 Swagger pass-through ops — wrapping each provider × method
    individually would be busywork.

    Known providers: `anthropic`, `openai`, `vertex_ai`, `vertex_ai/discovery`,
    `gemini` (Google AI Studio), `cohere`, `vllm`, `mistral`, `milvus`,
    `bedrock`, `assemblyai`, `eu.assemblyai`, `azure`, `azure_ai`, `cursor`,
    `langfuse`. The list is fixed by what the running LiteLLM build compiles in.

    Returns the upstream payload unmodified (no verbosity filtering — provider
    responses are too varied to filter, and the value of pass-through is
    byte-faithful access). Streaming responses are not supported in this slice.

    Args:
        provider: provider identifier — the path prefix after `/`.
            Forward slashes inside the value are preserved
            (e.g. `vertex_ai/discovery`).
        endpoint: the rest of the upstream path, e.g. `v1/models`,
            `v1/messages`, `v1/chat/completions`.
        method: HTTP method, default `GET`. Case-insensitive.
        body: optional JSON body for POST/PUT/PATCH.
        params: optional query parameters.
        headers: optional headers to forward (e.g. `{"anthropic-version": "..."}`).
            Do not override `Authorization` — it's set by the client.
    """
    return await get_client().passthrough(provider, endpoint, method, body, params, headers)


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
