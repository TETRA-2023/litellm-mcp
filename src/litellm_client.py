"""LiteLLM proxy HTTP client."""

import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


class LiteLLMAPIError(Exception):
    """Raised when the LiteLLM proxy returns an error."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class LiteLLMClient:
    """Async client for LiteLLM proxy admin and OpenAI-compatible endpoints.

    - Auth: `Authorization: Bearer <key>` (master or virtual key).
    - Retry: simple bounded retry on transient HTTP errors.
    - Error mapping: 401/403/404 raise typed errors with status code preserved.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 30.0,
        max_retries: int = 2,
    ):
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        transport = httpx.AsyncHTTPTransport(retries=0)
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            transport=transport,
        )

    async def close(self) -> None:
        await self._client.aclose()

    def _check_response(self, response: httpx.Response) -> None:
        if response.status_code == 401:
            raise LiteLLMAPIError("Unauthorized: invalid or missing API key", 401)
        if response.status_code == 403:
            raise LiteLLMAPIError("Forbidden: insufficient permissions", 403)
        if response.status_code == 404:
            raise LiteLLMAPIError("Not found", 404)
        if response.status_code >= 400:
            try:
                payload = response.json()
                detail = payload.get("error", payload)
                message = (
                    detail.get("message") if isinstance(detail, dict) else str(detail)
                ) or response.text
            except ValueError:
                message = response.text or f"HTTP {response.status_code}"
            raise LiteLLMAPIError(message, response.status_code)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict] = None,
        json: Optional[dict] = None,
        headers: Optional[dict] = None,
    ) -> Any:
        attempt = 0
        last_exc: Optional[Exception] = None
        while attempt <= self.max_retries:
            try:
                response = await self._client.request(
                    method, path, params=params, json=json, headers=headers
                )
            except httpx.TransportError as exc:
                last_exc = exc
                logger.warning(
                    "LiteLLM transport error on %s %s (attempt %d): %s",
                    method,
                    path,
                    attempt + 1,
                    exc,
                )
                attempt += 1
                continue

            if response.status_code in _RETRYABLE_STATUS and attempt < self.max_retries:
                logger.warning(
                    "LiteLLM %s %s -> %d (retrying)",
                    method,
                    path,
                    response.status_code,
                )
                attempt += 1
                continue

            self._check_response(response)
            if not response.content:
                return None
            try:
                return response.json()
            except ValueError:
                return response.text

        raise LiteLLMAPIError(
            f"LiteLLM {method} {path} failed after {self.max_retries + 1} attempts: {last_exc}"
        )

    # ── Model operations ──

    async def list_models(self) -> list[dict]:
        """List models exposed by the LiteLLM proxy.

        Calls the OpenAI-compatible `GET /v1/models` endpoint, which returns
        `{"data": [{"id", "object", "created", "owned_by"}, ...], "object": "list"}`.
        Returns the unwrapped `data` array.
        """
        payload = await self._request("GET", "/v1/models")
        if isinstance(payload, dict) and "data" in payload:
            return payload["data"]
        if isinstance(payload, list):
            return payload
        return []

    async def get_model(self, model_id: str) -> dict:
        """Get a single model entry from the OpenAI-compatible models endpoint.

        Calls `GET /v1/models/{model_id}`.

        Note: LiteLLM only resolves `model_id` against router-registered
        deployments here, *not* the union returned by `GET /v1/models`. Calling
        with a passthrough id from `list_models()` will 404 — that's upstream
        behavior, not a wrapper bug. Use `get_model_info()` for the admin-side
        view that includes router deployments by their internal litellm id.
        """
        return await self._request("GET", f"/v1/models/{model_id}")

    async def get_model_info(self, litellm_model_id: Optional[str] = None) -> dict:
        """Get admin-side model info (all deployments, or one by litellm internal id).

        Calls `GET /model/info` with optional `litellm_model_id` query param.
        Returns the upstream payload as-is (typically `{"data": [...]}`).
        """
        params = {"litellm_model_id": litellm_model_id} if litellm_model_id else None
        return await self._request("GET", "/model/info", params=params)

    async def add_model(
        self,
        model_name: str,
        litellm_params: dict,
        model_info: dict,
    ) -> dict:
        """Register a new model deployment (`POST /model/new`).

        The Deployment schema requires three fields:
        - `model_name`: alias clients call (e.g. `gpt-4o`).
        - `litellm_params`: provider routing (`{"model": "openai/gpt-4o", "api_key": ...}`).
        - `model_info`: deployment metadata (`{"id": "...", "db_model": false, ...}`).
        """
        body = {
            "model_name": model_name,
            "litellm_params": litellm_params,
            "model_info": model_info,
        }
        return await self._request("POST", "/model/new", json=body)

    async def update_model(
        self,
        model_id: str,
        model_name: Optional[str] = None,
        litellm_params: Optional[dict] = None,
        model_info: Optional[dict] = None,
    ) -> dict:
        """Patch an existing model deployment (`PATCH /model/{model_id}/update`).

        All body fields are optional; only those passed will be sent.
        """
        body: dict[str, Any] = {}
        if model_name is not None:
            body["model_name"] = model_name
        if litellm_params is not None:
            body["litellm_params"] = litellm_params
        if model_info is not None:
            body["model_info"] = model_info
        return await self._request("PATCH", f"/model/{model_id}/update", json=body)

    async def delete_model(self, model_id: str) -> dict:
        """Delete a model deployment (`POST /model/delete`).

        Note: the deployment id goes in the body as `{"id": ...}`, not the path.
        """
        return await self._request("POST", "/model/delete", json={"id": model_id})

    # ── Model Hub operations ──

    async def list_public_models(self) -> Any:
        """List models published to the public Model Hub (`GET /public/model_hub`)."""
        return await self._request("GET", "/public/model_hub")

    async def get_public_hub_info(self) -> dict:
        """Get Model Hub metadata: title, description, useful links (`GET /public/model_hub/info`)."""
        return await self._request("GET", "/public/model_hub/info")

    async def get_model_cost_map(self) -> dict:
        """Get LiteLLM's static model cost / capability map (`GET /public/litellm_model_cost_map`)."""
        return await self._request("GET", "/public/litellm_model_cost_map")

    async def make_model_group_public(self, model_groups: list[str]) -> dict:
        """Publish model groups to the public Model Hub (`POST /model_group/make_public`).

        Replaces (not appends) the current set of published groups with `model_groups`.
        """
        return await self._request(
            "POST", "/model_group/make_public", json={"model_groups": model_groups}
        )

    async def update_model_hub_links(self, useful_links: dict) -> dict:
        """Update the Model Hub useful-links section (`POST /model_hub/update_useful_links`).

        `useful_links` is a free-form mapping (e.g. `{"Documentation": "https://..."}`).
        """
        return await self._request(
            "POST", "/model_hub/update_useful_links", json={"useful_links": useful_links}
        )

    # ── Model Access Group operations ──

    async def list_model_access_groups(self) -> Any:
        """List all model access groups (`GET /access_group/list`)."""
        return await self._request("GET", "/access_group/list")

    async def get_model_access_group(self, access_group: str) -> dict:
        """Get a single model access group by name (`GET /access_group/{access_group}/info`)."""
        return await self._request("GET", f"/access_group/{access_group}/info")

    async def create_model_access_group(
        self,
        access_group: str,
        model_names: Optional[list[str]] = None,
        model_ids: Optional[list[str]] = None,
    ) -> dict:
        """Create a new model access group (`POST /access_group/new`).

        Membership can be specified by `model_names` (model_name aliases) and/or
        `model_ids` (deployment ids). Both default to None.
        """
        body: dict[str, Any] = {"access_group": access_group}
        if model_names is not None:
            body["model_names"] = model_names
        if model_ids is not None:
            body["model_ids"] = model_ids
        return await self._request("POST", "/access_group/new", json=body)

    async def update_model_access_group(
        self,
        access_group: str,
        model_names: Optional[list[str]] = None,
        model_ids: Optional[list[str]] = None,
    ) -> dict:
        """Update membership of a model access group (`PUT /access_group/{access_group}/update`).

        Both fields are optional; only those passed will be sent. The upstream
        endpoint replaces (not appends) membership.
        """
        body: dict[str, Any] = {}
        if model_names is not None:
            body["model_names"] = model_names
        if model_ids is not None:
            body["model_ids"] = model_ids
        return await self._request("PUT", f"/access_group/{access_group}/update", json=body)

    async def delete_model_access_group(self, access_group: str) -> dict:
        """Delete a model access group by name (`DELETE /access_group/{access_group}/delete`)."""
        return await self._request("DELETE", f"/access_group/{access_group}/delete")

    # ── Credential operations ──

    async def list_credentials(self) -> Any:
        """List provider credentials (`GET /credentials`).

        Returns the upstream payload. Credential values themselves are not
        included — use `get_credential` for details.
        """
        return await self._request("GET", "/credentials")

    async def get_credential(self, credential_name: str) -> dict:
        """Get a credential by name (`GET /credentials/by_name/{credential_name}`)."""
        return await self._request("GET", f"/credentials/by_name/{credential_name}")

    async def get_credential_by_model(self, model_id: str) -> dict:
        """Get the credential bound to a deployment (`GET /credentials/by_model/{model_id}`)."""
        return await self._request("GET", f"/credentials/by_model/{model_id}")

    async def create_credential(
        self,
        credential_name: str,
        credential_info: dict,
        credential_values: Optional[dict] = None,
        model_id: Optional[str] = None,
    ) -> dict:
        """Create a credential (`POST /credentials`).

        Either provide raw `credential_values` (e.g. `{"api_key": "sk-..."}`) or
        bind to an existing deployment via `model_id`.
        """
        body: dict[str, Any] = {
            "credential_name": credential_name,
            "credential_info": credential_info,
        }
        if credential_values is not None:
            body["credential_values"] = credential_values
        if model_id is not None:
            body["model_id"] = model_id
        return await self._request("POST", "/credentials", json=body)

    async def update_credential(
        self,
        credential_name: str,
        credential_info: dict,
        credential_values: dict,
    ) -> dict:
        """Update a credential (`PATCH /credentials/{credential_name}`).

        Despite the PATCH verb, this is *not* a partial update — the upstream
        `CredentialItem` schema requires all three fields. Callers must
        re-send the full current state of `credential_info` and
        `credential_values` (e.g. via `get_credential(..., verbosity='full')`),
        not just the keys they want to change. Path id and body `credential_name`
        should match.
        """
        body = {
            "credential_name": credential_name,
            "credential_info": credential_info,
            "credential_values": credential_values,
        }
        return await self._request("PATCH", f"/credentials/{credential_name}", json=body)

    async def delete_credential(self, credential_name: str) -> dict:
        """Delete a credential (`DELETE /credentials/{credential_name}`)."""
        return await self._request("DELETE", f"/credentials/{credential_name}")

    # ── Key operations ──

    async def list_keys(
        self,
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
    ) -> dict:
        """List virtual keys (`GET /key/list`).

        Returns a paginated payload. All filter args are optional.
        """
        params = {
            k: v
            for k, v in {
                "page": page,
                "size": size,
                "user_id": user_id,
                "team_id": team_id,
                "organization_id": organization_id,
                "key_alias": key_alias,
                "return_full_object": return_full_object,
                "include_team_keys": include_team_keys,
                "sort_by": sort_by,
                "sort_order": sort_order,
            }.items()
            if v is not None
        }
        return await self._request("GET", "/key/list", params=params or None)

    async def list_key_aliases(
        self,
        page: Optional[int] = None,
        size: Optional[int] = None,
        search: Optional[str] = None,
        team_id: Optional[str] = None,
    ) -> dict:
        """List key aliases (`GET /key/aliases`).

        Lighter-weight listing than `list_keys`; returns only alias metadata.
        """
        params = {
            k: v
            for k, v in {"page": page, "size": size, "search": search, "team_id": team_id}.items()
            if v is not None
        }
        return await self._request("GET", "/key/aliases", params=params or None)

    async def get_key_info(self, key: Optional[str] = None) -> dict:
        """Get info about a key (`GET /key/info`).

        If `key` is omitted, returns info about the caller's own key.
        """
        params = {"key": key} if key else None
        return await self._request("GET", "/key/info", params=params)

    @staticmethod
    def _build_key_body(
        common: dict[str, Any],
        extras: Optional[dict],
    ) -> dict[str, Any]:
        """Build a key request body: drop None entries from common, merge extras."""
        body = {k: v for k, v in common.items() if v is not None}
        if extras:
            body.update(extras)
        return body

    async def generate_key(
        self,
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

        All fields are optional. Use `extras` for the ~30 less common fields not
        surfaced as explicit args (see GenerateKeyRequest in upstream Swagger).
        """
        body = self._build_key_body(
            {
                "key_alias": key_alias,
                "duration": duration,
                "models": models,
                "max_budget": max_budget,
                "budget_duration": budget_duration,
                "user_id": user_id,
                "team_id": team_id,
                "tpm_limit": tpm_limit,
                "rpm_limit": rpm_limit,
                "metadata": metadata,
                "guardrails": guardrails,
                "blocked": blocked,
            },
            extras,
        )
        return await self._request("POST", "/key/generate", json=body)

    async def generate_service_account_key(
        self,
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

        Same body shape as `generate_key`, but the upstream tags the key as a
        service account (no human user binding required).
        """
        body = self._build_key_body(
            {
                "key_alias": key_alias,
                "duration": duration,
                "models": models,
                "max_budget": max_budget,
                "budget_duration": budget_duration,
                "user_id": user_id,
                "team_id": team_id,
                "tpm_limit": tpm_limit,
                "rpm_limit": rpm_limit,
                "metadata": metadata,
                "guardrails": guardrails,
                "blocked": blocked,
            },
            extras,
        )
        return await self._request("POST", "/key/service-account/generate", json=body)

    async def update_key(
        self,
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

        Only `key` is required. All other fields are merged sparsely.
        """
        body = self._build_key_body(
            {
                "key": key,
                "key_alias": key_alias,
                "duration": duration,
                "models": models,
                "max_budget": max_budget,
                "budget_duration": budget_duration,
                "tpm_limit": tpm_limit,
                "rpm_limit": rpm_limit,
                "metadata": metadata,
                "guardrails": guardrails,
                "blocked": blocked,
            },
            extras,
        )
        return await self._request("POST", "/key/update", json=body)

    async def regenerate_key(
        self,
        key: str,
        new_master_key: Optional[str] = None,
        duration: Optional[str] = None,
        extras: Optional[dict] = None,
    ) -> dict:
        """Regenerate a virtual key (`POST /key/{key}/regenerate`).

        Returns a payload containing the new key value. The body is optional;
        if you want to atomically update settings on the new key, pass them via
        `extras` (matches RegenerateKeyRequest in upstream Swagger).
        """
        body = self._build_key_body(
            {"new_master_key": new_master_key, "duration": duration},
            extras,
        )
        return await self._request("POST", f"/key/{key}/regenerate", json=body or None)

    async def set_key_blocked(self, key: str, blocked: bool) -> dict:
        """Block or unblock a virtual key.

        Routes to `POST /key/block` when `blocked=True`, otherwise `POST /key/unblock`.
        """
        path = "/key/block" if blocked else "/key/unblock"
        return await self._request("POST", path, json={"key": key})

    async def delete_keys(
        self,
        keys: Optional[list[str]] = None,
        key_aliases: Optional[list[str]] = None,
    ) -> dict:
        """Batch-delete virtual keys (`POST /key/delete`).

        Provide either `keys` (raw key values) or `key_aliases` (alias names);
        at least one must be non-empty.
        """
        body: dict[str, Any] = {}
        if keys is not None:
            body["keys"] = keys
        if key_aliases is not None:
            body["key_aliases"] = key_aliases
        return await self._request("POST", "/key/delete", json=body)

    async def reset_key_spend(self, key: str, reset_to: float = 0.0) -> dict:
        """Reset a key's accumulated spend (`POST /key/{key}/reset_spend`).

        `reset_to` defaults to 0.0 (typical use); pass a positive value to set
        a starting balance.
        """
        return await self._request("POST", f"/key/{key}/reset_spend", json={"reset_to": reset_to})

    async def key_health(self) -> dict:
        """Probe the health of the caller's key (`POST /key/health`).

        Returns connectivity / permission status for the bearer key. No body.
        """
        return await self._request("POST", "/key/health")

    # ── Internal User operations ──

    @staticmethod
    def _build_body(common: dict[str, Any], extras: Optional[dict] = None) -> dict[str, Any]:
        """Build a request body: drop None entries from common, merge extras."""
        body = {k: v for k, v in common.items() if v is not None}
        if extras:
            body.update(extras)
        return body

    async def list_users(
        self,
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
    ) -> dict:
        """List internal LiteLLM users (`GET /user/list`).

        All filters are optional. `user_ids`, `sso_user_ids`, `organization_ids`
        are upstream-comma-separated strings (not arrays).
        """
        params = {
            k: v
            for k, v in {
                "role": role,
                "user_ids": user_ids,
                "sso_user_ids": sso_user_ids,
                "user_email": user_email,
                "team": team,
                "page": page,
                "page_size": page_size,
                "sort_by": sort_by,
                "sort_order": sort_order,
                "organization_ids": organization_ids,
            }.items()
            if v is not None
        }
        return await self._request("GET", "/user/list", params=params or None)

    async def get_user_info(self, user_id: Optional[str] = None) -> dict:
        """Get info about an internal user (`GET /user/info`).

        If `user_id` is omitted, the upstream returns info about the caller.
        """
        params = {"user_id": user_id} if user_id else None
        return await self._request("GET", "/user/info", params=params)

    async def create_user(
        self,
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

        ~12 common NewUserRequest fields are surfaced as named args; pass any
        other upstream field via `extras` (merged last into the body). All args
        are optional — upstream defaults apply.
        """
        body = self._build_body(
            {
                "user_email": user_email,
                "user_id": user_id,
                "user_alias": user_alias,
                "user_role": user_role,
                "teams": teams,
                "organizations": organizations,
                "models": models,
                "max_budget": max_budget,
                "budget_duration": budget_duration,
                "tpm_limit": tpm_limit,
                "rpm_limit": rpm_limit,
                "metadata": metadata,
                "guardrails": guardrails,
                "blocked": blocked,
                "auto_create_key": auto_create_key,
                "send_invite_email": send_invite_email,
            },
            extras,
        )
        return await self._request("POST", "/user/new", json=body)

    async def update_user(
        self,
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

        Only `user_id` is required; other args are merged sparsely. Use
        `extras` for less-common UpdateUserRequest fields.
        """
        body = self._build_body(
            {
                "user_id": user_id,
                "user_email": user_email,
                "user_alias": user_alias,
                "user_role": user_role,
                "password": password,
                "models": models,
                "max_budget": max_budget,
                "budget_duration": budget_duration,
                "tpm_limit": tpm_limit,
                "rpm_limit": rpm_limit,
                "metadata": metadata,
                "guardrails": guardrails,
                "blocked": blocked,
            },
            extras,
        )
        return await self._request("POST", "/user/update", json=body)

    async def delete_user(self, user_ids: list[str]) -> dict:
        """Batch-delete internal users (`POST /user/delete`).

        Body is `{"user_ids": [...]}`. At least one id is required.
        """
        return await self._request("POST", "/user/delete", json={"user_ids": user_ids})

    # ── Customer (end-user) operations ──

    async def list_customers(self) -> Any:
        """List end-user customers (`GET /customer/list`)."""
        return await self._request("GET", "/customer/list")

    async def get_customer_info(self, end_user_id: str) -> dict:
        """Get a customer by end_user_id (`GET /customer/info`)."""
        return await self._request("GET", "/customer/info", params={"end_user_id": end_user_id})

    async def create_customer(
        self,
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

        `user_id` is required (the customer's stable end-user id). Other fields
        are optional; pass uncommon NewCustomerRequest fields via `extras`.
        """
        body = self._build_body(
            {
                "user_id": user_id,
                "alias": alias,
                "max_budget": max_budget,
                "soft_budget": soft_budget,
                "budget_id": budget_id,
                "budget_duration": budget_duration,
                "tpm_limit": tpm_limit,
                "rpm_limit": rpm_limit,
                "max_parallel_requests": max_parallel_requests,
                "blocked": blocked,
                "allowed_model_region": allowed_model_region,
                "default_model": default_model,
            },
            extras,
        )
        return await self._request("POST", "/customer/new", json=body)

    async def update_customer(
        self,
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

        `user_id` is required. UpdateCustomerRequest is intentionally narrower
        than NewCustomerRequest upstream — pass less-common fields via `extras`.
        """
        body = self._build_body(
            {
                "user_id": user_id,
                "alias": alias,
                "max_budget": max_budget,
                "budget_id": budget_id,
                "blocked": blocked,
                "allowed_model_region": allowed_model_region,
                "default_model": default_model,
            },
            extras,
        )
        return await self._request("POST", "/customer/update", json=body)

    async def delete_customer(self, user_ids: list[str]) -> dict:
        """Batch-delete customers (`POST /customer/delete`).

        Body is `{"user_ids": [...]}`.
        """
        return await self._request("POST", "/customer/delete", json={"user_ids": user_ids})

    async def set_customer_blocked(self, user_ids: list[str], blocked: bool) -> dict:
        """Block or unblock customers.

        Routes to `POST /customer/block` when `blocked=True`, else
        `POST /customer/unblock`. Body in both cases is `{"user_ids": [...]}`.
        """
        path = "/customer/block" if blocked else "/customer/unblock"
        return await self._request("POST", path, json={"user_ids": user_ids})

    async def get_customer_daily_activity(
        self,
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

        `start_date` and `end_date` (ISO `YYYY-MM-DD`) are required upstream —
        omitting either returns HTTP 400, despite the OpenAPI spec marking
        them optional. `end_user_ids` and `exclude_end_user_ids` are
        upstream-comma-separated strings (not arrays).
        """
        params = {
            k: v
            for k, v in {
                "end_user_ids": end_user_ids,
                "start_date": start_date,
                "end_date": end_date,
                "model": model,
                "api_key": api_key,
                "page": page,
                "page_size": page_size,
                "exclude_end_user_ids": exclude_end_user_ids,
            }.items()
            if v is not None
        }
        return await self._request("GET", "/customer/daily/activity", params=params or None)

    # ── Organization operations ──

    async def list_organizations(
        self,
        org_id: Optional[str] = None,
        org_alias: Optional[str] = None,
    ) -> Any:
        """List organizations (`GET /organization/list`)."""
        params = {
            k: v for k, v in {"org_id": org_id, "org_alias": org_alias}.items() if v is not None
        }
        return await self._request("GET", "/organization/list", params=params or None)

    async def get_organization_info(self, organization_id: str) -> dict:
        """Get a single organization by id (`GET /organization/info`)."""
        return await self._request(
            "GET", "/organization/info", params={"organization_id": organization_id}
        )

    async def create_organization(
        self,
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

        `organization_alias` is the only required field. Other common
        NewOrganizationRequest fields are surfaced as named args; pass less
        common ones via `extras`.
        """
        body = self._build_body(
            {
                "organization_alias": organization_alias,
                "organization_id": organization_id,
                "models": models,
                "max_budget": max_budget,
                "soft_budget": soft_budget,
                "budget_id": budget_id,
                "budget_duration": budget_duration,
                "tpm_limit": tpm_limit,
                "rpm_limit": rpm_limit,
                "max_parallel_requests": max_parallel_requests,
                "metadata": metadata,
            },
            extras,
        )
        return await self._request("POST", "/organization/new", json=body)

    async def update_organization(
        self,
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

        Upstream OpenAPI omits the request body schema for this endpoint, but
        the implementation accepts an UpdateOrganization payload mirroring
        NewOrganizationRequest minus the alias requirement. `organization_id`
        is required in the body to identify the row; other fields are merged
        sparsely. Use `extras` for any field not surfaced here.
        """
        body = self._build_body(
            {
                "organization_id": organization_id,
                "organization_alias": organization_alias,
                "models": models,
                "max_budget": max_budget,
                "budget_duration": budget_duration,
                "tpm_limit": tpm_limit,
                "rpm_limit": rpm_limit,
                "metadata": metadata,
            },
            extras,
        )
        return await self._request("PATCH", "/organization/update", json=body)

    async def delete_organization(self, organization_ids: list[str]) -> dict:
        """Batch-delete organizations (`DELETE /organization/delete`).

        Note: this is a `DELETE` with a JSON body — `{"organization_ids": [...]}`.
        """
        return await self._request(
            "DELETE", "/organization/delete", json={"organization_ids": organization_ids}
        )

    async def add_org_member(
        self,
        organization_id: str,
        member: dict,
        max_budget_in_organization: Optional[float] = None,
    ) -> dict:
        """Add a member to an organization (`POST /organization/member_add`).

        `member` is the upstream Member shape — at minimum
        `{"user_id" or "user_email": ..., "role": ...}`.
        """
        body: dict[str, Any] = {"organization_id": organization_id, "member": member}
        if max_budget_in_organization is not None:
            body["max_budget_in_organization"] = max_budget_in_organization
        return await self._request("POST", "/organization/member_add", json=body)

    async def update_org_member(
        self,
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
        body = self._build_body(
            {
                "organization_id": organization_id,
                "user_id": user_id,
                "user_email": user_email,
                "role": role,
                "max_budget_in_organization": max_budget_in_organization,
            }
        )
        return await self._request("PATCH", "/organization/member_update", json=body)

    async def delete_org_member(
        self,
        organization_id: str,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
    ) -> dict:
        """Remove an org member (`DELETE /organization/member_delete`).

        Identify the member by `user_id` or `user_email`. DELETE with JSON
        body — supported by the upstream.
        """
        body = self._build_body(
            {"organization_id": organization_id, "user_id": user_id, "user_email": user_email}
        )
        return await self._request("DELETE", "/organization/member_delete", json=body)

    async def get_org_daily_activity(
        self,
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

        `start_date` and `end_date` (ISO `YYYY-MM-DD`) are required upstream —
        omitting either returns HTTP 400, despite the OpenAPI spec marking
        them optional. `organization_ids` and `exclude_organization_ids` are
        upstream-comma-separated strings (not arrays).
        """
        params = {
            k: v
            for k, v in {
                "organization_ids": organization_ids,
                "start_date": start_date,
                "end_date": end_date,
                "model": model,
                "api_key": api_key,
                "page": page,
                "page_size": page_size,
                "exclude_organization_ids": exclude_organization_ids,
            }.items()
            if v is not None
        }
        return await self._request("GET", "/organization/daily/activity", params=params or None)

    # ── Project operations ──

    async def list_projects(self) -> Any:
        """List projects (`GET /project/list`)."""
        return await self._request("GET", "/project/list")

    async def get_project_info(self, project_id: str) -> dict:
        """Get a project by id (`GET /project/info`)."""
        return await self._request("GET", "/project/info", params={"project_id": project_id})

    async def create_project(
        self,
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

        `team_id` is required (projects belong to teams). Other fields are
        optional; pass less-common NewProjectRequest fields via `extras`.
        """
        body = self._build_body(
            {
                "team_id": team_id,
                "project_id": project_id,
                "project_alias": project_alias,
                "description": description,
                "models": models,
                "max_budget": max_budget,
                "soft_budget": soft_budget,
                "budget_duration": budget_duration,
                "tpm_limit": tpm_limit,
                "rpm_limit": rpm_limit,
                "tags": tags,
                "metadata": metadata,
                "guardrails": guardrails,
                "blocked": blocked,
            },
            extras,
        )
        return await self._request("POST", "/project/new", json=body)

    async def update_project(
        self,
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

        Only `project_id` is required; other fields are merged sparsely.
        """
        body = self._build_body(
            {
                "project_id": project_id,
                "project_alias": project_alias,
                "description": description,
                "models": models,
                "max_budget": max_budget,
                "budget_duration": budget_duration,
                "tpm_limit": tpm_limit,
                "rpm_limit": rpm_limit,
                "tags": tags,
                "metadata": metadata,
                "guardrails": guardrails,
                "blocked": blocked,
            },
            extras,
        )
        return await self._request("POST", "/project/update", json=body)

    async def delete_project(self, project_ids: list[str]) -> dict:
        """Batch-delete projects (`DELETE /project/delete`).

        DELETE with JSON body — `{"project_ids": [...]}`.
        """
        return await self._request("DELETE", "/project/delete", json={"project_ids": project_ids})

    # ── Unified User Access Group operations ──
    #
    # Wraps `/v1/unified_access_group/*`. Distinct from the model-access-group
    # family (`/access_group/*`) — unified access groups gate users/teams
    # against models, MCP servers, and agents in one shape.

    async def list_user_access_groups(self) -> Any:
        """List unified user access groups (`GET /v1/unified_access_group`)."""
        return await self._request("GET", "/v1/unified_access_group")

    async def get_user_access_group(self, access_group_id: str) -> dict:
        """Get a unified access group by id (`GET /v1/unified_access_group/{id}`)."""
        return await self._request("GET", f"/v1/unified_access_group/{access_group_id}")

    async def create_user_access_group(
        self,
        access_group_name: str,
        description: Optional[str] = None,
        access_model_names: Optional[list[str]] = None,
        access_mcp_server_ids: Optional[list[str]] = None,
        access_agent_ids: Optional[list[str]] = None,
        assigned_team_ids: Optional[list[str]] = None,
        assigned_key_ids: Optional[list[str]] = None,
    ) -> dict:
        """Create a unified user access group (`POST /v1/unified_access_group`).

        `access_group_name` is required; the access_* and assigned_* lists are
        all optional and can be added later via update.
        """
        body = self._build_body(
            {
                "access_group_name": access_group_name,
                "description": description,
                "access_model_names": access_model_names,
                "access_mcp_server_ids": access_mcp_server_ids,
                "access_agent_ids": access_agent_ids,
                "assigned_team_ids": assigned_team_ids,
                "assigned_key_ids": assigned_key_ids,
            }
        )
        return await self._request("POST", "/v1/unified_access_group", json=body)

    async def update_user_access_group(
        self,
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
        body = self._build_body(
            {
                "access_group_name": access_group_name,
                "description": description,
                "access_model_names": access_model_names,
                "access_mcp_server_ids": access_mcp_server_ids,
                "access_agent_ids": access_agent_ids,
                "assigned_team_ids": assigned_team_ids,
                "assigned_key_ids": assigned_key_ids,
            }
        )
        return await self._request(
            "PUT", f"/v1/unified_access_group/{access_group_id}", json=body or None
        )

    async def delete_user_access_group(self, access_group_id: str) -> dict:
        """Delete a unified user access group (`DELETE /v1/unified_access_group/{id}`)."""
        return await self._request("DELETE", f"/v1/unified_access_group/{access_group_id}")

    # ── Budget operations ──

    async def list_budgets(self) -> Any:
        """List configured budgets (`GET /budget/list`)."""
        return await self._request("GET", "/budget/list")

    async def get_budget_info(self, budgets: list[str]) -> Any:
        """Get info about one or more budgets (`POST /budget/info`).

        Body shape is `{"budgets": [...]}` — the upstream `BudgetRequest` schema
        treats this as a batch lookup, not a single-id GET.
        """
        return await self._request("POST", "/budget/info", json={"budgets": budgets})

    async def create_budget(
        self,
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

        All fields optional per upstream `BudgetNewRequest`. `model_max_budget`
        is a per-model cap mapping like `{"gpt-4o": {"budget_limit": 10.0}}`.
        """
        body = self._build_body(
            {
                "budget_id": budget_id,
                "max_budget": max_budget,
                "soft_budget": soft_budget,
                "max_parallel_requests": max_parallel_requests,
                "tpm_limit": tpm_limit,
                "rpm_limit": rpm_limit,
                "budget_duration": budget_duration,
                "model_max_budget": model_max_budget,
                "budget_reset_at": budget_reset_at,
            },
            extras,
        )
        return await self._request("POST", "/budget/new", json=body)

    async def update_budget(
        self,
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
        body = self._build_body(
            {
                "budget_id": budget_id,
                "max_budget": max_budget,
                "soft_budget": soft_budget,
                "max_parallel_requests": max_parallel_requests,
                "tpm_limit": tpm_limit,
                "rpm_limit": rpm_limit,
                "budget_duration": budget_duration,
                "model_max_budget": model_max_budget,
                "budget_reset_at": budget_reset_at,
            },
            extras,
        )
        return await self._request("POST", "/budget/update", json=body)

    async def delete_budget(self, budget_id: str) -> dict:
        """Delete a budget (`POST /budget/delete`).

        Body is `{"id": <budget_id>}` per upstream `BudgetDeleteRequest`.
        """
        return await self._request("POST", "/budget/delete", json={"id": budget_id})

    async def get_budget_settings(self, budget_id: str) -> dict:
        """Get effective budget settings (`GET /budget/settings`).

        `budget_id` is a required query param.
        """
        return await self._request("GET", "/budget/settings", params={"budget_id": budget_id})

    # ── Spend operations ──

    async def get_global_spend_report(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        group_by: Optional[str] = None,
        api_key: Optional[str] = None,
        internal_user_id: Optional[str] = None,
        team_id: Optional[str] = None,
        customer_id: Optional[str] = None,
    ) -> Any:
        """Aggregated global spend report (`GET /global/spend/report`).

        All filters are optional query params. `group_by` accepts upstream-defined
        values like `team`, `customer`, `api_key`, `model`.
        """
        params = {
            k: v
            for k, v in {
                "start_date": start_date,
                "end_date": end_date,
                "group_by": group_by,
                "api_key": api_key,
                "internal_user_id": internal_user_id,
                "team_id": team_id,
                "customer_id": customer_id,
            }.items()
            if v is not None
        }
        return await self._request("GET", "/global/spend/report", params=params or None)

    async def list_spend_logs(
        self,
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        summarize: Optional[bool] = None,
    ) -> Any:
        """List per-request spend logs (`GET /spend/logs`).

        All filters are optional. `summarize=True` returns aggregated rows.
        """
        params = {
            k: v
            for k, v in {
                "api_key": api_key,
                "user_id": user_id,
                "request_id": request_id,
                "start_date": start_date,
                "end_date": end_date,
                "summarize": summarize,
            }.items()
            if v is not None
        }
        return await self._request("GET", "/spend/logs", params=params or None)

    async def list_spend_tags(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Any:
        """List distinct spend tags within a date window (`GET /spend/tags`)."""
        params = {
            k: v
            for k, v in {"start_date": start_date, "end_date": end_date}.items()
            if v is not None
        }
        return await self._request("GET", "/spend/tags", params=params or None)

    async def calculate_spend(
        self,
        model: Optional[str] = None,
        messages: Optional[list[dict]] = None,
        completion_response: Optional[dict] = None,
    ) -> dict:
        """Estimate spend for a request (`POST /spend/calculate`).

        Per upstream `SpendCalculateRequest`, callers can either:
        - pass `model + messages` for a prospective cost estimate, or
        - pass `model + completion_response` for a retrospective re-cost.
        """
        body = self._build_body(
            {
                "model": model,
                "messages": messages,
                "completion_response": completion_response,
            }
        )
        return await self._request("POST", "/spend/calculate", json=body)

    async def get_user_daily_activity(
        self,
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

        Mirrors the customer/org daily-activity shape but scoped to the
        internal-user dimension. Supplying `start_date` / `end_date` is
        recommended (the upstream has been observed to reject open-ended ranges
        on related endpoints).
        """
        params = {
            k: v
            for k, v in {
                "start_date": start_date,
                "end_date": end_date,
                "model": model,
                "api_key": api_key,
                "user_id": user_id,
                "page": page,
                "page_size": page_size,
                "timezone": timezone,
            }.items()
            if v is not None
        }
        return await self._request("GET", "/user/daily/activity", params=params or None)

    # ── Execution operations ──

    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        body: Optional[dict] = None,
    ) -> dict:
        """Chat completion (`POST /v1/chat/completions`).

        Synchronous only — streaming is not exposed in this slice (would need a
        different transport contract on the MCP side). Pass any extra OpenAI /
        LiteLLM body fields (temperature, max_tokens, tools, etc.) via `body`.
        """
        payload: dict[str, Any] = dict(body) if body else {}
        payload["model"] = model
        payload["messages"] = messages
        payload.pop("stream", None)
        return await self._request("POST", "/v1/chat/completions", json=payload)

    async def completion(
        self,
        model: str,
        prompt: str,
        body: Optional[dict] = None,
    ) -> dict:
        """Legacy text completion (`POST /v1/completions`).

        OpenAPI declares only a `model` query param and no body schema, but the
        upstream still accepts the OpenAI-shaped body (`prompt`, `max_tokens`,
        etc.). We send `model` in both query and body to satisfy both routing
        paths.
        """
        payload: dict[str, Any] = dict(body) if body else {}
        payload["model"] = model
        payload["prompt"] = prompt
        payload.pop("stream", None)
        return await self._request("POST", "/v1/completions", params={"model": model}, json=payload)

    async def embed(
        self,
        model: str,
        input: list[str],
        body: Optional[dict] = None,
    ) -> dict:
        """Generate embeddings (`POST /v1/embeddings`).

        Pass any extra fields (e.g. `dimensions`, `encoding_format`, `user`)
        via `body`. `input` is a list of strings; single-string callers should
        wrap as `[text]`.
        """
        payload: dict[str, Any] = dict(body) if body else {}
        payload["model"] = model
        payload["input"] = input
        return await self._request("POST", "/v1/embeddings", json=payload)

    # ── Health operations ──

    async def check_health(
        self,
        model: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> dict:
        """Run upstream health checks against deployments (`GET /health`).

        Optionally narrow to a single deployment via `model` (model_name alias)
        or `model_id` (litellm internal id).
        """
        params = {k: v for k, v in {"model": model, "model_id": model_id}.items() if v is not None}
        return await self._request("GET", "/health", params=params or None)

    async def check_health_backlog(self) -> dict:
        """Get health-check queue backlog (`GET /health/backlog`)."""
        return await self._request("GET", "/health/backlog")

    async def get_health_history(
        self,
        model: Optional[str] = None,
        status_filter: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> dict:
        """Historical health check results (`GET /health/history`).

        `status_filter` accepts upstream values (`healthy` / `unhealthy`).
        """
        params = {
            k: v
            for k, v in {
                "model": model,
                "status_filter": status_filter,
                "limit": limit,
                "offset": offset,
            }.items()
            if v is not None
        }
        return await self._request("GET", "/health/history", params=params or None)

    async def get_health_latest(self) -> dict:
        """Latest health check snapshot (`GET /health/latest`)."""
        return await self._request("GET", "/health/latest")

    async def test_model_connection(
        self,
        litellm_params: dict,
        mode: Optional[str] = None,
        model_info: Optional[dict] = None,
    ) -> dict:
        """Test a candidate deployment connection (`POST /health/test_connection`).

        `litellm_params` is the provider routing dict (same shape as `add_model`).
        `mode` is the upstream test mode (e.g. `chat`, `embedding`, `completion`).
        `model_info` is optional deployment metadata for the probe.
        """
        body: dict[str, Any] = {"litellm_params": litellm_params}
        if mode is not None:
            body["mode"] = mode
        if model_info is not None:
            body["model_info"] = model_info
        return await self._request("POST", "/health/test_connection", json=body)

    # ── MCP Gateway operations ──
    #
    # The LiteLLM proxy can act as an MCP-of-MCPs: register upstream HTTP-transport
    # MCP servers and broker tool listing / invocation through the proxy. These
    # tools wrap the `/v1/mcp/*` and `/mcp-rest/*` admin surfaces.

    @staticmethod
    def _build_mcp_server_body(
        server_id: Optional[str],
        server_name: Optional[str],
        alias: Optional[str],
        description: Optional[str],
        transport: Optional[str],
        url: Optional[str],
        auth_type: Optional[str],
        spec_path: Optional[str],
        mcp_info: Optional[dict],
        mcp_access_groups: Optional[list[str]],
        allowed_tools: Optional[list[str]],
        credentials: Optional[dict],
        extras: Optional[dict],
    ) -> dict[str, Any]:
        """Build a NewMCPServerRequest / UpdateMCPServerRequest body."""
        body = {
            k: v
            for k, v in {
                "server_id": server_id,
                "server_name": server_name,
                "alias": alias,
                "description": description,
                "transport": transport,
                "url": url,
                "auth_type": auth_type,
                "spec_path": spec_path,
                "mcp_info": mcp_info,
                "mcp_access_groups": mcp_access_groups,
                "allowed_tools": allowed_tools,
                "credentials": credentials,
            }.items()
            if v is not None
        }
        if extras:
            body.update(extras)
        return body

    async def list_mcp_servers(self, team_id: Optional[str] = None) -> Any:
        """List registered upstream MCP servers (`GET /v1/mcp/server`).

        `team_id` optionally narrows to servers visible to a single team.
        """
        params = {"team_id": team_id} if team_id else None
        return await self._request("GET", "/v1/mcp/server", params=params)

    async def get_mcp_server(self, server_id: str) -> dict:
        """Get a single MCP server by id (`GET /v1/mcp/server/{server_id}`)."""
        return await self._request("GET", f"/v1/mcp/server/{server_id}")

    async def add_mcp_server(
        self,
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
        """Admin-register a new upstream MCP server (`POST /v1/mcp/server`).

        `transport` is required (`http`, `sse`, `stdio`). For HTTP/SSE provide
        `url`; for stdio supply upstream `command` / `args` / `env` fields
        via `extras`. Pass any of the ~30 NewMCPServerRequest fields not
        surfaced as named args through `extras` (e.g. `static_headers`,
        `extra_headers`, `tool_name_to_display_name`, `oauth2_flow`).
        """
        body = self._build_mcp_server_body(
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
        return await self._request("POST", "/v1/mcp/server", json=body)

    async def update_mcp_server(
        self,
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

        `server_id` identifies the row and is sent in the body, not the path —
        UpdateMCPServerRequest treats it as required.
        """
        body = self._build_mcp_server_body(
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
        return await self._request("PUT", "/v1/mcp/server", json=body)

    async def delete_mcp_server(self, server_id: str) -> dict:
        """Delete an MCP server (`DELETE /v1/mcp/server/{server_id}`)."""
        return await self._request("DELETE", f"/v1/mcp/server/{server_id}")

    async def register_mcp_server(
        self,
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

        Distinct from `add_mcp_server`: `register` is the user-facing submission
        path that creates a server in `pending` approval state, awaiting an
        admin's `approve_mcp_server_submission`. `add_mcp_server` is the admin
        path that creates servers in approved state directly.
        """
        body = self._build_mcp_server_body(
            None,
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
        return await self._request("POST", "/v1/mcp/server/register", json=body)

    async def list_mcp_server_submissions(self) -> Any:
        """List pending MCP server submissions (`GET /v1/mcp/server/submissions`)."""
        return await self._request("GET", "/v1/mcp/server/submissions")

    async def approve_mcp_server_submission(self, server_id: str) -> dict:
        """Approve a pending MCP submission (`PUT /v1/mcp/server/{id}/approve`)."""
        return await self._request("PUT", f"/v1/mcp/server/{server_id}/approve")

    async def reject_mcp_server_submission(
        self, server_id: str, review_notes: Optional[str] = None
    ) -> dict:
        """Reject a pending MCP submission (`PUT /v1/mcp/server/{id}/reject`).

        Optional `review_notes` is captured per upstream `RejectMCPServerRequest`.
        """
        body = {"review_notes": review_notes} if review_notes is not None else {}
        return await self._request("PUT", f"/v1/mcp/server/{server_id}/reject", json=body)

    async def check_mcp_servers_health(self, server_ids: Optional[str] = None) -> dict:
        """Probe upstream MCP server health (`GET /v1/mcp/server/health`).

        `server_ids` is an upstream-comma-separated string (not an array) to
        narrow the probe to specific servers.
        """
        params = {"server_ids": server_ids} if server_ids else None
        return await self._request("GET", "/v1/mcp/server/health", params=params)

    async def list_mcp_tools(self) -> Any:
        """List tools across all registered MCP servers (`GET /v1/mcp/tools`)."""
        return await self._request("GET", "/v1/mcp/tools")

    async def list_mcp_tools_rest(self, server_id: Optional[str] = None) -> Any:
        """List tools (REST shape) for a registered MCP server (`GET /mcp-rest/tools/list`).

        Lighter-weight than `list_mcp_tools`; per-server. `server_id` is a query
        param.
        """
        params = {"server_id": server_id} if server_id else None
        return await self._request("GET", "/mcp-rest/tools/list", params=params)

    async def call_mcp_tool(
        self,
        server_id: str,
        name: str,
        arguments: Optional[dict] = None,
    ) -> Any:
        """Invoke a tool on a registered MCP server (`POST /mcp-rest/tools/call`).

        Body shape: `{"server_id": ..., "name": ..., "arguments": {...}}`.
        Upstream returns the provider-shaped tool response (typically the MCP
        `tools/call` result envelope: `{"content": [...], "isError": bool}`).
        """
        body: dict[str, Any] = {"server_id": server_id, "name": name}
        if arguments is not None:
            body["arguments"] = arguments
        return await self._request("POST", "/mcp-rest/tools/call", json=body)

    async def test_mcp_connection(
        self,
        transport: str,
        url: Optional[str] = None,
        auth_type: Optional[str] = None,
        spec_path: Optional[str] = None,
        credentials: Optional[dict] = None,
        extras: Optional[dict] = None,
    ) -> dict:
        """Test a candidate MCP server connection (`POST /mcp-rest/test/connection`).

        Body is a NewMCPServerRequest — pass the same fields you would to
        `add_mcp_server`. Useful for validating provider URL + auth before
        registering.
        """
        body = self._build_mcp_server_body(
            None,
            None,
            None,
            None,
            transport,
            url,
            auth_type,
            spec_path,
            None,
            None,
            None,
            credentials,
            extras,
        )
        return await self._request("POST", "/mcp-rest/test/connection", json=body)

    async def discover_mcp_servers(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
    ) -> Any:
        """Search the public MCP server registry (`GET /v1/mcp/discover`).

        Both filters are optional query params.
        """
        params = {k: v for k, v in {"query": query, "category": category}.items() if v is not None}
        return await self._request("GET", "/v1/mcp/discover", params=params or None)

    async def get_mcp_openapi_registry(self) -> Any:
        """Get the OpenAPI registry of MCP servers (`GET /v1/mcp/openapi-registry`)."""
        return await self._request("GET", "/v1/mcp/openapi-registry")

    async def get_mcp_registry(self) -> Any:
        """Get the MCP servers registry JSON (`GET /v1/mcp/registry.json`)."""
        return await self._request("GET", "/v1/mcp/registry.json")

    async def list_mcp_access_groups(self) -> Any:
        """List MCP access groups (`GET /v1/mcp/access_groups`).

        Distinct from model access groups (`/access_group/list`) and
        unified user access groups (`/v1/unified_access_group`) — these
        gate keys/teams against MCP servers specifically.
        """
        return await self._request("GET", "/v1/mcp/access_groups")

    async def make_mcp_servers_public(self, mcp_server_ids: list[str]) -> dict:
        """Publish MCP servers to the public hub (`POST /v1/mcp/make_public`).

        Body is `{"mcp_server_ids": [...]}` per upstream `MakeMCPServersPublicRequest`.
        """
        return await self._request(
            "POST", "/v1/mcp/make_public", json={"mcp_server_ids": mcp_server_ids}
        )

    async def get_public_mcp_hub(self) -> Any:
        """Get the public MCP hub catalog (`GET /public/mcp_hub`)."""
        return await self._request("GET", "/public/mcp_hub")

    async def list_mcp_user_credentials(self) -> Any:
        """List the caller's stored MCP user credentials (`GET /v1/mcp/user-credentials`).

        Returns one entry per server the caller has provided credentials for;
        credential values are not included.
        """
        return await self._request("GET", "/v1/mcp/user-credentials")

    async def set_mcp_user_credential(
        self,
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

        Routes to `POST /v1/mcp/server/{server_id}/oauth-user-credential` when
        `oauth=True`, else `POST /v1/mcp/server/{server_id}/user-credential`.

        Non-oauth body (`MCPUserCredentialRequest`): `credential` (required),
        `save` (optional bool).

        OAuth body (`MCPOAuthUserCredentialRequest`): `access_token` (required),
        `refresh_token`, `expires_in`, `scopes`.
        """
        if oauth:
            body = self._build_body(
                {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                    "expires_in": expires_in,
                    "scopes": scopes,
                }
            )
            path = f"/v1/mcp/server/{server_id}/oauth-user-credential"
        else:
            body = self._build_body({"credential": credential, "save": save})
            path = f"/v1/mcp/server/{server_id}/user-credential"
        return await self._request("POST", path, json=body)

    async def delete_mcp_user_credential(self, server_id: str, oauth: bool = False) -> dict:
        """Delete the caller's user credential for an MCP server.

        Routes to `DELETE /v1/mcp/server/{server_id}/oauth-user-credential`
        when `oauth=True`, else `DELETE /v1/mcp/server/{server_id}/user-credential`.
        """
        suffix = "oauth-user-credential" if oauth else "user-credential"
        return await self._request("DELETE", f"/v1/mcp/server/{server_id}/{suffix}")

    async def get_mcp_oauth_user_credential_status(self, server_id: str) -> dict:
        """Get OAuth credential status for an MCP server (`GET /v1/mcp/server/{id}/oauth-user-credential/status`).

        Returns whether the caller has a valid OAuth credential for the server
        and (optionally) when it expires.
        """
        return await self._request(
            "GET", f"/v1/mcp/server/{server_id}/oauth-user-credential/status"
        )

    async def get_mcp_client_ip(self) -> dict:
        """Get the caller's resolved client IP (`GET /v1/mcp/network/client-ip`).

        Useful for diagnosing IP-allowlist / proxy-header issues without
        bouncing through an external service.
        """
        return await self._request("GET", "/v1/mcp/network/client-ip")

    # ── MCP Toolset operations ──
    #
    # Toolsets are named bundles of tools sourced from one or more registered
    # MCP servers (e.g. a `research` toolset combining `resolve-library-id`
    # from Context7 with `search` from another upstream). The proxy then
    # exposes each toolset as a brokered MCP endpoint at `/toolset/{name}/mcp`
    # — that transport route is permanent-skip (transport-level, not admin).

    async def list_mcp_toolsets(self) -> Any:
        """List defined MCP toolsets (`GET /v1/mcp/toolset`)."""
        return await self._request("GET", "/v1/mcp/toolset")

    async def get_mcp_toolset(self, toolset_id: str) -> dict:
        """Get a single MCP toolset by id (`GET /v1/mcp/toolset/{toolset_id}`)."""
        return await self._request("GET", f"/v1/mcp/toolset/{toolset_id}")

    async def add_mcp_toolset(
        self,
        toolset_name: str,
        tools: list[str],
        description: Optional[str] = None,
    ) -> dict:
        """Create an MCP toolset (`POST /v1/mcp/toolset`).

        Body shape per `NewMCPToolsetRequest`: `toolset_name` (required),
        `tools` (required list of tool identifiers — typically
        `<server_alias>/<tool_name>` strings), optional `description`.
        """
        body: dict[str, Any] = {"toolset_name": toolset_name, "tools": tools}
        if description is not None:
            body["description"] = description
        return await self._request("POST", "/v1/mcp/toolset", json=body)

    async def update_mcp_toolset(
        self,
        toolset_id: str,
        toolset_name: Optional[str] = None,
        description: Optional[str] = None,
        tools: Optional[list[str]] = None,
    ) -> dict:
        """Update an MCP toolset (`PUT /v1/mcp/toolset`).

        Like `update_mcp_server`, the id goes in the body (per
        `UpdateMCPToolsetRequest`), not the path. Only provided fields are sent.
        """
        body = self._build_body(
            {
                "toolset_id": toolset_id,
                "toolset_name": toolset_name,
                "description": description,
                "tools": tools,
            }
        )
        return await self._request("PUT", "/v1/mcp/toolset", json=body)

    async def delete_mcp_toolset(self, toolset_id: str) -> dict:
        """Delete an MCP toolset (`DELETE /v1/mcp/toolset/{toolset_id}`)."""
        return await self._request("DELETE", f"/v1/mcp/toolset/{toolset_id}")

    # ── Provider passthrough ──

    async def passthrough(
        self,
        provider: str,
        endpoint: str,
        method: str = "GET",
        body: Optional[dict] = None,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
    ) -> Any:
        """Proxy a request to a provider's native API via LiteLLM (`<METHOD> /<provider>/<endpoint>`).

        LiteLLM exposes 15+ providers' native APIs at `/<provider>/{endpoint:path}`
        with all 5 HTTP methods. One generic tool covers ~85 Swagger ops.

        Known providers: `anthropic`, `openai`, `vertex_ai`, `vertex_ai/discovery`,
        `gemini` (Google AI Studio), `cohere`, `vllm`, `mistral`, `milvus`,
        `bedrock`, `assemblyai`, `eu.assemblyai`, `azure`, `azure_ai`, `cursor`,
        `langfuse`. The provider list is fixed by what LiteLLM compiles in.

        Args:
            provider: provider identifier (path prefix). Forward slashes inside
                the provider value are preserved (e.g. `vertex_ai/discovery`).
            endpoint: everything after `/<provider>/` — e.g. `v1/models` or
                `v1/messages`. Leading slashes are stripped.
            method: HTTP method (`GET`, `POST`, `PUT`, `DELETE`, `PATCH`).
                Defaults to GET.
            body: optional JSON body (for POST/PUT/PATCH).
            params: optional query parameters.
            headers: optional headers to forward (e.g. `{"anthropic-version": "..."}`).
                The `Authorization` Bearer is set by the client and should not
                be overridden here.

        Returns:
            The upstream payload as-is — JSON if the response is JSON, else text.
            Streaming responses are not supported in this slice.
        """
        path = f"/{provider.strip('/')}/{endpoint.lstrip('/')}"
        return await self._request(method.upper(), path, params=params, json=body, headers=headers)
