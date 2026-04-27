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
    ) -> Any:
        attempt = 0
        last_exc: Optional[Exception] = None
        while attempt <= self.max_retries:
            try:
                response = await self._client.request(method, path, params=params, json=json)
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
    # Wraps `/v1/unified_access_group/*`. Distinct from model-access-groups
    # (`/access_group/*`, in #535) — unified access groups gate users/teams
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
