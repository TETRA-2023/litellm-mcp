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

        The CredentialItem schema requires all three fields. Path id and body
        `credential_name` should match.
        """
        body = {
            "credential_name": credential_name,
            "credential_info": credential_info,
            "credential_values": credential_values,
        }
        return await self._request(
            "PATCH", f"/credentials/{credential_name}", json=body
        )

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
