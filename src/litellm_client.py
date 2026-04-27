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
