"""Unit tests for LiteLLM MCP server."""

from unittest.mock import AsyncMock

import pytest

import src.server
from src.litellm_client import LiteLLMAPIError, LiteLLMClient


@pytest.fixture
def mock_client():
    """Inject an AsyncMock LiteLLMClient as the module-level client."""
    client = AsyncMock(spec=LiteLLMClient)
    original = src.server._client
    src.server._client = client
    yield client
    src.server._client = original


class TestListModels:
    @pytest.mark.asyncio
    async def test_returns_data_array(self, mock_client):
        mock_client.list_models.return_value = [
            {"id": "gpt-4o", "object": "model", "owned_by": "openai", "created": 0},
            {"id": "claude-opus-4-7", "object": "model", "owned_by": "anthropic", "created": 0},
        ]
        result = await src.server.list_models("standard")
        assert len(result) == 2
        assert result[0]["id"] == "gpt-4o"
        assert "owned_by" in result[0]
        mock_client.list_models.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_minimal_strips_fields(self, mock_client):
        mock_client.list_models.return_value = [
            {"id": "gpt-4o", "object": "model", "owned_by": "openai", "created": 1},
        ]
        result = await src.server.list_models("minimal")
        assert result == [{"id": "gpt-4o"}]

    @pytest.mark.asyncio
    async def test_full_passthrough(self, mock_client):
        raw = [{"id": "x", "object": "model", "owned_by": "y", "created": 0, "extra": True}]
        mock_client.list_models.return_value = raw
        result = await src.server.list_models("full")
        assert result == raw

    @pytest.mark.asyncio
    async def test_invalid_verbosity_falls_back(self, mock_client):
        mock_client.list_models.return_value = [
            {"id": "x", "object": "model", "owned_by": "y", "created": 0, "extra": True},
        ]
        result = await src.server.list_models("bogus")
        assert "extra" not in result[0]
        assert result[0]["id"] == "x"


class TestClientGuard:
    def test_get_client_unset_raises(self):
        original = src.server._client
        src.server._client = None
        try:
            with pytest.raises(RuntimeError):
                src.server.get_client()
        finally:
            src.server._client = original


class TestErrors:
    def test_litellm_api_error_carries_status(self):
        err = LiteLLMAPIError("nope", 401)
        assert err.status_code == 401
        assert "nope" in str(err)
