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


class TestGetModel:
    @pytest.mark.asyncio
    async def test_returns_model_dict(self, mock_client):
        mock_client.get_model.return_value = {
            "id": "gpt-4o",
            "object": "model",
            "owned_by": "openai",
            "created": 0,
        }
        result = await src.server.get_model("gpt-4o", "standard")
        assert result["id"] == "gpt-4o"
        mock_client.get_model.assert_awaited_once_with("gpt-4o")

    @pytest.mark.asyncio
    async def test_minimal_strips_fields(self, mock_client):
        mock_client.get_model.return_value = {
            "id": "gpt-4o",
            "object": "model",
            "owned_by": "openai",
            "created": 1,
        }
        result = await src.server.get_model("gpt-4o", "minimal")
        assert result == {"id": "gpt-4o"}


class TestGetModelInfo:
    @pytest.mark.asyncio
    async def test_passthrough_no_filter(self, mock_client):
        payload = {"data": [{"model_name": "gpt-4o", "litellm_params": {"model": "openai/gpt-4o"}}]}
        mock_client.get_model_info.return_value = payload
        result = await src.server.get_model_info()
        assert result == payload
        mock_client.get_model_info.assert_awaited_once_with(None)

    @pytest.mark.asyncio
    async def test_passes_litellm_model_id(self, mock_client):
        mock_client.get_model_info.return_value = {"data": []}
        await src.server.get_model_info("abc-123")
        mock_client.get_model_info.assert_awaited_once_with("abc-123")


class TestAddModel:
    @pytest.mark.asyncio
    async def test_passes_full_body(self, mock_client):
        mock_client.add_model.return_value = {"model_id": "abc-123"}
        result = await src.server.add_model(
            "gpt-4o",
            {"model": "openai/gpt-4o", "api_key": "sk-x"},
            {"id": "abc-123"},
        )
        assert result == {"model_id": "abc-123"}
        mock_client.add_model.assert_awaited_once_with(
            "gpt-4o",
            {"model": "openai/gpt-4o", "api_key": "sk-x"},
            {"id": "abc-123"},
        )


class TestUpdateModel:
    @pytest.mark.asyncio
    async def test_only_passes_set_fields(self, mock_client):
        mock_client.update_model.return_value = {"ok": True}
        await src.server.update_model("abc-123", model_name="gpt-4o-mini")
        mock_client.update_model.assert_awaited_once_with("abc-123", "gpt-4o-mini", None, None)

    @pytest.mark.asyncio
    async def test_passes_all_fields(self, mock_client):
        mock_client.update_model.return_value = {"ok": True}
        await src.server.update_model(
            "abc-123",
            model_name="x",
            litellm_params={"model": "y"},
            model_info={"id": "abc-123"},
        )
        mock_client.update_model.assert_awaited_once_with(
            "abc-123", "x", {"model": "y"}, {"id": "abc-123"}
        )


class TestDeleteModel:
    @pytest.mark.asyncio
    async def test_passes_model_id(self, mock_client):
        mock_client.delete_model.return_value = {"deleted": True}
        result = await src.server.delete_model("abc-123")
        assert result == {"deleted": True}
        mock_client.delete_model.assert_awaited_once_with("abc-123")


class TestListPublicModels:
    @pytest.mark.asyncio
    async def test_passthrough_unfiltered(self, mock_client):
        payload = [{"model_group": "gpt-4o", "providers": ["openai"]}]
        mock_client.list_public_models.return_value = payload
        result = await src.server.list_public_models("full")
        assert result == payload
        mock_client.list_public_models.assert_awaited_once()


class TestGetPublicHubInfo:
    @pytest.mark.asyncio
    async def test_passthrough(self, mock_client):
        payload = {"title": "Tetra Models", "description": "...", "useful_links": {}}
        mock_client.get_public_hub_info.return_value = payload
        result = await src.server.get_public_hub_info()
        assert result == payload
        mock_client.get_public_hub_info.assert_awaited_once()


class TestGetModelCostMap:
    @pytest.mark.asyncio
    async def test_passthrough(self, mock_client):
        payload = {"gpt-4o": {"input_cost_per_token": 1e-6}}
        mock_client.get_model_cost_map.return_value = payload
        result = await src.server.get_model_cost_map()
        assert result == payload
        mock_client.get_model_cost_map.assert_awaited_once()


class TestMakeModelGroupPublic:
    @pytest.mark.asyncio
    async def test_passes_model_groups(self, mock_client):
        mock_client.make_model_group_public.return_value = {"ok": True}
        await src.server.make_model_group_public(["gpt-4o", "claude-opus-4-7"])
        mock_client.make_model_group_public.assert_awaited_once_with(
            ["gpt-4o", "claude-opus-4-7"]
        )


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
