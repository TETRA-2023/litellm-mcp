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


class TestUpdateModelHubLinks:
    @pytest.mark.asyncio
    async def test_passes_links(self, mock_client):
        mock_client.update_model_hub_links.return_value = {"ok": True}
        await src.server.update_model_hub_links({"Docs": "https://example.com"})
        mock_client.update_model_hub_links.assert_awaited_once_with(
            {"Docs": "https://example.com"}
        )


class TestListModelAccessGroups:
    @pytest.mark.asyncio
    async def test_passthrough(self, mock_client):
        payload = [{"access_group": "engineering", "models": ["gpt-4o"]}]
        mock_client.list_model_access_groups.return_value = payload
        result = await src.server.list_model_access_groups("full")
        assert result == payload
        mock_client.list_model_access_groups.assert_awaited_once()


class TestGetModelAccessGroup:
    @pytest.mark.asyncio
    async def test_passes_access_group(self, mock_client):
        mock_client.get_model_access_group.return_value = {
            "access_group": "engineering",
            "models": ["gpt-4o"],
        }
        result = await src.server.get_model_access_group("engineering", "full")
        assert result["access_group"] == "engineering"
        mock_client.get_model_access_group.assert_awaited_once_with("engineering")


class TestCreateModelAccessGroup:
    @pytest.mark.asyncio
    async def test_minimum_required(self, mock_client):
        mock_client.create_model_access_group.return_value = {"ok": True}
        await src.server.create_model_access_group("engineering")
        mock_client.create_model_access_group.assert_awaited_once_with(
            "engineering", None, None
        )

    @pytest.mark.asyncio
    async def test_with_members(self, mock_client):
        mock_client.create_model_access_group.return_value = {"ok": True}
        await src.server.create_model_access_group(
            "engineering", model_names=["gpt-4o"], model_ids=["abc-123"]
        )
        mock_client.create_model_access_group.assert_awaited_once_with(
            "engineering", ["gpt-4o"], ["abc-123"]
        )


class TestUpdateModelAccessGroup:
    @pytest.mark.asyncio
    async def test_replaces_models(self, mock_client):
        mock_client.update_model_access_group.return_value = {"ok": True}
        await src.server.update_model_access_group(
            "engineering", model_names=["claude-opus-4-7"]
        )
        mock_client.update_model_access_group.assert_awaited_once_with(
            "engineering", ["claude-opus-4-7"], None
        )


class TestDeleteModelAccessGroup:
    @pytest.mark.asyncio
    async def test_passes_access_group(self, mock_client):
        mock_client.delete_model_access_group.return_value = {"deleted": True}
        await src.server.delete_model_access_group("engineering")
        mock_client.delete_model_access_group.assert_awaited_once_with("engineering")


class TestListCredentials:
    @pytest.mark.asyncio
    async def test_passthrough(self, mock_client):
        payload = [{"credential_name": "openai-prod", "credential_info": {}}]
        mock_client.list_credentials.return_value = payload
        result = await src.server.list_credentials("full")
        assert result == payload
        mock_client.list_credentials.assert_awaited_once()


class TestGetCredential:
    @pytest.mark.asyncio
    async def test_passes_credential_name(self, mock_client):
        mock_client.get_credential.return_value = {
            "credential_name": "openai-prod",
            "credential_info": {},
        }
        result = await src.server.get_credential("openai-prod", "full")
        assert result["credential_name"] == "openai-prod"
        mock_client.get_credential.assert_awaited_once_with("openai-prod")


class TestGetCredentialByModel:
    @pytest.mark.asyncio
    async def test_passes_model_id(self, mock_client):
        mock_client.get_credential_by_model.return_value = {
            "credential_name": "openai-prod",
            "credential_info": {},
        }
        result = await src.server.get_credential_by_model("abc-123", "full")
        assert result["credential_name"] == "openai-prod"
        mock_client.get_credential_by_model.assert_awaited_once_with("abc-123")


class TestCreateCredential:
    @pytest.mark.asyncio
    async def test_with_values(self, mock_client):
        mock_client.create_credential.return_value = {"ok": True}
        await src.server.create_credential(
            "openai-prod",
            {"custom_llm_provider": "openai"},
            credential_values={"api_key": "sk-x"},
        )
        mock_client.create_credential.assert_awaited_once_with(
            "openai-prod",
            {"custom_llm_provider": "openai"},
            {"api_key": "sk-x"},
            None,
        )

    @pytest.mark.asyncio
    async def test_with_model_id(self, mock_client):
        mock_client.create_credential.return_value = {"ok": True}
        await src.server.create_credential(
            "openai-prod", {"custom_llm_provider": "openai"}, model_id="abc-123"
        )
        mock_client.create_credential.assert_awaited_once_with(
            "openai-prod", {"custom_llm_provider": "openai"}, None, "abc-123"
        )


class TestUpdateCredential:
    @pytest.mark.asyncio
    async def test_passes_full_body(self, mock_client):
        mock_client.update_credential.return_value = {"ok": True}
        await src.server.update_credential(
            "openai-prod",
            {"custom_llm_provider": "openai"},
            {"api_key": "sk-y"},
        )
        mock_client.update_credential.assert_awaited_once_with(
            "openai-prod",
            {"custom_llm_provider": "openai"},
            {"api_key": "sk-y"},
        )


class TestDeleteCredential:
    @pytest.mark.asyncio
    async def test_passes_credential_name(self, mock_client):
        mock_client.delete_credential.return_value = {"deleted": True}
        await src.server.delete_credential("openai-prod")
        mock_client.delete_credential.assert_awaited_once_with("openai-prod")


class TestListKeys:
    @pytest.mark.asyncio
    async def test_no_filters(self, mock_client):
        mock_client.list_keys.return_value = {"keys": [], "total_count": 0}
        result = await src.server.list_keys()
        assert result == {"keys": [], "total_count": 0}
        mock_client.list_keys.assert_awaited_once_with(
            None, None, None, None, None, None, None, None, None, None
        )

    @pytest.mark.asyncio
    async def test_filter_by_team(self, mock_client):
        mock_client.list_keys.return_value = {"keys": [{"key_name": "sk-x"}]}
        await src.server.list_keys(team_id="t-1", page=2)
        mock_client.list_keys.assert_awaited_once_with(
            2, None, None, "t-1", None, None, None, None, None, None
        )


class TestListKeyAliases:
    @pytest.mark.asyncio
    async def test_search_filter(self, mock_client):
        mock_client.list_key_aliases.return_value = {"data": []}
        await src.server.list_key_aliases(search="prod")
        mock_client.list_key_aliases.assert_awaited_once_with(None, None, "prod", None)


class TestGetKeyInfo:
    @pytest.mark.asyncio
    async def test_no_key(self, mock_client):
        mock_client.get_key_info.return_value = {"key_name": "sk-x", "spend": 0}
        await src.server.get_key_info()
        mock_client.get_key_info.assert_awaited_once_with(None)

    @pytest.mark.asyncio
    async def test_with_key(self, mock_client):
        mock_client.get_key_info.return_value = {"key_name": "sk-x"}
        await src.server.get_key_info(key="sk-x", verbosity="full")
        mock_client.get_key_info.assert_awaited_once_with("sk-x")


class TestGenerateKey:
    @pytest.mark.asyncio
    async def test_minimal(self, mock_client):
        mock_client.generate_key.return_value = {"key": "sk-123", "key_name": "auto"}
        result = await src.server.generate_key()
        assert result["key"] == "sk-123"

    @pytest.mark.asyncio
    async def test_common_args(self, mock_client):
        mock_client.generate_key.return_value = {"key": "sk-x"}
        await src.server.generate_key(
            key_alias="prod-bot",
            duration="30d",
            models=["gpt-4o"],
            max_budget=10.0,
            team_id="t-1",
        )
        mock_client.generate_key.assert_awaited_once_with(
            "prod-bot", "30d", ["gpt-4o"], 10.0, None,
            None, "t-1", None, None, None, None, None, None,
        )

    @pytest.mark.asyncio
    async def test_extras_merged(self, mock_client):
        mock_client.generate_key.return_value = {"key": "sk-x"}
        await src.server.generate_key(
            key_alias="x",
            extras={"agent_id": "a-1", "max_parallel_requests": 4},
        )
        call = mock_client.generate_key.await_args
        assert call.args[0] == "x"
        assert call.args[12] == {"agent_id": "a-1", "max_parallel_requests": 4}


class TestGenerateServiceAccountKey:
    @pytest.mark.asyncio
    async def test_passes_args(self, mock_client):
        mock_client.generate_service_account_key.return_value = {"key": "sk-svc"}
        await src.server.generate_service_account_key(
            key_alias="ci-bot", duration="365d", team_id="t-1"
        )
        mock_client.generate_service_account_key.assert_awaited_once_with(
            "ci-bot", "365d", None, None, None,
            None, "t-1", None, None, None, None, None, None,
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
