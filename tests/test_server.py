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
        mock_client.make_model_group_public.assert_awaited_once_with(["gpt-4o", "claude-opus-4-7"])


class TestUpdateModelHubLinks:
    @pytest.mark.asyncio
    async def test_passes_links(self, mock_client):
        mock_client.update_model_hub_links.return_value = {"ok": True}
        await src.server.update_model_hub_links({"Docs": "https://example.com"})
        mock_client.update_model_hub_links.assert_awaited_once_with({"Docs": "https://example.com"})


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
        mock_client.create_model_access_group.assert_awaited_once_with("engineering", None, None)

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
        await src.server.update_model_access_group("engineering", model_names=["claude-opus-4-7"])
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
            "prod-bot",
            "30d",
            ["gpt-4o"],
            10.0,
            None,
            None,
            "t-1",
            None,
            None,
            None,
            None,
            None,
            None,
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
            "ci-bot",
            "365d",
            None,
            None,
            None,
            None,
            "t-1",
            None,
            None,
            None,
            None,
            None,
            None,
        )


class TestUpdateKey:
    @pytest.mark.asyncio
    async def test_minimal(self, mock_client):
        mock_client.update_key.return_value = {"key": "sk-x"}
        await src.server.update_key("sk-x", max_budget=50.0)
        mock_client.update_key.assert_awaited_once_with(
            "sk-x", None, None, None, 50.0, None, None, None, None, None, None, None
        )

    @pytest.mark.asyncio
    async def test_extras(self, mock_client):
        mock_client.update_key.return_value = {"key": "sk-x"}
        await src.server.update_key("sk-x", extras={"max_parallel_requests": 8})
        call = mock_client.update_key.await_args
        assert call.args[0] == "sk-x"
        assert call.args[11] == {"max_parallel_requests": 8}


class TestRegenerateKey:
    @pytest.mark.asyncio
    async def test_no_body(self, mock_client):
        mock_client.regenerate_key.return_value = {"key": "sk-new"}
        await src.server.regenerate_key("sk-old")
        mock_client.regenerate_key.assert_awaited_once_with("sk-old", None, None, None)

    @pytest.mark.asyncio
    async def test_with_extras(self, mock_client):
        mock_client.regenerate_key.return_value = {"key": "sk-new"}
        await src.server.regenerate_key("sk-old", extras={"max_budget": 100.0})
        mock_client.regenerate_key.assert_awaited_once_with(
            "sk-old", None, None, {"max_budget": 100.0}
        )


class TestSetKeyBlocked:
    @pytest.mark.asyncio
    async def test_block(self, mock_client):
        mock_client.set_key_blocked.return_value = {"ok": True}
        await src.server.set_key_blocked("sk-x", True)
        mock_client.set_key_blocked.assert_awaited_once_with("sk-x", True)

    @pytest.mark.asyncio
    async def test_unblock(self, mock_client):
        mock_client.set_key_blocked.return_value = {"ok": True}
        await src.server.set_key_blocked("sk-x", False)
        mock_client.set_key_blocked.assert_awaited_once_with("sk-x", False)


class TestDeleteKeys:
    @pytest.mark.asyncio
    async def test_by_keys(self, mock_client):
        mock_client.delete_keys.return_value = {"deleted": 2}
        await src.server.delete_keys(keys=["sk-1", "sk-2"])
        mock_client.delete_keys.assert_awaited_once_with(["sk-1", "sk-2"], None)

    @pytest.mark.asyncio
    async def test_by_aliases(self, mock_client):
        mock_client.delete_keys.return_value = {"deleted": 1}
        await src.server.delete_keys(key_aliases=["prod-bot"])
        mock_client.delete_keys.assert_awaited_once_with(None, ["prod-bot"])


class TestResetKeySpend:
    @pytest.mark.asyncio
    async def test_default_zero(self, mock_client):
        mock_client.reset_key_spend.return_value = {"ok": True}
        await src.server.reset_key_spend("sk-x")
        mock_client.reset_key_spend.assert_awaited_once_with("sk-x", 0.0)

    @pytest.mark.asyncio
    async def test_explicit_value(self, mock_client):
        mock_client.reset_key_spend.return_value = {"ok": True}
        await src.server.reset_key_spend("sk-x", 5.5)
        mock_client.reset_key_spend.assert_awaited_once_with("sk-x", 5.5)


class TestKeyHealth:
    @pytest.mark.asyncio
    async def test_no_args(self, mock_client):
        mock_client.key_health.return_value = {"status": "healthy"}
        result = await src.server.key_health()
        assert result == {"status": "healthy"}
        mock_client.key_health.assert_awaited_once_with()


class TestListUsers:
    @pytest.mark.asyncio
    async def test_no_filters(self, mock_client):
        mock_client.list_users.return_value = {"users": [], "total_count": 0}
        result = await src.server.list_users()
        assert result == {"users": [], "total_count": 0}
        mock_client.list_users.assert_awaited_once_with(
            None, None, None, None, None, None, None, None, None, None
        )

    @pytest.mark.asyncio
    async def test_filters_users_by_verbosity(self, mock_client):
        mock_client.list_users.return_value = {
            "users": [
                {
                    "user_id": "u-1",
                    "user_email": "a@x.com",
                    "user_role": "internal_user",
                    "spend": 0.0,
                    "extra": True,
                }
            ]
        }
        result = await src.server.list_users(verbosity="minimal")
        assert result["users"] == [{"user_id": "u-1", "user_email": "a@x.com"}]

    @pytest.mark.asyncio
    async def test_filters_list_payload(self, mock_client):
        mock_client.list_users.return_value = [
            {"user_id": "u-1", "user_email": "a@x.com", "extra": True}
        ]
        result = await src.server.list_users(verbosity="minimal")
        assert result == [{"user_id": "u-1", "user_email": "a@x.com"}]


class TestGetUserInfo:
    @pytest.mark.asyncio
    async def test_no_user_id(self, mock_client):
        mock_client.get_user_info.return_value = {"user_id": "u-self", "user_email": "me@x.com"}
        await src.server.get_user_info()
        mock_client.get_user_info.assert_awaited_once_with(None)

    @pytest.mark.asyncio
    async def test_with_user_id(self, mock_client):
        mock_client.get_user_info.return_value = {
            "user_id": "u-1",
            "user_email": "a@x.com",
            "user_role": "internal_user",
        }
        result = await src.server.get_user_info("u-1", "minimal")
        assert result == {"user_id": "u-1", "user_email": "a@x.com"}
        mock_client.get_user_info.assert_awaited_once_with("u-1")


class TestCreateUser:
    @pytest.mark.asyncio
    async def test_minimal(self, mock_client):
        mock_client.create_user.return_value = {"user_id": "u-1"}
        await src.server.create_user(user_email="a@x.com")
        call = mock_client.create_user.await_args
        assert call.args[0] == "a@x.com"

    @pytest.mark.asyncio
    async def test_extras_passthrough(self, mock_client):
        mock_client.create_user.return_value = {"user_id": "u-1"}
        await src.server.create_user(
            user_email="a@x.com", extras={"permissions": {"can_create_keys": True}}
        )
        call = mock_client.create_user.await_args
        assert call.args[16] == {"permissions": {"can_create_keys": True}}


class TestUpdateUser:
    @pytest.mark.asyncio
    async def test_required_user_id(self, mock_client):
        mock_client.update_user.return_value = {"user_id": "u-1"}
        await src.server.update_user("u-1", max_budget=50.0)
        mock_client.update_user.assert_awaited_once_with(
            "u-1", None, None, None, None, None, 50.0, None, None, None, None, None, None, None
        )


class TestDeleteUser:
    @pytest.mark.asyncio
    async def test_passes_ids(self, mock_client):
        mock_client.delete_user.return_value = {"deleted": 2}
        await src.server.delete_user(["u-1", "u-2"])
        mock_client.delete_user.assert_awaited_once_with(["u-1", "u-2"])


class TestListCustomers:
    @pytest.mark.asyncio
    async def test_passthrough_full(self, mock_client):
        payload = [{"user_id": "c-1", "alias": "Acme", "extra": True}]
        mock_client.list_customers.return_value = payload
        assert await src.server.list_customers("full") == payload

    @pytest.mark.asyncio
    async def test_minimal_filters(self, mock_client):
        mock_client.list_customers.return_value = [
            {"user_id": "c-1", "alias": "Acme", "max_budget": 100.0}
        ]
        result = await src.server.list_customers("minimal")
        assert result == [{"user_id": "c-1", "alias": "Acme"}]


class TestGetCustomerInfo:
    @pytest.mark.asyncio
    async def test_passes_end_user_id(self, mock_client):
        mock_client.get_customer_info.return_value = {"user_id": "c-1", "alias": "Acme"}
        await src.server.get_customer_info("c-1", "full")
        mock_client.get_customer_info.assert_awaited_once_with("c-1")


class TestCreateCustomer:
    @pytest.mark.asyncio
    async def test_required_user_id(self, mock_client):
        mock_client.create_customer.return_value = {"user_id": "c-1"}
        await src.server.create_customer("c-1", alias="Acme", max_budget=100.0)
        call = mock_client.create_customer.await_args
        assert call.args[0] == "c-1"
        assert call.args[1] == "Acme"
        assert call.args[2] == 100.0


class TestUpdateCustomer:
    @pytest.mark.asyncio
    async def test_minimal(self, mock_client):
        mock_client.update_customer.return_value = {"user_id": "c-1"}
        await src.server.update_customer("c-1", max_budget=200.0)
        mock_client.update_customer.assert_awaited_once_with(
            "c-1", None, 200.0, None, None, None, None, None
        )


class TestDeleteCustomer:
    @pytest.mark.asyncio
    async def test_passes_ids(self, mock_client):
        mock_client.delete_customer.return_value = {"deleted": 1}
        await src.server.delete_customer(["c-1"])
        mock_client.delete_customer.assert_awaited_once_with(["c-1"])


class TestSetCustomerBlocked:
    @pytest.mark.asyncio
    async def test_block(self, mock_client):
        mock_client.set_customer_blocked.return_value = {"ok": True}
        await src.server.set_customer_blocked(["c-1"], True)
        mock_client.set_customer_blocked.assert_awaited_once_with(["c-1"], True)

    @pytest.mark.asyncio
    async def test_unblock(self, mock_client):
        mock_client.set_customer_blocked.return_value = {"ok": True}
        await src.server.set_customer_blocked(["c-1", "c-2"], False)
        mock_client.set_customer_blocked.assert_awaited_once_with(["c-1", "c-2"], False)


class TestGetCustomerDailyActivity:
    @pytest.mark.asyncio
    async def test_passes_dates(self, mock_client):
        mock_client.get_customer_daily_activity.return_value = {"data": []}
        await src.server.get_customer_daily_activity(
            start_date="2026-04-01", end_date="2026-04-30", page=1
        )
        mock_client.get_customer_daily_activity.assert_awaited_once_with(
            None, "2026-04-01", "2026-04-30", None, None, 1, None, None
        )


class TestListOrganizations:
    @pytest.mark.asyncio
    async def test_no_filters(self, mock_client):
        mock_client.list_organizations.return_value = [{"organization_id": "o-1"}]
        await src.server.list_organizations()
        mock_client.list_organizations.assert_awaited_once_with(None, None)

    @pytest.mark.asyncio
    async def test_filter_by_alias(self, mock_client):
        mock_client.list_organizations.return_value = []
        await src.server.list_organizations(org_alias="acme")
        mock_client.list_organizations.assert_awaited_once_with(None, "acme")


class TestGetOrganizationInfo:
    @pytest.mark.asyncio
    async def test_passes_id(self, mock_client):
        mock_client.get_organization_info.return_value = {
            "organization_id": "o-1",
            "organization_alias": "Acme",
        }
        result = await src.server.get_organization_info("o-1", "minimal")
        assert result == {"organization_id": "o-1", "organization_alias": "Acme"}


class TestCreateOrganization:
    @pytest.mark.asyncio
    async def test_required_alias(self, mock_client):
        mock_client.create_organization.return_value = {"organization_id": "o-1"}
        await src.server.create_organization("Acme", max_budget=1000.0)
        call = mock_client.create_organization.await_args
        assert call.args[0] == "Acme"
        assert call.args[3] == 1000.0


class TestUpdateOrganization:
    @pytest.mark.asyncio
    async def test_required_id(self, mock_client):
        mock_client.update_organization.return_value = {"ok": True}
        await src.server.update_organization("o-1", max_budget=2000.0)
        mock_client.update_organization.assert_awaited_once_with(
            "o-1", None, None, 2000.0, None, None, None, None, None
        )

    @pytest.mark.asyncio
    async def test_extras_merged(self, mock_client):
        mock_client.update_organization.return_value = {"ok": True}
        await src.server.update_organization("o-1", extras={"object_permission": {}})
        call = mock_client.update_organization.await_args
        assert call.args[8] == {"object_permission": {}}


class TestDeleteOrganization:
    @pytest.mark.asyncio
    async def test_passes_ids(self, mock_client):
        mock_client.delete_organization.return_value = {"deleted": 1}
        await src.server.delete_organization(["o-1"])
        mock_client.delete_organization.assert_awaited_once_with(["o-1"])


class TestAddOrgMember:
    @pytest.mark.asyncio
    async def test_minimal(self, mock_client):
        mock_client.add_org_member.return_value = {"ok": True}
        await src.server.add_org_member("o-1", {"user_id": "u-1", "role": "internal_user"})
        mock_client.add_org_member.assert_awaited_once_with(
            "o-1", {"user_id": "u-1", "role": "internal_user"}, None
        )

    @pytest.mark.asyncio
    async def test_with_budget(self, mock_client):
        mock_client.add_org_member.return_value = {"ok": True}
        await src.server.add_org_member("o-1", {"user_email": "a@x.com", "role": "org_admin"}, 50.0)
        mock_client.add_org_member.assert_awaited_once_with(
            "o-1", {"user_email": "a@x.com", "role": "org_admin"}, 50.0
        )


class TestUpdateOrgMember:
    @pytest.mark.asyncio
    async def test_role_change(self, mock_client):
        mock_client.update_org_member.return_value = {"ok": True}
        await src.server.update_org_member("o-1", user_id="u-1", role="org_admin")
        mock_client.update_org_member.assert_awaited_once_with(
            "o-1", "u-1", None, "org_admin", None
        )


class TestDeleteOrgMember:
    @pytest.mark.asyncio
    async def test_by_email(self, mock_client):
        mock_client.delete_org_member.return_value = {"ok": True}
        await src.server.delete_org_member("o-1", user_email="a@x.com")
        mock_client.delete_org_member.assert_awaited_once_with("o-1", None, "a@x.com")


class TestGetOrgDailyActivity:
    @pytest.mark.asyncio
    async def test_passes_filters(self, mock_client):
        mock_client.get_org_daily_activity.return_value = {"data": []}
        await src.server.get_org_daily_activity(organization_ids="o-1,o-2", start_date="2026-04-01")
        mock_client.get_org_daily_activity.assert_awaited_once_with(
            "o-1,o-2", "2026-04-01", None, None, None, None, None, None
        )


class TestListProjects:
    @pytest.mark.asyncio
    async def test_passthrough(self, mock_client):
        mock_client.list_projects.return_value = [{"project_id": "p-1"}]
        result = await src.server.list_projects("full")
        assert result == [{"project_id": "p-1"}]


class TestGetProjectInfo:
    @pytest.mark.asyncio
    async def test_passes_id(self, mock_client):
        mock_client.get_project_info.return_value = {
            "project_id": "p-1",
            "project_alias": "Phoenix",
        }
        result = await src.server.get_project_info("p-1", "minimal")
        assert result == {"project_id": "p-1", "project_alias": "Phoenix"}


class TestCreateProject:
    @pytest.mark.asyncio
    async def test_required_team_id(self, mock_client):
        mock_client.create_project.return_value = {"project_id": "p-1"}
        await src.server.create_project("t-1", project_alias="Phoenix", max_budget=500.0)
        call = mock_client.create_project.await_args
        assert call.args[0] == "t-1"
        assert call.args[2] == "Phoenix"
        assert call.args[5] == 500.0


class TestUpdateProject:
    @pytest.mark.asyncio
    async def test_required_id(self, mock_client):
        mock_client.update_project.return_value = {"ok": True}
        await src.server.update_project("p-1", max_budget=1000.0)
        mock_client.update_project.assert_awaited_once_with(
            "p-1", None, None, None, 1000.0, None, None, None, None, None, None, None, None
        )


class TestDeleteProject:
    @pytest.mark.asyncio
    async def test_passes_ids(self, mock_client):
        mock_client.delete_project.return_value = {"deleted": 1}
        await src.server.delete_project(["p-1", "p-2"])
        mock_client.delete_project.assert_awaited_once_with(["p-1", "p-2"])


class TestListUserAccessGroups:
    @pytest.mark.asyncio
    async def test_passthrough(self, mock_client):
        mock_client.list_user_access_groups.return_value = [
            {"access_group_id": "ag-1", "access_group_name": "engineering"}
        ]
        result = await src.server.list_user_access_groups("full")
        assert result == [{"access_group_id": "ag-1", "access_group_name": "engineering"}]


class TestGetUserAccessGroup:
    @pytest.mark.asyncio
    async def test_passes_id(self, mock_client):
        mock_client.get_user_access_group.return_value = {
            "access_group_id": "ag-1",
            "access_group_name": "engineering",
        }
        result = await src.server.get_user_access_group("ag-1", "minimal")
        assert result == {"access_group_id": "ag-1", "access_group_name": "engineering"}


class TestCreateUserAccessGroup:
    @pytest.mark.asyncio
    async def test_minimal(self, mock_client):
        mock_client.create_user_access_group.return_value = {"access_group_id": "ag-1"}
        await src.server.create_user_access_group("engineering")
        mock_client.create_user_access_group.assert_awaited_once_with(
            "engineering", None, None, None, None, None, None
        )

    @pytest.mark.asyncio
    async def test_with_members(self, mock_client):
        mock_client.create_user_access_group.return_value = {"access_group_id": "ag-1"}
        await src.server.create_user_access_group(
            "engineering",
            description="eng team",
            access_model_names=["gpt-4o"],
            assigned_team_ids=["t-1"],
        )
        mock_client.create_user_access_group.assert_awaited_once_with(
            "engineering", "eng team", ["gpt-4o"], None, None, ["t-1"], None
        )


class TestUpdateUserAccessGroup:
    @pytest.mark.asyncio
    async def test_replace_models(self, mock_client):
        mock_client.update_user_access_group.return_value = {"ok": True}
        await src.server.update_user_access_group("ag-1", access_model_names=["claude-opus-4-7"])
        mock_client.update_user_access_group.assert_awaited_once_with(
            "ag-1", None, None, ["claude-opus-4-7"], None, None, None, None
        )


class TestDeleteUserAccessGroup:
    @pytest.mark.asyncio
    async def test_passes_id(self, mock_client):
        mock_client.delete_user_access_group.return_value = {"deleted": True}
        await src.server.delete_user_access_group("ag-1")
        mock_client.delete_user_access_group.assert_awaited_once_with("ag-1")


# ── Budget tools ──


class TestListBudgets:
    @pytest.mark.asyncio
    async def test_minimal_strips_fields(self, mock_client):
        mock_client.list_budgets.return_value = [
            {"budget_id": "b-1", "max_budget": 100.0, "tpm_limit": 1000},
        ]
        result = await src.server.list_budgets("minimal")
        assert result == [{"budget_id": "b-1"}]


class TestGetBudgetInfo:
    @pytest.mark.asyncio
    async def test_passes_list(self, mock_client):
        mock_client.get_budget_info.return_value = [{"budget_id": "b-1", "max_budget": 50.0}]
        result = await src.server.get_budget_info(["b-1"], "standard")
        mock_client.get_budget_info.assert_awaited_once_with(["b-1"])
        assert result[0]["budget_id"] == "b-1"


class TestCreateBudget:
    @pytest.mark.asyncio
    async def test_all_fields_optional(self, mock_client):
        mock_client.create_budget.return_value = {"budget_id": "b-1"}
        await src.server.create_budget(max_budget=100.0, budget_duration="30d")
        mock_client.create_budget.assert_awaited_once_with(
            None, 100.0, None, None, None, None, "30d", None, None, None
        )


class TestUpdateBudget:
    @pytest.mark.asyncio
    async def test_required_id(self, mock_client):
        mock_client.update_budget.return_value = {"ok": True}
        await src.server.update_budget("b-1", max_budget=200.0)
        mock_client.update_budget.assert_awaited_once_with(
            "b-1", 200.0, None, None, None, None, None, None, None, None
        )


class TestDeleteBudget:
    @pytest.mark.asyncio
    async def test_passes_id(self, mock_client):
        mock_client.delete_budget.return_value = {"deleted": True}
        await src.server.delete_budget("b-1")
        mock_client.delete_budget.assert_awaited_once_with("b-1")


class TestGetBudgetSettings:
    @pytest.mark.asyncio
    async def test_passes_id(self, mock_client):
        mock_client.get_budget_settings.return_value = {"budget_id": "b-1"}
        await src.server.get_budget_settings("b-1")
        mock_client.get_budget_settings.assert_awaited_once_with("b-1")


# ── Spend tools ──


class TestGetGlobalSpendReport:
    @pytest.mark.asyncio
    async def test_filters_passed(self, mock_client):
        mock_client.get_global_spend_report.return_value = []
        await src.server.get_global_spend_report(
            start_date="2026-04-01", end_date="2026-04-27", group_by="team"
        )
        mock_client.get_global_spend_report.assert_awaited_once_with(
            "2026-04-01", "2026-04-27", "team", None, None, None, None
        )


class TestListSpendLogs:
    @pytest.mark.asyncio
    async def test_filters_to_spend_record(self, mock_client):
        mock_client.list_spend_logs.return_value = [
            {
                "request_id": "r-1",
                "model": "gpt-4o",
                "spend": 0.012,
                "total_tokens": 100,
                "extra": "drop",
            }
        ]
        result = await src.server.list_spend_logs(api_key="sk-x", verbosity="standard")
        assert result == [
            {"request_id": "r-1", "model": "gpt-4o", "spend": 0.012, "total_tokens": 100}
        ]
        mock_client.list_spend_logs.assert_awaited_once_with("sk-x", None, None, None, None, None)

    @pytest.mark.asyncio
    async def test_minimal(self, mock_client):
        mock_client.list_spend_logs.return_value = [
            {"request_id": "r-1", "model": "gpt-4o", "spend": 0.012}
        ]
        result = await src.server.list_spend_logs(verbosity="minimal")
        assert result == [{"request_id": "r-1", "model": "gpt-4o"}]


class TestListSpendTags:
    @pytest.mark.asyncio
    async def test_passthrough(self, mock_client):
        mock_client.list_spend_tags.return_value = ["tag-a", "tag-b"]
        result = await src.server.list_spend_tags()
        assert result == ["tag-a", "tag-b"]


class TestCalculateSpend:
    @pytest.mark.asyncio
    async def test_prospective(self, mock_client):
        mock_client.calculate_spend.return_value = {"cost": 0.0023}
        await src.server.calculate_spend(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}]
        )
        mock_client.calculate_spend.assert_awaited_once_with(
            "gpt-4o", [{"role": "user", "content": "hi"}], None
        )

    @pytest.mark.asyncio
    async def test_retrospective(self, mock_client):
        mock_client.calculate_spend.return_value = {"cost": 0.0042}
        resp = {"choices": [{"message": {"content": "hello"}}]}
        await src.server.calculate_spend(model="gpt-4o", completion_response=resp)
        mock_client.calculate_spend.assert_awaited_once_with("gpt-4o", None, resp)


class TestGetUserDailyActivity:
    @pytest.mark.asyncio
    async def test_filters_passed(self, mock_client):
        mock_client.get_user_daily_activity.return_value = {"results": []}
        await src.server.get_user_daily_activity(
            start_date="2026-04-01", end_date="2026-04-27", user_id="u-1", timezone="Europe/Paris"
        )
        mock_client.get_user_daily_activity.assert_awaited_once_with(
            "2026-04-01", "2026-04-27", None, None, "u-1", None, None, "Europe/Paris"
        )


# ── Execution tools ──


class TestChatCompletion:
    @pytest.mark.asyncio
    async def test_basic(self, mock_client):
        raw = {
            "id": "cmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o",
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "usage": {"total_tokens": 5},
            "system_fingerprint": "fp_drop_me",
        }
        mock_client.chat_completion.return_value = raw
        result = await src.server.chat_completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
            verbosity="standard",
        )
        assert "system_fingerprint" not in result
        assert result["id"] == "cmpl-1"
        assert result["choices"][0]["message"]["content"] == "hi"
        mock_client.chat_completion.assert_awaited_once_with(
            "gpt-4o", [{"role": "user", "content": "hello"}], None
        )

    @pytest.mark.asyncio
    async def test_minimal_strips_to_id_model(self, mock_client):
        mock_client.chat_completion.return_value = {
            "id": "cmpl-1",
            "model": "gpt-4o",
            "choices": [],
            "usage": {},
        }
        result = await src.server.chat_completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            verbosity="minimal",
        )
        assert result == {"id": "cmpl-1", "model": "gpt-4o"}

    @pytest.mark.asyncio
    async def test_passes_extra_body(self, mock_client):
        mock_client.chat_completion.return_value = {"id": "cmpl-1", "model": "x"}
        await src.server.chat_completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            body={"temperature": 0.0, "max_tokens": 64},
        )
        mock_client.chat_completion.assert_awaited_once_with(
            "gpt-4o",
            [{"role": "user", "content": "hi"}],
            {"temperature": 0.0, "max_tokens": 64},
        )


class TestCompletion:
    @pytest.mark.asyncio
    async def test_basic(self, mock_client):
        mock_client.completion.return_value = {"id": "cmpl-2", "model": "gpt-3.5-turbo-instruct"}
        await src.server.completion(model="gpt-3.5-turbo-instruct", prompt="The sky is")
        mock_client.completion.assert_awaited_once_with(
            "gpt-3.5-turbo-instruct", "The sky is", None
        )


class TestEmbed:
    @pytest.mark.asyncio
    async def test_basic(self, mock_client):
        mock_client.embed.return_value = {
            "object": "list",
            "model": "text-embedding-3-small",
            "data": [{"embedding": [0.0, 0.1]}],
            "usage": {"total_tokens": 2},
        }
        result = await src.server.embed(
            model="text-embedding-3-small", input=["hi", "there"], verbosity="standard"
        )
        assert result["model"] == "text-embedding-3-small"
        assert len(result["data"]) == 1
        mock_client.embed.assert_awaited_once_with("text-embedding-3-small", ["hi", "there"], None)


# ── Health tools ──


class TestCheckHealth:
    @pytest.mark.asyncio
    async def test_no_filter(self, mock_client):
        mock_client.check_health.return_value = {
            "healthy_count": 3,
            "unhealthy_count": 0,
            "healthy_endpoints": [{"model": "gpt-4o"}],
            "unhealthy_endpoints": [],
            "extra": "drop",
        }
        result = await src.server.check_health(verbosity="standard")
        assert "extra" not in result
        assert result["healthy_count"] == 3
        mock_client.check_health.assert_awaited_once_with(None, None)

    @pytest.mark.asyncio
    async def test_with_model_filter(self, mock_client):
        mock_client.check_health.return_value = {"healthy_count": 1, "unhealthy_count": 0}
        await src.server.check_health(model="gpt-4o")
        mock_client.check_health.assert_awaited_once_with("gpt-4o", None)


class TestCheckHealthBacklog:
    @pytest.mark.asyncio
    async def test_passthrough(self, mock_client):
        mock_client.check_health_backlog.return_value = {"queue_depth": 0}
        result = await src.server.check_health_backlog()
        assert result == {"queue_depth": 0}


class TestGetHealthHistory:
    @pytest.mark.asyncio
    async def test_filters_passed(self, mock_client):
        mock_client.get_health_history.return_value = []
        await src.server.get_health_history(
            model="gpt-4o", status_filter="unhealthy", limit=50, offset=0
        )
        mock_client.get_health_history.assert_awaited_once_with("gpt-4o", "unhealthy", 50, 0)


class TestGetHealthLatest:
    @pytest.mark.asyncio
    async def test_filters_to_health(self, mock_client):
        mock_client.get_health_latest.return_value = {
            "healthy_count": 5,
            "unhealthy_count": 1,
            "healthy_endpoints": [],
            "unhealthy_endpoints": [],
            "internal_state": "drop",
        }
        result = await src.server.get_health_latest("standard")
        assert "internal_state" not in result


class TestTestModelConnection:
    @pytest.mark.asyncio
    async def test_required_litellm_params(self, mock_client):
        mock_client.test_model_connection.return_value = {"status": "ok"}
        await src.server.test_model_connection(
            litellm_params={"model": "openai/gpt-4o", "api_key": "sk-x"},
            mode="chat",
        )
        mock_client.test_model_connection.assert_awaited_once_with(
            {"model": "openai/gpt-4o", "api_key": "sk-x"}, "chat", None
        )


# ── MCP Gateway: server CRUD ──


class TestListMCPServers:
    @pytest.mark.asyncio
    async def test_no_filter(self, mock_client):
        mock_client.list_mcp_servers.return_value = [
            {"server_id": "s-1", "alias": "context7", "description": "drop"}
        ]
        result = await src.server.list_mcp_servers(verbosity="minimal")
        assert result == [{"server_id": "s-1", "alias": "context7"}]
        mock_client.list_mcp_servers.assert_awaited_once_with(None)

    @pytest.mark.asyncio
    async def test_team_filter(self, mock_client):
        mock_client.list_mcp_servers.return_value = []
        await src.server.list_mcp_servers(team_id="t-1")
        mock_client.list_mcp_servers.assert_awaited_once_with("t-1")


class TestGetMCPServer:
    @pytest.mark.asyncio
    async def test_passes_id(self, mock_client):
        mock_client.get_mcp_server.return_value = {
            "server_id": "s-1",
            "alias": "context7",
            "transport": "http",
            "url": "https://mcp.context7.com/mcp",
        }
        result = await src.server.get_mcp_server("s-1", "standard")
        assert result["transport"] == "http"
        mock_client.get_mcp_server.assert_awaited_once_with("s-1")


class TestAddMCPServer:
    @pytest.mark.asyncio
    async def test_required_transport(self, mock_client):
        mock_client.add_mcp_server.return_value = {"server_id": "s-1"}
        await src.server.add_mcp_server(
            transport="http",
            alias="context7",
            url="https://mcp.context7.com/mcp",
            auth_type="none",
        )
        call = mock_client.add_mcp_server.await_args
        assert call.args[0] == "http"
        assert call.args[2] == "context7"
        assert call.args[4] == "https://mcp.context7.com/mcp"
        assert call.args[5] == "none"


class TestUpdateMCPServer:
    @pytest.mark.asyncio
    async def test_required_id(self, mock_client):
        mock_client.update_mcp_server.return_value = {"ok": True}
        await src.server.update_mcp_server("s-1", description="updated")
        mock_client.update_mcp_server.assert_awaited_once_with(
            "s-1", None, None, "updated", None, None, None, None, None, None, None, None, None
        )


class TestDeleteMCPServer:
    @pytest.mark.asyncio
    async def test_passes_id(self, mock_client):
        mock_client.delete_mcp_server.return_value = {"deleted": True}
        await src.server.delete_mcp_server("s-1")
        mock_client.delete_mcp_server.assert_awaited_once_with("s-1")


class TestRegisterMCPServer:
    @pytest.mark.asyncio
    async def test_minimal(self, mock_client):
        mock_client.register_mcp_server.return_value = {
            "server_id": "s-1",
            "approval_status": "pending",
        }
        await src.server.register_mcp_server(
            transport="http",
            alias="my-mcp",
            url="https://example.com/mcp",
        )
        call = mock_client.register_mcp_server.await_args
        assert call.args[0] == "http"
        assert call.args[2] == "my-mcp"


# ── MCP Gateway: submissions ──


class TestListMCPServerSubmissions:
    @pytest.mark.asyncio
    async def test_filters_to_submission_shape(self, mock_client):
        mock_client.list_mcp_server_submissions.return_value = [
            {
                "server_id": "s-1",
                "approval_status": "pending",
                "alias": "candidate",
                "transport": "http",
                "url": "https://x.com/mcp",
                "drop_me": True,
            }
        ]
        result = await src.server.list_mcp_server_submissions("standard")
        assert "drop_me" not in result[0]
        assert result[0]["approval_status"] == "pending"


class TestApproveMCPServerSubmission:
    @pytest.mark.asyncio
    async def test_passes_id(self, mock_client):
        mock_client.approve_mcp_server_submission.return_value = {"ok": True}
        await src.server.approve_mcp_server_submission("s-1")
        mock_client.approve_mcp_server_submission.assert_awaited_once_with("s-1")


class TestRejectMCPServerSubmission:
    @pytest.mark.asyncio
    async def test_with_notes(self, mock_client):
        mock_client.reject_mcp_server_submission.return_value = {"ok": True}
        await src.server.reject_mcp_server_submission("s-1", review_notes="duplicate")
        mock_client.reject_mcp_server_submission.assert_awaited_once_with("s-1", "duplicate")

    @pytest.mark.asyncio
    async def test_no_notes(self, mock_client):
        mock_client.reject_mcp_server_submission.return_value = {"ok": True}
        await src.server.reject_mcp_server_submission("s-1")
        mock_client.reject_mcp_server_submission.assert_awaited_once_with("s-1", None)


# ── MCP Gateway: health ──


class TestCheckMCPServersHealth:
    @pytest.mark.asyncio
    async def test_filters_passed(self, mock_client):
        mock_client.check_mcp_servers_health.return_value = [
            {"server_id": "s-1", "status": "healthy", "latency_ms": 42, "internal": "drop"}
        ]
        result = await src.server.check_mcp_servers_health(server_ids="s-1,s-2")
        assert "internal" not in result[0]
        mock_client.check_mcp_servers_health.assert_awaited_once_with("s-1,s-2")


# ── MCP Gateway: tool discovery & invocation ──


class TestListMCPTools:
    @pytest.mark.asyncio
    async def test_filters_to_tool_shape(self, mock_client):
        mock_client.list_mcp_tools.return_value = [
            {
                "name": "resolve-library-id",
                "description": "...",
                "server_id": "s-1",
                "inputSchema": {"type": "object"},
                "drop_me": True,
            }
        ]
        result = await src.server.list_mcp_tools("standard")
        assert "drop_me" not in result[0]
        assert result[0]["name"] == "resolve-library-id"


class TestListMCPToolsRest:
    @pytest.mark.asyncio
    async def test_passes_server_id_and_filters_tools_array(self, mock_client):
        mock_client.list_mcp_tools_rest.return_value = {
            "tools": [
                {"name": "tool-a", "description": "...", "drop_me": True},
                {"name": "tool-b", "description": "..."},
            ]
        }
        result = await src.server.list_mcp_tools_rest(server_id="s-1", verbosity="minimal")
        assert result["tools"] == [{"name": "tool-a"}, {"name": "tool-b"}]
        mock_client.list_mcp_tools_rest.assert_awaited_once_with("s-1")

    @pytest.mark.asyncio
    async def test_handles_list_payload(self, mock_client):
        mock_client.list_mcp_tools_rest.return_value = [
            {"name": "tool-a"},
        ]
        result = await src.server.list_mcp_tools_rest()
        assert result == [{"name": "tool-a"}]


class TestCallMCPTool:
    @pytest.mark.asyncio
    async def test_basic(self, mock_client):
        mock_client.call_mcp_tool.return_value = {"content": [{"type": "text", "text": "ok"}]}
        await src.server.call_mcp_tool(
            server_id="s-1", name="resolve-library-id", arguments={"libraryName": "react"}
        )
        mock_client.call_mcp_tool.assert_awaited_once_with(
            "s-1", "resolve-library-id", {"libraryName": "react"}
        )

    @pytest.mark.asyncio
    async def test_no_arguments(self, mock_client):
        mock_client.call_mcp_tool.return_value = {"content": []}
        await src.server.call_mcp_tool(server_id="s-1", name="ping")
        mock_client.call_mcp_tool.assert_awaited_once_with("s-1", "ping", None)


class TestTestMCPConnection:
    @pytest.mark.asyncio
    async def test_required_transport(self, mock_client):
        mock_client.test_mcp_connection.return_value = {"status": "ok"}
        await src.server.test_mcp_connection(
            transport="http", url="https://example.com/mcp", auth_type="none"
        )
        mock_client.test_mcp_connection.assert_awaited_once_with(
            "http", "https://example.com/mcp", "none", None, None, None
        )


# ── MCP Gateway: discovery / registry / hub ──


class TestDiscoverMCPServers:
    @pytest.mark.asyncio
    async def test_filters_passed(self, mock_client):
        mock_client.discover_mcp_servers.return_value = []
        await src.server.discover_mcp_servers(query="docs", category="dev-tools")
        mock_client.discover_mcp_servers.assert_awaited_once_with("docs", "dev-tools")


class TestGetMCPOpenAPIRegistry:
    @pytest.mark.asyncio
    async def test_passthrough(self, mock_client):
        mock_client.get_mcp_openapi_registry.return_value = {"version": 1}
        result = await src.server.get_mcp_openapi_registry()
        assert result == {"version": 1}


class TestGetMCPRegistry:
    @pytest.mark.asyncio
    async def test_passthrough(self, mock_client):
        mock_client.get_mcp_registry.return_value = {"servers": []}
        result = await src.server.get_mcp_registry()
        assert result == {"servers": []}


class TestListMCPAccessGroups:
    @pytest.mark.asyncio
    async def test_passthrough(self, mock_client):
        mock_client.list_mcp_access_groups.return_value = ["engineering", "ops"]
        result = await src.server.list_mcp_access_groups()
        assert result == ["engineering", "ops"]


class TestMakeMCPServersPublic:
    @pytest.mark.asyncio
    async def test_passes_ids(self, mock_client):
        mock_client.make_mcp_servers_public.return_value = {"ok": True}
        await src.server.make_mcp_servers_public(["s-1", "s-2"])
        mock_client.make_mcp_servers_public.assert_awaited_once_with(["s-1", "s-2"])


class TestGetPublicMCPHub:
    @pytest.mark.asyncio
    async def test_passthrough(self, mock_client):
        mock_client.get_public_mcp_hub.return_value = {"servers": []}
        result = await src.server.get_public_mcp_hub()
        assert result == {"servers": []}


# ── MCP Gateway: user credentials ──


class TestListMCPUserCredentials:
    @pytest.mark.asyncio
    async def test_filters_to_credential_shape(self, mock_client):
        mock_client.list_mcp_user_credentials.return_value = [
            {
                "server_id": "s-1",
                "has_credential": True,
                "oauth_enabled": False,
                "drop_me": True,
            }
        ]
        result = await src.server.list_mcp_user_credentials("standard")
        assert "drop_me" not in result[0]


class TestSetMCPUserCredential:
    @pytest.mark.asyncio
    async def test_non_oauth(self, mock_client):
        mock_client.set_mcp_user_credential.return_value = {"ok": True}
        await src.server.set_mcp_user_credential(server_id="s-1", credential="sk-x", save=True)
        mock_client.set_mcp_user_credential.assert_awaited_once_with(
            "s-1", False, "sk-x", True, None, None, None, None
        )

    @pytest.mark.asyncio
    async def test_oauth(self, mock_client):
        mock_client.set_mcp_user_credential.return_value = {"ok": True}
        await src.server.set_mcp_user_credential(
            server_id="s-1",
            oauth=True,
            access_token="at-x",
            refresh_token="rt-x",
            expires_in=3600,
            scopes=["read"],
        )
        mock_client.set_mcp_user_credential.assert_awaited_once_with(
            "s-1", True, None, None, "at-x", "rt-x", 3600, ["read"]
        )


class TestDeleteMCPUserCredential:
    @pytest.mark.asyncio
    async def test_non_oauth(self, mock_client):
        mock_client.delete_mcp_user_credential.return_value = {"deleted": True}
        await src.server.delete_mcp_user_credential("s-1")
        mock_client.delete_mcp_user_credential.assert_awaited_once_with("s-1", False)

    @pytest.mark.asyncio
    async def test_oauth(self, mock_client):
        mock_client.delete_mcp_user_credential.return_value = {"deleted": True}
        await src.server.delete_mcp_user_credential("s-1", oauth=True)
        mock_client.delete_mcp_user_credential.assert_awaited_once_with("s-1", True)


class TestGetMCPOAuthUserCredentialStatus:
    @pytest.mark.asyncio
    async def test_passes_id(self, mock_client):
        mock_client.get_mcp_oauth_user_credential_status.return_value = {"valid": True}
        await src.server.get_mcp_oauth_user_credential_status("s-1")
        mock_client.get_mcp_oauth_user_credential_status.assert_awaited_once_with("s-1")


# ── MCP Gateway: utility ──


class TestGetMCPClientIp:
    @pytest.mark.asyncio
    async def test_passthrough(self, mock_client):
        mock_client.get_mcp_client_ip.return_value = {"client_ip": "10.0.0.1"}
        result = await src.server.get_mcp_client_ip()
        assert result == {"client_ip": "10.0.0.1"}


# ── MCP Toolset tools ──


class TestListMCPToolsets:
    @pytest.mark.asyncio
    async def test_filters_to_toolset_shape(self, mock_client):
        mock_client.list_mcp_toolsets.return_value = [
            {
                "toolset_id": "ts-1",
                "toolset_name": "research",
                "description": "...",
                "tools": ["context7/resolve-library-id"],
                "drop_me": True,
            }
        ]
        result = await src.server.list_mcp_toolsets("standard")
        assert "drop_me" not in result[0]
        assert result[0]["toolset_name"] == "research"

    @pytest.mark.asyncio
    async def test_minimal(self, mock_client):
        mock_client.list_mcp_toolsets.return_value = [
            {"toolset_id": "ts-1", "toolset_name": "research", "tools": []}
        ]
        result = await src.server.list_mcp_toolsets("minimal")
        assert result == [{"toolset_id": "ts-1", "toolset_name": "research"}]


class TestGetMCPToolset:
    @pytest.mark.asyncio
    async def test_passes_id(self, mock_client):
        mock_client.get_mcp_toolset.return_value = {
            "toolset_id": "ts-1",
            "toolset_name": "research",
        }
        result = await src.server.get_mcp_toolset("ts-1", "standard")
        assert result["toolset_name"] == "research"
        mock_client.get_mcp_toolset.assert_awaited_once_with("ts-1")


class TestAddMCPToolset:
    @pytest.mark.asyncio
    async def test_required_fields(self, mock_client):
        mock_client.add_mcp_toolset.return_value = {"toolset_id": "ts-1"}
        await src.server.add_mcp_toolset(
            toolset_name="research",
            tools=["context7/resolve-library-id"],
            description="docs lookup",
        )
        mock_client.add_mcp_toolset.assert_awaited_once_with(
            "research", ["context7/resolve-library-id"], "docs lookup"
        )

    @pytest.mark.asyncio
    async def test_no_description(self, mock_client):
        mock_client.add_mcp_toolset.return_value = {"toolset_id": "ts-1"}
        await src.server.add_mcp_toolset(
            toolset_name="research", tools=["context7/resolve-library-id"]
        )
        mock_client.add_mcp_toolset.assert_awaited_once_with(
            "research", ["context7/resolve-library-id"], None
        )


class TestUpdateMCPToolset:
    @pytest.mark.asyncio
    async def test_required_id(self, mock_client):
        mock_client.update_mcp_toolset.return_value = {"ok": True}
        await src.server.update_mcp_toolset("ts-1", description="updated")
        mock_client.update_mcp_toolset.assert_awaited_once_with("ts-1", None, "updated", None)

    @pytest.mark.asyncio
    async def test_replace_tools(self, mock_client):
        mock_client.update_mcp_toolset.return_value = {"ok": True}
        await src.server.update_mcp_toolset(
            "ts-1", tools=["context7/resolve-library-id", "context7/query-docs"]
        )
        mock_client.update_mcp_toolset.assert_awaited_once_with(
            "ts-1", None, None, ["context7/resolve-library-id", "context7/query-docs"]
        )


class TestDeleteMCPToolset:
    @pytest.mark.asyncio
    async def test_passes_id(self, mock_client):
        mock_client.delete_mcp_toolset.return_value = {"deleted": True}
        await src.server.delete_mcp_toolset("ts-1")
        mock_client.delete_mcp_toolset.assert_awaited_once_with("ts-1")


# ── Provider passthrough ──


class TestPassthrough:
    @pytest.mark.asyncio
    async def test_default_get(self, mock_client):
        mock_client.passthrough.return_value = {"data": [{"id": "claude-opus-4-7"}]}
        result = await src.server.passthrough(provider="anthropic", endpoint="v1/models")
        assert result["data"][0]["id"] == "claude-opus-4-7"
        mock_client.passthrough.assert_awaited_once_with(
            "anthropic", "v1/models", "GET", None, None, None
        )

    @pytest.mark.asyncio
    async def test_post_with_body_and_headers(self, mock_client):
        mock_client.passthrough.return_value = {"id": "msg_1"}
        await src.server.passthrough(
            provider="anthropic",
            endpoint="v1/messages",
            method="POST",
            body={"model": "claude-opus-4-7", "messages": [{"role": "user", "content": "hi"}]},
            headers={"anthropic-version": "2023-06-01"},
        )
        mock_client.passthrough.assert_awaited_once_with(
            "anthropic",
            "v1/messages",
            "POST",
            {"model": "claude-opus-4-7", "messages": [{"role": "user", "content": "hi"}]},
            None,
            {"anthropic-version": "2023-06-01"},
        )

    @pytest.mark.asyncio
    async def test_method_case_insensitive(self, mock_client):
        mock_client.passthrough.return_value = {}
        await src.server.passthrough(provider="openai", endpoint="v1/models", method="get")
        mock_client.passthrough.assert_awaited_once_with(
            "openai", "v1/models", "get", None, None, None
        )

    @pytest.mark.asyncio
    async def test_nested_provider(self, mock_client):
        # Vertex AI Discovery uses a two-segment provider prefix
        mock_client.passthrough.return_value = {}
        await src.server.passthrough(
            provider="vertex_ai/discovery",
            endpoint="v1/projects/x/locations/global/dataStores",
        )
        mock_client.passthrough.assert_awaited_once_with(
            "vertex_ai/discovery",
            "v1/projects/x/locations/global/dataStores",
            "GET",
            None,
            None,
            None,
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
