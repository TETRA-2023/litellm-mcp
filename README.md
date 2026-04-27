# litellm-mcp

MCP server for [LiteLLM](https://github.com/BerriAI/litellm) proxy administration and execution.

**Upstream API reference:** [LiteLLM proxy Swagger](https://litellm-api.up.railway.app/).

Foundation (US #534) shipped `list_models`. Admin slice (US #535) added 32 tools
covering models, model hub, model access groups, credentials, and virtual keys.
Identity slice (US #536) added 31 tools covering internal users, customers,
organizations (with member management), projects, and unified user access
groups. Spend/execution/health slice (US #596) added 19 tools covering budgets,
spend reporting, the three core execution verbs (chat / completion / embed),
and admin health probes. MCP-Gateway slice (US #558) added 25 tools that let
the LiteLLM proxy act as an MCP-of-MCPs: register upstream HTTP-transport MCP
servers and list / invoke their tools through the proxy. MCP-Toolsets slice
(US #688) adds 5 tools to manage named bundles of cross-server tools.
Governance and passthrough (#538) is the remaining deferred slice.

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- A reachable LiteLLM proxy and a key (master or virtual)

### Installation

```bash
git clone https://github.com/TETRA-2023/litellm-mcp.git
cd litellm-mcp
uv sync
```

### Configuration

```bash
cp .env.example .env
# Edit .env with your LiteLLM proxy URL and API key
```

| Variable | Description | Default |
|----------|-------------|---------|
| `LITELLM_PROXY_URL` | LiteLLM proxy base URL | `http://localhost:4000` |
| `LITELLM_API_KEY` | Master key or virtual key (sent as Bearer) | *required* |
| `LITELLM_TRANSPORT` | `stdio` or `streamable-http` | `stdio` |
| `LITELLM_TIMEOUT_SECONDS` | HTTP timeout | `30` |
| `LITELLM_MAX_RETRIES` | Retry budget on transient errors | `2` |

## Usage

### stdio (Claude Code / local)

```bash
uv run python src/server.py
```

### Claude Code configuration

Add to your Claude Code MCP settings (project-scoped `.mcp.json` recommended):

```json
{
  "mcpServers": {
    "litellm": {
      "command": "uv",
      "args": ["run", "python", "src/server.py"],
      "cwd": "/path/to/litellm-mcp",
      "env": {
        "LITELLM_PROXY_URL": "${LITELLM_PROXY_URL}",
        "LITELLM_API_KEY": "${LITELLM_API_KEY}"
      }
    }
  }
}
```

## Tools

All tools accept a `verbosity` arg (`minimal` / `standard` / `full`) where it makes sense; see `RESPONSE_FIELDS` in `src/server.py` for the shapes. The `credential` shape deliberately omits `credential_values` at `standard` to avoid leaking secrets into agent transcripts — use `full` to inspect.

### Models (6)

| Tool | Endpoint |
|------|----------|
| `list_models` | `GET /v1/models` |
| `get_model` | `GET /v1/models/{model_id}` |
| `get_model_info` | `GET /model/info` (admin, full deployment details) |
| `add_model` | `POST /model/new` |
| `update_model` | `PATCH /model/{model_id}/update` |
| `delete_model` | `POST /model/delete` |

### Model Hub (5)

| Tool | Endpoint |
|------|----------|
| `list_public_models` | `GET /public/model_hub` |
| `get_public_hub_info` | `GET /public/model_hub/info` |
| `get_model_cost_map` | `GET /public/litellm_model_cost_map` |
| `make_model_group_public` | `POST /model_group/make_public` |
| `update_model_hub_links` | `POST /model_hub/update_useful_links` |

### Model Access Groups (5)

| Tool | Endpoint |
|------|----------|
| `list_model_access_groups` | `GET /access_group/list` |
| `get_model_access_group` | `GET /access_group/{access_group}/info` |
| `create_model_access_group` | `POST /access_group/new` |
| `update_model_access_group` | `PUT /access_group/{access_group}/update` |
| `delete_model_access_group` | `DELETE /access_group/{access_group}/delete` |

### Credentials (6)

| Tool | Endpoint |
|------|----------|
| `list_credentials` | `GET /credentials` |
| `get_credential` | `GET /credentials/by_name/{credential_name}` |
| `get_credential_by_model` | `GET /credentials/by_model/{model_id}` |
| `create_credential` | `POST /credentials` |
| `update_credential` | `PATCH /credentials/{credential_name}` |
| `delete_credential` | `DELETE /credentials/{credential_name}` |

### Keys (11)

| Tool | Endpoint |
|------|----------|
| `list_keys` | `GET /key/list` |
| `list_key_aliases` | `GET /key/aliases` |
| `get_key_info` | `GET /key/info` |
| `generate_key` | `POST /key/generate` |
| `generate_service_account_key` | `POST /key/service-account/generate` |
| `update_key` | `POST /key/update` |
| `regenerate_key` | `POST /key/{key}/regenerate` |
| `set_key_blocked(blocked: bool)` | `POST /key/block` ∣ `POST /key/unblock` |
| `delete_keys` | `POST /key/delete` |
| `reset_key_spend` | `POST /key/{key}/reset_spend` |
| `key_health` | `POST /key/health` |

`generate_key`, `generate_service_account_key`, `update_key`, and `regenerate_key`
expose the most common ~12 fields as named args; pass any other upstream field
via the `extras: dict` argument (merged into the request body).

### Internal Users (5)

| Tool | Endpoint |
|------|----------|
| `list_users` | `GET /user/list` |
| `get_user_info` | `GET /user/info` |
| `create_user` | `POST /user/new` |
| `update_user` | `POST /user/update` |
| `delete_user` | `POST /user/delete` |

### Customers (7)

| Tool | Endpoint |
|------|----------|
| `list_customers` | `GET /customer/list` |
| `get_customer_info` | `GET /customer/info` |
| `create_customer` | `POST /customer/new` |
| `update_customer` | `POST /customer/update` |
| `delete_customer` | `POST /customer/delete` |
| `set_customer_blocked(blocked: bool)` | `POST /customer/block` ∣ `POST /customer/unblock` |
| `get_customer_daily_activity` | `GET /customer/daily/activity` |

### Organizations (9)

| Tool | Endpoint |
|------|----------|
| `list_organizations` | `GET /organization/list` |
| `get_organization_info` | `GET /organization/info` |
| `create_organization` | `POST /organization/new` |
| `update_organization` | `PATCH /organization/update` |
| `delete_organization` | `DELETE /organization/delete` |
| `add_org_member` | `POST /organization/member_add` |
| `update_org_member` | `PATCH /organization/member_update` |
| `delete_org_member` | `DELETE /organization/member_delete` |
| `get_org_daily_activity` | `GET /organization/daily/activity` |

### Projects (5)

| Tool | Endpoint |
|------|----------|
| `list_projects` | `GET /project/list` |
| `get_project_info` | `GET /project/info` |
| `create_project` | `POST /project/new` |
| `update_project` | `POST /project/update` |
| `delete_project` | `DELETE /project/delete` |

### Unified User Access Groups (5)

| Tool | Endpoint |
|------|----------|
| `list_user_access_groups` | `GET /v1/unified_access_group` |
| `get_user_access_group` | `GET /v1/unified_access_group/{id}` |
| `create_user_access_group` | `POST /v1/unified_access_group` |
| `update_user_access_group` | `PUT /v1/unified_access_group/{id}` |
| `delete_user_access_group` | `DELETE /v1/unified_access_group/{id}` |

Distinct from the model-access-groups family above — unified user access
groups gate users/teams against models, MCP servers, and agents in one shape.
`create_user`, `update_user`, `create_customer`, `update_customer`,
`create_organization`, `update_organization`, `create_project`, and
`update_project` accept an `extras: dict` argument for the long tail of
upstream fields not surfaced as named args.

### Budgets (6)

| Tool | Endpoint |
|------|----------|
| `list_budgets` | `GET /budget/list` |
| `get_budget_info` | `POST /budget/info` (batch lookup by id list) |
| `create_budget` | `POST /budget/new` |
| `update_budget` | `POST /budget/update` |
| `delete_budget` | `POST /budget/delete` |
| `get_budget_settings` | `GET /budget/settings` |

`create_budget` and `update_budget` accept an `extras: dict` for any
`BudgetNewRequest` field not surfaced as a named arg.

### Spend (5)

| Tool | Endpoint |
|------|----------|
| `get_global_spend_report` | `GET /global/spend/report` (LiteLLM Enterprise only) |
| `list_spend_logs` | `GET /spend/logs` |
| `list_spend_tags` | `GET /spend/tags` |
| `calculate_spend` | `POST /spend/calculate` (prospective or retrospective) |
| `get_user_daily_activity` | `GET /user/daily/activity` |

### Execution (3)

| Tool | Endpoint |
|------|----------|
| `chat_completion` | `POST /v1/chat/completions` |
| `completion` | `POST /v1/completions` (legacy text) |
| `embed` | `POST /v1/embeddings` |

Synchronous only — `stream` is stripped from the body. Pass any extra
OpenAI/LiteLLM body fields (temperature, max_tokens, tools, dimensions, etc.)
via the `body: dict` argument.

### Health (5)

| Tool | Endpoint |
|------|----------|
| `check_health` | `GET /health` (probes router-registered deployments) |
| `check_health_backlog` | `GET /health/backlog` |
| `get_health_history` | `GET /health/history` |
| `get_health_latest` | `GET /health/latest` |
| `test_model_connection` | `POST /health/test_connection` |

### MCP Gateway — Server CRUD (6)

| Tool | Endpoint |
|------|----------|
| `list_mcp_servers` | `GET /v1/mcp/server` |
| `get_mcp_server` | `GET /v1/mcp/server/{server_id}` |
| `add_mcp_server` | `POST /v1/mcp/server` (admin path, approved) |
| `update_mcp_server` | `PUT /v1/mcp/server` (server_id in body) |
| `delete_mcp_server` | `DELETE /v1/mcp/server/{server_id}` |
| `register_mcp_server` | `POST /v1/mcp/server/register` (user path, pending review) |

### MCP Gateway — Submissions (3)

| Tool | Endpoint |
|------|----------|
| `list_mcp_server_submissions` | `GET /v1/mcp/server/submissions` |
| `approve_mcp_server_submission` | `PUT /v1/mcp/server/{id}/approve` |
| `reject_mcp_server_submission` | `PUT /v1/mcp/server/{id}/reject` |

### MCP Gateway — Tool Discovery & Invocation (4) + Health (1)

| Tool | Endpoint |
|------|----------|
| `check_mcp_servers_health` | `GET /v1/mcp/server/health` |
| `list_mcp_tools` | `GET /v1/mcp/tools` |
| `list_mcp_tools_rest` | `GET /mcp-rest/tools/list` |
| `call_mcp_tool` | `POST /mcp-rest/tools/call` |
| `test_mcp_connection` | `POST /mcp-rest/test/connection` |

`call_mcp_tool` body shape: `{"server_id": ..., "name": ..., "arguments": {...}}`. Returns the provider-shaped MCP envelope (`{"content": [...], "isError": bool}`) unmodified.

### MCP Gateway — Discovery / Registry / Hub (6)

| Tool | Endpoint |
|------|----------|
| `discover_mcp_servers` | `GET /v1/mcp/discover` |
| `get_mcp_openapi_registry` | `GET /v1/mcp/openapi-registry` |
| `get_mcp_registry` | `GET /v1/mcp/registry.json` |
| `list_mcp_access_groups` | `GET /v1/mcp/access_groups` |
| `make_mcp_servers_public` | `POST /v1/mcp/make_public` |
| `get_public_mcp_hub` | `GET /public/mcp_hub` |

### MCP Gateway — User Credentials (4) + Utility (1)

| Tool | Endpoint |
|------|----------|
| `list_mcp_user_credentials` | `GET /v1/mcp/user-credentials` |
| `set_mcp_user_credential(server_id, credential, oauth: bool)` | `POST /v1/mcp/server/{id}/user-credential` ∣ `POST .../oauth-user-credential` |
| `delete_mcp_user_credential(server_id, oauth: bool)` | `DELETE /v1/mcp/server/{id}/user-credential` ∣ `DELETE .../oauth-user-credential` |
| `get_mcp_oauth_user_credential_status` | `GET /v1/mcp/server/{id}/oauth-user-credential/status` |
| `get_mcp_client_ip` | `GET /v1/mcp/network/client-ip` |

The `oauth: bool` discriminator on `set_mcp_user_credential` and `delete_mcp_user_credential` compresses the OAuth + non-OAuth endpoints into one tool each. Non-OAuth body: `{"credential": ..., "save": ...}`. OAuth body: `{"access_token": ..., "refresh_token": ..., "expires_in": ..., "scopes": [...]}`.

### MCP Gateway — Toolsets (5)

| Tool | Endpoint |
|------|----------|
| `list_mcp_toolsets` | `GET /v1/mcp/toolset` |
| `get_mcp_toolset` | `GET /v1/mcp/toolset/{toolset_id}` |
| `add_mcp_toolset` | `POST /v1/mcp/toolset` |
| `update_mcp_toolset` | `PUT /v1/mcp/toolset` (toolset_id in body) |
| `delete_mcp_toolset` | `DELETE /v1/mcp/toolset/{toolset_id}` |

Toolsets are named bundles of tools sourced from one or more registered MCP servers (e.g. a `research` toolset combining `resolve-library-id` from Context7 with `search` from another MCP). Once defined, the proxy exposes each toolset as a brokered MCP endpoint at `/toolset/{name}/mcp` — that transport route is intentionally not wrapped here.

`add_mcp_server` / `update_mcp_server` / `register_mcp_server` / `test_mcp_connection` accept an `extras: dict` argument for the long tail of `NewMCPServerRequest` fields not surfaced as named args (~20 less common fields like `static_headers`, `oauth2_flow`, `tool_name_to_display_name`, `allow_all_keys`).

## Development

```bash
uv sync --all-extras --dev
uv run pre-commit install
uv run pytest -v
uv run ruff check src/ tests/
```

## License

MIT — see [LICENSE](LICENSE).
