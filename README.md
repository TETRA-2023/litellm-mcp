# litellm-mcp

MCP server for [LiteLLM](https://github.com/BerriAI/litellm) proxy administration and execution.

**Upstream API reference:** [LiteLLM proxy Swagger](https://litellm-api.up.railway.app/).

Foundation (US #534) shipped `list_models`. Admin slice (US #535) adds 32 tools
covering models, model hub, model access groups, credentials, and virtual keys.
Subsequent slices add identity/teams/spend/execution (#536), governance and
passthrough (#538), and the MCP-of-MCPs gateway (#558).

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

## Development

```bash
uv sync --all-extras --dev
uv run pre-commit install
uv run pytest -v
uv run ruff check src/ tests/
```

## License

MIT — see [LICENSE](LICENSE).
