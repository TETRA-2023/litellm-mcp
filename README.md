# litellm-mcp

MCP server for [LiteLLM](https://github.com/BerriAI/litellm) proxy administration and execution.

**Upstream API reference:** [LiteLLM proxy Swagger](https://litellm-api.up.railway.app/).

Foundation slice (US #534): exposes a single tool — `list_models` — against a running LiteLLM proxy.
Subsequent slices add models/credentials/keys/users/spend/teams/guardrails/routing/passthrough/MCP-gateway tools.

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

| Tool | Description |
|------|-------------|
| `list_models` | List models exposed by the proxy (`GET /v1/models`). Verbosity: `minimal` / `standard` / `full`. |

## Development

```bash
uv sync --all-extras --dev
uv run pre-commit install
uv run pytest -v
uv run ruff check src/ tests/
```

## License

MIT — see [LICENSE](LICENSE).
