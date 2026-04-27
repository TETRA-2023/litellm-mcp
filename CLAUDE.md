# CLAUDE.md — litellm-mcp

MCP server for the LiteLLM proxy. Python 3.12, fastmcp (`mcp.server.fastmcp`), httpx async client, pydantic-settings config.

**Upstream API reference:** https://litellm-api.up.railway.app/ (LiteLLM proxy Swagger; canonical source for endpoint shapes, auth, and request/response schemas — consult this before adding any tool).

## Layout

- `src/server.py` — FastMCP entrypoint, `lifespan` initialises the shared `LiteLLMClient`.
- `src/config.py` — `LiteLLMSettings` (pydantic-settings, loads `.env`).
- `src/litellm_client.py` — async httpx client; Bearer auth, retry, error mapping.
- `tests/test_server.py` — pytest-asyncio unit tests, mocking the client.

## Conventions

- All tools accept a `verbosity: str` argument (`minimal` / `standard` / `full`) and route through `_filter_response`.
- Add new resource shapes to `RESPONSE_FIELDS` in `src/server.py`.
- New endpoints go on `LiteLLMClient` first, then a thin `@mcp.tool()` wrapper in `server.py`.
- Keep transport agnostic: never call `print` / write to stdout in stdio mode.

## Testing

```bash
uv run pytest -v
```

Smoke test (requires a running LiteLLM proxy):

```bash
LITELLM_PROXY_URL=http://localhost:4000 LITELLM_API_KEY=sk-... \
  uv run python -c "import asyncio; from src.litellm_client import LiteLLMClient; \
    asyncio.run(LiteLLMClient('http://localhost:4000', 'sk-...').list_models())"
```
