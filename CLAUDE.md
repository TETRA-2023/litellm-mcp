# CLAUDE.md — litellm-mcp

MCP server for the LiteLLM proxy. Python 3.12, fastmcp (`mcp.server.fastmcp`), httpx async client, pydantic-settings config.

**Upstream API reference:** https://litellm-api.up.railway.app/ (LiteLLM proxy Swagger; canonical source for endpoint shapes, auth, and request/response schemas — consult this before adding any tool).

## Layout

- `src/server.py` — FastMCP entrypoint, `lifespan` initialises the shared `LiteLLMClient`.
- `src/config.py` — `LiteLLMSettings` (pydantic-settings, loads `.env`).
- `src/litellm_client.py` — async httpx client; Bearer auth, retry, error mapping.
- `tests/test_server.py` — pytest-asyncio unit tests, mocking the client.

## Conventions

- All read-shaped tools accept a `verbosity: str` argument (`minimal` / `standard` / `full`) and route through `_filter_response`. Write/admin tools generally don't filter (`update_model`, `add_model`, etc. return upstream as-is).
- Add new resource shapes to `RESPONSE_FIELDS` in `src/server.py`. Current shapes: `model`, `access_group`, `credential`, `key`, `public_hub`, `user`, `customer`, `organization`, `project`, `user_access_group`, `budget`, `spend_record`, `chat_completion`, `embedding`, `health`, `mcp_server`, `mcp_tool`, `mcp_submission`, `mcp_credential`, `mcp_health`. The `credential.standard` and `mcp_credential.standard` shapes intentionally drop credential values to avoid leaking secrets — use `full` to inspect.
- New endpoints go on `LiteLLMClient` first, then a thin `@mcp.tool()` wrapper in `server.py`.
- For body schemas with many optional fields (`GenerateKeyRequest`, `UpdateKeyRequest`, `RegenerateKeyRequest`, `NewUserRequest`, `NewCustomerRequest`, `NewOrganizationRequest`, `NewProjectRequest`), expose ~12 common fields as named args and accept the long tail through an `extras: Optional[dict]` argument (merged into the body via `_build_body`/`_build_key_body`).
- Keep transport agnostic: never call `print` / write to stdout in stdio mode.

## Current scope (US #534 + #535 + #536 + #596 + #558)

- **#534 (v1.0.0):** foundation, `list_models`.
- **#535 (v1.1.0):** 32 admin tools — Models (5), Model Hub (5), Access Groups (5), Credentials (6), Keys (11). All wrapped with unit tests against `AsyncMock(LiteLLMClient)`.
- **#536 (v1.2.0):** 31 identity tools — Internal Users (5), Customers (7), Organizations (9, incl. member CRUD), Projects (5), Unified User Access Groups (5).
- **#596 (v1.3.0):** 19 tools — Budgets (6), Spend (5), Execution (3 — chat/completion/embed, synchronous), Health (5).
- **#558:** 25 MCP-Gateway tools — Server CRUD (6), Submissions (3), Health (1), Tool discovery & invocation (4), Discovery/registry/hub (6), User credentials (4), Utility (1). Lets the proxy broker upstream HTTP-transport MCP servers and list/invoke their tools.

### Upstream quirks

- `GET /v1/models/{id}` only resolves router-registered ids; passthrough ids 404. Use `get_model_info` for admin lookups.
- `GET /key/info` 404s on the master key (not stored in `LiteLLM_VerificationToken`). Probe a virtual key.
- `PATCH /credentials/{name}` is a full replace despite the verb; `CredentialItem` requires all three body fields.
- `GET /customer/daily/activity` and `GET /organization/daily/activity` require `start_date` and `end_date` despite the OpenAPI marking them optional — omit either and you get HTTP 400.
- `PATCH /organization/update` has no documented request body in the OpenAPI spec; the wrapper sends an UpdateOrganization payload mirroring NewOrganizationRequest.
- `POST /budget/info` is a batch GET — it takes a body `{"budgets": [...]}` rather than a single-id query, despite the GET-shaped name.
- `POST /v1/completions` only documents a `model` query param in the OpenAPI spec but accepts the full OpenAI-shaped body. The wrapper sends `model` in both query and body.
- `GET /global/spend/report` is gated behind a LiteLLM Enterprise license — community-tier proxies return HTTP 400 "You must be a LiteLLM Enterprise user". The wrapper passes the call through unchanged.
- Execution tools (`chat_completion` / `completion` / `embed`) are synchronous; `stream` is stripped from the body. Streaming would require a different MCP transport contract — revisit as a follow-on if needed.
- `PUT /v1/mcp/server` carries `server_id` in the body (per `UpdateMCPServerRequest`), not the path — opposite convention from the GET/DELETE variants.
- `POST /mcp-rest/tools/call` has no documented body schema in OpenAPI but expects `{"server_id": ..., "name": ..., "arguments": {...}}` per the MCP `tools/call` shape.
- MCP tool invocation (`call_mcp_tool`) does **not** rely on the registered server's `allow_all_keys` flag alone — there is a per-user MCP-access check upstream that returns "User not allowed to call this tool" even for the master key, unless the calling user is a member of the server's `mcp_access_groups`. Wrapper sends a correct request; granting access is a deployment concern (assign the calling user to an MCP access group, or attach `mcp_servers=[<id>]` to the calling key).
- `GET /v1/mcp/registry.json` is optional upstream — older / minimal proxy configs return 404. Wrapper returns the upstream payload as-is.

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
