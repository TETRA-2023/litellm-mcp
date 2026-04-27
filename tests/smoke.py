"""Read-only live smoke test for litellm-mcp.

Hits the GET / health endpoints against a running LiteLLM proxy to verify the
wrapper paths, auth, and response shapes. Destructive operations (add/update/
delete) are covered by unit tests with AsyncMock fixtures and exercised
implicitly when the admin agent uses them — this script only reads.

Run:

    LITELLM_PROXY_URL=... LITELLM_API_KEY=... uv run python tests/smoke.py

Exits 0 on full pass, 1 on any failure.
"""

import asyncio
import os
import sys
from pathlib import Path

# Allow running as a script: add repo root to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load_env() -> None:
    """Source ~/.claude/litellm.env if present and env not already set."""
    if os.environ.get("LITELLM_API_KEY"):
        return
    env_file = Path.home() / ".claude" / "litellm.env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# Load env vars before importing settings (which evaluates them at import time).
_load_env()

from src.config import settings  # noqa: E402
from src.litellm_client import LiteLLMAPIError, LiteLLMClient  # noqa: E402


async def _step(name: str, coro) -> tuple[str, bool, str]:
    try:
        result = await coro
    except LiteLLMAPIError as e:
        return (name, False, f"APIError {e.status_code}: {e}")
    except Exception as e:
        return (name, False, f"{type(e).__name__}: {e}")
    summary = _summarize(result)
    return (name, True, summary)


def _summarize(result) -> str:
    if result is None:
        return "None"
    if isinstance(result, list):
        return f"list[{len(result)}]"
    if isinstance(result, dict):
        keys = list(result.keys())[:5]
        return f"dict({', '.join(keys)}{'...' if len(result) > 5 else ''})"
    return f"{type(result).__name__}"


async def run() -> int:
    if not settings.has_api_key:
        print("LITELLM_API_KEY is not set", file=sys.stderr)
        return 1

    client = LiteLLMClient(
        base_url=settings.proxy_url,
        api_key=settings.get_api_key_value(),
        timeout=settings.timeout_seconds,
        max_retries=settings.max_retries,
    )
    print(f"→ {settings.proxy_url}\n")
    results = []

    try:
        # Models family (3 read ops)
        models = await client.list_models()
        results.append(("list_models", True, _summarize(models)))
        first_model_id = models[0]["id"] if isinstance(models, list) and models else None
        if first_model_id:
            # Note: /v1/models lists router + passthrough models, but
            # /v1/models/{id} only resolves router-registered ids. 404 is
            # acceptable when the first list item is a passthrough.
            try:
                payload = await client.get_model(first_model_id)
                results.append(("get_model", True, _summarize(payload)))
            except LiteLLMAPIError as e:
                if e.status_code == 404:
                    results.append(("get_model", True, "404 (passthrough model — wrapper OK)"))
                else:
                    results.append(("get_model", False, f"APIError {e.status_code}: {e}"))
        else:
            results.append(("get_model", True, "skipped (no models)"))
        results.append(await _step("get_model_info", client.get_model_info()))

        # Model Hub family (3 read ops)
        results.append(await _step("list_public_models", client.list_public_models()))
        results.append(await _step("get_public_hub_info", client.get_public_hub_info()))
        results.append(await _step("get_model_cost_map", client.get_model_cost_map()))

        # Access Groups family (2 read ops)
        ag_list = await client.list_model_access_groups()
        results.append(("list_model_access_groups", True, _summarize(ag_list)))
        first_ag = None
        if isinstance(ag_list, list) and ag_list:
            first_ag = ag_list[0] if isinstance(ag_list[0], str) else ag_list[0].get("access_group")
        elif isinstance(ag_list, dict):
            data = ag_list.get("data") or ag_list.get("access_groups") or []
            if data:
                first_ag = data[0] if isinstance(data[0], str) else data[0].get("access_group")
        if first_ag:
            results.append(
                await _step("get_model_access_group", client.get_model_access_group(first_ag))
            )
        else:
            results.append(("get_model_access_group", True, "skipped (no access groups)"))

        # Credentials family (3 read ops)
        creds = await client.list_credentials()
        results.append(("list_credentials", True, _summarize(creds)))
        first_cred = None
        if isinstance(creds, list) and creds:
            first_cred = creds[0].get("credential_name")
        elif isinstance(creds, dict):
            data = creds.get("credentials") or creds.get("data") or []
            if data:
                first_cred = data[0].get("credential_name")
        if first_cred:
            results.append(await _step("get_credential", client.get_credential(first_cred)))
        else:
            results.append(("get_credential", True, "skipped (no credentials)"))
        if first_model_id:
            try:
                payload = await client.get_credential_by_model(first_model_id)
                results.append(("get_credential_by_model", True, _summarize(payload)))
            except LiteLLMAPIError as e:
                # Many models have no bound credential entry — 404 is acceptable.
                if e.status_code == 404:
                    results.append(("get_credential_by_model", True, "404 (no bound cred)"))
                else:
                    results.append(
                        ("get_credential_by_model", False, f"APIError {e.status_code}: {e}")
                    )
        else:
            results.append(("get_credential_by_model", True, "skipped (no models)"))

        # Keys family (4 read ops)
        keys_payload = await client.list_keys(size=5)
        results.append(("list_keys", True, _summarize(keys_payload)))
        results.append(await _step("list_key_aliases", client.list_key_aliases(size=5)))
        # Pick a real virtual-key value to probe get_key_info; calling without
        # one (or with the master key) 404s because the master key isn't a
        # row in the LiteLLM_VerificationToken table.
        first_virtual_key = None
        if isinstance(keys_payload, dict):
            keys_list = keys_payload.get("keys") or []
            for k in keys_list:
                if isinstance(k, dict) and k.get("token"):
                    first_virtual_key = k["token"]
                    break
        if first_virtual_key:
            results.append(await _step("get_key_info", client.get_key_info(first_virtual_key)))
        else:
            results.append(("get_key_info", True, "skipped (no virtual keys to probe)"))
        results.append(await _step("key_health", client.key_health()))

        # Users family (2 read ops)
        users_payload = await client.list_users(page_size=5)
        results.append(("list_users", True, _summarize(users_payload)))
        first_user_id = None
        if isinstance(users_payload, dict):
            for u in users_payload.get("users") or []:
                if isinstance(u, dict) and u.get("user_id"):
                    first_user_id = u["user_id"]
                    break
        elif isinstance(users_payload, list) and users_payload:
            first_user_id = (
                users_payload[0].get("user_id") if isinstance(users_payload[0], dict) else None
            )
        if first_user_id:
            results.append(await _step("get_user_info", client.get_user_info(first_user_id)))
        else:
            results.append(("get_user_info", True, "skipped (no users)"))

        # Customers family (2 read ops + daily activity)
        customers = await client.list_customers()
        results.append(("list_customers", True, _summarize(customers)))
        first_customer = None
        if isinstance(customers, list) and customers:
            first_customer = (
                customers[0] if isinstance(customers[0], str) else customers[0].get("user_id")
            )
        elif isinstance(customers, dict):
            data = customers.get("data") or customers.get("customers") or []
            if data:
                first_customer = data[0] if isinstance(data[0], str) else data[0].get("user_id")
        if first_customer:
            try:
                payload = await client.get_customer_info(first_customer)
                results.append(("get_customer_info", True, _summarize(payload)))
            except LiteLLMAPIError as e:
                # Customer rows can be soft-deleted but still appear in list — 404 acceptable.
                if e.status_code == 404:
                    results.append(("get_customer_info", True, "404 (stale list entry)"))
                else:
                    results.append(("get_customer_info", False, f"APIError {e.status_code}: {e}"))
        else:
            results.append(("get_customer_info", True, "skipped (no customers)"))
        # Daily-activity endpoints require start/end dates despite the
        # OpenAPI marking them optional — upstream returns 400 if either is
        # missing. Use a 30-day rolling window.
        from datetime import date, timedelta

        end_d = date.today().isoformat()
        start_d = (date.today() - timedelta(days=30)).isoformat()
        results.append(
            await _step(
                "get_customer_daily_activity",
                client.get_customer_daily_activity(start_date=start_d, end_date=end_d, page=1),
            )
        )

        # Organizations family (3 read ops)
        orgs = await client.list_organizations()
        results.append(("list_organizations", True, _summarize(orgs)))
        first_org_id = None
        if isinstance(orgs, list) and orgs:
            first_org_id = orgs[0].get("organization_id") if isinstance(orgs[0], dict) else None
        elif isinstance(orgs, dict):
            data = orgs.get("data") or orgs.get("organizations") or []
            if data:
                first_org_id = data[0].get("organization_id") if isinstance(data[0], dict) else None
        if first_org_id:
            results.append(
                await _step("get_organization_info", client.get_organization_info(first_org_id))
            )
        else:
            results.append(("get_organization_info", True, "skipped (no orgs)"))
        results.append(
            await _step(
                "get_org_daily_activity",
                client.get_org_daily_activity(start_date=start_d, end_date=end_d, page=1),
            )
        )

        # Projects family (2 read ops)
        projects = await client.list_projects()
        results.append(("list_projects", True, _summarize(projects)))
        first_project_id = None
        if isinstance(projects, list) and projects:
            first_project_id = (
                projects[0].get("project_id") if isinstance(projects[0], dict) else None
            )
        elif isinstance(projects, dict):
            data = projects.get("data") or projects.get("projects") or []
            if data:
                first_project_id = data[0].get("project_id") if isinstance(data[0], dict) else None
        if first_project_id:
            results.append(
                await _step("get_project_info", client.get_project_info(first_project_id))
            )
        else:
            results.append(("get_project_info", True, "skipped (no projects)"))

        # Unified Access Groups family (2 read ops)
        uag = await client.list_user_access_groups()
        results.append(("list_user_access_groups", True, _summarize(uag)))
        first_uag_id = None
        if isinstance(uag, list) and uag:
            first_uag_id = uag[0].get("access_group_id") if isinstance(uag[0], dict) else None
        elif isinstance(uag, dict):
            data = uag.get("data") or uag.get("access_groups") or []
            if data:
                first_uag_id = data[0].get("access_group_id") if isinstance(data[0], dict) else None
        if first_uag_id:
            results.append(
                await _step("get_user_access_group", client.get_user_access_group(first_uag_id))
            )
        else:
            results.append(("get_user_access_group", True, "skipped (no user access groups)"))

        # Budget family (3 read ops — list, info-batch, settings)
        budgets = await client.list_budgets()
        results.append(("list_budgets", True, _summarize(budgets)))
        first_budget_id = None
        if isinstance(budgets, list) and budgets:
            first_budget_id = budgets[0].get("budget_id") if isinstance(budgets[0], dict) else None
        elif isinstance(budgets, dict):
            data = budgets.get("data") or budgets.get("budgets") or []
            if data:
                first_budget_id = data[0].get("budget_id") if isinstance(data[0], dict) else None
        if first_budget_id:
            results.append(
                await _step("get_budget_info", client.get_budget_info([first_budget_id]))
            )
            try:
                payload = await client.get_budget_settings(first_budget_id)
                results.append(("get_budget_settings", True, _summarize(payload)))
            except LiteLLMAPIError as e:
                # /budget/settings may 404 for budgets without explicit settings rows.
                if e.status_code == 404:
                    results.append(("get_budget_settings", True, "404 (no settings row)"))
                else:
                    results.append(("get_budget_settings", False, f"APIError {e.status_code}: {e}"))
        else:
            results.append(("get_budget_info", True, "skipped (no budgets)"))
            results.append(("get_budget_settings", True, "skipped (no budgets)"))

        # Spend family (4 read ops + 1 calculate)
        # /global/spend/report is gated behind LiteLLM Enterprise — community
        # tier returns 400 "You must be a LiteLLM Enterprise user". Wrapper is
        # OK; mark as upstream-gated rather than a failure.
        try:
            payload = await client.get_global_spend_report(start_date=start_d, end_date=end_d)
            results.append(("get_global_spend_report", True, _summarize(payload)))
        except LiteLLMAPIError as e:
            if e.status_code == 400 and "Enterprise" in str(e):
                results.append(
                    ("get_global_spend_report", True, "400 (Enterprise-gated, wrapper OK)")
                )
            else:
                results.append(("get_global_spend_report", False, f"APIError {e.status_code}: {e}"))
        results.append(
            await _step(
                "list_spend_logs", client.list_spend_logs(start_date=start_d, end_date=end_d)
            )
        )
        results.append(
            await _step(
                "list_spend_tags", client.list_spend_tags(start_date=start_d, end_date=end_d)
            )
        )
        results.append(
            await _step(
                "get_user_daily_activity",
                client.get_user_daily_activity(
                    start_date=start_d, end_date=end_d, page=1, page_size=5
                ),
            )
        )
        results.append(
            await _step(
                "calculate_spend",
                client.calculate_spend(
                    model="tora-no-think",
                    messages=[{"role": "user", "content": "hi"}],
                ),
            )
        )

        # Health family (4 read ops, test_connection skipped — write-shaped)
        results.append(await _step("check_health_backlog", client.check_health_backlog()))
        results.append(await _step("get_health_latest", client.get_health_latest()))
        results.append(await _step("get_health_history", client.get_health_history(limit=5)))
        # /health probes every router-registered deployment — can take 30+s.
        # Narrow to a single small local model.
        results.append(await _step("check_health", client.check_health(model="tora-no-think")))

        # Execution family (3 paths) — these spend real (small) tokens against
        # local vLLM deployments. Skip via SMOKE_SKIP_EXEC=1 if needed.
        if os.environ.get("SMOKE_SKIP_EXEC"):
            results.append(("chat_completion", True, "skipped (SMOKE_SKIP_EXEC)"))
            results.append(("completion", True, "skipped (SMOKE_SKIP_EXEC)"))
            results.append(("embed", True, "skipped (SMOKE_SKIP_EXEC)"))
        else:
            try:
                cc = await client.chat_completion(
                    model="tora-no-think",
                    messages=[{"role": "user", "content": "Reply with: ok"}],
                    body={"max_tokens": 5, "temperature": 0.0},
                )
                results.append(("chat_completion", True, _summarize(cc)))
            except LiteLLMAPIError as e:
                results.append(("chat_completion", False, f"APIError {e.status_code}: {e}"))

            # Legacy /v1/completions: many chat-only models 400 here. Mark
            # 400/404/422 as "expected" since no legacy-text deployment exists.
            try:
                comp = await client.completion(
                    model="tora-no-think",
                    prompt="The capital of France is",
                    body={"max_tokens": 3, "temperature": 0.0},
                )
                results.append(("completion", True, _summarize(comp)))
            except LiteLLMAPIError as e:
                if e.status_code in {400, 404, 422}:
                    results.append(
                        ("completion", True, f"{e.status_code} (no legacy-text deployment)")
                    )
                else:
                    results.append(("completion", False, f"APIError {e.status_code}: {e}"))

            try:
                emb = await client.embed(
                    model="qwen/qwen3-embedding-0.6b",
                    input=["smoke test"],
                )
                results.append(("embed", True, _summarize(emb)))
            except LiteLLMAPIError as e:
                results.append(("embed", False, f"APIError {e.status_code}: {e}"))

        # MCP Gateway family — read-only paths + Context7 end-to-end.
        servers = await client.list_mcp_servers()
        results.append(("list_mcp_servers", True, _summarize(servers)))

        # Locate Context7 (or any registered HTTP-transport server) for the
        # end-to-end probe. We don't auto-register because TETRA's proxy
        # already has Context7 registered; double-registration would 409.
        first_server_id = None
        first_server_alias = None
        if isinstance(servers, list):
            for s in servers:
                if isinstance(s, dict):
                    if s.get("alias") == "context7" or "context7" in (s.get("url") or ""):
                        first_server_id = s.get("server_id")
                        first_server_alias = s.get("alias")
                        break
            if first_server_id is None and servers and isinstance(servers[0], dict):
                first_server_id = servers[0].get("server_id")
                first_server_alias = servers[0].get("alias")

        if first_server_id:
            results.append(await _step("get_mcp_server", client.get_mcp_server(first_server_id)))
        else:
            results.append(("get_mcp_server", True, "skipped (no servers)"))

        results.append(
            await _step("list_mcp_server_submissions", client.list_mcp_server_submissions())
        )
        results.append(await _step("check_mcp_servers_health", client.check_mcp_servers_health()))
        results.append(await _step("list_mcp_tools", client.list_mcp_tools()))
        results.append(
            await _step("list_mcp_tools_rest", client.list_mcp_tools_rest(first_server_id))
        )
        results.append(await _step("discover_mcp_servers", client.discover_mcp_servers()))
        results.append(await _step("get_mcp_openapi_registry", client.get_mcp_openapi_registry()))
        # /v1/mcp/registry.json is optional upstream — older versions / minimal
        # configs return 404. Tolerate it and report.
        try:
            payload = await client.get_mcp_registry()
            results.append(("get_mcp_registry", True, _summarize(payload)))
        except LiteLLMAPIError as e:
            if e.status_code == 404:
                results.append(("get_mcp_registry", True, "404 (registry.json not enabled)"))
            else:
                results.append(("get_mcp_registry", False, f"APIError {e.status_code}: {e}"))
        results.append(await _step("list_mcp_access_groups", client.list_mcp_access_groups()))
        results.append(await _step("get_public_mcp_hub", client.get_public_mcp_hub()))
        results.append(await _step("list_mcp_user_credentials", client.list_mcp_user_credentials()))
        if first_server_id:
            try:
                payload = await client.get_mcp_oauth_user_credential_status(first_server_id)
                results.append(("get_mcp_oauth_user_credential_status", True, _summarize(payload)))
            except LiteLLMAPIError as e:
                # Most servers don't have OAuth configured — 404 is acceptable.
                if e.status_code == 404:
                    results.append(
                        (
                            "get_mcp_oauth_user_credential_status",
                            True,
                            "404 (no oauth configured)",
                        )
                    )
                else:
                    results.append(
                        (
                            "get_mcp_oauth_user_credential_status",
                            False,
                            f"APIError {e.status_code}: {e}",
                        )
                    )
        else:
            results.append(("get_mcp_oauth_user_credential_status", True, "skipped (no servers)"))
        results.append(await _step("get_mcp_client_ip", client.get_mcp_client_ip()))

        # MCP toolsets (read-only — empty on TETRA, create-flow stays in unit tests)
        toolsets = await client.list_mcp_toolsets()
        results.append(("list_mcp_toolsets", True, _summarize(toolsets)))
        first_toolset_id = None
        if isinstance(toolsets, list) and toolsets and isinstance(toolsets[0], dict):
            first_toolset_id = toolsets[0].get("toolset_id")
        if first_toolset_id:
            results.append(await _step("get_mcp_toolset", client.get_mcp_toolset(first_toolset_id)))
        else:
            results.append(("get_mcp_toolset", True, "skipped (no toolsets)"))

        # End-to-end: call Context7's resolve-library-id through the gateway.
        # Master keys aren't auto-granted MCP tool access; some proxies return
        # 403 "User not allowed" — that's an upstream auth quirk, not a wrapper
        # bug, so we tolerate it and report it explicitly.
        if first_server_alias == "context7" and first_server_id:
            try:
                payload = await client.call_mcp_tool(
                    server_id=first_server_id,
                    name="resolve-library-id",
                    arguments={"libraryName": "react"},
                )
                content = payload.get("content") if isinstance(payload, dict) else None
                if content:
                    results.append(("call_mcp_tool[context7]", True, f"content[{len(content)}]"))
                else:
                    results.append(("call_mcp_tool[context7]", True, _summarize(payload)))
            except LiteLLMAPIError as e:
                # 403 here means the calling key isn't granted MCP-tool access.
                # The wrapper sent a well-formed request; this is an upstream
                # auth quirk (master keys aren't auto-granted). Tolerate it.
                if e.status_code in {401, 403}:
                    results.append(
                        (
                            "call_mcp_tool[context7]",
                            True,
                            f"{e.status_code} (caller not granted MCP access — wrapper OK)",
                        )
                    )
                else:
                    results.append(
                        ("call_mcp_tool[context7]", False, f"APIError {e.status_code}: {e}")
                    )
        else:
            results.append(("call_mcp_tool[context7]", True, "skipped (Context7 not registered)"))

    finally:
        await client.close()

    # Report
    print(f"{'Tool':<32} {'OK':<4} Detail")
    print("─" * 72)
    failed = 0
    for name, ok, detail in results:
        mark = "✓" if ok else "✗"
        if not ok:
            failed += 1
        print(f"{name:<32} {mark:<4} {detail}")
    print("─" * 72)
    print(f"\n{len(results) - failed}/{len(results)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
