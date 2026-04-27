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
        first_model_id = (
            models[0]["id"] if isinstance(models, list) and models else None
        )
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
                    results.append(("get_credential_by_model", False, f"APIError {e.status_code}: {e}"))
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
