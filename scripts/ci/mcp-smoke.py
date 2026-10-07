#!/usr/bin/env python3
"""Container smoke test for an MCP server image: initialize + tools/list.

A green unit-test suite does not prove the server can start: langfuse-mcp
v1.1.6/v1.1.7 passed CI, shipped :stable, and crash-looped in prod on
2026-10-07 because the entrypoint failed at import. This test talks to the
real image, through its real entrypoint and transport, and fails unless the
server answers tools/list with at least one tool.

Stdlib only, Python >= 3.8 (the self-hosted runner image ships 3.8).

  stdio:  mcp-smoke.py stdio -- docker run -i --rm IMAGE [ARGS...]
  http:   mcp-smoke.py http URL [--bearer TOKEN] [--wait SECONDS]

In http mode run this script where URL is reachable, e.g. in a sidecar that
shares the server's network namespace, with the script piped on stdin (no
host port to collide on, no bind mount for a self-hosted runner's host
daemon to miss):
  docker run --rm -i --network container:SUT python:3.12-alpine \
    python - http http://127.0.0.1:8000/mcp < scripts/ci/mcp-smoke.py
"""

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

PROTOCOL = "2025-03-26"
INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": PROTOCOL,
        "capabilities": {},
        "clientInfo": {"name": "ci-smoke", "version": "1"},
    },
}
INITIALIZED = {"jsonrpc": "2.0", "method": "notifications/initialized"}
TOOLS = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}


def fail(msg):
    """Print a failure and exit non-zero."""
    print("SMOKE FAIL: " + msg, file=sys.stderr)
    sys.exit(1)


def check_tools(resp):
    """Fail unless a tools/list response carries at least one tool."""
    if "error" in resp:
        fail("tools/list returned an error: " + json.dumps(resp["error"]))
    tools = resp.get("result", {}).get("tools", [])
    if not tools:
        fail("tools/list returned no tools")
    names = sorted(t.get("name", "?") for t in tools)
    print(
        "SMOKE OK: {} tools ({}{})".format(
            len(names), ", ".join(names[:5]), ", ..." if len(names) > 5 else ""
        )
    )


def run_stdio(cmd, timeout):
    """Drive a stdio server: initialize, initialized, tools/list."""
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True
    )
    payload = "\n".join(json.dumps(m) for m in (INIT, INITIALIZED, TOOLS)) + "\n"
    try:
        # Keep stdin open until tools/list answers: some servers exit on EOF.
        proc.stdin.write(payload)
        proc.stdin.flush()
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                fail(f"server exited before answering tools/list (exit code {proc.poll()})")
            line = line.strip()
            if not line.startswith("{"):
                continue
            msg = json.loads(line)
            if msg.get("id") == 1 and "error" in msg:
                fail("initialize returned an error: " + json.dumps(msg["error"]))
            if msg.get("id") == 2:
                check_tools(msg)
                return
        fail(f"no tools/list answer within {timeout}s")
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.kill()


def post(url, body, headers):
    """POST one JSON-RPC message; return (message, session id). Handles SSE."""
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers=headers, method="POST"
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        raw = r.read().decode()
        ctype = r.headers.get("Content-Type", "")
        sid = r.headers.get("Mcp-Session-Id")
    if not raw.strip():
        return None, sid
    if "text/event-stream" in ctype:
        msgs = [
            json.loads(line[5:].strip())
            for line in raw.splitlines()
            if line.startswith("data:") and line[5:].strip()
        ]
        msgs = [m for m in msgs if m.get("id") == body.get("id")]
        return (msgs[-1] if msgs else None), sid
    return json.loads(raw), sid


def run_http(url, bearer, wait):
    """Drive a streamable-http server, waiting up to `wait`s for it to listen."""
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    if bearer:
        headers["Authorization"] = "Bearer " + bearer
    deadline = time.time() + wait
    last = None
    while True:
        try:
            resp, sid = post(url, INIT, headers)
            break
        except urllib.error.HTTPError as e:
            fail(f"initialize -> HTTP {e.code}: {e.read()[:300]}")
        except Exception as e:  # not listening yet
            last = e
            if time.time() > deadline:
                fail(f"server not reachable at {url} after {wait}s: {last}")
            time.sleep(1)
    if not resp or "error" in resp:
        fail("bad initialize response: " + json.dumps(resp))
    if sid:
        headers["Mcp-Session-Id"] = sid
    headers["Mcp-Protocol-Version"] = resp.get("result", {}).get("protocolVersion", PROTOCOL)
    try:
        post(url, INITIALIZED, headers)
        resp, _ = post(url, TOOLS, headers)
    except urllib.error.HTTPError as e:
        fail(f"tools/list -> HTTP {e.code}: {e.read()[:300]}")
    if not resp:
        fail("empty tools/list response")
    check_tools(resp)


def main(argv):
    """Parse argv and run the selected transport."""
    if len(argv) >= 3 and argv[1] == "stdio" and argv[2] == "--":
        run_stdio(argv[3:], timeout=60)
    elif len(argv) >= 3 and argv[1] == "http":
        url, bearer, wait = argv[2], None, 60
        rest = argv[3:]
        while rest:
            flag = rest.pop(0)
            if flag == "--bearer":
                bearer = rest.pop(0)
            elif flag == "--wait":
                wait = int(rest.pop(0))
            else:
                fail("unknown flag " + flag)
        run_http(url, bearer, wait)
    else:
        print(__doc__, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main(sys.argv)
