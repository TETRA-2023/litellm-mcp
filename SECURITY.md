# Security Policy

## Reporting a vulnerability

Do not open a public issue. Open a private security advisory on this repository:
[Security → Report a vulnerability](https://github.com/TETRA-2023/litellm-mcp/security/advisories/new).
Include the affected version or image digest and the steps to reproduce.

## Supported versions

Only the latest release receives fixes.

## Scope

The server exposes the LiteLLM proxy admin and execution API (models, keys, users, teams, budgets, MCP
gateway, pass-through) through MCP tools.

- **Transport**: the intended use is `stdio`. The `streamable-http` transport has no authentication of its
  own and binds to `127.0.0.1` by default. Do not expose it on a network unless an authenticating gateway
  sits in front of it.
- **API key**: `LITELLM_API_KEY` decides what every tool can do. A master key gives the full admin surface
  (key generation, user and budget changes, model and credential edits). Prefer a virtual key with the
  least rights the agent needs, and never commit `.env` files.
- **Pass-through**: the `passthrough` tool forwards requests to the providers configured in LiteLLM with
  the proxy's credentials. Leave it out of the agent's allowed tools if it is not needed.

## Out of scope

- LiteLLM itself: report to [BerriAI/litellm](https://github.com/BerriAI/litellm).
