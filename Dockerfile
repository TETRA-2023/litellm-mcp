FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN groupadd --system appgroup && useradd --system --gid appgroup appuser

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project

COPY src/ src/

RUN uv sync --frozen --no-editable

ENV UV_CACHE_DIR=/tmp/uv-cache

USER appuser

ENTRYPOINT ["/app/.venv/bin/python", "src/server.py"]
