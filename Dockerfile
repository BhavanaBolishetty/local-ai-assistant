# Single shared image for both services (backend + Streamlit frontend);
# docker-compose.yml picks which command each container runs. Based on
# uv's own recommended base image (uv preinstalled, no extra install step).
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Copy dependency manifests first so `uv sync` is cached across rebuilds
# unless pyproject.toml/uv.lock actually change.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# No EXPOSE/CMD here — docker-compose.yml sets the port and command per
# service (uvicorn for the backend, streamlit for the frontend), since
# both run from this same image.
