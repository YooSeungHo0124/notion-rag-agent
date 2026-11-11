# notion-rag-agent app image: React UI (built in stage 1) + FastAPI/RAG backend on CUDA.
# torch cu128 wheels bundle the CUDA runtime, so a slim Python base + the NVIDIA container
# runtime (--gpus) is enough. Models are NOT baked in: mount the Hugging Face cache.
# In a container the LLM is the Anthropic API (LLM_PROVIDER=anthropic_api): the personal
# `claude -p` subscription login is never copied into an image.

FROM node:24.11-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

FROM python:3.12.8-slim AS app
COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 HF_HOME=/models \
    LLM_PROVIDER=anthropic_api MLFLOW_DISABLE_AGENT_HINT=1
WORKDIR /app
# deps first (cached layer), then the project itself
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project \
      --extra ingest --extra rag --extra api --extra api-llm
COPY src/ src/
COPY config/ config/
COPY scenario/ scenario/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra ingest --extra rag --extra api --extra api-llm
COPY --from=web /web/dist web/dist
RUN useradd -m -u 1000 nra && mkdir -p /app/data /models && chown -R nra /app/data /models
USER nra
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
# 0.0.0.0 inside the container only; publish to 127.0.0.1 on the host (see compose)
CMD ["nra", "serve", "--host", "0.0.0.0", "--port", "8000"]
