# notion-rag-agent

Personal Notion AI agent: a NotebookLM-style assistant over my own Notion, with hybrid RAG,
LangGraph tool-calling, MCP, grounded citations and evaluation.

> Status: **F0** (project skeleton + LLM adapter). Roadmap and design notes live in [`docs/`](docs/).

## Architecture (target)

```
Notion ──sync──► chunk + context prefix ──► Qwen3-Embedding-4B (dense) + bge-m3 (sparse) ──► Qdrant
query ─► rewrite ─► hybrid top-50 ─► bge-reranker ─► top-5 ─► LangGraph agents ─► grounded answer + citations
                                                    └─ tool registry / executor (writes need approval)
```

- **App** runs in a local `uv` venv. **Infra** (Qdrant, Phoenix tracing) runs in Docker, bound to `127.0.0.1`.
- **LLM** goes through `LLMClient`: `claude_cli` (Claude subscription via `claude -p`, personal use only)
  or `anthropic_api`. Built-in tools are disabled on every call; tools are executed by our own code.

## Quickstart

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # once
uv sync                       # core
uv sync --extra rag --extra ingest   # embeddings, Qdrant, Notion
cp .env.example .env          # fill NOTION_TOKEN
docker compose up -d          # optional: Qdrant server + Phoenix
uv run nra doctor
uv run nra ask "hello"
```

## Roadmap

| Step | Scope |
|---|---|
| F0 | Skeleton, `LLMClient` (claude CLI) ✅ |
| F1 | Notion ingestion + incremental sync |
| F2 | Chunking, hybrid embeddings, Qdrant, reranking, grounded Q&A |
| F3 | `my-memory` MCP server |
| F4 | Tool registry/executor, LangGraph multi-agent, write approval |
| F5 | Memory (checkpointer + long-term profile) |
| F6 | Evaluation (retrieval, faithfulness, tool selection) + tracing |
| F7 | FastAPI + NotebookLM-style web UI |
