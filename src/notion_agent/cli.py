"""Command line entry point: `nra <command>`."""

from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=True, help="Personal Notion RAG agent")
console = Console()


@app.command()
def ask(question: str) -> None:
    """Ask the LLM directly (F0 smoke test; RAG comes in F2)."""
    from notion_agent.llm import get_llm

    resp = get_llm().complete(question)
    console.print(resp.text)
    console.print(f"[dim]{resp.model} · {resp.latency_ms} ms[/dim]")


@app.command()
def doctor() -> None:
    """Check the local environment (LLM CLI, GPU, config)."""
    import shutil

    from notion_agent.config import get_settings

    s = get_settings()
    console.print(f"llm provider : {s.llm_provider} ({s.llm_model})")
    console.print(f"claude CLI   : {shutil.which('claude') or 'NOT FOUND'}")
    console.print(f"notion token : {'set' if s.notion_token else 'missing'}")
    console.print(f"qdrant       : {s.qdrant_url or f'embedded at {s.qdrant_path}'}")
    try:
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable"
        console.print(f"torch        : {torch.__version__} · cuda {torch.version.cuda} · {gpu}")
    except ImportError:
        console.print("torch        : not installed (uv sync --extra rag)")


if __name__ == "__main__":
    app()
