"""Command line entry point: `nra <command>`."""

from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=True, help="Personal Notion RAG agent")
console = Console()


@app.command()
def llm(prompt: str) -> None:
    """Call the LLM directly, without retrieval (adapter smoke test)."""
    from notion_agent.llm import get_llm

    resp = get_llm().complete(prompt)
    console.print(resp.text)
    console.print(f"[dim]{resp.model} · {resp.latency_ms} ms[/dim]")


@app.command()
def ask(question: str, k: int = typer.Option(0, help="sources to retrieve (0 = config)")) -> None:
    """Answer from my Notion with citations and a grounding check."""
    from notion_agent.llm import get_llm
    from notion_agent.qa.grounded import GroundedQA
    from notion_agent.qa.render import render_answer
    from notion_agent.rag.factory import build_retriever, build_verifier

    retriever = build_retriever()
    if k:
        retriever.cfg = retriever.cfg.model_copy(update={"final_k": k})
    qa = GroundedQA(retriever, get_llm(), build_verifier())
    render_answer(console, qa.answer(question))


@app.command()
def agent(question: str) -> None:
    """LangGraph agent: plan -> tools -> grounded answer; asks before any Notion write."""
    import json
    import uuid

    from notion_agent.agent.factory import build_agent
    from notion_agent.qa.render import render_answer_json

    ag = build_agent()
    tid = uuid.uuid4().hex
    r = ag.run(question, tid)
    while r["status"] == "awaiting_approval":
        ap = r["approval"]
        console.print(f"[yellow]승인 필요[/yellow] {ap['tool']} "
                      f"{json.dumps(ap['args'], ensure_ascii=False)}", markup=True)
        r = ag.resume(tid, typer.confirm("실행할까요?", default=False))
    for i, s in enumerate(r["steps"], 1):
        console.print(f"[dim]{i}. {s['agent']}·{s['tool']} {s['args']} ok={s['ok']}[/dim]")
    render_answer_json(console, r["answer"])
    console.print(f"[dim]{r['done_reason']} · {r['total_ms']} ms[/dim]")


@app.command()
def mcp() -> None:
    """Run the read-only `my-memory` MCP server on stdio."""
    import logging

    logging.basicConfig(level=logging.WARNING)  # stdout is the MCP channel
    from notion_agent.mcp_server import main

    main()


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


@app.command()
def sync(
    full: bool = typer.Option(False, "--full", help="Reprocess every page"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print actions, write nothing"),
) -> None:
    """Incrementally sync Notion pages into the index (F1)."""
    import logging

    from notion_agent.config import get_settings
    from notion_agent.ingest.notion_reader import NotionReader
    from notion_agent.ingest.state import SyncState
    from notion_agent.ingest.sync import PrintSink
    from notion_agent.ingest.sync import sync as run_sync

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    s = get_settings()
    try:
        reader = NotionReader.from_settings(s)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    if dry_run:
        sink = PrintSink(lambda msg: console.print(msg, markup=False, highlight=False))
    else:
        try:
            from notion_agent.rag.factory import build_sink
        except ImportError as exc:
            console.print(f"[red]RAG sink unavailable (F2 not installed?): {exc}[/red]")
            console.print("Use --dry-run to test ingestion only.")
            raise typer.Exit(1) from exc
        sink = build_sink(s)

    state = SyncState(s.state_db)
    try:
        report = run_sync(reader, sink, state, full=full, commit=not dry_run)
    finally:
        state.close()
    console.print(report.summary())
    for err in report.errors:
        console.print(f"[yellow]  {err}[/yellow]")
    if report.failed:
        raise typer.Exit(2)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind address (keep local: no auth)"),
    port: int = typer.Option(8000, help="Port"),
    demo: bool = typer.Option(False, "--demo", help="In-memory fake data, no models (UI dev)"),
) -> None:
    """Run the web UI + API server (F7)."""
    import logging

    try:
        import uvicorn

        from notion_agent.api.app import create_app
    except ImportError as exc:
        console.print(f"[red]API extra not installed: {exc}[/red] (uv sync --extra api)")
        raise typer.Exit(1) from exc

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if host not in ("127.0.0.1", "localhost", "::1"):
        console.print(f"[yellow]경고: {host} 로 바인딩합니다. 인증이 없으니 주의하세요.[/yellow]")
    loader = None
    if demo:
        from notion_agent.api.demo import load_demo_services

        loader = load_demo_services
    console.print(f"http://{host}:{port}  (models load in the background; see /api/health)")
    uvicorn.run(create_app(loader), host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
