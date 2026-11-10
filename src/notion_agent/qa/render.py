"""Terminal rendering of a grounded Answer."""

from __future__ import annotations

from rich.console import Console
from rich.markup import escape

from notion_agent.qa.grounded import Answer

BADGE = {
    "high": "[green]● 신뢰도 높음[/green]",
    "medium": "[yellow]◐ 신뢰도 보통[/yellow]",
    "low": "[red]○ 신뢰도 낮음[/red]",
    "none": "[dim]–[/dim]",
}


def render_answer(console: Console, a: Answer) -> None:
    if not a.answerable:
        console.print(
            f"[yellow]기록에서 근거를 찾지 못했습니다.[/yellow] "
            f"[dim](최고 검색 점수 {a.retrieval_score:.2f})[/dim]"
        )
        return

    for s in a.sentences:
        cites = "".join(f"[{n}]" for n in s.citations)
        text = escape(s.text)
        if s.support == "unsupported":
            text = f"[yellow underline]{text}[/yellow underline]"
        elif s.support == "partial":
            text = f"[underline]{text}[/underline]"
        console.print(f"{text} [cyan]{escape(cites)}[/cyan]")

    console.print()
    n_sup = sum(s.support == "supported" for s in a.sentences)
    console.print(
        f"{BADGE[a.confidence]}  [dim]근거 문장 {n_sup}/{len(a.sentences)} · "
        f"검색 점수 {a.retrieval_score:.2f} · {a.latency_ms} ms"
        + (f" · 무효 인용 {a.dropped_citations}건 제거" if a.dropped_citations else "")
        + "[/dim]"
    )
    console.print("[bold]출처[/bold]")
    for i, h in enumerate(a.sources, 1):
        where = " › ".join([h.page_title, *h.heading_path])
        console.print(
            f"  [{i}] {escape(where)} [dim]({h.last_edited_time:%Y-%m-%d}, "
            f"관련도 {h.rerank_score:.2f})[/dim]\n      [link={h.cite_url}]{h.cite_url}[/link]",
            highlight=False,
        )
    note = "신뢰도는 '내 노트에 근거하는가'를 뜻하며 사실 여부를 보장하지 않습니다."
    console.print(f"[dim]{note}[/dim]")


def render_answer_json(console: Console, a: dict | None) -> None:
    """Plain rendering of the agent's serialised answer (dict from answer_to_json)."""
    if not a:
        console.print("(no answer)")
        return
    if a.get("kind") == "action":
        for s in a["steps"]:
            console.print(f"{s['tool']}: {'완료' if s['ok'] else '실패/거부'} → {s['observation']}",
                          markup=False)
        return
    if not a.get("answerable"):
        console.print("기록에서 근거를 찾지 못했습니다.")
        return
    for s in a["sentences"]:
        cites = "".join(f"[{n}]" for n in s["citations"])
        console.print(f"{s['text']} {cites} ({s['support']})", markup=False)
    for i, h in enumerate(a["sources"], 1):
        console.print(f"  [{i}] {h['title']} {h['cite_url']}", markup=False)
    console.print(f"신뢰도: {a['confidence']}", markup=False)
