"""Chunking quality report (no LLM): size distribution and structural red flags.

usage: uv run python -m eval.chunk_stats
"""

from __future__ import annotations

import re
import statistics
from collections import Counter

from eval.common import all_chunks, open_store
from notion_agent.config import get_settings


def pct(xs: list[int], q: float) -> int:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main() -> None:
    cfg = get_settings().file.chunking
    chunks = all_chunks(open_store())
    sizes = [len(c["text"]) for c in chunks]
    per_page = Counter(c["page_id"] for c in chunks)
    n = len(chunks)

    def share(pred) -> str:  # type: ignore[no-untyped-def]
        k = sum(1 for c in chunks if pred(c))
        return f"{k} ({k / n:.1%})"

    print(f"chunks={n} pages={len(per_page)} max_chars={cfg.max_chars}")
    print(
        f"size chars: p10={pct(sizes, .1)} p50={pct(sizes, .5)} p90={pct(sizes, .9)} "
        f"max={max(sizes)} mean={statistics.mean(sizes):.0f}"
    )
    print(f"chunks/page: p50={pct(list(per_page.values()), .5)} max={max(per_page.values())}")
    print("red flags:")
    print("  tiny (<80 chars, little meaning)      ", share(lambda c: len(c["text"]) < 80))
    print("  over max_chars (unsplit big block)     ", share(lambda c: len(c["text"]) > cfg.max_chars * 1.2))
    print("  likely truncated at embed (>4000 chars)", share(lambda c: len(c["text"]) > 4000))
    print("  properties-only (DB row, no body)      ", share(lambda c: c["text"].count("\n") < 6 and re.match(r"^[^\n]+: ", c["text"])))
    print("  no heading context                     ", share(lambda c: not c.get("heading_path")))
    print("  unbalanced code fence (split mid-code) ", share(lambda c: c["text"].count("```") % 2 == 1))


if __name__ == "__main__":
    main()
