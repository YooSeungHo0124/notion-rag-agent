"""Keep gold labels valid across re-chunking.

`--save`  : store each gold chunk's text in the dataset (run before re-indexing).
(default) : point gold_chunk_id at the new chunk on the same page whose text overlaps the
            saved gold text the most (share of gold 5-grams it contains).

usage: uv run python -m eval.remap_gold --save ; nra sync --full ; uv run python -m eval.remap_gold
"""

from __future__ import annotations

import argparse

from eval.common import DATASET, all_chunks, open_store, read_jsonl, write_jsonl


def grams(t: str, n: int = 5) -> set[str]:
    t = " ".join(t.split())
    return {t[i : i + n] for i in range(max(len(t) - n + 1, 1))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    rows = read_jsonl(DATASET)
    chunks = all_chunks(open_store())
    if args.save:
        by_id = {c["chunk_id"]: c for c in chunks}
        for r in rows:
            if r.get("gold_chunk_id") and "gold_text" not in r:
                r["gold_text"] = by_id[r["gold_chunk_id"]]["text"]
        write_jsonl(DATASET, rows)
        print("saved gold texts")
        return
    by_page: dict[str, list[dict]] = {}
    for c in chunks:
        by_page.setdefault(c["page_id"], []).append(c)
    moved = 0
    for r in rows:
        if not r.get("gold_text"):
            continue
        g = grams(r["gold_text"])
        best = max(by_page.get(r["gold_page_id"], []),
                   key=lambda c: len(g & grams(c["text"])) / len(g),  # containment
                   default=None)
        if best and best["chunk_id"] != r["gold_chunk_id"]:
            r["gold_chunk_id"] = best["chunk_id"]
            moved += 1
    write_jsonl(DATASET, rows)
    print(f"remapped {moved} gold chunk ids")


if __name__ == "__main__":
    main()
