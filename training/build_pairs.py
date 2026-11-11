"""Synthetic (question -> chunk) pairs for reranker fine-tuning.

Claude writes one natural question per chunk (same prompt as the eval set builder).
Pages that hold a gold chunk of eval/dataset.jsonl are excluded entirely, so the
eval set never leaks into training. Pages are then split into train / test by a hash of
the page id, giving a held-out synthetic test set much larger than the 40-question eval.

Output (gitignored, contains note text): data/rerank/pairs.jsonl

    uv run python -m training.build_pairs --n 900 [--store data/qdrant_copy]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eval.build_dataset import ITEMS, SYSTEM
from eval.common import DATASET, all_chunks, read_jsonl
from notion_agent.config import PROJECT_ROOT, get_settings
from notion_agent.llm import get_llm
from notion_agent.rag.store import QdrantStore

OUT = PROJECT_ROOT / "data" / "rerank" / "pairs.jsonl"


def split_of(page_id: str, test_frac: float = 0.2) -> str:
    h = int(hashlib.sha1(page_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test_frac else "train"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--store", type=Path, default=None, help="copy of the qdrant dir")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    s = get_settings()
    store = QdrantStore(s.file.retrieval.collection, path=args.store or s.qdrant_path)
    eval_pages = {r["gold_page_id"] for r in read_jsonl(DATASET) if r.get("gold_page_id")}
    chunks = [c for c in all_chunks(store) if len(c["text"]) >= 150
              and c["page_id"] not in eval_pages and not c["page_id"].startswith("doc:")]
    random.Random(args.seed).shuffle(chunks)
    pool = chunks[: int(args.n * 1.3)]
    print(f"{len(chunks)} eligible chunks, generating from {len(pool)}")

    llm = get_llm()
    batches = [pool[i:i + args.batch] for i in range(0, len(pool), args.batch)]

    def gen(batch: list[dict]) -> list[dict]:
        body = "\n\n".join(f'<chunk id="{j}" title="{c["page_title"]}">\n{c["text"][:1500]}\n'
                           f"</chunk>" for j, c in enumerate(batch))
        try:
            out = llm.complete(body, system=SYSTEM, json_schema=ITEMS).structured or {}
        except Exception as e:  # one failed batch should not kill the run
            print("batch failed:", str(e)[:120])
            return []
        rows = []
        for it in out.get("items", []):
            if it.get("usable") and it["id"].isdigit() and int(it["id"]) < len(batch):
                c = batch[int(it["id"])]
                rows.append({"question": it["question"], "chunk_id": c["chunk_id"],
                             "page_id": c["page_id"], "split": split_of(c["page_id"])})
        return rows

    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with ThreadPoolExecutor(args.workers) as ex, OUT.open("w") as f:
        for rows in ex.map(gen, batches):
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += len(rows)
            f.flush()
            print(f"{n} pairs", flush=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
