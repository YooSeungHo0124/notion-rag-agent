"""Smoke test of the real RAG pipeline (real models, temp embedded Qdrant).

Usage: uv run python scripts/smoke_rag.py [--dense Qwen/Qwen3-Embedding-0.6B] [--device cuda]
"""

from __future__ import annotations

import argparse
import tempfile
import time
from datetime import UTC, datetime

import torch

from notion_agent.config import get_settings
from notion_agent.ingest.models import Block, NotionPage
from notion_agent.rag.embedder import DenseEmbedder, SparseEmbedder
from notion_agent.rag.reranker import Reranker
from notion_agent.rag.retriever import Retriever
from notion_agent.rag.sink import QdrantPageSink
from notion_agent.rag.store import QdrantStore

DOCS = {
    "Kubernetes 운영 메모": [
        ("heading_2", "배포 재시작", 2),
        (
            "paragraph",
            "파드가 멈추면 kubectl rollout restart deployment/<name> 으로 재시작한다.",
            None,
        ),
        (
            "code",
            "kubectl rollout restart deployment/api\nkubectl rollout status deployment/api",
            None,
        ),
    ],
    "김치찌개 레시피": [
        (
            "paragraph",
            "돼지고기 앞다리살을 볶고 묵은지와 두부를 넣어 20분 끓인다. 고춧가루 한 스푼.",
            None,
        ),
    ],
    "Reading list 2026": [
        ("heading_2", "Distributed systems", 2),
        ("paragraph", "Designing Data-Intensive Applications; the Raft consensus paper.", None),
    ],
    "회의록 - 검색 품질": [
        (
            "paragraph",
            "하이브리드 검색(dense+sparse)과 리랭커를 적용해 한국어 질의 정확도를 개선하기로 함.",
            None,
        ),
    ],
}
QUERIES = ["파드 재시작 명령어가 뭐였지?", "what book did I note about consensus?"]


def main() -> None:
    ap = argparse.ArgumentParser()
    s = get_settings()
    ap.add_argument("--dense", default=s.file.embedding.dense_model)
    ap.add_argument("--device", default=s.file.embedding.device)
    args = ap.parse_args()
    dev = args.device
    t = datetime(2026, 11, 1, tzinfo=UTC)
    pages = [
        NotionPage(
            page_id=f"page{i}",
            title=title,
            url=f"https://www.notion.so/page{i}",
            created_time=t,
            last_edited_time=t,
            blocks=[
                Block(block_id=f"p{i}b{j}", type=ty, text=tx, heading_level=lv)
                for j, (ty, tx, lv) in enumerate(blocks)
            ],
        )
        for i, (title, blocks) in enumerate(DOCS.items())
    ]
    dense = DenseEmbedder(args.dense, dev)
    sparse = SparseEmbedder(s.file.embedding.sparse_model, dev)
    rer = Reranker(s.file.retrieval.reranker_model, dev)
    for name, load in (
        ("dense", lambda: dense.model),
        ("sparse", lambda: sparse.model),
        ("reranker", lambda: rer.model),
    ):
        t0 = time.perf_counter()
        load()
        print(
            f"load {name}: {time.perf_counter() - t0:.1f}s, "
            f"alloc={torch.cuda.memory_allocated() / 2**30:.2f} GiB"
        )
    with tempfile.TemporaryDirectory() as d:
        store = QdrantStore("smoke", path=d)
        sink = QdrantPageSink(store, dense, sparse, s.file.chunking)
        t0 = time.perf_counter()
        n = sum(sink.upsert_page(p) for p in pages)
        print(f"indexed {n} chunks in {time.perf_counter() - t0:.2f}s")
        retr = Retriever(store, dense, sparse, rer, s.file.retrieval)
        for q in QUERIES:
            retr.retrieve(q)  # warm-up
            t0 = time.perf_counter()
            hits = retr.retrieve(q, k=3)
            ms = (time.perf_counter() - t0) * 1000
            print(f"\nQ: {q}  ({ms:.0f} ms)")
            for h in hits:
                print(
                    f"  {h.rerank_score:.3f} fus={h.fusion_score:.3f} ans={h.answerable} "
                    f"{h.page_title} :: {h.text[:50]!r}"
                )
        store.client.close()
    print(f"\npeak GPU allocated: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
