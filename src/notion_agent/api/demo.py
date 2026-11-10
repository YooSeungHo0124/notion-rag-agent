"""In-memory demo services for UI development (`nra serve --demo`).

No models, no GPU, no Notion: hash-based encoders, a token-overlap reranker and a canned
"LLM" that quotes the retrieved sources. Never used by the real server path.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from datetime import UTC, datetime
from typing import Any

from qdrant_client import QdrantClient, models

from notion_agent.api.app import Services
from notion_agent.config import ChunkingConfig, RetrievalConfig
from notion_agent.ingest.models import Block, NotionPage
from notion_agent.ingest.sync import SyncReport
from notion_agent.llm.base import LLMClient, LLMResponse
from notion_agent.qa.grounded import GroundedQA
from notion_agent.rag.retriever import Retriever
from notion_agent.rag.sink import QdrantPageSink
from notion_agent.rag.store import QdrantStore


def _tok(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _h(t: str, mod: int) -> int:
    return int(hashlib.md5(t.encode()).hexdigest(), 16) % mod


class _Dense:
    dim = 64

    def _v(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for t in _tok(text):
            v[_h(t, self.dim)] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._v(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._v(text)


class _Sparse:
    def _v(self, text: str) -> models.SparseVector:
        w: dict[int, float] = {}
        for t in _tok(text):
            k = _h(t, 250000)
            w[k] = w.get(k, 0.0) + 1.0
        ks = sorted(w)
        return models.SparseVector(indices=ks, values=[w[k] for k in ks])

    def embed_documents(self, texts: list[str]) -> list[models.SparseVector]:
        return [self._v(t) for t in texts]

    def embed_query(self, text: str) -> models.SparseVector:
        return self._v(text)


class _Overlap:
    def score(self, query: str, passages: list[str]) -> list[float]:
        q = set(_tok(query))
        return [len(q & set(_tok(p))) / max(len(q), 1) for p in passages]


class _EchoLLM(LLMClient):
    """Answers by quoting the first line of the first two sources, plus one ungrounded line."""

    def complete(
        self, prompt: str, *, system: str | None = None, json_schema: dict[str, Any] | None = None
    ) -> LLMResponse:
        time.sleep(1.5)
        blocks = re.split(r"\n\n(?=\[S\d+\])", prompt.split("</sources>")[0])
        sentences: list[dict[str, Any]] = []
        for b in blocks[:2]:
            m = re.search(r"\[(S\d+)\]", b)
            body = [ln for ln in b.splitlines()[1:] if ln.strip()]
            if m and body:
                sentences.append({"text": max(body, key=len).strip(), "citations": [m.group(1)]})
        if sentences:
            sentences.append({"text": "데모용 근거 없는 문장입니다.", "citations": []})
        return LLMResponse(
            text="",
            structured={"answerable": bool(sentences), "sentences": sentences},
            model="demo",
            latency_ms=1500,
        )


def _page(
    pid: str,
    title: str,
    path: list[str],
    tags: list[str],
    day: str,
    blocks: list[tuple[str, str, int | None]],
) -> NotionPage:
    t = datetime.fromisoformat(day).replace(tzinfo=UTC)
    return NotionPage(
        page_id=pid,
        title=title,
        url=f"https://www.notion.so/{pid}",
        path=path,
        tags=tags,
        created_time=t,
        last_edited_time=t,
        blocks=[
            Block(block_id=f"{pid}b{i}", type=ty, text=tx, heading_level=lv)
            for i, (ty, tx, lv) in enumerate(blocks)
        ],
    )


DEMO_PAGES = [
    _page(
        "demo1",
        "학습 로그 9/12",
        ["프로젝트DB"],
        ["ml"],
        "2026-09-12",
        [
            ("heading_2", "OOM 대응", 2),
            ("paragraph", "GPU OOM 이 나서 배치 크기를 16 에서 8 로 줄였다.", None),
            ("paragraph", "gradient checkpointing 을 켜서 메모리를 40% 아꼈다.", None),
        ],
    ),
    _page(
        "demo2",
        "주간 회의록 10/02",
        ["회의록"],
        ["meeting"],
        "2026-10-02",
        [
            ("heading_2", "결정 사항", 2),
            ("paragraph", "평가 데이터셋은 200 문항으로 늘리기로 했다.", None),
        ],
    ),
    _page(
        "demo3",
        "김치찌개 레시피",
        ["개인메모"],
        ["food"],
        "2026-05-01",
        [("paragraph", "김치찌개: 돼지고기, 두부, 김치를 넣고 끓인다.", None)],
    ),
]


def load_demo_services() -> Services:
    store = QdrantStore("demo", client=QdrantClient(location=":memory:"))
    dense, sparse, rr = _Dense(), _Sparse(), _Overlap()
    sink = QdrantPageSink(store, dense, sparse, ChunkingConfig(max_chars=300, overlap_chars=20))
    for p in DEMO_PAGES:
        sink.upsert_page(p)
    synced = {"at": datetime.now(UTC).isoformat()}
    retr = Retriever(
        store, dense, sparse, rr, RetrievalConfig(candidate_k=10, final_k=3, min_rerank_score=0.2)
    )

    def run_sync(full: bool) -> SyncReport:
        time.sleep(3)
        synced["at"] = datetime.now(UTC).isoformat()
        return SyncReport(scanned=len(DEMO_PAGES), unchanged=len(DEMO_PAGES))

    return Services(
        store=store,
        qa=GroundedQA(retr, _EchoLLM(), rr),
        run_sync=run_sync,
        last_sync=lambda: synced["at"],
    )
