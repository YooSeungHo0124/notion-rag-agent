"""FastAPI server for the web UI (F7).

Heavy components (embedders, reranker, Qdrant client, LLM) are loaded once, in a background
thread started by the lifespan, so the UI is reachable while models load (`/api/health`
reports ``ready=false`` until then).

Qdrant runs in embedded mode (single-process lock), so sync reuses the server's own store and
embedders instead of spawning ``nra sync``. A single lock serialises ask and sync.
"""

from __future__ import annotations

import json
import logging
import queue
import sqlite3
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from notion_agent.ingest.sync import SyncReport
from notion_agent.qa.grounded import GroundedQA
from notion_agent.qa.serialize import answer_to_json, hit_to_json  # noqa: F401 (re-export)
from notion_agent.rag.store import QdrantStore, SearchFilters

log = logging.getLogger(__name__)

WEB_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"
ASK_LOCK_TIMEOUT_S = 600.0


@dataclass
class Services:
    """Everything the API needs; built once by a loader (real or fake)."""

    store: QdrantStore
    qa: GroundedQA
    run_sync: Callable[[bool], SyncReport] | None = None  # arg: full
    agent: Any | None = None  # notion_agent.agent.graph.NotionAgent
    pages: Any | None = None  # notion_agent.api.pages.PageLoader
    images: Any | None = None  # notion_agent.qa.images.ImageService
    studio: Any | None = None  # notion_agent.qa.studio.Studio
    last_sync: Callable[[], str | None] = lambda: None


@dataclass
class SyncStatus:
    running: bool = False
    started_at: str | None = None
    finished_at: str | None = None
    ok: bool | None = None
    summary: str | None = None
    errors: list[str] = field(default_factory=list)
    error: str | None = None


class AppState:
    def __init__(self, loader: Callable[[], Services]) -> None:
        self.loader = loader
        self.services: Services | None = None
        self.load_error: str | None = None
        self.lock = threading.Lock()  # guards the embedded Qdrant store + models
        self.sync = SyncStatus()
        self.sources_cache: list[dict[str, Any]] | None = None

    def load(self) -> None:
        try:
            self.services = self.loader()
        except Exception as exc:  # noqa: BLE001 - surfaced via /api/health
            log.exception("failed to load services")
            self.load_error = f"{type(exc).__name__}: {exc}"

    def require(self) -> Services:
        if self.services is None:
            detail = self.load_error or "모델을 불러오는 중입니다"
            raise HTTPException(status_code=503, detail=detail)
        return self.services


# --- serialisation ---------------------------------------------------------------------------


def scan_sources(store: QdrantStore) -> list[dict[str, Any]]:
    """Aggregate chunk payloads into one entry per page."""
    pages: dict[str, dict[str, Any]] = {}
    client = store.client
    if not client.collection_exists(store.collection):
        return []
    offset = None
    keys = ["page_id", "page_title", "path", "tags", "last_edited_time", "page_url"]
    while True:
        points, offset = client.scroll(
            store.collection, limit=512, offset=offset, with_payload=keys, with_vectors=False
        )
        for p in points:
            pl = p.payload or {}
            pid = pl.get("page_id")
            if not pid:
                continue
            entry = pages.setdefault(
                pid,
                {
                    "page_id": pid,
                    "title": pl.get("page_title") or "(제목 없음)",
                    "path": pl.get("path") or [],
                    "tags": pl.get("tags") or [],
                    "last_edited_time": pl.get("last_edited_time"),
                    "url": pl.get("page_url"),
                    "chunks": 0,
                },
            )
            entry["chunks"] += 1
        if offset is None:
            break
    return sorted(pages.values(), key=lambda e: ([*e["path"], e["title"]]))


def read_last_sync(state_db: Path) -> str | None:
    """Most recent ``indexed_at`` in the sync state DB (read-only)."""
    if not state_db.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{state_db}?mode=ro", uri=True)
        try:
            row = conn.execute("SELECT MAX(indexed_at) FROM pages").fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    return row[0] if row else None


# --- request models --------------------------------------------------------------------------


class NoteIn(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    title: str = Field(default="", max_length=200)
    text: str = Field(min_length=1, max_length=20_000)
    updated_at: str | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    page_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    notes: list[NoteIn] = Field(default_factory=list, max_length=50)


def notes_to_hits(notes: list[NoteIn]) -> list[Any]:
    """User memos (kept in the browser) as pseudo-hits with page_id 'note:<id>'."""
    from notion_agent.rag.retriever import Hit

    out = []
    for n in notes:
        try:
            ts = datetime.fromisoformat(n.updated_at) if n.updated_at else datetime.now(UTC)
        except ValueError:
            ts = datetime.now(UTC)
        out.append(Hit(chunk_id=f"note:{n.id}", page_id=f"note:{n.id}",
                       page_title=f"메모: {n.title or n.text[:30]}", page_url="", path=["내 메모"],
                       text=n.text[:4000], last_edited_time=ts, fusion_score=0.0,
                       rerank_score=0.0))
    return out


class AgentRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = None


class ResumeRequest(BaseModel):
    thread_id: str
    approved: bool
    args: dict[str, Any] | None = None


class ImagesRequest(BaseModel):
    sources: list[dict[str, Any]] = Field(default_factory=list)  # {page_id, block_ids}
    sentences: list[dict[str, Any]] = Field(default_factory=list)  # {text, citations, type}


class StudioRequest(BaseModel):
    kind: str
    page_ids: list[str] = Field(default_factory=list, max_length=2000)
    topic: str = Field(default="", max_length=300)


class SyncRequest(BaseModel):
    full: bool = False


# --- app -------------------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(UTC).isoformat()


def create_app(
    loader: Callable[[], Services] | None = None,
    *,
    background_load: bool = True,
    static_dir: Path | None = WEB_DIST,
) -> FastAPI:
    st = AppState(loader or load_real_services)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if background_load:
            threading.Thread(target=st.load, name="nra-load", daemon=True).start()
        else:
            st.load()
        yield

    app = FastAPI(title="notion-rag-agent", lifespan=lifespan)
    app.state.nra = st

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        svc = st.services
        cache = st.sources_cache
        if svc is not None and cache is None and st.lock.acquire(timeout=2):
            try:
                cache = _sources_locked(svc)
            finally:
                st.lock.release()
        cache = cache or []
        chunks, pages = sum(e["chunks"] for e in cache), len(cache)
        return {
            "ready": svc is not None,
            "error": st.load_error,
            "chunks": chunks,
            "pages": pages,
            "last_sync": svc.last_sync() if svc else None,
            "syncing": st.sync.running,
        }

    def _sources_locked(svc: Services) -> list[dict[str, Any]]:
        if st.sources_cache is None:
            st.sources_cache = scan_sources(svc.store)
        return st.sources_cache

    @app.get("/api/sources")
    def sources(refresh: bool = False) -> dict[str, Any]:
        svc = st.require()
        if not refresh and st.sources_cache is not None:
            return {"sources": st.sources_cache}
        if not st.lock.acquire(timeout=30):
            if st.sources_cache is not None:
                return {"sources": st.sources_cache}
            raise HTTPException(status_code=409, detail="동기화 중입니다. 잠시 후 다시 시도하세요")
        try:
            if refresh:
                st.sources_cache = None
            return {"sources": _sources_locked(svc)}
        finally:
            st.lock.release()

    def _ask(req: AskRequest) -> dict[str, Any]:
        svc = st.require()
        filters = None
        if req.page_ids or req.tags:
            filters = SearchFilters(page_ids=req.page_ids or None, tags=req.tags or None)
        if st.sync.running:
            raise HTTPException(status_code=409, detail="동기화 중입니다. 끝난 뒤 다시 질문하세요")
        if not st.lock.acquire(timeout=ASK_LOCK_TIMEOUT_S):
            raise HTTPException(status_code=503, detail="서버가 바쁩니다")
        try:
            t0 = time.perf_counter()
            ans = svc.qa.answer(req.question.strip(), filters=filters,
                                extra=notes_to_hits(req.notes))
            out = answer_to_json(ans)
            out["total_ms"] = int((time.perf_counter() - t0) * 1000)
            return out
        finally:
            st.lock.release()

    @app.post("/api/ask")
    async def ask(req: AskRequest) -> dict[str, Any]:
        return await run_in_threadpool(_ask, req)

    @app.get("/api/page/{page_id}")
    def page(page_id: str, refresh: bool = False, live: bool = True) -> dict[str, Any]:
        """Source viewer content. live=0 -> indexed chunks (instant, reads Qdrant under the
        store lock); live=1 -> Notion API (cached 5 min, can take seconds on big pages; falls
        back to the index)."""
        svc = st.require()
        if svc.pages is None:
            raise HTTPException(501, "page viewer not configured")
        if live:
            p = svc.pages.get(page_id, refresh=refresh, live=True)
            if p is None:
                raise HTTPException(502, "노션에서 페이지를 가져오지 못했습니다")
        else:
            if not st.lock.acquire(timeout=30):
                raise HTTPException(503, "서버가 바쁩니다")
            try:
                p = svc.pages.get(page_id, live=False)
            finally:
                st.lock.release()
        if p is None:
            raise HTTPException(404, "페이지를 찾을 수 없습니다")
        return p

    @app.post("/api/answer/images")
    def answer_images(req: ImagesRequest) -> dict[str, Any]:
        """Images near the cited blocks that the reranker judges relevant to the citing
        sentence (see qa/images.py). Called by the UI after the answer is shown."""
        svc = st.require()
        if svc.images is None or not req.sources:
            return {"images": []}
        try:
            cands = svc.images.candidates(req.sources[:8])  # Notion API, no lock
        except Exception as e:  # noqa: BLE001 - images are optional
            log.warning("image candidates failed: %s", e)
            return {"images": []}
        if not cands:
            return {"images": []}
        if not st.lock.acquire(timeout=60):
            return {"images": []}
        try:
            chosen = svc.images.select(cands, req.sentences)  # reranker on GPU
        finally:
            st.lock.release()
        return {"images": [{
            "block_id": c.block_id, "page_id": c.page_id, "source": c.source,
            "after_block": c.after_block, "caption": c.caption, "context": c.context[:200],
            "heading": c.heading, "score": round(c.score, 3),
            "src": f"/api/image/{c.block_id}",
        } for c in chosen]}

    @app.get("/api/image/{block_id}")
    def image(block_id: str) -> Response:
        """Proxy for an image we proposed (Notion file URLs expire after ~1 h)."""
        svc = st.require()
        if svc.images is None:
            raise HTTPException(404)
        try:
            got = svc.images.image(block_id)
        except Exception as e:  # noqa: BLE001
            log.warning("image fetch failed: %s", e)
            raise HTTPException(502, "이미지를 가져오지 못했습니다") from e
        if got is None:
            raise HTTPException(404)
        data, ctype = got
        return Response(data, media_type=ctype, headers={"Cache-Control": "private, max-age=3000"})

    @app.post("/api/ask/stream")
    def ask_stream(req: AskRequest) -> StreamingResponse:
        """Server-sent events: stage updates (retrieving/retrieved/generating/verifying),
        then `answer` (or `error`). Lets the UI show progress during the ~7 s pipeline."""
        q: queue.Queue[tuple[str, Any] | None] = queue.Queue()

        def work() -> None:
            try:
                svc = st.require()
                filters = None
                if req.page_ids or req.tags:
                    filters = SearchFilters(page_ids=req.page_ids or None, tags=req.tags or None)
                if st.sync.running:
                    raise HTTPException(409, "동기화 중입니다. 끝난 뒤 다시 질문하세요")
                if not st.lock.acquire(timeout=ASK_LOCK_TIMEOUT_S):
                    raise HTTPException(503, "서버가 바쁩니다")
                try:
                    t0 = time.perf_counter()
                    ans = svc.qa.answer(req.question.strip(), filters=filters,
                                        on_event=lambda e, d: q.put((e, d)),
                                        extra=notes_to_hits(req.notes))
                    out = answer_to_json(ans)
                    out["total_ms"] = int((time.perf_counter() - t0) * 1000)
                    q.put(("answer", out))
                finally:
                    st.lock.release()
            except HTTPException as e:
                q.put(("error", {"detail": e.detail}))
            except Exception as e:  # noqa: BLE001 - reported to the client
                log.exception("ask stream failed")
                q.put(("error", {"detail": f"{type(e).__name__}: {e}"}))
            finally:
                q.put(None)

        threading.Thread(target=work, daemon=True).start()

        def events() -> Iterator[str]:
            while (item := q.get()) is not None:
                name, data = item
                yield f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    @app.post("/api/studio/stream")
    def studio_stream(req: StudioRequest) -> StreamingResponse:
        """Studio document over a scope (SSE: collecting -> generating -> verifying -> result).
        The store lock is held only to collect chunks and to verify, not during the LLM call."""
        from notion_agent.qa.studio import KINDS, result_to_json

        if req.kind not in KINDS:
            raise HTTPException(422, f"unknown kind: {req.kind}")
        q: queue.Queue[tuple[str, Any] | None] = queue.Queue()
        emit = lambda e, d: q.put((e, d))  # noqa: E731

        def locked(fn: Callable[[], Any]) -> Any:
            if st.sync.running:
                raise HTTPException(409, "동기화 중입니다. 끝난 뒤 다시 시도하세요")
            if not st.lock.acquire(timeout=ASK_LOCK_TIMEOUT_S):
                raise HTTPException(503, "서버가 바쁩니다")
            try:
                return fn()
            finally:
                st.lock.release()

        def work() -> None:
            try:
                svc = st.require()
                if svc.studio is None:
                    raise HTTPException(501, "studio not configured")
                emit("collecting", {})
                hits, cov = locked(lambda: svc.studio.material(req.page_ids, req.topic))
                emit("collected", {"chunks": len(hits), **cov.__dict__})
                r = svc.studio.generate(req.kind, hits, cov, emit)
                r = locked(lambda: svc.studio.verify(r, hits, emit))
                q.put(("result", result_to_json(r)))
            except HTTPException as e:
                q.put(("error", {"detail": e.detail}))
            except Exception as e:  # noqa: BLE001 - reported to the client
                log.exception("studio failed")
                q.put(("error", {"detail": f"{type(e).__name__}: {e}"}))
            finally:
                q.put(None)

        threading.Thread(target=work, daemon=True).start()

        def events() -> Iterator[str]:
            while (item := q.get()) is not None:
                name, data = item
                yield f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    def _agent_call(fn: Callable[[Any], dict[str, Any]]) -> dict[str, Any]:
        svc = st.require()
        if svc.agent is None:
            raise HTTPException(status_code=501, detail="agent not configured")
        if st.sync.running:
            raise HTTPException(status_code=409, detail="동기화 중입니다. 끝난 뒤 다시 시도하세요")
        if not st.lock.acquire(timeout=ASK_LOCK_TIMEOUT_S):
            raise HTTPException(status_code=503, detail="서버가 바쁩니다")
        try:
            return fn(svc.agent)
        finally:
            st.lock.release()

    @app.post("/api/agent")
    async def agent_run(req: AgentRequest) -> dict[str, Any]:
        tid = req.thread_id or uuid.uuid4().hex
        return await run_in_threadpool(_agent_call, lambda a: a.run(req.question.strip(), tid))

    @app.post("/api/agent/resume")
    async def agent_resume(req: ResumeRequest) -> dict[str, Any]:
        return await run_in_threadpool(
            _agent_call, lambda a: a.resume(req.thread_id, req.approved, req.args))

    def _run_sync(svc: Services, full: bool) -> None:
        assert svc.run_sync is not None
        try:
            with st.lock:
                report = svc.run_sync(full)
                st.sources_cache = None
            st.sync.ok = report.failed == 0
            st.sync.summary = report.summary()
            st.sync.errors = report.errors[:50]
        except Exception as exc:  # noqa: BLE001 - reported via status
            log.exception("sync failed")
            st.sync.ok = False
            st.sync.error = f"{type(exc).__name__}: {exc}"
        finally:
            st.sync.finished_at = _now()
            st.sync.running = False

    @app.post("/api/sync", status_code=202)
    def sync(req: SyncRequest | None = None) -> dict[str, Any]:
        svc = st.require()
        if svc.run_sync is None:
            raise HTTPException(status_code=501, detail="이 서버에서는 동기화를 쓸 수 없습니다")
        if st.sync.running:
            raise HTTPException(status_code=409, detail="이미 동기화 중입니다")
        st.sync = SyncStatus(running=True, started_at=_now())
        full = bool(req and req.full)
        threading.Thread(target=_run_sync, args=(svc, full), name="nra-sync", daemon=True).start()
        return sync_status()

    @app.get("/api/sync/status")
    def sync_status() -> dict[str, Any]:
        s = st.sync
        svc = st.services
        return {
            "running": s.running,
            "started_at": s.started_at,
            "finished_at": s.finished_at,
            "ok": s.ok,
            "summary": s.summary,
            "errors": s.errors,
            "error": s.error,
            "last_sync": svc.last_sync() if svc else None,
        }

    if static_dir is not None and (static_dir / "index.html").exists():
        index = static_dir / "index.html"
        if (static_dir / "assets").is_dir():
            app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            if path.startswith("api/"):
                raise HTTPException(status_code=404)
            f = (static_dir / path).resolve()
            if path and f.is_file() and f.is_relative_to(static_dir.resolve()):
                return FileResponse(f)
            return FileResponse(index)

    return app


def load_real_services() -> Services:
    """Load models + store once (takes minutes on first call)."""
    from notion_agent.config import get_settings
    from notion_agent.llm import get_llm
    from notion_agent.rag.factory import build_retriever, build_sink, build_verifier

    s = get_settings()
    retriever = build_retriever(s)
    qa = GroundedQA(retriever, get_llm(), build_verifier(s))

    def run_sync(full: bool) -> SyncReport:
        from notion_agent.ingest.notion_reader import NotionReader
        from notion_agent.ingest.state import SyncState
        from notion_agent.ingest.sync import sync as do_sync

        reader = NotionReader.from_settings(s)
        state = SyncState(s.state_db)
        try:
            return do_sync(reader, build_sink(s), state, full=full)  # shares cached store/models
        finally:
            state.close()

    from notion_agent.agent.factory import build_agent
    from notion_agent.api.pages import PageLoader, index_fetcher, live_fetcher
    from notion_agent.qa.images import ImageService

    live = images = None
    if s.notion_token:
        from notion_agent.ingest.notion_reader import NotionReader

        live = live_fetcher(NotionReader.from_settings(s))
        images = ImageService(NotionReader.from_settings(s), qa.verifier)

    from notion_agent.llm.claude_cli import ClaudeCLIClient
    from notion_agent.qa.studio import Studio

    studio_llm = ClaudeCLIClient(model=s.llm_model, timeout_s=480, max_retries=1) \
        if s.llm_provider == "claude_cli" else qa.llm

    return Services(
        store=retriever.store,
        qa=qa,
        studio=Studio(qa, studio_llm),
        agent=build_agent(s, qa),
        pages=PageLoader(live, index_fetcher(retriever.store)),
        images=images,
        run_sync=run_sync,
        last_sync=lambda: read_last_sync(s.state_db),
    )
