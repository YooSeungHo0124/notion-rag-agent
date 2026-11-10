"""Attach relevant Notion images to an answer (no re-index; images are not searched).

An image is shown only when ALL of these hold — irrelevant images are worse than none:
  1. proximity : it is a top-level image block within MAX_DISTANCE blocks of a block the
                 answer actually cited, with no heading in between (same section);
  2. context   : it has a caption, or >= MIN_CONTEXT_CHARS of neighbouring text (otherwise
                 we cannot tell what it is);
  3. relevance : the cross-encoder scores (answer sentence citing that source) vs
                 (caption + section heading + neighbouring text) >= MIN_SCORE.
At most MAX_IMAGES per answer. The image itself is never seen by a model, so relevance is
inferred from its context; when in doubt, the image is dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from notion_agent.ingest.notion_reader import block_to_text, rich_text
from notion_agent.rag.reranker import RerankerProtocol

MAX_DISTANCE = 2
MIN_SCORE = 0.5
MIN_CONTEXT_CHARS = 20  # without a caption, a one-word neighbour is too little to judge
MAX_IMAGES = 2
CONTEXT_CHARS = 600


@dataclass
class ImageCandidate:
    block_id: str
    page_id: str
    source: int  # 1-based answer source index
    caption: str
    context: str
    heading: str
    distance: int
    score: float = 0.0
    after_block: int = -1  # index into answer blocks (sentences) to insert after


def _is_heading(b: dict[str, Any]) -> bool:
    return str(b.get("type", "")).startswith("heading_")


def _plain(b: dict[str, Any]) -> str:
    if b.get("type") == "image":
        return ""
    return (block_to_text(b) or "").strip()


def find_candidates(
    blocks: list[dict[str, Any]], cited: set[str], page_id: str, source: int
) -> list[ImageCandidate]:
    """Image blocks in the same section as, and close to, a cited block."""
    pos = {b["id"]: i for i, b in enumerate(blocks)}
    cited_pos = sorted(pos[c] for c in cited if c in pos)
    out: list[ImageCandidate] = []
    if not cited_pos:
        return out
    for i, b in enumerate(blocks):
        if b.get("type") != "image":
            continue
        best = None
        for c in cited_pos:
            lo, hi = sorted((i, c))
            if hi - lo > MAX_DISTANCE:
                continue
            if any(_is_heading(blocks[k]) for k in range(lo + 1, hi + (1 if c > i else 0))):
                continue  # a heading separates them -> different section
            best = hi - lo if best is None else min(best, hi - lo)
        if best is None:
            continue
        caption = rich_text((b.get("image") or {}).get("caption")).strip()
        heading = ""
        for k in range(i - 1, -1, -1):
            if _is_heading(blocks[k]):
                heading = _plain(blocks[k]).lstrip("#").strip()
                break
        near = [_plain(blocks[k]) for k in (i - 1, i + 1) if 0 <= k < len(blocks)]
        context = " ".join(x for x in near if x)[:CONTEXT_CHARS]
        if not caption and len(context) < MIN_CONTEXT_CHARS:
            continue
        out.append(ImageCandidate(b["id"], page_id, source, caption, context, heading, best))
    return out


NESTED_DEPTH = 3
MAX_EXPAND = 6  # cited blocks whose children we fetch (Notion API cost)


def find_nested_candidates(
    parent: dict[str, Any],
    children_of: Any,  # callable(block_id) -> list[dict]
    page_id: str,
    source: int,
    heading: str = "",
    depth: int = 0,
) -> list[ImageCandidate]:
    """Images inside a cited block's children (toggles, columns, callouts...).

    Being inside a block the answer cited is the proximity signal; context = the image
    caption + its sibling text + the parent block's own first line.
    """
    if depth >= NESTED_DEPTH or not parent.get("has_children"):
        return []
    if parent.get("type") in ("child_page", "child_database"):
        return []
    kids = children_of(parent["id"])
    parent_line = (_plain(parent).splitlines() or [""])[0].lstrip("#").strip()[:200]
    out: list[ImageCandidate] = []
    for i, b in enumerate(kids):
        if b.get("type") == "image":
            caption = rich_text((b.get("image") or {}).get("caption")).strip()
            near = [_plain(kids[k]) for k in (i - 1, i + 1) if 0 <= k < len(kids)]
            ctx_near = " ".join(x for x in near if x)
            context = " ".join(x for x in [parent_line, ctx_near] if x)[:CONTEXT_CHARS]
            if caption or len(ctx_near) >= MIN_CONTEXT_CHARS:
                out.append(ImageCandidate(b["id"], page_id, source, caption, context,
                                          heading, distance=depth + 1))
        elif b.get("has_children"):
            out += find_nested_candidates(b, children_of, page_id, source, heading, depth + 1)
    return out


def select_images(
    candidates: list[ImageCandidate],
    sentences: list[dict[str, Any]],
    scorer: RerankerProtocol,
) -> list[ImageCandidate]:
    """Score each candidate against the answer blocks that cite its source; keep the best."""
    kept: list[ImageCandidate] = []
    for c in candidates:
        citing = [(j, s) for j, s in enumerate(sentences)
                  if c.source in (s.get("citations") or []) and s.get("type") != "heading"]
        if not citing:
            continue
        passage = " ".join(x for x in (c.caption, c.heading, c.context) if x)
        scores = [scorer.score(s.get("text", "")[:1000], [passage])[0] for _, s in citing]
        best = max(range(len(scores)), key=scores.__getitem__)
        c.score = float(scores[best])
        c.after_block = citing[best][0]
        if c.score >= MIN_SCORE:
            kept.append(c)
    kept.sort(key=lambda c: (-c.score, c.distance))
    seen: set[str] = set()
    out = []
    for c in kept:
        if c.block_id not in seen:
            seen.add(c.block_id)
            out.append(c)
    return out[:MAX_IMAGES]


def image_url(block: dict[str, Any]) -> str | None:
    img = block.get("image") or {}
    if img.get("type") == "external":
        return (img.get("external") or {}).get("url")
    return (img.get("file") or {}).get("url")


class ImageService:
    """Finds answer images via the Notion API and proxies their bytes (file URLs expire)."""

    def __init__(self, reader: Any, scorer: RerankerProtocol, fetch: Any = None) -> None:
        import threading
        import time

        self.reader, self.scorer = reader, scorer
        self._time = time.monotonic
        self._blocks: dict[str, tuple[float, list[dict[str, Any]]]] = {}
        self._allowed: dict[str, str] = {}  # block_id -> page_id (only ids we proposed)
        self._bytes: dict[str, tuple[bytes, str]] = {}
        self._lock = threading.Lock()
        self._fetch = fetch or _http_get

    def _page_blocks(self, page_id: str) -> list[dict[str, Any]]:
        now = self._time()
        hit = self._blocks.get(page_id)
        if hit and now - hit[0] < 300:
            return hit[1]
        blocks = list(self.reader.list_children(page_id))
        self._blocks[page_id] = (now, blocks)
        return blocks

    def _children(self, block_id: str) -> list[dict[str, Any]]:
        return self._page_blocks(block_id)  # same cache; a block id works as parent

    def candidates(self, sources: list[dict[str, Any]]) -> list[ImageCandidate]:
        """Notion API only (no GPU) — safe to run without the store lock."""
        out: list[ImageCandidate] = []
        expanded = 0
        for n, s in enumerate(sources, 1):
            pid, cited = s.get("page_id"), set(s.get("block_ids") or [])
            if not pid or not cited:
                continue
            blocks = self._page_blocks(pid)
            out += find_candidates(blocks, cited, pid, n)
            heading = ""
            for b in blocks:  # images nested inside the cited blocks themselves
                if _is_heading(b):
                    heading = _plain(b).lstrip("#").strip()
                if b["id"] in cited and b.get("has_children") and expanded < MAX_EXPAND:
                    expanded += 1
                    out += find_nested_candidates(b, self._children, pid, n, heading)
        return out

    def select(self, cands: list[ImageCandidate], sentences: list[dict[str, Any]]):
        chosen = select_images(cands, sentences, self.scorer)
        with self._lock:
            for c in chosen:
                self._allowed[c.block_id] = c.page_id
        return chosen

    def image(self, block_id: str) -> tuple[bytes, str] | None:
        with self._lock:
            if block_id not in self._allowed:
                return None
            if block_id in self._bytes:
                return self._bytes[block_id]
        block = self.reader._call(self.reader.client.blocks.retrieve, block_id=block_id)
        url = image_url(block)
        if not url:
            return None
        data = self._fetch(url)
        with self._lock:
            if len(self._bytes) > 64:
                self._bytes.pop(next(iter(self._bytes)))
            self._bytes[block_id] = data
        return data


def _http_get(url: str) -> tuple[bytes, str]:
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "notion-rag-agent"})
    with urllib.request.urlopen(req, timeout=20) as r:  # noqa: S310 - Notion/external image URL
        ctype = r.headers.get("Content-Type", "application/octet-stream")
        if not ctype.startswith("image/"):
            raise ValueError(f"not an image: {ctype}")
        return r.read(15 * 1024 * 1024), ctype
