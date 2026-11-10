import { ArrowUpRight, ChevronLeft, ChevronRight, Clock3, Lightbulb, Loader2, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { PageBlock, PageDoc, SourceHit } from "../types";
import { fmtDate } from "../util";

const cache = new Map<string, PageDoc>();

/** Indexed version first (instant), then the live Notion page replaces it when it arrives. */
function useDoc(pageId: string | undefined) {
  const [doc, setDoc] = useState<PageDoc | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [liveLoading, setLiveLoading] = useState(false);
  useEffect(() => {
    if (!pageId) return;
    let alive = true;
    setErr(null);
    const hit = cache.get(pageId);
    setDoc(hit ?? null);
    if (hit?.source === "notion") return;
    if (!hit)
      api
        .page(pageId, false)
        .then((d) => alive && !cache.has(pageId) && setDoc(d))
        .catch((e) => alive && setErr((e as Error).message));
    setLiveLoading(true);
    api
      .page(pageId, true)
      .then((d) => {
        cache.set(pageId, d);
        if (alive) setDoc(d);
      })
      .catch(() => {
        /* keep the indexed version */
      })
      .finally(() => alive && setLiveLoading(false));
    return () => {
      alive = false;
    };
  }, [pageId]);
  return { doc, err, liveLoading };
}

function stripMd(t: string) {
  return t.replace(/^#{1,3}\s+/, "");
}

function BlockView({ b, quote }: { b: PageBlock; quote: boolean }) {
  const cls = `blk${quote ? " quote" : ""}`;
  const t = b.text;
  if (b.type === "properties") return <div className={`${cls} props`}>{t}</div>;
  if (b.type.startsWith("heading_")) return <h3 className={cls}>{stripMd(t.split("\n")[0])}</h3>;
  if (b.type === "code" || t.startsWith("```")) {
    const body = t.replace(/^```[^\n]*\n?/, "").replace(/\n?```$/, "");
    return <pre className={cls}>{body}</pre>;
  }
  if (b.type === "callout" || b.type === "quote")
    return (
      <div className={`callout ${cls}`}>
        <Lightbulb />
        <span style={{ whiteSpace: "pre-wrap" }}>{t.replace(/^>\s?/gm, "")}</span>
      </div>
    );
  if (b.type === "table" || /^\|.*\|$/m.test(t)) {
    const rows = t.split("\n").filter((l) => l.startsWith("|")).map((l) => l.slice(1, -1).split(" | "));
    return (
      <div className={cls} style={{ overflowX: "auto" }}>
        <table>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                {r.map((c, j) => (i === 0 ? <th key={j}>{c.trim()}</th> : <td key={j}>{c.trim()}</td>))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }
  if (b.type === "divider") return <hr className={cls} style={{ border: 0, borderTop: "1px solid var(--border)" }} />;
  return <p className={cls}>{t}</p>;
}

export default function SourcePanel({
  sources,
  idx,
  onIdx,
  onClose,
  openNotion,
}: {
  sources: SourceHit[];
  idx: number; // 1-based
  onIdx: (n: number) => void;
  onClose: () => void;
  openNotion: (url: string, title: string) => void;
}) {
  const hit = sources[idx - 1];
  const { doc, err, liveLoading } = useDoc(hit?.page_id);
  const docRef = useRef<HTMLDivElement>(null);
  const [pulse, setPulse] = useState(0);

  const quoted = new Set(hit?.block_ids ?? []);
  const isQuoted = (b: PageBlock) =>
    quoted.has(b.id) || (b.block_ids ?? []).some((x) => quoted.has(x)) || (doc?.source === "index" && b.text === hit?.text);

  useEffect(() => setPulse((p) => p + 1), [idx]);
  useEffect(() => {
    const el = docRef.current?.querySelector<HTMLElement>(".quote");
    const d = docRef.current;
    if (!el || !d) return;
    requestAnimationFrame(() => {
      d.scrollTop = el.offsetTop - d.offsetTop - 60;
      el.classList.remove("pulse");
      void el.offsetWidth;
      el.classList.add("pulse");
    });
  }, [doc, pulse]);

  if (!hit) return null;
  const crumb = [...(hit.path ?? []), ...(hit.heading_path ?? [])];
  return (
    <>
      <div className="tabs" role="tablist">
        {sources.map((s, i) => (
          <button key={s.chunk_id} className="tab" role="tab" aria-selected={i + 1 === idx} onClick={() => onIdx(i + 1)}>
            <span className="num">{i + 1}</span>
            <span className="t">{s.title}</span>
          </button>
        ))}
        <span className="spacer" />
        <button className="icon-btn" onClick={onClose} aria-label="출처 패널 닫기 (Esc)">
          <X />
        </button>
      </div>
      <div className="doc-head">
        <h2>{doc?.title ?? hit.title}</h2>
        {crumb.length > 0 && (
          <div className="crumb">
            {crumb.map((c, i) => (
              <span key={i} style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                {i > 0 && <ChevronRight style={{ width: 12, height: 12 }} />}
                {c}
              </span>
            ))}
          </div>
        )}
        <div className="meta">
          <Clock3 style={{ width: 13, height: 13 }} />
          {fmtDate(doc?.last_edited_time ?? hit.last_edited_time)} 수정
          <button
            className="btn"
            title="같은 노션 창에서 열기 (Shift+클릭)"
            onClick={() => openNotion(hit.cite_url, hit.title)}
          >
            <ArrowUpRight />
            노션에서 열기
          </button>
        </div>
      </div>
      <div className="doc" role="tabpanel" ref={docRef}>
        {err && <div className="err">원문을 불러오지 못했습니다: {err}</div>}
        {!doc && !err && (
          <div className="loading">
            <Loader2 className="spin" /> 노션에서 원문을 불러오는 중…
          </div>
        )}
        {doc?.source === "index" && (
          <div className="src-note" style={liveLoading ? { color: "var(--text-muted)" } : undefined}>
            {liveLoading
              ? "색인된 내용입니다 · 노션 최신본을 불러오는 중…"
              : "노션에 연결하지 못해 색인된 내용으로 보여줍니다. 최신 내용과 다를 수 있습니다."}
          </div>
        )}
        {doc && doc.blocks.length === 0 && <p className="blk">본문이 비어 있습니다.</p>}
        {doc?.blocks.map((b, i) => <BlockView key={b.id + i} b={b} quote={isQuoted(b)} />)}
      </div>
      <div className="panel-foot">
        <span>
          출처 <b>{idx}/{sources.length}</b>
        </span>
        <span className="sp" />
        <button className="btn ghost" onClick={() => onIdx(((idx - 2 + sources.length) % sources.length) + 1)}>
          <ChevronLeft />
          이전
        </button>
        <button className="btn ghost" onClick={() => onIdx((idx % sources.length) + 1)}>
          다음
          <ChevronRight />
        </button>
      </div>
    </>
  );
}
