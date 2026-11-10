import { Bookmark, ChevronDown, ChevronRight, FolderSearch, Moon, Plus, RefreshCw, Search, Sun } from "lucide-react";
import { useMemo, useRef, useState } from "react";
import type { Conv, Health, PageSource, SyncStatus } from "../types";
import { relTime } from "../util";

function bucket(ts: number): string {
  const d = new Date(ts);
  const now = new Date();
  const day = 86400000;
  const start = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  if (d.getTime() >= start) return "오늘";
  if (d.getTime() >= start - 6 * day) return "이번 주";
  return "이전";
}

export function Sidebar(p: {
  convs: Conv[];
  current: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  savedCount: number;
  onSaved: () => void;
  scopeLabel: string;
  onScope: (el: HTMLElement) => void;
  health: Health | null;
  sync: SyncStatus | null;
  onSync: () => void;
  theme: "light" | "dark";
  onTheme: () => void;
}) {
  const [f, setF] = useState("");
  const groups = useMemo(() => {
    const g = new Map<string, Conv[]>();
    for (const c of [...p.convs].sort((a, b) => b.updatedAt - a.updatedAt)) {
      if (f && !c.title.toLowerCase().includes(f.toLowerCase())) continue;
      const k = bucket(c.updatedAt);
      g.set(k, [...(g.get(k) ?? []), c]);
    }
    return [...g.entries()];
  }, [p.convs, f]);
  const syncing = p.sync?.running || p.health?.syncing;
  return (
    <aside className="sidebar" aria-label="사이드바">
      <div className="brand">
        <span className="mark" aria-hidden="true">
          N
        </span>
        <span className="lbl">내 노트</span>
      </div>
      <button className="side-btn" onClick={p.onNew} title="새 질문 (Ctrl+K)">
        <Plus />
        <span className="lbl">새 질문</span>
        <span className="kbd lbl">Ctrl K</span>
      </button>
      <button className="side-btn" onClick={p.onSaved} title="저장한 답변">
        <Bookmark />
        <span className="lbl">저장한 답변 {p.savedCount ? `(${p.savedCount})` : ""}</span>
      </button>
      <nav className="hist" aria-label="대화 기록">
        <label className="hist-search">
          <Search />
          <span className="sr-only">대화 검색</span>
          <input placeholder="대화 검색" value={f} onChange={(e) => setF(e.target.value)} />
        </label>
        {groups.length === 0 && <h4>{f ? "일치하는 대화가 없습니다" : "아직 대화가 없습니다"}</h4>}
        {groups.map(([k, cs]) => (
          <div key={k}>
            <h4>{k}</h4>
            {cs.map((c) => (
              <button key={c.id} aria-current={c.id === p.current} onClick={() => p.onSelect(c.id)} title={c.title}>
                {c.title}
              </button>
            ))}
          </div>
        ))}
      </nav>
      <div className="side-foot">
        <button className="side-btn" onClick={(e) => p.onScope(e.currentTarget)} title="검색 범위">
          <FolderSearch />
          <span className="lbl">범위: {p.scopeLabel}</span>
          <ChevronDown className="lbl" style={{ marginLeft: "auto" }} />
        </button>
        <div className="meta">
          <span className="dot" aria-hidden="true" style={syncing ? { background: "var(--warn)" } : undefined} />
          {syncing ? "동기화 중…" : `동기화 ${relTime(p.health?.last_sync)} · ${p.health?.pages ?? 0} 페이지`}
          {!syncing && (
            <button onClick={p.onSync} title="지금 동기화" aria-label="지금 동기화">
              <RefreshCw style={{ width: 13, height: 13 }} />
            </button>
          )}
        </div>
        <button className="side-btn" onClick={p.onTheme} title="테마 전환">
          {p.theme === "dark" ? <Sun /> : <Moon />}
          <span className="lbl">{p.theme === "dark" ? "라이트 모드" : "다크 모드"}</span>
        </button>
      </div>
    </aside>
  );
}

/* ---------------- scope popover ---------------- */

const UNTITLED = /^(\(제목 없음\)|-+|[-\s\d:~.]+.*|v\d+)$/;

interface Group {
  name: string;
  pages: PageSource[];
  untitled: PageSource[];
}

function buildGroups(sources: PageSource[]): Group[] {
  const m = new Map<string, Group>();
  for (const s of sources) {
    const name = s.path[0] ?? "(최상위)";
    const g = m.get(name) ?? { name, pages: [], untitled: [] };
    (UNTITLED.test(s.title.trim()) ? g.untitled : g.pages).push(s);
    m.set(name, g);
  }
  return [...m.values()].sort((a, b) => b.pages.length + b.untitled.length - (a.pages.length + a.untitled.length));
}

export function ScopePopover(p: {
  anchor: HTMLElement | null;
  sources: PageSource[];
  current: string | null;
  onPick: (label: string | null, ids: string[]) => void;
  onClose: () => void;
}) {
  const [f, setF] = useState("");
  const [open, setOpen] = useState<Set<string>>(new Set());
  const ref = useRef<HTMLDivElement>(null);
  const groups = useMemo(() => buildGroups(p.sources), [p.sources]);
  if (!p.anchor) return null;
  const r = p.anchor.getBoundingClientRect();
  const q = f.trim().toLowerCase();
  const style: React.CSSProperties = {
    left: Math.min(innerWidth - 310, Math.max(8, r.left)),
    bottom: Math.max(8, innerHeight - r.top + 8),
    maxHeight: Math.min(520, r.top - 16),
    overflow: "auto",
  };
  return (
    <div className="popover show" ref={ref} role="dialog" aria-label="검색 범위 선택" style={style}>
      <label className="pop-search">
        <Search />
        <span className="sr-only">페이지 검색</span>
        <input autoFocus placeholder="페이지 검색" value={f} onChange={(e) => setF(e.target.value)} />
      </label>
      {!q && (
        <button className="opt" role="menuitemradio" aria-checked={!p.current} onClick={() => p.onPick(null, [])}>
          <span className="radio" />
          전체 노트<small>{p.sources.length}</small>
        </button>
      )}
      <h5>폴더 / 데이터베이스</h5>
      {groups.map((g) => {
        const pages = q ? g.pages.filter((s) => s.title.toLowerCase().includes(q)) : g.pages;
        if (q && !pages.length && !g.name.toLowerCase().includes(q)) return null;
        const expanded = open.has(g.name) || !!q;
        const all = [...g.pages, ...g.untitled];
        return (
          <div key={g.name}>
            <div style={{ display: "flex" }}>
              <button
                className="opt"
                aria-expanded={expanded}
                style={{ width: 28, padding: "6px 4px", justifyContent: "center" }}
                aria-label="펼치기"
                onClick={() => {
                  const n = new Set(open);
                  if (n.has(g.name)) n.delete(g.name);
                  else n.add(g.name);
                  setOpen(n);
                }}
              >
                <ChevronRight className="car" />
              </button>
              <button
                className="opt"
                aria-checked={p.current === g.name}
                role="menuitemradio"
                onClick={() => p.onPick(g.name, all.map((s) => s.page_id))}
              >
                {g.name}
                <small>{all.length}</small>
              </button>
            </div>
            {expanded && (
              <div className="sub show">
                {g.untitled.length > 0 && !q && (
                  <div className="grp-note">날짜·제목 없는 기록 {g.untitled.length}개는 폴더 선택에 포함됩니다</div>
                )}
                {pages.slice(0, 60).map((s) => (
                  <button
                    key={s.page_id}
                    className="opt"
                    aria-checked={p.current === s.title}
                    onClick={() => p.onPick(s.title, [s.page_id])}
                  >
                    <span className="radio" />
                    <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{s.title}</span>
                    <small>{s.chunks}</small>
                  </button>
                ))}
                {pages.length > 60 && <div className="grp-note">… {pages.length - 60}개 더 (검색으로 좁히세요)</div>}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
