import { ArrowUp, Check, Folder, FolderSearch, Info, Menu, PanelRight, PenLine, Search, Square, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import { answerText } from "./components/Answer";
import { ScopePopover, Sidebar } from "./components/Sidebar";
import SourcePanel from "./components/SourcePanel";
import TurnView, { type TurnActions } from "./components/TurnView";
import type { AgentRun, Answer, Conv, Health, PageSource, SourceHit, SyncStatus, Turn } from "./types";
import { loadJSON, saveJSON, uid } from "./util";

const K_CONVS = "nra.convs.v2";
const K_SAVED = "nra.saved.v2";
const K_NOTICE = "nra.notice.dismissed";
const NOTION_WIN = "nra-notion";
const EXAMPLES: [string, string][] = [
  ["TECH_SEE v3 backend 포트가 몇이었지?", "기억 찾기"],
  ["OOM 났을 때 batch 몇으로 줄였지?", "기억 찾기"],
  ["쓰러짐 모듈 지금까지 시도한 방법 정리해줘", "정리·비교"],
];

type Scope = { label: string; page_ids: string[] } | null;

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [sources, setSources] = useState<PageSource[]>([]);
  const [sync, setSync] = useState<SyncStatus | null>(null);
  const [convs, setConvs] = useState<Conv[]>(() => loadJSON<Conv[]>(K_CONVS, []));
  const [cur, setCur] = useState<string | null>(null);
  const [saved, setSaved] = useState<string[]>(() => loadJSON<string[]>(K_SAVED, []));
  const [busy, setBusy] = useState<string | null>(null);
  const [agent, setAgent] = useState(false);
  const [scope, setScope] = useState<Scope>(null);
  const [scopeAnchor, setScopeAnchor] = useState<HTMLElement | null>(null);
  const [panel, setPanel] = useState<{ turnId: string; idx: number; closest?: boolean } | null>(null);
  const [drawer, setDrawer] = useState(false);
  const [theme, setTheme] = useState<"light" | "dark">(
    () => (document.documentElement.dataset.theme as "light" | "dark") || "light",
  );
  const [notice, setNotice] = useState(() => !loadJSON<boolean>(K_NOTICE, false));
  const [input, setInput] = useState("");
  const [toastMsg, setToastMsg] = useState<string | null>(null);
  const [hover, setHover] = useState<{ x: number; y: number; hit: SourceHit } | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const appRef = useRef<HTMLDivElement>(null);
  const toastT = useRef<number | undefined>(undefined);
  const hoverT = useRef<number | undefined>(undefined);

  const conv = convs.find((c) => c.id === cur) ?? null;
  const ready = !!health?.ready;

  const toast = useCallback((m: string) => {
    setToastMsg(m);
    clearTimeout(toastT.current);
    toastT.current = window.setTimeout(() => setToastMsg(null), 1600);
  }, []);

  useEffect(() => saveJSON(K_CONVS, convs.slice(-100)), [convs]);
  useEffect(() => saveJSON(K_SAVED, saved), [saved]);

  /* ---------- health / sources / sync ---------- */
  const refreshHealth = useCallback(async () => {
    try {
      const h = await api.health();
      setHealth(h);
      return h;
    } catch {
      return null;
    }
  }, []);
  const refreshSources = useCallback(async (refresh = false) => {
    try {
      setSources(await api.sources(refresh));
    } catch {
      /* surfaced via health */
    }
  }, []);
  useEffect(() => {
    let stop = false;
    let t: number | undefined;
    const tick = async () => {
      const h = await refreshHealth();
      if (stop) return;
      if (h?.ready) refreshSources();
      else t = window.setTimeout(tick, 3000);
    };
    tick();
    return () => {
      stop = true;
      clearTimeout(t);
    };
  }, [refreshHealth, refreshSources]);

  const onSync = async () => {
    try {
      setSync(await api.sync());
      toast("노션 동기화를 시작했습니다");
      const poll = async () => {
        const s = await api.syncStatus().catch(() => null);
        if (s) setSync(s);
        if (s?.running) setTimeout(poll, 2000);
        else {
          refreshSources(true);
          refreshHealth();
          if (s) toast(s.ok ? "동기화 완료" : `동기화 실패: ${s.error ?? ""}`);
        }
      };
      poll();
    } catch (e) {
      toast((e as Error).message);
    }
  };

  /* ---------- turns ---------- */
  const patchTurn = useCallback((cid: string, tid: string, patch: Partial<Turn>) => {
    setConvs((cs) =>
      cs.map((c) =>
        c.id !== cid
          ? c
          : { ...c, updatedAt: Date.now(), turns: c.turns.map((t) => (t.id === tid ? { ...t, ...patch } : t)) },
      ),
    );
  }, []);

  const applyRun = (cid: string, tid: string, run: AgentRun, t0: number) => {
    const a = run.answer;
    patchTurn(cid, tid, {
      run,
      answer: a && a.kind === "answer" ? (a as Answer) : undefined,
      doneMs: run.status === "done" ? Date.now() - t0 : undefined,
    });
  };

  const ask = async (q: string, opts: { agentMode?: boolean; scope?: Scope; newConv?: boolean } = {}) => {
    q = q.trim();
    if (!q || busy || !ready) return;
    const mode: Turn["mode"] = (opts.agentMode ?? agent) ? "agent" : "ask";
    const sc = opts.scope !== undefined ? opts.scope : scope;
    let cid = opts.newConv ? null : cur;
    const turn: Turn = { id: uid(), q, mode, startedAt: Date.now() };
    if (!cid) {
      cid = uid();
      const c: Conv = { id: cid, title: q.slice(0, 60), updatedAt: Date.now(), turns: [turn], scope: sc };
      setConvs((cs) => [...cs, c]);
      setCur(cid);
    } else {
      const id = cid;
      setConvs((cs) => cs.map((c) => (c.id === id ? { ...c, updatedAt: Date.now(), turns: [...c.turns, turn] } : c)));
    }
    setPanel(null);
    setBusy(turn.id);
    setDrawer(false);
    requestAnimationFrame(() =>
      scrollRef.current?.querySelector(`[data-turn="${turn.id}"]`)?.scrollIntoView({ block: "start", behavior: "smooth" }),
    );
    const t0 = Date.now();
    const ctl = new AbortController();
    abortRef.current = ctl;
    const id = cid;
    try {
      if (mode === "agent") {
        applyRun(id, turn.id, await api.agent(q, id), t0);
      } else {
        const a = await api.askStream(q, sc?.page_ids ?? [], (stage) => patchTurn(id, turn.id, stage.name === "retrieved" ? { stage, retrieved: stage } : { stage }), ctl.signal);
        patchTurn(id, turn.id, { answer: a, doneMs: Date.now() - t0 });
      }
    } catch (e) {
      if ((e as Error).name === "AbortError") patchTurn(id, turn.id, { stopped: true });
      else patchTurn(id, turn.id, { error: (e as Error).message });
    } finally {
      setBusy(null);
      abortRef.current = null;
    }
  };

  const stop = () => abortRef.current?.abort();

  const newConv = () => {
    stop();
    setCur(null);
    setPanel(null);
    setDrawer(false);
    setInput("");
    setTimeout(() => inputRef.current?.focus(), 0);
  };

  /* ---------- source panel ---------- */
  const panelSources: SourceHit[] = useMemo(() => {
    if (!panel || !conv) return [];
    const t = conv.turns.find((x) => x.id === panel.turnId);
    if (!t?.answer) return [];
    if (panel.closest) return t.answer.closest ? [t.answer.closest] : [];
    return t.answer.sources;
  }, [panel, conv]);

  const openNotion = useCallback(
    (url: string, title: string) => {
      window.open(url, NOTION_WIN);
      toast(`노션 창에서 열기 · ${title}`);
    },
    [toast],
  );

  // linked highlight: sentences citing the open source + active cite chips
  useEffect(() => {
    const root = scrollRef.current;
    if (!root) return;
    root.querySelectorAll(".s.linked").forEach((e) => e.classList.remove("linked"));
    root.querySelectorAll(".cite.on").forEach((e) => e.classList.remove("on"));
    if (!panel || panel.closest) return;
    const turnEl = root.querySelector(`[data-turn="${panel.turnId}"]`);
    if (!turnEl) return;
    const n = String(panel.idx);
    turnEl.querySelectorAll<HTMLElement>(".s").forEach((el) => {
      if ((el.dataset.src ?? "").split(" ").includes(n)) el.classList.add("linked");
    });
    turnEl.querySelectorAll<HTMLElement>(".cite").forEach((el) => el.classList.toggle("on", el.dataset.c === n));
    const first = turnEl.querySelector(".s.linked");
    if (first && !matchMedia("(max-width:768px)").matches) first.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [panel, convs]);

  const act: TurnActions = {
    openSource: (turnId, n) => setPanel({ turnId, idx: n }),
    openClosest: (turnId) => setPanel({ turnId, idx: 1, closest: true }),
    followup: (q) => ask(q, { agentMode: false }),
    rephrase: (q) => {
      setInput(q);
      setTimeout(() => inputRef.current?.focus(), 0);
    },
    widen: (q) => {
      setScope(null);
      toast("범위를 전체 노트로 넓혔습니다");
      ask(q, { scope: null });
    },
    copy: (a) => {
      navigator.clipboard?.writeText(answerText(a)).catch(() => {});
      toast("답변을 복사했습니다");
    },
    toggleSave: (t) => {
      const was = saved.includes(t.id);
      setSaved((s) => (was ? s.filter((x) => x !== t.id) : [...s, t.id]));
      toast(was ? "저장을 취소했습니다" : "저장한 답변에 추가했습니다");
    },
    isSaved: (t) => saved.includes(t.id),
    decide: async (turnId, approved, args) => {
      const c = conv;
      const t = c?.turns.find((x) => x.id === turnId);
      if (!c || !t?.run) return;
      const t0 = Date.now();
      patchTurn(c.id, turnId, { run: { ...t.run, status: "done", approval: null } });
      setBusy(turnId);
      try {
        applyRun(c.id, turnId, await api.resume(t.run.thread_id, approved, args), t0);
      } catch (e) {
        patchTurn(c.id, turnId, { error: (e as Error).message });
      } finally {
        setBusy(null);
      }
    },
    retry: (t) => ask(t.q, { agentMode: t.mode === "agent" }),
    openNotion,
  };

  /* ---------- thread events: cites, hover, copy ---------- */
  const turnOf = (el: Element) => {
    const tid = el.closest<HTMLElement>("[data-turn]")?.dataset.turn;
    return conv?.turns.find((t) => t.id === tid);
  };
  const onThreadClick = (e: React.MouseEvent) => {
    const el = e.target as HTMLElement;
    const c = el.closest<HTMLElement>(".cite");
    if (c) {
      const t = turnOf(c);
      const n = Number(c.dataset.c);
      setHover(null);
      if (!t?.answer) return;
      if (e.shiftKey) {
        const h = t.answer.sources[n - 1];
        if (h) openNotion(h.cite_url, h.title);
        return;
      }
      setPanel({ turnId: t.id, idx: n });
      return;
    }
    const cp = el.closest<HTMLElement>("[data-copy]");
    if (cp) {
      navigator.clipboard?.writeText(cp.dataset.copy ?? "").catch(() => {});
      toast("코드를 복사했습니다");
    }
  };
  const onThreadOver = (e: React.MouseEvent) => {
    const c = (e.target as HTMLElement).closest<HTMLElement>(".cite");
    clearTimeout(hoverT.current);
    if (!c || matchMedia("(max-width:768px)").matches) {
      if (!c) setHover(null);
      return;
    }
    hoverT.current = window.setTimeout(() => {
      const hit = turnOf(c)?.answer?.sources[Number(c.dataset.c) - 1];
      if (!hit) return;
      const r = c.getBoundingClientRect();
      setHover({
        hit,
        x: Math.min(innerWidth - 310, Math.max(8, r.left - 20)),
        y: r.bottom + 8 + 140 > innerHeight ? r.top - 150 : r.bottom + 8,
      });
    }, 250);
  };
  const onDbl = (e: React.MouseEvent) => {
    const v = (e.target as HTMLElement).closest("code.v");
    if (!v) return;
    navigator.clipboard?.writeText(v.textContent ?? "").catch(() => {});
    getSelection()?.removeAllRanges();
    toast(`복사됨 · ${v.textContent}`);
  };

  /* ---------- keyboard ---------- */
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const typing = /INPUT|TEXTAREA/.test((document.activeElement as HTMLElement | null)?.tagName ?? "");
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        newConv();
        return;
      }
      if ((e.ctrlKey || e.metaKey) && e.key === "/") {
        e.preventDefault();
        toast(!agent ? "에이전트 모드 켜짐" : "에이전트 모드 꺼짐");
        setAgent(!agent);
        return;
      }
      if (e.key === "Escape") {
        if (scopeAnchor) return setScopeAnchor(null);
        if (busy && abortRef.current) return stop();
        if (panel) return setPanel(null);
        setDrawer(false);
        return;
      }
      if (!typing && panel && panelSources.length) {
        const n = panelSources.length;
        if (e.key === "[") setPanel({ ...panel, idx: ((panel.idx - 2 + n) % n) + 1 });
        if (e.key === "]") setPanel({ ...panel, idx: (panel.idx % n) + 1 });
      }
    };
    addEventListener("keydown", onKey);
    return () => removeEventListener("keydown", onKey);
  });

  /* ---------- panel resize ---------- */
  const onResize = (e: React.PointerEvent) => {
    e.preventDefault();
    const app = appRef.current!;
    app.classList.add("dragging");
    const move = (ev: PointerEvent) => {
      const total = app.clientWidth - 56;
      const w = Math.min(total - 480, Math.max(320, app.getBoundingClientRect().right - ev.clientX));
      app.style.setProperty("--panel-w", `${w}px`);
    };
    const up = () => {
      app.classList.remove("dragging");
      removeEventListener("pointermove", move);
      removeEventListener("pointerup", up);
    };
    addEventListener("pointermove", move);
    addEventListener("pointerup", up);
  };

  const toggleTheme = () => {
    const n = theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = n;
    try {
      localStorage.setItem("nra.theme", n);
    } catch {
      /* ignore */
    }
    setTheme(n);
  };

  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(160, el.scrollHeight)}px`;
  }, [input]);

  const submit = () => {
    if (busy && abortRef.current) return stop();
    if (!input.trim()) return;
    const q = input;
    setInput("");
    ask(q);
  };

  const disabledReason = !health
    ? "서버에 연결하는 중…"
    : health.error
      ? `모델 로딩 실패: ${health.error}`
      : !health.ready
        ? "모델을 불러오는 중입니다 (1~2분)…"
        : null;

  const savedTurns = convs.flatMap((c) => c.turns.filter((t) => saved.includes(t.id)).map((t) => ({ c, t })));

  return (
    <div className={`app${panel ? " panel-open" : ""}${drawer ? " drawer-open" : ""}`} ref={appRef}>
      <div
        className="backdrop"
        onClick={() => {
          setPanel(null);
          setDrawer(false);
        }}
      />
      <Sidebar
        convs={convs}
        current={cur}
        onSelect={(id) => {
          setCur(id);
          setPanel(null);
          setDrawer(false);
        }}
        onNew={newConv}
        savedCount={saved.length}
        onSaved={() => {
          if (!savedTurns.length) return toast("저장한 답변이 없습니다");
          setCur(savedTurns[savedTurns.length - 1].c.id);
          toast(`저장한 답변 ${savedTurns.length}개 · 가장 최근 대화로 이동`);
        }}
        scopeLabel={scope?.label ?? "전체"}
        onScope={(el) => setScopeAnchor(el)}
        health={health}
        sync={sync}
        onSync={onSync}
        theme={theme}
        onTheme={toggleTheme}
      />

      <main className="main">
        <div className="topbar">
          <button className="icon-btn menu-btn" onClick={() => setDrawer(!drawer)} aria-label="사이드바 열기">
            <Menu />
          </button>
          <div className="spacer" />
          <button
            className="icon-btn"
            aria-label="출처 패널 열기/닫기"
            title="출처 패널"
            onClick={() => {
              if (panel) return setPanel(null);
              const t = [...(conv?.turns ?? [])].reverse().find((x) => x.answer?.sources.length);
              if (t) setPanel({ turnId: t.id, idx: 1 });
              else toast("열 수 있는 출처가 없습니다");
            }}
          >
            <PanelRight />
          </button>
        </div>

        <div className="scroll" ref={scrollRef} onClick={onThreadClick} onMouseOver={onThreadOver} onDoubleClick={onDbl}>
          {!conv ? (
            <div className="empty">
              <h2>무엇을 찾고 있나요?</h2>
              <p>
                {health?.pages ?? 0}개 페이지에서 찾고, 출처를 바로 대조할 수 있게 보여드립니다.
                {disabledReason && <> · {disabledReason}</>}
              </p>
              <div className="ex">
                {EXAMPLES.map(([q, t]) => (
                  <button key={q} onClick={() => ask(q, { newConv: true })} disabled={!ready}>
                    <Search />
                    {q}
                    <small>{t}</small>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="thread">
              {conv.turns.map((t) => (
                <TurnView key={t.id} t={t} act={act} generating={busy === t.id} />
              ))}
            </div>
          )}
        </div>

        <div className="composer-wrap">
          <div className="composer-inner">
            {notice && (
              <div className="notice">
                <Info />
                답변은 내 노트에서만 찾습니다. 중요한 값은 출처 번호를 눌러 원문과 대조하세요.
                <button
                  onClick={() => {
                    setNotice(false);
                    saveJSON(K_NOTICE, true);
                  }}
                >
                  확인
                </button>
              </div>
            )}
            <div className="chips">
              {scope ? (
                <span className="chip">
                  <Folder />
                  범위: {scope.label}
                  <button onClick={() => setScope(null)} aria-label="범위 해제">
                    <X />
                  </button>
                </span>
              ) : (
                <button
                  className="chip"
                  style={{ background: "var(--bg-subtle)", color: "var(--text-muted)", padding: "0 10px" }}
                  onClick={(e) => setScopeAnchor(e.currentTarget)}
                >
                  <FolderSearch />
                  범위: 전체 노트
                </button>
              )}
            </div>
            <form
              className="composer"
              onSubmit={(e) => {
                e.preventDefault();
                submit();
              }}
            >
              <label htmlFor="q" className="sr-only">
                질문 입력
              </label>
              <textarea
                id="q"
                ref={inputRef}
                rows={1}
                value={input}
                disabled={!!disabledReason}
                placeholder={
                  disabledReason ?? (agent ? "노션에 무엇을 기록할까요? 실행 전 확인을 받습니다" : "무엇이든 물어보세요")
                }
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                    e.preventDefault();
                    submit();
                  }
                }}
              />
              <button
                type="button"
                className="toggle"
                aria-pressed={agent}
                title="에이전트 모드 (Ctrl+/)"
                onClick={() => setAgent(!agent)}
              >
                <PenLine />
                <span className="lbl">에이전트</span>
              </button>
              <button
                type="submit"
                className={`send${busy ? " stop" : ""}`}
                aria-label={busy ? "생성 중단 (Esc)" : "보내기"}
                disabled={busy ? false : !input.trim() || !!disabledReason}
              >
                {busy ? <Square /> : <ArrowUp />}
              </button>
            </form>
            <div className="hint">
              <span>
                <span className="kbd">Ctrl K</span> 새 질문
              </span>
              <span>
                <span className="kbd">Ctrl /</span> 에이전트
              </span>
              <span>
                <span className="kbd">Esc</span> 중단·닫기
              </span>
              <span>
                <span className="kbd">[</span> <span className="kbd">]</span> 출처 전환
              </span>
            </div>
          </div>
        </div>
      </main>

      <aside className="panel" aria-label="출처 패널" aria-hidden={!panel}>
        <div
          className="resizer"
          role="separator"
          aria-orientation="vertical"
          aria-label="패널 폭 조절"
          onPointerDown={onResize}
        />
        <div className="sheet-grip" aria-hidden="true" />
        {panel && panelSources.length > 0 && (
          <SourcePanel
            sources={panelSources}
            idx={Math.min(panel.idx, panelSources.length)}
            onIdx={(n) => setPanel({ ...panel, idx: n })}
            onClose={() => setPanel(null)}
            openNotion={openNotion}
          />
        )}
      </aside>

      {hover && (
        <div className="hovercard show" role="tooltip" style={{ left: hover.x, top: hover.y }}>
          <b>{hover.hit.title}</b>
          <div className="p">{[...(hover.hit.path ?? []), ...(hover.hit.heading_path ?? [])].join(" › ")}</div>
          <div className="x">{hover.hit.text}</div>
          <div className="k">클릭: 출처 패널 · Shift+클릭: 노션 창</div>
        </div>
      )}
      <div className={`toast${toastMsg ? " show" : ""}`} role="status" aria-live="polite">
        <Check />
        {toastMsg}
      </div>
      {scopeAnchor && (
        <>
          <div style={{ position: "fixed", inset: 0, zIndex: 39 }} onClick={() => setScopeAnchor(null)} />
          <ScopePopover
            anchor={scopeAnchor}
            sources={sources}
            current={scope?.label ?? null}
            onPick={(label, ids) => {
              setScope(label ? { label, page_ids: ids } : null);
              setScopeAnchor(null);
              toast(`범위: ${label ?? "전체 노트"}`);
            }}
            onClose={() => setScopeAnchor(null)}
          />
        </>
      )}
    </div>
  );
}
