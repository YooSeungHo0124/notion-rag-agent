import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import ChatPanel, { type Message } from "./components/ChatPanel";
import { answerPlainText } from "./components/AnswerView";
import SourcesPanel from "./components/SourcesPanel";
import StudioPanel from "./components/StudioPanel";
import type { AgentRun, Answer, Health, PageSource, SavedNote, SyncStatus } from "./types";
import { loadJSON, saveJSON, uid } from "./util";

const K_EXCLUDED = "nra.excludedSources";
const K_NOTES = "nra.savedNotes";

type Drawer = "none" | "sources" | "studio";

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [sources, setSources] = useState<PageSource[]>([]);
  const [sourcesError, setSourcesError] = useState<string | null>(null);
  const [excluded, setExcluded] = useState<Set<string>>(() => new Set(loadJSON<string[]>(K_EXCLUDED, [])));
  const [sync, setSync] = useState<SyncStatus | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [notes, setNotesState] = useState<SavedNote[]>(() => loadJSON<SavedNote[]>(K_NOTES, []));
  const [drawer, setDrawer] = useState<Drawer>("none");
  const [offline, setOffline] = useState(false);
  const syncTimer = useRef<number | null>(null);

  const selected = useMemo(
    () => new Set(sources.map((s) => s.page_id).filter((id) => !excluded.has(id))),
    [sources, excluded],
  );
  const setSelected = (s: Set<string>) => {
    const ex = new Set(sources.map((x) => x.page_id).filter((id) => !s.has(id)));
    setExcluded(ex);
    saveJSON(K_EXCLUDED, [...ex]);
  };
  const setNotes = (n: SavedNote[]) => {
    setNotesState(n);
    saveJSON(K_NOTES, n);
  };

  const refreshHealth = useCallback(async () => {
    try {
      const h = await api.health();
      setHealth(h);
      setOffline(false);
      return h;
    } catch {
      setOffline(true);
      return null;
    }
  }, []);

  const refreshSources = useCallback(async (refresh = false) => {
    try {
      setSources(await api.sources(refresh));
      setSourcesError(null);
    } catch (e) {
      setSourcesError((e as Error).message);
    }
  }, []);

  // poll health until the models are loaded, then fetch sources once
  useEffect(() => {
    let stop = false;
    let t: number | undefined;
    const tick = async () => {
      const h = await refreshHealth();
      if (stop) return;
      if (h?.ready) {
        refreshSources();
        if (h.syncing) pollSync();
      } else {
        t = window.setTimeout(tick, 3000);
      }
    };
    tick();
    return () => {
      stop = true;
      if (t) clearTimeout(t);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const pollSync = useCallback(() => {
    if (syncTimer.current) clearTimeout(syncTimer.current);
    const tick = async () => {
      try {
        const s = await api.syncStatus();
        setSync(s);
        if (s.running) {
          syncTimer.current = window.setTimeout(tick, 2000);
        } else {
          refreshSources(true);
          refreshHealth();
        }
      } catch {
        syncTimer.current = window.setTimeout(tick, 4000);
      }
    };
    tick();
  }, [refreshHealth, refreshSources]);

  const onSync = async () => {
    try {
      setSync(await api.sync());
      pollSync();
    } catch (e) {
      setSync({ running: false, ok: false, error: (e as Error).message, started_at: null, finished_at: null, summary: null, errors: [], last_sync: health?.last_sync ?? null });
    }
  };

  const threadId = useRef(uid());
  const busy = messages.some((m) => !m.answer && !m.error && m.run?.status !== "awaiting_approval");

  const applyRun = (id: string, run: AgentRun) =>
    setMessages((ms) =>
      ms.map((m) => {
        if (m.id !== id) return m;
        const a = run.answer;
        const answer = a && a.kind === "answer" ? (a as Answer) : undefined;
        const done = run.status === "done";
        return { ...m, run, answer, actionDone: done && (!a || a.kind === "action") };
      }),
    );

  const onDecide = async (id: string, approved: boolean) => {
    const m = messages.find((x) => x.id === id);
    if (!m?.run) return;
    setMessages((ms) => ms.map((x) => (x.id === id ? { ...x, run: { ...m.run!, status: "done", approval: null } } : x)));
    try {
      applyRun(id, await api.resume(m.run.thread_id, approved));
    } catch (e) {
      setMessages((ms) => ms.map((x) => (x.id === id ? { ...x, error: (e as Error).message } : x)));
    }
  };
  const allSelected = selected.size === sources.length;

  const onAsk = async (question: string, agentMode = false) => {
    const id = uid();
    const pageIds = allSelected ? [] : [...selected];
    const scope = allSelected ? "전체 노션" : `선택한 ${selected.size}개 페이지`;
    setMessages((ms) => [...ms, { id, question, scope: agentMode ? "에이전트 · 전체 노션" : scope, startedAt: Date.now(), agentMode }]);
    setDrawer("none");
    if (agentMode) {
      try {
        applyRun(id, await api.agent(question, threadId.current));
      } catch (e) {
        setMessages((ms) => ms.map((m) => (m.id === id ? { ...m, error: (e as Error).message } : m)));
      }
      return;
    }
    try {
      const answer = await api.askStream(question, pageIds, (stage) =>
        setMessages((ms) => ms.map((m) => (m.id === id ? { ...m, stage } : m))),
      );
      setMessages((ms) => ms.map((m) => (m.id === id ? { ...m, answer } : m)));
    } catch (e) {
      setMessages((ms) => ms.map((m) => (m.id === id ? { ...m, error: (e as Error).message } : m)));
    }
  };

  const onSave = (m: Message) => {
    if (!m.answer) return;
    const srcs = m.answer.sources.map((s, i) => `[${i + 1}] ${s.title} ${s.cite_url}`).join("\n");
    setNotes([
      {
        id: uid(),
        question: m.question,
        text: `${answerPlainText(m.answer)}\n\n${srcs}`,
        confidence: m.answer.confidence,
        savedAt: new Date().toISOString(),
      },
      ...notes,
    ]);
  };

  let disabledReason: string | null = null;
  if (offline) disabledReason = "서버에 연결할 수 없습니다 (nra serve 실행 확인)";
  else if (!health?.ready) disabledReason = health?.error ? `로딩 실패: ${health.error}` : "모델을 불러오는 중입니다 (최대 2분)…";
  else if (sources.length > 0 && selected.size === 0) disabledReason = "왼쪽에서 소스를 하나 이상 선택하세요";
  else if (sync?.running) disabledReason = "동기화가 끝난 뒤 질문할 수 있습니다";

  const sourcesPanel = (
    <SourcesPanel
      sources={sources}
      selected={selected}
      setSelected={setSelected}
      health={health}
      sync={sync}
      onSync={onSync}
      loadError={sourcesError}
    />
  );
  const studioPanel = <StudioPanel notes={notes} setNotes={setNotes} />;

  return (
    <div className="flex h-dvh flex-col">
      <header className="flex items-center gap-2 px-4 py-2.5">
        <button
          className="rounded-lg border border-stone-300 px-2.5 py-1 text-sm lg:hidden dark:border-stone-700"
          onClick={() => setDrawer(drawer === "sources" ? "none" : "sources")}
        >
          소스 {sources.length > 0 && `(${selected.size})`}
        </button>
        <span className="text-lg">📓</span>
        <span className="font-semibold">내 노션 노트북</span>
        <span
          className={`ml-2 hidden rounded-full px-2 py-0.5 text-[11px] sm:inline ${
            health?.ready
              ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300"
              : "bg-stone-200 text-stone-600 dark:bg-stone-800 dark:text-stone-400"
          }`}
        >
          {offline ? "오프라인" : health?.ready ? "준비됨" : "로딩 중"}
        </span>
        <button
          className="ml-auto rounded-lg border border-stone-300 px-2.5 py-1 text-sm xl:hidden dark:border-stone-700"
          onClick={() => setDrawer(drawer === "studio" ? "none" : "studio")}
        >
          작업 {notes.length > 0 && `(${notes.length})`}
        </button>
      </header>

      <main className="relative flex min-h-0 flex-1 gap-3 px-3 pb-3">
        <aside className="panel hidden w-72 shrink-0 overflow-hidden lg:block">{sourcesPanel}</aside>
        <section className="panel min-w-0 flex-1 overflow-hidden">
          <ChatPanel messages={messages} onAsk={onAsk} onDecide={onDecide} onSave={onSave} disabledReason={disabledReason} busy={busy} />
        </section>
        <aside className="panel hidden w-80 shrink-0 overflow-hidden xl:block">{studioPanel}</aside>

        {drawer !== "none" && (
          <div className="absolute inset-0 z-40 flex" onClick={() => setDrawer("none")}>
            <div className="absolute inset-0 bg-black/30" />
            <div
              className={`panel relative m-3 h-[calc(100%-1.5rem)] w-80 max-w-[calc(100%-1.5rem)] overflow-hidden shadow-2xl ${
                drawer === "studio" ? "ml-auto" : ""
              }`}
              onClick={(e) => e.stopPropagation()}
            >
              {drawer === "sources" ? sourcesPanel : studioPanel}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
