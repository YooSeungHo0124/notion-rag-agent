import { useEffect, useRef, useState } from "react";
import type { AgentRun, Answer, Stage } from "../types";
import AnswerView from "./AnswerView";

export interface Message {
  id: string;
  question: string;
  scope: string;
  startedAt: number;
  answer?: Answer;
  error?: string;
  agentMode?: boolean;
  stage?: Stage;
  run?: AgentRun;
  actionDone?: boolean;
}

interface Props {
  messages: Message[];
  onAsk: (q: string, agentMode: boolean) => void;
  onDecide: (id: string, approved: boolean) => void;
  onSave: (m: Message) => void;
  disabledReason: string | null;
  busy: boolean;
}

const STAGES: [Stage["name"], string][] = [
  ["retrieving", "노트 검색"],
  ["generating", "답변 작성"],
  ["verifying", "근거 검증"],
];

function StageLine({ m }: { m: Message }) {
  if (m.agentMode) return <>에이전트가 도구를 고르고 실행하는 중…</>;
  const cur = m.stage?.name === "retrieved" ? "generating" : (m.stage?.name ?? "retrieving");
  const idx = STAGES.findIndex(([n]) => n === cur);
  return (
    <span className="inline-flex flex-wrap items-center gap-1.5">
      {STAGES.map(([n, label], i) => (
        <span
          key={n}
          className={i < idx ? "text-emerald-600" : i === idx ? "font-medium text-indigo-600" : "text-stone-400"}
        >
          {i < idx ? "✓" : i === idx ? "●" : "○"} {label}
          {n === "retrieving" && i < idx && m.stage?.n !== undefined && ` (${m.stage.n}개)`}
          {i < STAGES.length - 1 && <span className="ml-1.5 text-stone-300">→</span>}
        </span>
      ))}
    </span>
  );
}

function Elapsed({ since }: { since: number }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 250);
    return () => clearInterval(t);
  }, []);
  return <>{((now - since) / 1000).toFixed(0)}초</>;
}

const EXAMPLES = ["GPU OOM 어떻게 해결했지?", "지난 회의에서 결정한 것 정리해줘", "최근에 읽은 책 목록"];

const TOOL_LABEL: Record<string, string> = {
  search_notes: "노트 검색",
  get_page: "페이지 읽기",
  get_timeline: "최근 수정 목록",
  append_to_page: "페이지에 추가",
  create_page: "새 페이지 생성",
};

function short(v: unknown): string {
  const s = typeof v === "string" ? v : JSON.stringify(v);
  return s.length > 120 ? s.slice(0, 120) + "…" : s;
}

function Steps({ run }: { run: AgentRun }) {
  if (!run.steps.length) return null;
  return (
    <details className="rounded-xl border border-stone-200 bg-stone-50 p-3 text-xs dark:border-stone-700 dark:bg-stone-900">
      <summary className="cursor-pointer text-stone-500">
        에이전트 실행 과정 {run.steps.length}단계
        {run.done_reason === "step_limit" ? " (단계 한도 도달)" : ""} · {(run.total_ms / 1000).toFixed(1)}초
      </summary>
      <ol className="mt-2 space-y-1.5">
        {run.steps.map((s, i) => (
          <li key={i} className="font-mono">
            <span className={s.ok ? "text-emerald-600" : "text-rose-600"}>{s.ok ? "✓" : "✗"}</span>{" "}
            <span className="text-stone-400">{s.agent}</span> {TOOL_LABEL[s.tool] ?? s.tool}{" "}
            <span className="text-stone-500">{short(s.args)}</span>
            {!s.ok && <div className="ml-4 text-rose-600">{short(s.observation)}</div>}
          </li>
        ))}
      </ol>
    </details>
  );
}

function Approval({ run, onDecide }: { run: AgentRun; onDecide: (ok: boolean) => void }) {
  const ap = run.approval!;
  return (
    <div className="rounded-xl border border-amber-300 bg-amber-50 p-4 text-sm dark:border-amber-800 dark:bg-amber-950/40">
      <p className="font-medium text-amber-900 dark:text-amber-200">
        노션에 쓰기 전에 승인이 필요합니다: {TOOL_LABEL[ap.tool] ?? ap.tool}
      </p>
      {ap.thought && <p className="mt-1 text-xs text-amber-800/80 dark:text-amber-300/80">{ap.thought}</p>}
      <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap rounded-lg bg-white p-2 text-xs dark:bg-stone-900">
        {JSON.stringify(ap.args, null, 2)}
      </pre>
      <div className="mt-3 flex gap-2">
        <button onClick={() => onDecide(true)} className="rounded-lg bg-amber-600 px-3 py-1.5 text-white hover:bg-amber-700">
          승인하고 실행
        </button>
        <button onClick={() => onDecide(false)} className="rounded-lg border border-stone-300 px-3 py-1.5 hover:bg-stone-100 dark:border-stone-600 dark:hover:bg-stone-800">
          거부
        </button>
      </div>
    </div>
  );
}

export default function ChatPanel({ messages, onAsk, onDecide, onSave, disabledReason, busy }: Props) {
  const [q, setQ] = useState("");
  const [agentMode, setAgentMode] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const submit = () => {
    const v = q.trim();
    if (!v || busy || disabledReason) return;
    onAsk(v, agentMode);
    setQ("");
  };

  return (
    <div className="flex h-full flex-col">
      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-6 sm:px-8">
        {messages.length === 0 && (
          <div className="mx-auto mt-16 max-w-md text-center">
            <h1 className="text-2xl font-semibold">내 노션에게 물어보기</h1>
            <p className="mt-2 text-sm text-stone-500">
              답변의 모든 문장은 내 노트의 원문에 연결됩니다. 근거가 없으면 지어내지 않습니다.
            </p>
            <div className="mt-6 flex flex-wrap justify-center gap-2">
              {EXAMPLES.map((e) => (
                <button
                  key={e}
                  onClick={() => setQ(e)}
                  className="rounded-full border border-stone-200 px-3 py-1.5 text-sm hover:bg-stone-100 dark:border-stone-700 dark:hover:bg-stone-800"
                >
                  {e}
                </button>
              ))}
            </div>
          </div>
        )}
        <div className="mx-auto max-w-3xl space-y-8">
          {messages.map((m) => (
            <div key={m.id} className="space-y-3">
              <div className="flex justify-end">
                <div className="max-w-[85%] rounded-2xl rounded-br-md bg-indigo-600 px-4 py-2 text-sm text-white">
                  {m.question}
                </div>
              </div>
              <div className="text-[11px] text-stone-400">범위: {m.scope}</div>
              {m.run && <Steps run={m.run} />}
              {m.run?.status === "awaiting_approval" && (
                <Approval run={m.run} onDecide={(ok) => onDecide(m.id, ok)} />
              )}
              {m.actionDone && m.run?.answer?.kind === "action" && (
                <div className="rounded-xl border border-stone-200 p-3 text-sm dark:border-stone-700">
                  {m.run.answer.steps.map((s, i) => (
                    <div key={i}>
                      {TOOL_LABEL[s.tool] ?? s.tool}: {s.ok ? "완료" : "실행 안 함"} — {short(s.observation)}
                    </div>
                  ))}
                </div>
              )}
              {!m.answer && !m.error && !m.actionDone && m.run?.status !== "awaiting_approval" && (
                <div className="flex items-center gap-3 text-sm text-stone-500">
                  <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-indigo-500 border-t-transparent" />
                  <StageLine m={m} /> <Elapsed since={m.startedAt} />
                </div>
              )}
              {m.error && (
                <div className="rounded-xl border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700 dark:border-rose-900 dark:bg-rose-950/40 dark:text-rose-300">
                  오류: {m.error}
                </div>
              )}
              {m.answer && (
                <div className="group/answer">
                  <AnswerView a={m.answer} onFollowup={busy ? undefined : (f) => onAsk(f, !!m.agentMode)} />
                  {m.answer.answerable && (
                    <button
                      onClick={() => onSave(m)}
                      className="mt-2 text-xs text-stone-400 hover:text-indigo-600"
                    >
                      📌 작업 패널에 저장
                    </button>
                  )}
                </div>
              )}
            </div>
          ))}
          <div ref={endRef} />
        </div>
      </div>

      <div className="border-t border-stone-200 p-3 sm:p-4 dark:border-stone-800">
        <label className="mx-auto mb-2 flex max-w-3xl items-center gap-2 text-xs text-stone-500">
          <input type="checkbox" checked={agentMode} onChange={(e) => setAgentMode(e.target.checked)} />
          에이전트 모드 (여러 번 검색·페이지 읽기, 노션 기록 요청 시 승인 후 실행)
        </label>
        <div className="mx-auto flex max-w-3xl items-end gap-2 rounded-2xl border border-stone-200 bg-stone-50 p-2 focus-within:border-indigo-400 dark:border-stone-700 dark:bg-stone-800">
          <textarea
            rows={1}
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                submit();
              }
            }}
            placeholder={disabledReason ?? "질문을 입력하세요 (Enter 전송, Shift+Enter 줄바꿈)"}
            className="max-h-40 min-h-[2.5rem] flex-1 resize-none bg-transparent px-2 py-2 text-sm outline-none"
          />
          <button
            onClick={submit}
            disabled={!q.trim() || busy || !!disabledReason}
            className="rounded-xl bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-40"
          >
            {busy ? "…" : "보내기"}
          </button>
        </div>
      </div>
    </div>
  );
}
