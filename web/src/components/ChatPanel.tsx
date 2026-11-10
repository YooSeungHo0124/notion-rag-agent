import { useEffect, useRef, useState } from "react";
import type { Answer } from "../types";
import AnswerView from "./AnswerView";

export interface Message {
  id: string;
  question: string;
  scope: string;
  startedAt: number;
  answer?: Answer;
  error?: string;
}

interface Props {
  messages: Message[];
  onAsk: (q: string) => void;
  onSave: (m: Message) => void;
  disabledReason: string | null;
  busy: boolean;
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

export default function ChatPanel({ messages, onAsk, onSave, disabledReason, busy }: Props) {
  const [q, setQ] = useState("");
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const submit = () => {
    const v = q.trim();
    if (!v || busy || disabledReason) return;
    onAsk(v);
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
              {!m.answer && !m.error && (
                <div className="flex items-center gap-3 text-sm text-stone-500">
                  <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-indigo-500 border-t-transparent" />
                  노트를 찾고 답변을 검증하는 중… <Elapsed since={m.startedAt} />
                </div>
              )}
              {m.error && (
                <div className="rounded-xl border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700 dark:border-rose-900 dark:bg-rose-950/40 dark:text-rose-300">
                  오류: {m.error}
                </div>
              )}
              {m.answer && (
                <div className="group/answer">
                  <AnswerView a={m.answer} />
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
