import { useState } from "react";
import type { SavedNote } from "../types";
import { uid } from "../util";

interface Props {
  notes: SavedNote[];
  setNotes: (n: SavedNote[]) => void;
}

export default function StudioPanel({ notes, setNotes }: Props) {
  const [draft, setDraft] = useState("");
  const add = () => {
    const t = draft.trim();
    if (!t) return;
    setNotes([{ id: uid(), text: t, savedAt: new Date().toISOString() }, ...notes]);
    setDraft("");
  };
  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-stone-200 p-4 dark:border-stone-800">
        <h2 className="text-sm font-semibold">작업</h2>
      </div>
      <div className="min-h-0 flex-1 space-y-6 overflow-y-auto p-4">
        <section>
          <h3 className="text-xs font-semibold text-stone-500">⏳ 승인 대기 (F4 예정)</h3>
          <p className="mt-2 rounded-lg border border-dashed border-stone-300 p-3 text-xs text-stone-400 dark:border-stone-700">
            노션 페이지 생성·수정 같은 쓰기 작업은 여기서 미리보기 후 승인하게 됩니다.
          </p>
        </section>

        <section>
          <h3 className="text-xs font-semibold text-stone-500">📌 저장한 답변 · 메모</h3>
          <div className="mt-2 flex gap-2">
            <input
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !e.nativeEvent.isComposing && add()}
              placeholder="메모 추가"
              className="min-w-0 flex-1 rounded-lg border border-stone-200 bg-stone-50 px-2 py-1 text-sm outline-none focus:border-indigo-400 dark:border-stone-700 dark:bg-stone-800"
            />
            <button onClick={add} className="rounded-lg border border-stone-300 px-2 text-sm dark:border-stone-700">
              +
            </button>
          </div>
          {notes.length === 0 && <p className="mt-3 text-xs text-stone-400">저장한 항목이 없습니다.</p>}
          <ul className="mt-3 space-y-2">
            {notes.map((n) => (
              <li key={n.id} className="group rounded-xl border border-stone-200 p-3 text-sm dark:border-stone-700">
                {n.question && <p className="mb-1 text-xs font-semibold text-indigo-600 dark:text-indigo-300">Q. {n.question}</p>}
                <p className="whitespace-pre-wrap leading-relaxed">{n.text}</p>
                <div className="mt-2 flex items-center justify-between text-[11px] text-stone-400">
                  <span>{new Date(n.savedAt).toLocaleString("ko-KR")}</span>
                  <span className="flex gap-2">
                    <button onClick={() => navigator.clipboard?.writeText(n.text)} className="hover:text-indigo-600">
                      복사
                    </button>
                    <button onClick={() => setNotes(notes.filter((x) => x.id !== n.id))} className="hover:text-rose-600">
                      삭제
                    </button>
                  </span>
                </div>
              </li>
            ))}
          </ul>
        </section>

        <section>
          <h3 className="text-xs font-semibold text-stone-500">🧪 평가 결과</h3>
          <p className="mt-2 text-xs text-stone-400">`eval/` 실행 결과 연동 예정</p>
        </section>
      </div>
    </div>
  );
}
