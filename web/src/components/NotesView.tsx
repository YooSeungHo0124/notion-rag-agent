import {
  ArrowLeft,
  Copy,
  MessageSquare,
  NotebookPen,
  Plus,
  StarOff,
  Trash2,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { Conv, Note, Turn } from "../types";
import { relTime } from "../util";
import { answerText } from "./Answer";

const CONF_TXT = {
  high: "근거 충분",
  medium: "일부만 확인",
  low: "근거 약함",
  none: "근거 없음",
} as const;

function MemoCard({
  n,
  autoFocus,
  onChange,
  onDelete,
}: {
  n: Note;
  autoFocus: boolean;
  onChange: (patch: Partial<Note>) => void;
  onDelete: () => void;
}) {
  const ta = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    const el = ta.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 420)}px`;
  }, [n.text]);
  return (
    <div className="memo">
      <div className="memo-hd">
        <input
          aria-label="메모 제목"
          placeholder="제목"
          value={n.title}
          maxLength={200}
          autoFocus={autoFocus}
          onChange={(e) => onChange({ title: e.target.value })}
        />
        <small>{relTime(n.updatedAt)}</small>
        <button
          className="icon-btn"
          aria-label="메모 삭제"
          title="메모 삭제"
          onClick={onDelete}
        >
          <Trash2 />
        </button>
      </div>
      <textarea
        ref={ta}
        aria-label="메모 내용"
        placeholder="노션에 없는 사실, 결정, 정정 내용을 적어 두면 질문할 때 출처로 함께 씁니다"
        value={n.text}
        maxLength={20000}
        onChange={(e) => onChange({ text: e.target.value })}
      />
      {n.fromQ && <div className="memo-from">답변에서 옮김 · “{n.fromQ}”</div>}
    </div>
  );
}

export default function NotesView(p: {
  notes: Note[];
  setNotes: (fn: (ns: Note[]) => Note[]) => void;
  useNotes: boolean;
  setUseNotes: (v: boolean) => void;
  saved: { c: Conv; t: Turn }[];
  onOpen: (convId: string, turnId: string) => void;
  onUnsave: (t: Turn) => void;
  onBack: () => void;
}) {
  const [fresh, setFresh] = useState<string | null>(null);
  const add = (init: Partial<Note> = {}) => {
    const n: Note = {
      id: Math.random().toString(36).slice(2, 10),
      title: "",
      text: "",
      updatedAt: new Date().toISOString(),
      ...init,
    };
    p.setNotes((ns) => [n, ...ns]);
    setFresh(n.id);
  };
  const patch = (id: string, x: Partial<Note>) =>
    p.setNotes((ns) =>
      ns.map((n) =>
        n.id === id ? { ...n, ...x, updatedAt: new Date().toISOString() } : n,
      ),
    );
  const usable = p.notes.filter((n) => n.text.trim()).length;
  return (
    <div className="notes-view fade">
      <div className="nv-head">
        <button
          className="icon-btn"
          onClick={p.onBack}
          aria-label="대화로 돌아가기"
        >
          <ArrowLeft />
        </button>
        <h2>
          <NotebookPen /> 노트
        </h2>
      </div>

      <section>
        <div className="nv-sec">
          <h3>
            내 메모 {p.notes.length > 0 && <small>{p.notes.length}</small>}
          </h3>
          <label
            className="nv-toggle"
            title="켜면 질문할 때 관련 있는 메모를 노션 기록과 함께 출처로 씁니다"
          >
            <input
              type="checkbox"
              checked={p.useNotes}
              onChange={(e) => p.setUseNotes(e.target.checked)}
            />
            질문할 때 메모도 출처로 쓰기
          </label>
          <button className="btn" onClick={() => add()}>
            <Plus />새 메모
          </button>
        </div>
        <p className="nv-note">
          메모는 이 브라우저에만 저장되고 노션에는 쓰지 않습니다.
          {p.useNotes
            ? ` 질문과 관련도가 기준 이상인 메모(최대 2개, 현재 ${usable}개 중)만 출처로 붙고, 다른 출처처럼 문장 검증을 거칩니다.`
            : " 지금은 질문에 쓰지 않습니다."}
        </p>
        {p.notes.length === 0 && (
          <div className="nv-empty">아직 메모가 없습니다.</div>
        )}
        {p.notes.map((n) => (
          <MemoCard
            key={n.id}
            n={n}
            autoFocus={fresh === n.id}
            onChange={(x) => patch(n.id, x)}
            onDelete={() => p.setNotes((ns) => ns.filter((x) => x.id !== n.id))}
          />
        ))}
      </section>

      <section>
        <div className="nv-sec">
          <h3>
            저장한 답변 {p.saved.length > 0 && <small>{p.saved.length}</small>}
          </h3>
        </div>
        {p.saved.length === 0 && (
          <div className="nv-empty">
            답변 아래 ‘저장’을 누르면 여기에 모입니다.
          </div>
        )}
        {[...p.saved].reverse().map(({ c, t }) => {
          const a = t.answer;
          const first =
            a?.sentences.find((s) => s.type !== "heading")?.text ?? "";
          return (
            <div key={t.id} className="saved-item">
              <div className="si-main">
                <b>{t.q}</b>
                <p>{first.replace(/\*\*|`/g, "")}</p>
                <small>
                  {relTime(new Date(t.startedAt).toISOString())}
                  {a
                    ? ` · ${CONF_TXT[a.confidence] ?? ""} · 출처 ${a.sources.length}`
                    : ""}
                </small>
              </div>
              <div className="si-acts">
                <button
                  className="btn ghost"
                  onClick={() => p.onOpen(c.id, t.id)}
                >
                  <MessageSquare />
                  대화 열기
                </button>
                {a && a.sentences.length > 0 && (
                  <button
                    className="btn ghost"
                    title="답변을 편집할 수 있는 메모로 복사"
                    onClick={() =>
                      add({
                        title: t.q.slice(0, 80),
                        text: answerText(a)
                          .replace(/\[\d+\]/g, "")
                          .trim(),
                        fromQ: t.q,
                      })
                    }
                  >
                    <Copy />
                    메모로
                  </button>
                )}
                <button className="btn ghost" onClick={() => p.onUnsave(t)}>
                  <StarOff />
                  저장 해제
                </button>
              </div>
            </div>
          );
        })}
      </section>
    </div>
  );
}
