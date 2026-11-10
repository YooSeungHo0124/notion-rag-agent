import { BookOpen, CalendarClock, ChevronRight, FileText, HelpCircle, Network, Sparkles } from "lucide-react";
import { useState, type ReactNode } from "react";
import type { MindNode, StudioKind } from "../types";
import { Cite } from "./Answer";

export const STUDIO_KINDS: { kind: StudioKind; title: string; desc: string; icon: ReactNode }[] = [
  { kind: "briefing", title: "브리핑 문서", desc: "핵심 내용·수치·남은 문제를 한 장으로", icon: <FileText /> },
  { kind: "study", title: "학습 가이드", desc: "핵심 개념 표, 동작 원리, 복습 질문", icon: <BookOpen /> },
  { kind: "faq", title: "자주 묻는 질문", desc: "기록으로 답할 수 있는 질문과 답", icon: <HelpCircle /> },
  { kind: "timeline", title: "타임라인", desc: "사건·변경·결정을 시간순으로", icon: <CalendarClock /> },
  { kind: "mindmap", title: "마인드맵", desc: "주제 구조를 한눈에, 노드를 눌러 질문", icon: <Network /> },
];

export const STUDIO_TITLE: Record<StudioKind, string> = {
  briefing: "브리핑 문서",
  study: "학습 가이드",
  faq: "자주 묻는 질문",
  timeline: "타임라인",
  guide: "소스 가이드",
  mindmap: "마인드맵",
};

/** Popover: pick a Studio document for the current scope, optionally narrowed by a topic. */
export function StudioMenu(p: {
  anchor: HTMLElement;
  scopeLabel: string;
  scopePages: number;
  onPick: (kind: StudioKind, topic: string) => void;
}) {
  const [topic, setTopic] = useState("");
  const r = p.anchor.getBoundingClientRect();
  const style: React.CSSProperties = {
    left: Math.min(innerWidth - 340, Math.max(8, r.left)),
    bottom: Math.max(8, innerHeight - r.top + 8),
    width: 330,
  };
  return (
    <div className="popover show studio-menu" role="dialog" aria-label="스튜디오" style={style}>
      <div className="sm-head">
        <Sparkles />
        <b>스튜디오</b>
        <small>
          범위: {p.scopeLabel}
          {p.scopePages ? ` · ${p.scopePages}페이지` : ""}
        </small>
      </div>
      <label className="pop-search">
        <span className="sr-only">주제 (선택)</span>
        <input
          autoFocus
          placeholder="주제로 좁히기 (선택) 예: OCR 전처리"
          value={topic}
          maxLength={300}
          onChange={(e) => setTopic(e.target.value)}
        />
      </label>
      {STUDIO_KINDS.map((k) => (
        <button key={k.kind} className="opt sm-opt" onClick={() => p.onPick(k.kind, topic)}>
          {k.icon}
          <span>
            <b>{k.title}</b>
            <small>{k.desc}</small>
          </span>
        </button>
      ))}
      <div className="grp-note">
        범위가 크면 최근에 수정한 페이지부터 읽습니다. 주제를 적으면 그 주제와 관련된 부분만 모읍니다.
      </div>
    </div>
  );
}

function Node({
  n,
  depth,
  onAsk,
  path,
}: {
  n: MindNode;
  depth: number;
  onAsk: (q: string) => void;
  path: string[];
}) {
  const [open, setOpen] = useState(depth < 2);
  const has = n.children.length > 0;
  const here = [...path, n.label];
  return (
    <li className={`mm-node d${Math.min(depth, 3)}`}>
      <div className="mm-row">
        {has ? (
          <button className="mm-tog" aria-expanded={open} aria-label={open ? "접기" : "펼치기"} onClick={() => setOpen(!open)}>
            <ChevronRight />
          </button>
        ) : (
          <span className="mm-tog" />
        )}
        <button
          className="mm-label"
          data-src={n.citations.join(" ")}
          title="이 주제로 질문하기"
          onClick={() => onAsk(`${here.slice(-2).join(" › ")}에 대해 내 노트 기준으로 정리해줘`)}
        >
          {n.label}
        </button>
        {n.citations.slice(0, 3).map((c) => (
          <Cite key={c} n={c} />
        ))}
        {has && !open && <small className="mm-count">{n.children.length}</small>}
      </div>
      {has && open && (
        <ul>
          {n.children.map((k, i) => (
            <Node key={i} n={k} depth={depth + 1} onAsk={onAsk} path={here} />
          ))}
        </ul>
      )}
    </li>
  );
}

export function MindMap({ root, onAsk }: { root: MindNode; onAsk: (q: string) => void }) {
  return (
    <div className="mindmap fade">
      <ul className="mm-root">
        <Node n={root} depth={0} onAsk={onAsk} path={[]} />
      </ul>
    </div>
  );
}
