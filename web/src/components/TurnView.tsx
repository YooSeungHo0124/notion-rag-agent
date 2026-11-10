import {
  Check,
  ChevronDown,
  ChevronRight,
  CircleStop,
  CheckCircle2,
  CornerDownRight,
  Copy,
  FileText,
  Info,
  Maximize2,
  PanelRightOpen,
  RotateCcw,
  ShieldCheck,
  Star,
  XCircle,
  ArrowUpRight,
} from "lucide-react";
import { useEffect, useState } from "react";
import type { AgentRun, Answer, Turn } from "../types";
import AnswerBody from "./Answer";
import { MindMap } from "./Studio";

const CONF = {
  high: { dot: "c-high", txt: "근거 충분" },
  medium: { dot: "c-med", txt: "일부만 확인" },
  low: { dot: "c-low", txt: "근거 약함 · 출처를 직접 확인하세요" },
  none: { dot: "c-low", txt: "근거 없음" },
} as const;

export interface TurnActions {
  openSource: (turnId: string, n: number) => void;
  openClosest: (turnId: string) => void;
  followup: (q: string) => void;
  rephrase: (q: string) => void;
  widen: (q: string) => void;
  copy: (a: Answer) => void;
  toggleSave: (t: Turn) => void;
  isSaved: (t: Turn) => boolean;
  decide: (
    turnId: string,
    approved: boolean,
    args?: Record<string, unknown>,
  ) => void;
  retry: (t: Turn) => void;
  openNotion: (url: string, title: string) => void;
}

function Elapsed({ since }: { since: number }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 200);
    return () => clearInterval(t);
  }, []);
  return <span className="timer">{Math.floor((now - since) / 1000)}초</span>;
}

function Progress({ t }: { t: Turn }) {
  const st = t.stage?.name ?? "retrieving";
  const order = ["retrieving", "generating", "verifying"];
  const cur = st === "retrieved" ? 1 : order.indexOf(st);
  const retrieved = t.stage && cur >= 1;
  const titles = t.retrieved?.titles ?? [];
  const row = (i: number, run: ReactLabel, done: ReactLabel) => (
    <div className={`step ${i < cur ? "done" : i === cur ? "run" : ""}`}>
      <span className="ic">
        {i < cur ? <Check /> : <span className="ring" />}
      </span>
      {i < cur ? done : run}
    </div>
  );
  if (t.mode === "studio") {
    const sorder = ["collecting", "generating", "verifying"];
    const sc = st === "collected" ? 1 : Math.max(0, sorder.indexOf(st));
    const c = t.retrieved;
    const srow = (i: number, run: ReactLabel, done: ReactLabel) => (
      <div className={`step ${i < sc ? "done" : i === sc ? "run" : ""}`}>
        <span className="ic">
          {i < sc ? <Check /> : <span className="ring" />}
        </span>
        {i < sc ? done : run}
      </div>
    );
    return (
      <div className="steps" aria-live="polite">
        {srow(
          0,
          <span>범위의 노트 모으는 중…</span>,
          <span>
            노트 모으기 · {c?.pages_used ?? 0}페이지 {c?.chunks ?? 0}개 조각
            {c?.truncated
              ? ` (전체 ${c.pages_total}페이지 중 최근 수정 순)`
              : ""}
          </span>,
        )}
        {srow(
          1,
          <>
            <span>
              {t.studio?.title ?? "문서"} 작성 중 · 긴 문서는 1~2분 걸립니다
            </span>
            <Elapsed since={t.startedAt} />
          </>,
          <span>{t.studio?.title ?? "문서"} 작성</span>,
        )}
        {t.studio?.kind !== "mindmap" &&
          srow(
            2,
            <span>{sc === 2 ? "근거 검증 중…" : "근거 검증"}</span>,
            <span>근거 검증</span>,
          )}
      </div>
    );
  }
  if (t.mode === "agent")
    return (
      <div className="steps" aria-live="polite">
        <div className="step run">
          <span className="ic">
            <span className="ring" />
          </span>
          <span>에이전트가 도구를 고르고 실행하는 중…</span>
          <Elapsed since={t.startedAt} />
        </div>
      </div>
    );
  return (
    <>
      <div className="steps" aria-live="polite">
        {row(
          0,
          <span>노트 검색 중…</span>,
          <>
            <span>노트 검색 · {t.retrieved?.n ?? 0}개 조각</span>
            {retrieved && titles.length > 0 && (
              <span className="docs">
                ({titles.slice(0, 2).join(", ")}
                {titles.length > 2 ? " …" : ""})
              </span>
            )}
          </>,
        )}
        {row(
          1,
          <>
            <span>답변 작성 중</span>
            <Elapsed since={t.startedAt} />
          </>,
          <span>답변 작성</span>,
        )}
        {row(
          2,
          <span>{cur === 2 ? "근거 검증 중…" : "근거 검증"}</span>,
          <span>근거 검증</span>,
        )}
      </div>
      {t.draft?.length ? (
        <div
          className="draft"
          aria-busy="true"
          title="작성 중 · 근거 검증 전 초안입니다"
        >
          <AnswerBody a={draftAnswer(t)} />
        </div>
      ) : (
        <div className="answer">
          <div className="skel-box">
            <div className="skel" style={{ width: "92%" }} />
            <div className="skel" style={{ width: "64%" }} />
          </div>
        </div>
      )}
    </>
  );
}

const draftAnswer = (t: Turn): Answer => ({
  question: t.q,
  answerable: true,
  sentences: t.draft ?? [],
  sources: [],
  confidence: "none",
  retrieval_score: 0,
  supported_ratio: 0,
  dropped_citations: 0,
  latency_ms: 0,
});
type ReactLabel = React.ReactNode;

function Trust({ t, a, act }: { t: Turn; a: Answer; act: TurnActions }) {
  const [open, setOpen] = useState(false);
  const c = CONF[a.confidence] ?? CONF.low;
  const weak = a.sentences.filter(
    (s) => s.type !== "heading" && s.support !== "supported",
  ).length;
  const label =
    a.confidence === "medium" && weak
      ? `${c.txt} · 확인 필요 ${weak}문장`
      : c.txt;
  const saved = act.isSaved(t);
  return (
    <>
      <div className="trust fade">
        <button
          className="pill"
          aria-expanded={open}
          onClick={() => setOpen(!open)}
        >
          <span className={`c-dot ${c.dot}`} aria-hidden="true" />
          {label} · 출처 {a.sources.length}
          <ChevronDown />
        </button>
        {a.sources.length > 0 && (
          <button className="act" onClick={() => act.openSource(t.id, 1)}>
            <PanelRightOpen />
            출처 보기
          </button>
        )}
        <span className="sp" />
        <button className="act" onClick={() => act.copy(a)}>
          <Copy />
          복사
        </button>
        <button
          className="act"
          aria-pressed={saved}
          onClick={() => act.toggleSave(t)}
        >
          <Star />
          저장
        </button>
      </div>
      {open && (
        <div className="scores">
          <div className="h">
            <span>문장</span>
            <span>근거</span>
            <span>검색</span>
          </div>
          {a.sentences
            .filter((s) => s.type !== "heading")
            .map((s, i) => {
              const hit = s.citations[0]
                ? a.sources[s.citations[0] - 1]
                : undefined;
              return (
                <div key={i}>
                  <span className="t" title={s.text}>
                    {s.text.replace(/\n/g, " ")}
                  </span>
                  <span
                    style={{
                      color: s.support_score < 0.5 ? "var(--warn)" : "inherit",
                    }}
                  >
                    {s.support_score.toFixed(2)}
                  </span>
                  <span>{hit ? hit.rerank_score.toFixed(2) : "—"}</span>
                </div>
              );
            })}
        </div>
      )}
      {(a.followups?.length ?? 0) > 0 && (
        <div className="follow fade">
          <h4>이어서 물어보기</h4>
          {a.followups!.map((f) => (
            <button key={f} onClick={() => act.followup(f)}>
              <CornerDownRight />
              {f}
            </button>
          ))}
        </div>
      )}
    </>
  );
}

function Refusal({ t, a, act }: { t: Turn; a: Answer; act: TurnActions }) {
  return (
    <div className="refuse fade" role="status">
      <div className="head">
        <Info />내 노트에서 근거를 찾지 못했습니다
      </div>
      {a.closest ? (
        <p>
          가장 가까운 기록: <b>“{a.closest.title}”</b> (관련도 낮음 ·{" "}
          {a.closest.rerank_score.toFixed(2)})
        </p>
      ) : (
        <p>관련 있는 기록이 없습니다.</p>
      )}
      <div className="acts">
        {a.closest && (
          <button className="btn" onClick={() => act.openClosest(t.id)}>
            <FileText />이 페이지 열기
          </button>
        )}
        <button className="btn" onClick={() => act.rephrase(t.q)}>
          <RotateCcw />
          다른 표현으로 다시 묻기
        </button>
        <button className="btn" onClick={() => act.widen(t.q)}>
          <Maximize2 />
          범위 넓히기
        </button>
      </div>
    </div>
  );
}

const TOOL_LABEL: Record<string, string> = {
  search_notes: "노트 검색",
  get_page: "페이지 읽기",
  get_timeline: "최근 수정 목록",
  append_to_page: "페이지 끝에 추가",
  create_page: "새 하위 페이지 생성",
  remember: "기억하기",
};

function Trace({ run }: { run: AgentRun }) {
  if (!run.steps.length) return null;
  return (
    <details className="trace">
      <summary>
        <ChevronRight />
        실행 과정 {run.steps.length}단계 (
        {run.steps.map((s) => TOOL_LABEL[s.tool] ?? s.tool).join(" → ")})
      </summary>
      <ol>
        {run.steps.map((s, i) => (
          <li key={i}>
            {TOOL_LABEL[s.tool] ?? s.tool}:{" "}
            {JSON.stringify(s.args).slice(0, 120)}
            {!s.ok && (
              <span style={{ color: "var(--bad)" }}>
                {" "}
                — {String(s.observation).slice(0, 80)}
              </span>
            )}
          </li>
        ))}
      </ol>
    </details>
  );
}

function Approval({
  t,
  run,
  act,
}: {
  t: Turn;
  run: AgentRun;
  act: TurnActions;
}) {
  const ap = run.approval!;
  const orig = String(ap.args.text ?? "");
  const [text, setText] = useState(orig);
  const edited = text !== orig;
  const target =
    ap.tool === "create_page"
      ? `새 페이지 “${String(ap.args.title ?? "")}”`
      : `페이지 ${String(ap.args.page_id ?? "").slice(0, 8)}…`;
  return (
    <>
      <div className="approve fade" role="group" aria-label="노션 기록 승인">
        <div className="hd">
          <ShieldCheck />
          노션에 기록하기 전에 확인해 주세요
        </div>
        <dl>
          <dt>대상</dt>
          <dd>
            <span className="tgt">
              <FileText />
              {target} <small>({TOOL_LABEL[ap.tool] ?? ap.tool})</small>
            </span>
          </dd>
          <dt>내용</dt>
          <dd>
            <textarea
              aria-label="추가할 내용"
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
          </dd>
          {ap.thought && (
            <>
              <dt>이유</dt>
              <dd style={{ color: "var(--text-muted)" }}>{ap.thought}</dd>
            </>
          )}
        </dl>
        <div className="ft">
          {edited && <span className="edited">내용을 수정했습니다</span>}
          <button className="btn" onClick={() => act.decide(t.id, false)}>
            거부
          </button>
          <button
            className="btn primary"
            onClick={() =>
              act.decide(t.id, true, edited ? { ...ap.args, text } : undefined)
            }
          >
            {edited ? "수정 후 실행" : "실행"}
          </button>
        </div>
      </div>
      <Trace run={run} />
    </>
  );
}

function ActionResult({ run, act }: { run: AgentRun; act: TurnActions }) {
  const a = run.answer;
  if (!a || a.kind !== "action") return null;
  return (
    <>
      {a.steps.map((s, i) => {
        const obs = (s.observation ?? {}) as { url?: string };
        return (
          <div key={i} className="result fade" style={{ marginTop: i ? 8 : 0 }}>
            {s.ok ? <CheckCircle2 className="ok" /> : <XCircle />}
            <span>
              {s.ok ? "완료" : "실행하지 않음"} · {TOOL_LABEL[s.tool] ?? s.tool}
              {!s.ok && ` — ${String(s.observation)}`}
            </span>
            {s.ok && obs.url && (
              <a
                href="#"
                onClick={(e) => (
                  e.preventDefault(),
                  act.openNotion(obs.url!, "새 페이지")
                )}
              >
                노션에서 보기
                <ArrowUpRight />
              </a>
            )}
          </div>
        );
      })}
      <Trace run={run} />
    </>
  );
}

function StudioLine({ t }: { t: Turn }) {
  const cv = t.studio?.coverage;
  return (
    <div className="summary-line fade">
      <Check />
      {t.doneMs !== undefined ? `${(t.doneMs / 1000).toFixed(0)}초 · ` : ""}
      {cv ? `${cv.pages_used}페이지 ${cv.chunks_used}개 조각을 읽음` : ""}
      {cv?.truncated ? ` (전체 ${cv.pages_total}페이지 중 최근 수정 순)` : ""}
      {t.studio?.topic ? ` · 주제: ${t.studio.topic}` : ""}
      {t.studio?.kind !== "mindmap" ? " · 근거 검증 완료" : ""}
    </div>
  );
}

export default function TurnView({
  t,
  act,
  generating,
}: {
  t: Turn;
  act: TurnActions;
  generating: boolean;
}) {
  const a = t.answer;
  return (
    <article className="turn" data-turn={t.id}>
      <h1 className="q-title">{t.q}</h1>
      <div className="q-rule" />
      {t.error && <div className="err">오류: {t.error}</div>}
      {t.stopped && !a && (
        <div className="stopped fade">
          <CircleStop />
          생성을 중단했습니다.
          <button
            className="btn ghost"
            style={{ marginLeft: "auto" }}
            onClick={() => act.retry(t)}
          >
            다시 생성
          </button>
        </div>
      )}
      {generating &&
        !a &&
        !t.error &&
        !t.stopped &&
        t.run?.status !== "awaiting_approval" && <Progress t={t} />}
      {t.run?.status === "awaiting_approval" && (
        <Approval t={t} run={t.run} act={act} />
      )}
      {t.run?.status === "done" && t.run.answer?.kind === "action" && (
        <ActionResult run={t.run} act={act} />
      )}
      {a && !a.answerable && t.mode !== "studio" && (
        <Refusal t={t} a={a} act={act} />
      )}
      {a && !a.answerable && t.mode === "studio" && (
        <div className="refuse fade" role="status">
          <div className="head">
            <Info />이 범위에서 문서를 만들 만한 내용을 찾지 못했습니다
          </div>
          <p>범위를 넓히거나 주제를 바꿔 다시 시도해 보세요.</p>
        </div>
      )}
      {a && t.mindmap && (
        <>
          <StudioLine t={t} />
          <MindMap root={t.mindmap} onAsk={act.followup} />
          <div className="trust fade">
            <span className="pill" style={{ cursor: "default" }}>
              <span className="c-dot c-med" aria-hidden="true" />
              구조 요약 · 문장 검증 대상 아님 · 출처 {a.sources.length}
            </span>
            {a.sources.length > 0 && (
              <button className="act" onClick={() => act.openSource(t.id, 1)}>
                <PanelRightOpen />
                출처 보기
              </button>
            )}
            <span className="sp" />
            <button
              className="act"
              aria-pressed={act.isSaved(t)}
              onClick={() => act.toggleSave(t)}
            >
              <Star />
              저장
            </button>
          </div>
        </>
      )}
      {a && a.answerable && !t.mindmap && (
        <>
          {t.mode === "studio" && <StudioLine t={t} />}
          {t.doneMs !== undefined && t.mode !== "studio" && (
            <div className="summary-line fade">
              <Check />
              {t.mode === "agent" && t.run
                ? `에이전트 ${t.run.steps.length}단계 · `
                : ""}
              {(t.doneMs / 1000).toFixed(1)}초 · 근거 검증 완료
            </div>
          )}
          <AnswerBody a={a} images={t.images} />
          {t.run && <Trace run={t.run} />}
          <Trust t={t} a={a} act={act} />
        </>
      )}
    </article>
  );
}
