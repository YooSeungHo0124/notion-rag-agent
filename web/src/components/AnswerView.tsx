import type { Answer, Confidence, SourceHit } from "../types";
import { fmtDate } from "../util";

const BADGE: Record<Confidence, { label: string; dot: string; cls: string }> = {
  high: { label: "높음", dot: "●", cls: "bg-emerald-50 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300" },
  medium: { label: "보통", dot: "◐", cls: "bg-amber-50 text-amber-700 dark:bg-amber-950 dark:text-amber-300" },
  low: { label: "낮음", dot: "○", cls: "bg-rose-50 text-rose-700 dark:bg-rose-950 dark:text-rose-300" },
  none: { label: "–", dot: "–", cls: "bg-stone-100 text-stone-500 dark:bg-stone-800" },
};

export const DISCLAIMER = "신뢰도는 '내 노트에 근거하는가'를 뜻하며 사실 여부를 보장하지 않습니다.";

function where(h: SourceHit): string {
  return [h.title, ...h.heading_path].join(" › ");
}

function CitationChip({ n, hit }: { n: number; hit?: SourceHit }) {
  if (!hit) return null;
  return (
    <span className="group relative inline-block align-baseline">
      <a
        href={hit.cite_url}
        target="_blank"
        rel="noreferrer noopener"
        className="mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded-full bg-indigo-50 px-1.5 text-[11px] font-semibold text-indigo-700 no-underline hover:bg-indigo-100 focus:outline-none focus:ring-2 focus:ring-indigo-400 dark:bg-indigo-950 dark:text-indigo-300 dark:hover:bg-indigo-900"
        aria-label={`출처 ${n}: ${where(hit)}`}
      >
        {n}
      </a>
      <span
        role="tooltip"
        className="pointer-events-none invisible absolute bottom-full left-1/2 z-30 mb-2 w-80 max-w-[80vw] -translate-x-1/2 rounded-xl border border-stone-200 bg-white p-3 text-left text-xs font-normal text-stone-700 opacity-0 shadow-xl transition group-focus-within:visible group-focus-within:opacity-100 group-hover:visible group-hover:opacity-100 dark:border-stone-700 dark:bg-stone-800 dark:text-stone-200"
      >
        <span className="block font-semibold text-stone-900 dark:text-stone-50">{where(hit)}</span>
        <span className="mt-0.5 block text-stone-500 dark:text-stone-400">
          {fmtDate(hit.last_edited_time)} · 관련도 {hit.rerank_score.toFixed(2)}
        </span>
        <span className="mt-2 line-clamp-6 block whitespace-pre-wrap leading-relaxed">{hit.text}</span>
        <span className="mt-2 block text-indigo-600 dark:text-indigo-300">클릭하면 노션 원문으로 이동 ↗</span>
      </span>
    </span>
  );
}

export function Abstained({ score }: { score: number }) {
  return (
    <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm dark:border-amber-900 dark:bg-amber-950/40">
      <p className="font-medium text-amber-800 dark:text-amber-200">기록에서 근거를 찾지 못했습니다.</p>
      <p className="mt-1 text-amber-700/80 dark:text-amber-300/80">
        다른 키워드로 바꿔 묻거나, 왼쪽 소스 패널에서 검색 범위를 넓혀 보세요.
      </p>
      <p className="mt-2 text-xs text-stone-500">최고 검색 점수 {score.toFixed(2)}</p>
    </div>
  );
}

export function answerPlainText(a: Answer): string {
  return a.sentences.map((s) => s.text + s.citations.map((n) => `[${n}]`).join("")).join(" ");
}

export default function AnswerView({ a }: { a: Answer }) {
  if (!a.answerable) return <Abstained score={a.retrieval_score} />;
  const nSup = a.sentences.filter((s) => s.support === "supported").length;
  const badge = BADGE[a.confidence];
  return (
    <div className="space-y-4">
      <p className="text-[15px] leading-8">
        {a.sentences.map((s, i) => (
          <span key={i}>
            <span
              className={s.support === "unsupported" ? "unsupported" : s.support === "partial" ? "partial" : ""}
              title={
                s.support === "unsupported"
                  ? "기록에서 확인되지 않음"
                  : s.support === "partial"
                    ? "기록에서 일부만 확인됨"
                    : undefined
              }
            >
              {s.text}
            </span>
            {s.citations.map((n) => (
              <CitationChip key={n} n={n} hit={a.sources[n - 1]} />
            ))}{" "}
          </span>
        ))}
      </p>

      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className={`rounded-full px-2.5 py-1 font-medium ${badge.cls}`}>
          {badge.dot} 신뢰도 {badge.label}
        </span>
        <span className="text-stone-500 dark:text-stone-400">
          근거 문장 {nSup}/{a.sentences.length} · 검색 점수 {a.retrieval_score.toFixed(2)} ·{" "}
          {((a.total_ms ?? a.latency_ms) / 1000).toFixed(1)}초
          {a.dropped_citations > 0 && ` · 무효 인용 ${a.dropped_citations}건 제거`}
        </span>
      </div>

      {a.sources.length > 0 && (
        <div>
          <h4 className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-stone-500">출처</h4>
          <ol className="space-y-1">
            {a.sources.map((h, i) => (
              <li key={h.chunk_id}>
                <a
                  href={h.cite_url}
                  target="_blank"
                  rel="noreferrer noopener"
                  className="flex items-baseline gap-2 rounded-lg px-2 py-1 text-sm hover:bg-stone-100 dark:hover:bg-stone-800"
                >
                  <span className="text-xs font-semibold text-indigo-600 dark:text-indigo-300">[{i + 1}]</span>
                  <span className="min-w-0 flex-1 truncate">{where(h)}</span>
                  <span className="shrink-0 text-xs text-stone-400">
                    {fmtDate(h.last_edited_time)} · {h.rerank_score.toFixed(2)} ↗
                  </span>
                </a>
              </li>
            ))}
          </ol>
        </div>
      )}
      <p className="text-[11px] text-stone-400">{DISCLAIMER}</p>
    </div>
  );
}
