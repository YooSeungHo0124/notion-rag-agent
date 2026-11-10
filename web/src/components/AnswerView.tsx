import { useState } from "react";
import type { Answer, Confidence, Sentence, SourceHit } from "../types";
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

// ---------------------------------------------------------------------------
// Generative UI: the LLM returns typed blocks (Structured Outputs); each type maps to a
// component here. Unknown types fall back to plain text, so a schema change never crashes.
// ---------------------------------------------------------------------------

type BlockProps = { s: Sentence; a: Answer };

function supportCls(s: Sentence): string {
  return s.support === "unsupported" ? "unsupported" : s.support === "partial" ? "partial" : "";
}

function supportTitle(s: Sentence): string | undefined {
  if (s.note) return s.note;
  if (s.support === "unsupported") return "기록에서 확인되지 않음";
  if (s.support === "partial") return "기록에서 일부만 확인됨";
  return undefined;
}

function Cites({ s, a }: BlockProps) {
  return (
    <>
      {s.citations.map((n) => (
        <CitationChip key={n} n={n} hit={a.sources[n - 1]} />
      ))}
    </>
  );
}

function BlockFrame({ s, a, label, children }: BlockProps & { label: string; children: React.ReactNode }) {
  const warn = s.support !== "supported";
  return (
    <figure
      className={`overflow-hidden rounded-xl border ${warn ? "border-amber-300 dark:border-amber-800" : "border-stone-200 dark:border-stone-700"}`}
    >
      <figcaption className="flex items-center gap-2 border-b border-stone-200 bg-stone-50 px-3 py-1.5 text-[11px] text-stone-500 dark:border-stone-700 dark:bg-stone-900">
        <span className="font-medium uppercase tracking-wide">{label}</span>
        {warn && <span className="text-amber-700 dark:text-amber-300">⚠ {supportTitle(s)}</span>}
        <span className="ml-auto">
          <Cites s={s} a={a} />
        </span>
      </figcaption>
      <div className="p-3">{children}</div>
    </figure>
  );
}

function TableBlock({ s, a }: BlockProps) {
  const { columns = [], rows = [] } = s.data ?? {};
  return (
    <BlockFrame s={s} a={a} label="표">
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead>
            <tr>
              {columns.map((c, i) => (
                <th key={i} className="border-b border-stone-200 px-2 py-1.5 font-semibold dark:border-stone-700">
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} className="odd:bg-stone-50/60 dark:odd:bg-stone-800/40">
                {r.map((c, j) => (
                  <td key={j} className="px-2 py-1.5 align-top">
                    {c}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </BlockFrame>
  );
}

function StepsBlock({ s, a }: BlockProps) {
  return (
    <BlockFrame s={s} a={a} label="단계">
      <ol className="space-y-2">
        {(s.data?.items ?? []).map((it, i) => (
          <li key={i} className="flex gap-3 text-sm">
            <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-indigo-600 text-xs font-semibold text-white">
              {i + 1}
            </span>
            <span className="pt-0.5">{it}</span>
          </li>
        ))}
      </ol>
    </BlockFrame>
  );
}

function CodeBlock({ s, a }: BlockProps) {
  const [copied, setCopied] = useState(false);
  const code = s.data?.code ?? s.text;
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard blocked */
    }
  };
  return (
    <BlockFrame s={s} a={a} label={s.data?.lang || "code"}>
      <div className="relative">
        <button
          onClick={copy}
          className="absolute right-0 top-0 rounded-md border border-stone-300 bg-white px-2 py-0.5 text-[11px] hover:bg-stone-100 dark:border-stone-600 dark:bg-stone-800"
        >
          {copied ? "복사됨" : "복사"}
        </button>
        <pre className="overflow-x-auto pr-14 font-mono text-[13px] leading-6">
          <code>{code}</code>
        </pre>
      </div>
    </BlockFrame>
  );
}

function ChartBlock({ s, a }: BlockProps) {
  const { labels = [], values = [], unit = "" } = s.data ?? {};
  const max = Math.max(...values.map((v) => Math.abs(v)), 1e-9);
  return (
    <BlockFrame s={s} a={a} label="차트">
      <div className="space-y-1.5" role="img" aria-label={s.text}>
        {labels.map((l, i) => (
          <div key={i} className="grid grid-cols-[minmax(4rem,9rem)_1fr_auto] items-center gap-2 text-sm">
            <span className="truncate text-stone-600 dark:text-stone-300" title={l}>
              {l}
            </span>
            <div className="h-5 rounded bg-stone-100 dark:bg-stone-800">
              <div
                className="h-5 rounded bg-indigo-500"
                style={{ width: `${Math.max((Math.abs(values[i]) / max) * 100, 2)}%` }}
              />
            </div>
            <span className="tabular-nums text-stone-700 dark:text-stone-200">
              {values[i]}
              {unit}
            </span>
          </div>
        ))}
      </div>
    </BlockFrame>
  );
}

const REGISTRY: Record<string, (p: BlockProps) => React.ReactElement> = {
  table: TableBlock,
  steps: StepsBlock,
  code: CodeBlock,
  chart: ChartBlock,
};

function TextRun({ items, a }: { items: Sentence[]; a: Answer }) {
  return (
    <p className="text-[15px] leading-8">
      {items.map((s, i) => (
        <span key={i}>
          <span className={supportCls(s)} title={supportTitle(s)}>
            {s.text}
          </span>
          <Cites s={s} a={a} />{" "}
        </span>
      ))}
    </p>
  );
}

function Blocks({ a }: { a: Answer }) {
  // consecutive text blocks flow as one paragraph; other types render their component
  const out: React.ReactElement[] = [];
  let run: Sentence[] = [];
  const flush = () => {
    if (run.length) out.push(<TextRun key={`t${out.length}`} items={run} a={a} />);
    run = [];
  };
  for (const s of a.sentences) {
    const C = s.type ? REGISTRY[s.type] : undefined;
    if (!C) {
      run.push(s);
      continue;
    }
    flush();
    out.push(<C key={`b${out.length}`} s={s} a={a} />);
  }
  flush();
  return <div className="space-y-3">{out}</div>;
}

export default function AnswerView({ a }: { a: Answer }) {
  if (!a.answerable) return <Abstained score={a.retrieval_score} />;
  const nSup = a.sentences.filter((s) => s.support === "supported").length;
  const badge = BADGE[a.confidence];
  return (
    <div className="space-y-4">
      <Blocks a={a} />

      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className={`rounded-full px-2.5 py-1 font-medium ${badge.cls}`}>
          {badge.dot} 신뢰도 {badge.label}
        </span>
        <span className="text-stone-500 dark:text-stone-400">
          근거 확인 {nSup}/{a.sentences.length} · 검색 점수 {a.retrieval_score.toFixed(2)} ·{" "}
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
