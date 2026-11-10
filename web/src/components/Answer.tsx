import { Copy, GitCompare, Info, Lightbulb, Table2, BarChart3, ArrowRight, TriangleAlert } from "lucide-react";
import type { ReactNode } from "react";
import type { Answer, AnswerImage, Sentence } from "../types";

/* Generative UI: each typed block from the LLM maps to one component.
   `.s[data-src]` + `.cite` wire the answer to the source panel (linked highlight). */

export function Cite({ n }: { n: number }) {
  return (
    <button className="cite" data-c={n} aria-label={`출처 ${n}`}>
      {n}
    </button>
  );
}

const Cites = ({ s }: { s: Sentence }) => (
  <>
    {s.citations.map((n) => (
      <Cite key={n} n={n} />
    ))}
  </>
);

/** Safe inline markdown: **bold** and `code` (values are double-click-to-copy). */
export function Inline({ text }: { text: string }) {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
  return (
    <>
      {parts.map((p, i) =>
        p.startsWith("**") && p.endsWith("**") && p.length > 4 ? (
          <b key={i}>{p.slice(2, -2)}</b>
        ) : p.startsWith("`") && p.endsWith("`") && p.length > 2 ? (
          <code key={i} className="v">
            {p.slice(1, -1)}
          </code>
        ) : (
          <span key={i}>{p}</span>
        ),
      )}
    </>
  );
}

function weakTitle(s: Sentence): string | undefined {
  if (s.note) return `근거 약함: ${s.note}`;
  if (s.support === "unsupported") return "근거 약함: 인용한 원문에서 확인되지 않습니다";
  if (s.support === "partial") return "일부만 확인: 원문과 대조해 보세요";
  return undefined;
}

const sCls = (s: Sentence, extra = "") =>
  `s${s.support !== "supported" ? " weak" : ""}${extra ? " " + extra : ""}`;
const src = (s: Sentence) => s.citations.join(" ");
const isNum = (v: string) => /^[-+~≈]?\s*\d[\d,.]*\s*(%|ms|fps|대|초|GB|MB)?$/.test(v.trim());

function Label({ icon, text, s }: { icon: ReactNode; text: string; s: Sentence }) {
  return (
    <div className="blabel">
      {icon}
      {text}
      <span className="r">
        <Cites s={s} />
      </span>
    </div>
  );
}

function Block({ s }: { s: Sentence }) {
  const d = s.data ?? {};
  switch (s.type) {
    case "summary":
      return (
        <div className={sCls(s, "conclusion")} data-src={src(s)} title={weakTitle(s)}>
          <Inline text={s.text} />
          <Cites s={s} />
        </div>
      );
    case "heading":
      return <h3>{s.text}</h3>;
    case "callout": {
      const lvl = d.level ?? "note";
      if (lvl === "note")
        return (
          <div className="callout info">
            <Info />
            <span className={sCls(s)} data-src={src(s)} title={weakTitle(s)}>
              <Inline text={s.text} />
              <Cites s={s} />
            </span>
          </div>
        );
      return (
        <div className="callout mismatch">
          {lvl === "conflict" ? <GitCompare /> : <TriangleAlert />}
          <div className={sCls(s)} data-src={src(s)} title={weakTitle(s)}>
            <span className="lab">{lvl === "conflict" ? "기록 불일치" : "주의"}</span>
            <span className="row" style={{ color: "var(--text)" }}>
              <Inline text={s.text} />
              <Cites s={s} />
            </span>
          </div>
        </div>
      );
    }
    case "table": {
      const cols = d.columns ?? [];
      const rows = d.rows ?? [];
      const numCol = cols.map((_, j) => rows.length > 0 && rows.every((r) => !r[j] || isNum(r[j])));
      return (
        <div className="wide">
          <Label icon={<Table2 />} text="표" s={s} />
          <div className="table-scroll">
            <table className={sCls(s)} data-src={src(s)} title={weakTitle(s)}>
              <thead>
                <tr>
                  {cols.map((c, j) => (
                    <th key={j} className={numCol[j] ? "n" : undefined}>
                      {c}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => (
                  <tr key={i}>
                    {r.map((c, j) => (
                      <td key={j} className={numCol[j] ? "n" : undefined}>
                        <Inline text={c} />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      );
    }
    case "steps":
      return (
        <div>
          <Label icon={<ArrowRight />} text="단계" s={s} />
          <ol className={sCls(s, "steps-list")} data-src={src(s)} title={weakTitle(s)}>
            {(d.items ?? []).map((it, i) => (
              <li key={i}>
                <span>
                  <Inline text={it} />
                </span>
              </li>
            ))}
          </ol>
        </div>
      );
    case "code": {
      const code = d.code ?? s.text;
      return (
        <div className="codeblk">
          <Label icon={<Copy />} text={d.lang || "code"} s={s} />
          <button
            className="btn ghost copy"
            data-copy={code}
            aria-label="코드 복사"
            style={{ height: 26, padding: "0 8px" }}
          >
            복사
          </button>
          <pre className={sCls(s)} data-src={src(s)} title={weakTitle(s)}>
            {code}
          </pre>
        </div>
      );
    }
    case "chart": {
      const labels = d.labels ?? [];
      const values = d.values ?? [];
      const max = Math.max(...values.map((v) => Math.abs(v)), 1e-9);
      return (
        <div>
          <Label icon={<BarChart3 />} text={`차트${d.unit ? ` · ${d.unit}` : ""}`} s={s} />
          <div className={sCls(s, "bars")} data-src={src(s)} title={weakTitle(s)}>
            {labels.map((l, i) => (
              <Bar key={i} label={l} v={values[i]} max={max} unit={d.unit ?? ""} />
            ))}
          </div>
        </div>
      );
    }
    case "flow": {
      const nodes = d.nodes ?? [];
      const notes = d.notes ?? [];
      return (
        <div className={sCls(s, "flow")} data-src={src(s)} title={weakTitle(s)}>
          {nodes.map((n, i) => (
            <FlowNode key={i} name={n} note={notes[i]} last={i === nodes.length - 1} />
          ))}
        </div>
      );
    }
    default:
      return null;
  }
}

const Bar = ({ label, v, max, unit }: { label: string; v: number; max: number; unit: string }) => (
  <>
    <span className="lbl2">{label}</span>
    <div className="bar" style={{ width: `${Math.max((Math.abs(v) / max) * 100, 2)}%` }} />
    <span className="val">
      {v}
      {unit}
    </span>
  </>
);

const FlowNode = ({ name, note, last }: { name: string; note?: string; last: boolean }) => (
  <>
    <div className="node">
      <b>{name}</b>
      {note && <small>{note}</small>}
    </div>
    {!last && (
      <div className="arrow">
        <ArrowRight />
      </div>
    )}
  </>
);

function Figure({ img, a }: { img: AnswerImage; a: Answer }) {
  const hit = a.sources[img.source - 1];
  const where = [hit?.title, img.heading].filter(Boolean).join(" › ");
  return (
    <figure className="img-blk fade" data-img-src={img.source} title="클릭하면 출처 패널에서 원문 위치를 엽니다">
      <img src={img.src} alt={img.caption || img.context.slice(0, 80)} loading="lazy" />
      <figcaption>
        {img.caption && <span className="cap">{img.caption}</span>}
        <span className="from">
          출처 {img.source} · {where}
        </span>
      </figcaption>
    </figure>
  );
}

export default function AnswerBody({ a, images = [] }: { a: Answer; images?: AnswerImage[] }) {
  const out: ReactNode[] = [];
  let run: Sentence[] = [];
  const flush = () => {
    if (!run.length) return;
    const items = run;
    out.push(
      <p key={`p${out.length}`}>
        {items.map((s, i) => (
          <span key={i}>
            <span className={sCls(s)} data-src={src(s)} title={weakTitle(s)}>
              <Inline text={s.text} />
              <Cites s={s} />
            </span>{" "}
          </span>
        ))}
      </p>,
    );
    run = [];
  };
  a.sentences.forEach((s, j) => {
    // a text block that opens with a bold topic ("**주제** — ...") or "Q." starts its own line
    if ((!s.type || s.type === "text") && /^(\*\*|Q\.\s)/.test(s.text)) flush();
    if (!s.type || s.type === "text") run.push(s);
    else {
      flush();
      out.push(<Block key={`b${out.length}`} s={s} />);
    }
    const imgs = images.filter((im) => im.after_block === j);
    if (imgs.length) {
      flush();
      imgs.forEach((im) => out.push(<Figure key={`i${im.block_id}`} img={im} a={a} />));
    }
  });
  flush();
  return <div className="answer fade">{out}</div>;
}

export function answerText(a: Answer): string {
  return a.sentences
    .map((s) => (s.type === "heading" ? `\n## ${s.text}` : s.text) + s.citations.map((n) => `[${n}]`).join(""))
    .join("\n");
}

export { Lightbulb };
