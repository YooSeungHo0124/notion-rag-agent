"""Evaluate retrieval and answers on eval/dataset.jsonl.

Retrieval (no LLM): rank of the gold chunk/page before and after reranking
  -> Recall@1/5/10, MRR at chunk and page level.
Answers (Claude): GroundedQA on every question, then an LLM judge compares each answer
  with the reference -> correct / partial / wrong; abstention checked on unanswerables.

usage: uv run python -m eval.run_eval [--no-answers] [--tag baseline]
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

from eval.common import DATASET, EVAL_DIR, RUNS, read_jsonl
from notion_agent.llm import get_llm
from notion_agent.observability.tracing import flush, setup_tracing
from notion_agent.qa.grounded import GroundedQA
from notion_agent.rag.factory import build_retriever, build_verifier

JUDGE_SYSTEM = """너는 RAG 답변 채점자다. 각 항목의 질문, 참조 답, 시스템 답을 비교한다.
- correct: 참조 답의 핵심을 정확히 담음 (표현 차이 무관, 추가 정보는 틀리지 않으면 허용)
- partial: 일부만 맞거나 핵심 일부 누락
- wrong: 틀리거나 무관하거나, 답할 수 있는데 '기록 없음'으로 회피함
간단한 이유를 한 문장으로 쓴다."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"grades": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "qid": {"type": "string"},
            "grade": {"type": "string", "enum": ["correct", "partial", "wrong"]},
            "reason": {"type": "string"},
        },
        "required": ["qid", "grade", "reason"], "additionalProperties": False,
    }}},
    "required": ["grades"], "additionalProperties": False,
}


def rank_of(ids: list[str], gold: str | None) -> int | None:
    return ids.index(gold) + 1 if gold in ids else None


def recall(ranks: list[int | None], k: int) -> float:
    return sum(1 for r in ranks if r and r <= k) / len(ranks) if ranks else 0.0


def mrr(ranks: list[int | None]) -> float:
    return sum(1 / r for r in ranks if r) / len(ranks) if ranks else 0.0


def eval_retrieval(retriever, rows: list[dict]) -> tuple[dict, list[dict]]:
    per = []
    for r in rows:
        if not r["answerable"]:
            continue
        q = r["question"]
        cands = retriever.store.hybrid_search(
            retriever.dense.embed_query(q), retriever.sparse.embed_query(q),
            retriever.cfg.candidate_k, None,
        )
        fused_chunks = [str(c.payload.get("chunk_id", c.id)) for c in cands]
        fused_pages = list(dict.fromkeys(c.payload["page_id"] for c in cands))
        hits = retriever.retrieve(q, k=10)
        rr_chunks = [h.chunk_id for h in hits]
        rr_pages = list(dict.fromkeys(h.page_id for h in hits))
        per.append({
            "qid": r["qid"],
            "fused_chunk_rank": rank_of(fused_chunks, r["gold_chunk_id"]),
            "fused_page_rank": rank_of(fused_pages, r["gold_page_id"]),
            "chunk_rank": rank_of(rr_chunks, r["gold_chunk_id"]),
            "page_rank": rank_of(rr_pages, r["gold_page_id"]),
            "top_title": hits[0].page_title if hits else None,
            "top_score": hits[0].rerank_score if hits else 0.0,
        })
    m = {}
    for key in ("fused_chunk_rank", "fused_page_rank", "chunk_rank", "page_rank"):
        ranks = [p[key] for p in per]
        m[key.replace("_rank", "")] = {
            "R@1": recall(ranks, 1), "R@5": recall(ranks, 5), "R@10": recall(ranks, 10),
            "MRR": mrr(ranks),
        }
    return m, per


class AgentAsQA:
    """Adapter: run the LangGraph agent and expose its final answer like GroundedQA.answer."""

    def __init__(self, agent) -> None:
        self.agent = agent
        self.steps: list[int] = []

    def answer(self, question: str):
        import uuid
        from types import SimpleNamespace as NS

        r = self.agent.run(question, uuid.uuid4().hex)
        self.steps.append(len(r["steps"]))
        a = r["answer"] or {}
        if a.get("kind") != "answer":
            a = {"answerable": False, "sentences": [], "confidence": "none",
                 "supported_ratio": 0.0, "retrieval_score": 0.0}
        return NS(answerable=a["answerable"], confidence=a["confidence"],
                  supported_ratio=a["supported_ratio"], retrieval_score=a["retrieval_score"],
                  sentences=[NS(text=s["text"]) for s in a["sentences"]])


def eval_answers(qa: GroundedQA | AgentAsQA, rows: list[dict]) -> tuple[dict, list[dict]]:
    per = []
    for r in rows:
        t = time.time()
        a = qa.answer(r["question"])
        per.append({
            "qid": r["qid"], "answerable_gold": r["answerable"], "abstained": not a.answerable,
            "answer": " ".join(s.text for s in a.sentences), "confidence": a.confidence,
            "supported_ratio": a.supported_ratio, "retrieval_score": a.retrieval_score,
            "latency_s": round(time.time() - t, 2),
        })
        print(f"  {r['qid']} {'ABSTAIN' if not a.answerable else a.confidence}")

    judge_items = [
        (r, p) for r, p in zip([x for x in rows], per, strict=True)
        if r["answerable"] and not p["abstained"]
    ]
    grades: dict[str, dict] = {}
    llm = get_llm()
    for i in range(0, len(judge_items), 10):
        batch = judge_items[i : i + 10]
        body = "\n\n".join(
            f"[{r['qid']}]\n질문: {r['question']}\n참조 답: {r['reference']}\n시스템 답: {p['answer']}"
            for r, p in batch
        )
        out = llm.complete(body, system=JUDGE_SYSTEM, json_schema=JUDGE_SCHEMA).structured or {}
        grades.update({g["qid"]: g for g in out.get("grades", [])})
    for p in per:
        g = grades.get(p["qid"])
        if p["answerable_gold"] and p["abstained"]:
            p["grade"], p["reason"] = "wrong", "answerable but abstained"
        elif not p["answerable_gold"]:
            p["grade"] = "correct" if p["abstained"] else "wrong"
            p["reason"] = "abstention check"
        else:
            p["grade"], p["reason"] = (g["grade"], g["reason"]) if g else ("wrong", "not graded")

    ans = [p for p in per if p["answerable_gold"]]
    una = [p for p in per if not p["answerable_gold"]]
    answered = [p for p in ans if not p["abstained"]]
    m = {
        "answer_accuracy": sum(p["grade"] == "correct" for p in ans) / max(len(ans), 1),
        "answer_correct_or_partial": sum(p["grade"] != "wrong" for p in ans) / max(len(ans), 1),
        "false_abstain_rate": sum(p["abstained"] for p in ans) / max(len(ans), 1),
        "abstain_accuracy_unanswerable": sum(p["abstained"] for p in una) / max(len(una), 1),
        "mean_supported_ratio": sum(p["supported_ratio"] for p in answered) / max(len(answered), 1),
        "mean_latency_s": sum(p["latency_s"] for p in per) / max(len(per), 1),
    }
    if isinstance(qa, AgentAsQA):
        m["mean_agent_steps"] = sum(qa.steps) / max(len(qa.steps), 1)
    return m, per


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-answers", action="store_true")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--agent", action="store_true", help="answer via the LangGraph agent")
    args = ap.parse_args()
    setup_tracing(f"eval-{args.tag}")

    rows = [r for r in read_jsonl(DATASET) if not r.get("exclude")]
    retriever = build_retriever()
    report: dict = {"tag": args.tag, "n": len(rows), "config": retriever.cfg.model_dump()}
    report["retrieval"], ret_per = eval_retrieval(retriever, rows)
    ans_per: list[dict] = []
    if not args.no_answers:
        qa = GroundedQA(retriever, get_llm(), build_verifier())
        if args.agent:
            from notion_agent.agent.factory import build_agent

            qa = AgentAsQA(build_agent(qa=qa))
        report["answers"], ans_per = eval_answers(qa, rows)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"{stamp}_{args.tag}.json").write_text(json.dumps(
        {"report": report, "retrieval": ret_per, "answers": ans_per},
        ensure_ascii=False, indent=2, default=str,
    ))

    # metrics only (no note content) -> safe to commit
    lines = [f"\n## {stamp} · {args.tag} · n={len(rows)}\n", "| level | R@1 | R@5 | R@10 | MRR |",
             "|---|---|---|---|---|"]
    for lvl, v in report["retrieval"].items():
        lines.append(f"| {lvl} | {v['R@1']:.2f} | {v['R@5']:.2f} | {v['R@10']:.2f} | {v['MRR']:.2f} |")
    if "answers" in report:
        lines.append("")
        lines += [f"- {k}: {v:.2f}" for k, v in report["answers"].items()]
    md = EVAL_DIR / "results.md"
    if not md.exists():
        md.write_text("# Evaluation results\n\nfused = hybrid RRF only, plain = after rerank.\n")
    with md.open("a") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    from eval.tracking import file_hash, log_run

    log_run("notes-rag", args.tag,
            {"n": len(rows), "agent": args.agent, "dataset_sha": file_hash(DATASET),
             "config": report["config"]},
            {"retrieval": report["retrieval"], "answers": report.get("answers", {})})
    flush()


if __name__ == "__main__":
    main()
