"""Build a synthetic eval set from my own chunks (stored in eval/dataset.jsonl, gitignored).

For sampled chunks, Claude writes a question *I* would naturally ask whose answer is in that
chunk (paraphrased, not copied), plus a short reference answer. A few unanswerable
questions are added to test abstention. Review/edit the file by hand afterwards:
synthetic questions are biased toward word overlap, so hand-written ones are more valuable.

usage: uv run python -m eval.build_dataset --n 40 --unanswerable 8
"""

from __future__ import annotations

import argparse
import random

from eval.common import DATASET, all_chunks, open_store, write_jsonl
from notion_agent.llm import get_llm

SYSTEM = """너는 RAG 평가 데이터셋 작성자다. 사용자는 자신의 노션 노트를 검색하는 개인 에이전트를 쓴다.
각 노트 조각에 대해, 사용자가 '예전에 내가 정리한 걸' 떠올리며 실제로 물어볼 법한 질문 1개와 참조 답을 만든다.
규칙:
- 질문은 조각의 문구를 그대로 베끼지 말고 자연스럽게 바꿔 말한다(고유명사·명령어는 유지 가능).
- 질문만 보고 그 조각을 특정할 수 있을 만큼 구체적으로 쓴다. '이 문서는?' 같은 모호한 질문 금지.
- 조각에 의미 있는 정보가 없으면(빈 템플릿, 목차, 단순 링크) usable=false.
- 참조 답은 조각 내용만으로 1~3문장."""

ITEMS = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "usable": {"type": "boolean"},
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                },
                "required": ["id", "usable", "question", "answer"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}

UNANSWERABLE = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {"type": "string"}}},
    "required": ["questions"],
    "additionalProperties": False,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--unanswerable", type=int, default=8)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    llm = get_llm()
    chunks = [c for c in all_chunks(open_store()) if len(c["text"]) >= 150]
    random.Random(args.seed).shuffle(chunks)
    pool = chunks[: int(args.n * 1.5)]  # oversample: some come back unusable

    rows: list[dict] = []
    for i in range(0, len(pool), args.batch):
        batch = pool[i : i + args.batch]
        body = "\n\n".join(
            f'<chunk id="{j}" title="{c["page_title"]}">\n{c["text"][:1500]}\n</chunk>'
            for j, c in enumerate(batch)
        )
        out = llm.complete(body, system=SYSTEM, json_schema=ITEMS).structured or {}
        for it in out.get("items", []):
            if not it.get("usable") or not it["id"].isdigit() or int(it["id"]) >= len(batch):
                continue
            c = batch[int(it["id"])]
            rows.append({
                "qid": f"q{len(rows) + 1:03d}", "question": it["question"],
                "reference": it["answer"], "answerable": True,
                "gold_chunk_id": c["chunk_id"], "gold_page_id": c["page_id"],
                "gold_title": c["page_title"], "source": "synthetic",
            })
        print(f"generated {len(rows)} so far")
        if len(rows) >= args.n:
            break
    rows = rows[: args.n]

    titles = sorted({c["page_title"] for c in chunks})
    prompt = (
        "아래는 사용자의 노션 페이지 제목 목록이다. 이 노트 어디에도 답이 없을 것이 확실한, "
        f"그러나 개인 노트에 물어볼 법한 질문 {args.unanswerable}개를 만들어라. "
        "일부는 노트 주제와 비슷한 분야지만 다루지 않은 세부사항으로 만들어 난이도를 높여라.\n\n"
        + "\n".join(titles[:400])
    )
    qs = llm.complete(prompt, json_schema=UNANSWERABLE).structured or {}
    for q in qs.get("questions", [])[: args.unanswerable]:
        rows.append({
            "qid": f"q{len(rows) + 1:03d}", "question": q, "reference": "",
            "answerable": False, "gold_chunk_id": None, "gold_page_id": None,
            "gold_title": None, "source": "synthetic",
        })

    write_jsonl(DATASET, rows)
    print(f"wrote {len(rows)} rows -> {DATASET} (gitignored). Review it by hand.")


if __name__ == "__main__":
    main()
