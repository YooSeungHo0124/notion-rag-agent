"""ERP + company-doc agent eval against the running API (`nra serve`).

Grading is rule based: every `must` group needs one of its strings in the answer text,
every kind in `src` (erp / doc) must be among the cited sources, and `abstain` questions
pass when the agent refuses or says the records don't have it.

    uv run python eval/erp_eval.py [--api http://127.0.0.1:8000] [--label v1]
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
NO_INFO = ("없습니다", "찾을 수 없", "기록에 없", "확인되지 않", "나와 있지 않")


def ask(api: str, q: str, tid: str) -> dict:
    req = urllib.request.Request(f"{api}/api/agent", method="POST",
                                 data=json.dumps({"question": q, "thread_id": tid}).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def grade(item: dict, run: dict) -> dict:
    a = run.get("answer") or {}
    text = " ".join(s.get("text", "") for s in a.get("sentences", []))
    kinds = {s["page_id"].split(":")[0] for s in a.get("sources", []) if ":" in s["page_id"]}
    tools = [s["tool"] for s in run.get("steps", [])]
    if item.get("abstain"):
        ok = (not a.get("answerable")) or any(p in text for p in NO_INFO)
        return {"ok": ok, "text": text[:300], "tools": tools}
    miss = [g for g in item["must"] if not any(x in text for x in g)]
    miss_src = [k for k in item.get("src", []) if k not in kinds]
    return {"ok": not miss and not miss_src, "missing": miss, "missing_src": miss_src,
            "supported_ratio": a.get("supported_ratio"), "confidence": a.get("confidence"),
            "tools": tools, "text": text[:300]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--label", default="erp-v1")
    args = ap.parse_args()
    data = json.loads((HERE / "erp_dataset.json").read_text())
    rows = []
    for it in data:
        t0 = time.time()
        run = ask(args.api, it["q"], f"eval-{args.label}-{it['id']}-{int(t0)}")
        g = grade(it, run) | {"id": it["id"], "q": it["q"], "s": round(time.time() - t0, 1)}
        rows.append(g)
        print(f"{'PASS' if g['ok'] else 'FAIL'} {it['id']} {g['s']}s {g['tools']}"
              f" {g.get('missing') or ''}{g.get('missing_src') or ''}", flush=True)
    ans = [r for r in rows if not r["id"].startswith("u")]
    un = [r for r in rows if r["id"].startswith("u")]
    sr = [r["supported_ratio"] for r in ans if r.get("supported_ratio") is not None]
    summary = {
        "n": len(rows), "answer_accuracy": round(sum(r["ok"] for r in ans) / len(ans), 3),
        "abstain_accuracy": round(sum(r["ok"] for r in un) / max(len(un), 1), 3),
        "mean_supported_ratio": round(sum(sr) / max(len(sr), 1), 3),
        "mean_latency_s": round(sum(r["s"] for r in rows) / len(rows), 1),
    }
    print(json.dumps(summary, ensure_ascii=False))
    out = HERE / "runs" / f"erp_{time.strftime('%Y%m%d_%H%M%S')}_{args.label}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
