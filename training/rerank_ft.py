"""Fine-tune the reranker (bge-reranker-v2-m3) with LoRA on my own notes, track it in MLflow.

Pipeline (each step caches its output under data/rerank/, all gitignored):

  mine   hybrid search (same dense+sparse+RRF as production) for every synthetic question;
         hard negatives = top fused candidates from OTHER pages (same-page chunks are
         skipped: they are often also relevant, i.e. false negatives).
  train  LoRA on the cross-encoder, listwise softmax-CE over (1 positive + G-1 negatives).
  eval   rerank the 50 fused candidates and score R@1 / R@5 / MRR of the gold chunk, for
         base and fine-tuned, on (a) the held-out synthetic test pages and (b) the original
         40-question eval set (whose pages were never used for training).

The run is logged to MLflow (experiment `reranker-ft`); the adapter is registered as
model `nra-reranker` and gets the alias `champion` only when it beats the base on BOTH
test sets. Production picks it up via `retrieval.reranker_adapter` in settings.yaml.

    uv run python -m training.rerank_ft all            # mine -> train -> eval -> register
    uv run python -m training.rerank_ft eval --adapter data/rerank/adapter
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

from eval.common import DATASET, read_jsonl
from notion_agent.config import PROJECT_ROOT, get_settings

DIR = PROJECT_ROOT / "data" / "rerank"
PAIRS, MINED, ADAPTER = DIR / "pairs.jsonl", DIR / "mined.jsonl", DIR / "adapter"
CAND_K = 50


def _text(p: dict[str, Any]) -> str:
    return p.get("context_text") or p["text"]


# ---------------------------------------------------------------- mine
def mine() -> None:
    from notion_agent.rag.factory import build_retriever

    r = build_retriever(get_settings())
    pairs = read_jsonl(PAIRS)
    gold = [{"question": x["question"], "chunk_id": x["gold_chunk_id"],
             "page_id": x["gold_page_id"], "split": "eval40"}
            for x in read_jsonl(DATASET) if x.get("answerable")]
    out = []
    t0 = time.time()
    for i, x in enumerate(pairs + gold):
        q = x["question"]
        cands = r.store.hybrid_search(r.dense.embed_query(q), r.sparse.embed_query(q),
                                      CAND_K, None)
        ids = [str(c.payload.get("chunk_id", c.id)) for c in cands]
        texts = {str(c.payload.get("chunk_id", c.id)): _text(c.payload) for c in cands}
        pos_text = texts.get(x["chunk_id"])
        if pos_text is None:  # positive not retrieved: fetch its text for training anyway
            pts = r.store.client.retrieve(r.store.collection, [x["chunk_id"]], with_payload=True)
            pos_text = _text(pts[0].payload) if pts else None
        if pos_text is None:
            continue
        negs = [(cid, texts[cid]) for c, cid in zip(cands, ids, strict=True)
                if c.payload["page_id"] != x["page_id"]]
        out.append({**x, "pos": pos_text, "fused_rank": ids.index(x["chunk_id"]) + 1
                    if x["chunk_id"] in ids else None,
                    "cands": [[cid, texts[cid]] for cid in ids], "negs": negs[:30]})
        if i % 100 == 0:
            print(f"mined {i}/{len(pairs) + len(gold)} {time.time() - t0:.0f}s", flush=True)
    MINED.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out))
    print("wrote", MINED, len(out))


def mine_unanswerable() -> None:
    """Add the eval set's unanswerable questions (split `unans`, candidates only) to MINED,
    so the gate can check that a model still lets the agent say "not in my notes"."""
    from notion_agent.rag.factory import build_retriever

    r = build_retriever(get_settings())
    rows = [x for x in read_jsonl(MINED) if x["split"] != "unans"]
    for x in read_jsonl(DATASET):
        if x.get("answerable"):
            continue
        q = x["question"]
        cands = r.store.hybrid_search(r.dense.embed_query(q), r.sparse.embed_query(q),
                                      CAND_K, None)
        rows.append({"question": q, "split": "unans", "chunk_id": None, "fused_rank": None,
                     "cands": [[str(c.payload.get("chunk_id", c.id)), _text(c.payload)]
                               for c in cands]})
    MINED.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in rows))
    print("unanswerable rows:", sum(x["split"] == "unans" for x in rows))


# ---------------------------------------------------------------- model helpers
def load_model(base: str, adapter: Path | None = None, train: bool = False, lora_r: int = 16):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(base, torch_dtype=torch.float32)
    if train:
        from peft import LoraConfig, get_peft_model

        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05,
                         target_modules=["query", "key", "value", "dense"],
                         modules_to_save=["classifier"], task_type="SEQ_CLS")
        model = get_peft_model(model, cfg)
    elif adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    return tok, model.cuda()


def score(tok, model, q: str, passages: list[str], max_len: int = 1024, bs: int = 16):
    import torch

    model.eval()
    out: list[float] = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(passages), bs):
            enc = tok([q] * len(passages[i:i + bs]), passages[i:i + bs], truncation=True,
                      max_length=max_len, padding=True, return_tensors="pt").to("cuda")
            out += model(**enc).logits.view(-1).float().tolist()
    return out


# ---------------------------------------------------------------- train
def train(epochs: int = 1, group: int = 8, bs: int = 2, accum: int = 8, lr: float = 1e-4,
          lora_r: int = 16, max_len: int = 384, seed: int = 0,
          bce: float = 0.0) -> dict[str, Any]:
    """Listwise softmax-CE ranks the positive first; `bce` > 0 adds a pointwise BCE term
    (positive -> 1, negatives -> 0) that keeps sigmoid scores calibrated, because production
    uses an absolute threshold (min_rerank_score) to decide "not in my notes"."""
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    s = get_settings()
    rows = [r for r in read_jsonl(MINED) if r["split"] == "train" and len(r["negs"]) >= group - 1]
    tok, model = load_model(s.file.retrieval.reranker_model, train=True, lora_r=lora_r)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr,
                            weight_decay=0.01)
    steps = math.ceil(len(rows) * epochs / (bs * accum))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.1)
    model.train()
    losses, step, t0 = [], 0, time.time()
    for ep in range(epochs):
        random.shuffle(rows)
        for i in range(0, len(rows), bs):
            batch = rows[i:i + bs]
            qs, ps = [], []
            for r in batch:
                negs = random.sample(r["negs"][:20], min(group - 1, len(r["negs"][:20])))
                for p in [r["pos"]] + [n[1] for n in negs]:
                    qs.append(r["question"])
                    ps.append(p)
            enc = tok(qs, ps, truncation=True, max_length=max_len, padding=True,
                      return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**enc).logits.view(len(batch), group).float()
            loss = torch.nn.functional.cross_entropy(
                logits, torch.zeros(len(batch), dtype=torch.long, device="cuda"))
            if bce:
                target = torch.zeros_like(logits)
                target[:, 0] = 1.0
                loss = loss + bce * torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, target, pos_weight=torch.tensor(float(group - 1), device="cuda"))
            loss = loss / accum
            loss.backward()
            losses.append(loss.item() * accum)
            if (i // bs + 1) % accum == 0 and step < steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if step % 10 == 0:
                    avg = sum(losses[-80:]) / len(losses[-80:])
                    print(f"ep{ep} step {step}/{steps} loss {avg:.4f}"
                          f" {time.time() - t0:.0f}s", flush=True)
    model.save_pretrained(str(ADAPTER))
    tok.save_pretrained(str(ADAPTER))
    return {"train_rows": len(rows), "steps": step, "final_loss": sum(losses[-80:]) / 80,
            "train_seconds": round(time.time() - t0), "peak_vram_gb":
            round(torch.cuda.max_memory_allocated() / 2**30, 2)}


# ---------------------------------------------------------------- eval
def evaluate(adapter: Path | None) -> dict[str, dict[str, float]]:
    import torch

    s = get_settings()
    tok, model = load_model(s.file.retrieval.reranker_model, adapter)
    rows = [r for r in read_jsonl(MINED) if r["split"] in ("test", "eval40", "unans")]
    res: dict[str, list[int | None]] = {"test": [], "eval40": []}
    fused: dict[str, list[int | None]] = {"test": [], "eval40": []}
    top: dict[str, list[float]] = {"test": [], "eval40": [], "unans": []}
    for r in rows:
        ids = [c[0] for c in r["cands"]]
        sc = score(tok, model, r["question"], [c[1] for c in r["cands"]])
        top[r["split"]].append(1 / (1 + math.exp(-max(sc))))  # same sigmoid as production
        if r["split"] == "unans":
            continue
        order = [ids[i] for i in sorted(range(len(ids)), key=lambda i: -sc[i])]
        res[r["split"]].append(order.index(r["chunk_id"]) + 1 if r["chunk_id"] in order else None)
        fused[r["split"]].append(r["fused_rank"])
    del model
    torch.cuda.empty_cache()

    def m(ranks: list[int | None]) -> dict[str, float]:
        n = max(len(ranks), 1)
        return {"n": len(ranks), "R_at_1": sum(1 for x in ranks if x == 1) / n,
                "R_at_5": sum(1 for x in ranks if x and x <= 5) / n,
                "MRR": sum(1 / x for x in ranks if x) / n}
    out = {k: m(v) for k, v in res.items()} | {f"fused_{k}": m(v) for k, v in fused.items()}
    thr = s.file.retrieval.min_rerank_score
    ans = top["test"] + top["eval40"]
    out["calib"] = {"answerable_kept": sum(x >= thr for x in ans) / max(len(ans), 1),
                    "unans_abstain": sum(x < thr for x in top["unans"]) / max(len(top["unans"]), 1),
                    "min_answerable_top": min(ans) if ans else 0.0}
    out["_ranks"] = res  # per-question ranks, for the paired win/loss count
    return out


def paired(base: dict, ft: dict) -> dict[str, dict[str, int]]:
    """Per question: did the gold chunk move up (win) or down (loss) after fine-tuning?"""
    big = 10**6
    res = {}
    for k in ("test", "eval40"):
        b, f = base["_ranks"][k], ft["_ranks"][k]
        res[k] = {"wins": sum((y or big) < (x or big) for x, y in zip(b, f, strict=True)),
                  "losses": sum((y or big) > (x or big) for x, y in zip(b, f, strict=True))}
    return res


# ---------------------------------------------------------------- mlflow
class RerankerPyfunc:  # wrapped by mlflow.pyfunc.PythonModel in _pyfunc() (lazy mlflow import)
    def load_context(self, context):  # type: ignore[no-untyped-def]
        import json as _j

        cfg = _j.loads(Path(context.artifacts["meta"]).read_text())
        self.tok, self.model = load_model(cfg["base"], Path(context.artifacts["adapter"]))

    def predict(self, context, model_input, params=None):  # type: ignore[no-untyped-def]
        """model_input: DataFrame with columns query, passage -> list of logits."""
        out = []
        for q, grp in model_input.groupby("query", sort=False):
            out += score(self.tok, self.model, q, grp["passage"].tolist())
        return out


def _pyfunc():  # type: ignore[no-untyped-def]
    import mlflow.pyfunc

    return type("NraReranker", (RerankerPyfunc, mlflow.pyfunc.PythonModel), {})()


def gate(base: dict, ft: dict) -> bool:
    """Promotable only if ranking improves on BOTH sets and calibration does not regress:
    no extra answerable question falls under the threshold, no lost abstention."""
    rank_ok = all(ft[k]["MRR"] > base[k]["MRR"] for k in ("test", "eval40"))
    b, f = base.get("calib"), ft.get("calib")
    calib_ok = b is None or (f["answerable_kept"] >= b["answerable_kept"]
                             and f["unans_abstain"] >= b["unans_abstain"])
    return rank_ok and calib_ok


def score_key(m: dict) -> float:
    """Model selection key: mean MRR over the held-out synthetic test and the eval set."""
    return (m["test"]["MRR"] + m["eval40"]["MRR"]) / 2


def register(run_id: str, adapter: Path, base_model: str, ft: dict, beats_base: bool) -> None:
    """Log the adapter as a pyfunc model in `run_id`, register it, move `champion` if best."""
    import tempfile

    import mlflow

    with tempfile.TemporaryDirectory() as td:
        meta = Path(td) / "meta.json"
        meta.write_text(json.dumps({"base": base_model}))
        with mlflow.start_run(run_id=run_id):
            info = mlflow.pyfunc.log_model(
                name="reranker", python_model=_pyfunc(),
                artifacts={"adapter": str(adapter), "meta": str(meta)},
                registered_model_name="nra-reranker", pip_requirements=[
                    "torch", "transformers", "peft"])
    client = mlflow.MlflowClient()
    mv = info.registered_model_version
    client.set_model_version_tag("nra-reranker", str(mv), "score_key", f"{score_key(ft):.4f}")
    promote = beats_base
    try:
        cur = client.get_model_version_by_alias("nra-reranker", "champion")
        promote = promote and score_key(ft) > float(cur.tags.get("score_key", 0))
    except Exception:  # no champion yet
        pass
    if promote:
        client.set_registered_model_alias("nra-reranker", "champion", str(mv))
    print(f"registered nra-reranker v{mv} champion={promote}")


def track(params: dict, train_m: dict, base: dict, ft: dict) -> None:
    import mlflow

    from eval.tracking import STORE, _flat, _git

    (PROJECT_ROOT / "mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(STORE)
    mlflow.set_experiment("reranker-ft")
    pw = paired(base, ft)
    base = {k: v for k, v in base.items() if k != "_ranks"}
    ft = {k: v for k, v in ft.items() if k != "_ranks"}
    beats = gate(base, ft)
    name = f"lora-r{params['lora_r']}-ep{params['epochs']}-lr{params['lr']}-bce{params['bce']}"
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags(_git() | {"beats_base": str(beats)})
        mlflow.log_params(params)
        mlflow.log_metrics({k: float(v) for k, v in _flat(
            {"train": train_m, "base": base, "ft": ft, "paired": pw}).items()
            if isinstance(v, int | float)})
    register(run.info.run_id, ADAPTER, params["base"], ft, beats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["mine", "mine-unans", "train", "eval", "all"])
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--bce", type=float, default=0.0, help="pointwise BCE weight (calibration)")
    a = ap.parse_args()
    if a.cmd in ("mine", "all") and (a.cmd == "mine" or not MINED.exists()):
        mine()
    if a.cmd == "mine-unans":
        mine_unanswerable()
    if a.cmd == "eval":
        b, f = evaluate(None), evaluate(a.adapter or ADAPTER)
        strip = lambda d: {k: v for k, v in d.items() if k != "_ranks"}  # noqa: E731
        print(json.dumps({"paired": paired(b, f), "base": strip(b), "ft": strip(f)}, indent=1))
    if a.cmd in ("train", "all"):
        params = {"epochs": a.epochs, "lora_r": a.lora_r, "lr": a.lr, "group": a.group,
                  "bce": a.bce, "eval_max_len": 1024,
                  "base": get_settings().file.retrieval.reranker_model}
        tm = train(epochs=a.epochs, group=a.group, lr=a.lr, lora_r=a.lora_r, bce=a.bce)
        base, ft = evaluate(None), evaluate(ADAPTER)
        print(json.dumps({"train": tm, "paired": paired(base, ft),
                          "test": [base["test"], ft["test"]],
                          "eval40": [base["eval40"], ft["eval40"]],
                          "calib": [base["calib"], ft["calib"]]}, indent=1))
        track(params, tm, base, ft)


if __name__ == "__main__":
    main()
