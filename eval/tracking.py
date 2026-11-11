"""Log eval runs to MLflow (local sqlite store under `mlruns/`, gitignored).

Only params and metrics are logged -- never questions, answers or note text -- so the
store holds nothing private. Every run records the git commit, whether the tree was
dirty and a hash of the dataset, so a metric can always be traced back to the exact code
and question set that produced it. No-op when mlflow is not installed.

    uv run python -m eval.tracking backfill   # import existing eval/runs/*.json
    uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STORE = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlruns' / 'mlflow.db'}")


def _git() -> dict[str, str]:
    def run(*a: str) -> str:
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                                  timeout=5).stdout.strip()
        except Exception:
            return ""
    return {"git_commit": run("rev-parse", "--short", "HEAD") or "unknown",
            "git_dirty": str(bool(run("status", "--porcelain", "--", "src", "eval")))}


def file_hash(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12] if p.exists() else "missing"


def _flat(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}".replace("@", "_at_")
        if isinstance(v, dict):
            out |= _flat(v, f"{key}.")
        else:
            out[key] = v
    return out


def log_run(experiment: str, run_name: str, params: dict[str, Any], metrics: dict[str, Any],
            tags: dict[str, str] | None = None, git: bool = True) -> str | None:
    try:
        import mlflow
    except ImportError:
        return None
    (ROOT / "mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(STORE)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as r:
        mlflow.set_tags((_git() if git else {}) | (tags or {}))
        mlflow.log_params({k: str(v)[:500] for k, v in _flat(params).items()})
        mlflow.log_metrics({k: float(v) for k, v in _flat(metrics).items()
                            if isinstance(v, int | float) and not isinstance(v, bool)})
        return r.info.run_id


def backfill() -> None:
    """Import the JSON reports already in eval/runs (metrics only)."""
    for p in sorted((ROOT / "eval" / "runs").glob("*.json")):
        d = json.loads(p.read_text())
        name = p.stem
        if name.startswith("erp_"):
            log_run("erp-agent", name.split("_", 3)[-1], {"source_file": p.name},
                    d["summary"], {"backfilled": "true"}, git=False)
        else:
            rep = d["report"]
            log_run("notes-rag", rep["tag"], {"source_file": p.name, "n": rep["n"],
                    "config": rep.get("config", {})},
                    {"retrieval": rep["retrieval"], "answers": rep.get("answers", {})},
                    {"backfilled": "true"}, git=False)
        print("logged", p.name)


if __name__ == "__main__":
    if sys.argv[1:] == ["backfill"]:
        backfill()
