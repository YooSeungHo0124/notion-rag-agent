"""Nightly pipeline: Notion incremental sync -> ERP agent eval -> quality gate.

Airflow only orchestrates. The heavy work (GPU models, Qdrant, Notion reader) stays in
the running app, which Airflow drives over its HTTP API, so the scheduler needs no GPU
and no Notion/Odoo credentials. The eval run is logged to MLflow (eval/tracking.py), and
the gate fails the DAG run when answer quality drops below the thresholds, which is the
signal to look at the MLflow comparison before shipping a change.

Env: NRA_API (default http://127.0.0.1:8000), NRA_MIN_ACCURACY (default 0.85).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta

from airflow.sdk import dag, task

API = os.environ.get("NRA_API", "http://127.0.0.1:8000")
MIN_ACCURACY = float(os.environ.get("NRA_MIN_ACCURACY", "0.85"))


def _call(path: str, body: dict | None = None, timeout: int = 30) -> dict:
    req = urllib.request.Request(
        f"{API}{path}", method="POST" if body is not None else "GET",
        data=json.dumps(body).encode() if body is not None else None,
        headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


@dag(schedule="0 3 * * *", start_date=datetime(2026, 11, 1), catchup=False,
     max_active_runs=1, tags=["nra"],
     default_args={"retries": 1, "retry_delay": timedelta(minutes=5)})
def nra_nightly():
    @task
    def wait_ready() -> dict:
        for _ in range(60):  # models may still be loading after a restart
            try:
                h = _call("/api/health")
                if h.get("ready"):
                    return {"pages": h["pages"], "chunks": h["chunks"]}
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(10)
        raise RuntimeError("app not ready after 10 minutes")

    @task
    def notion_sync(_: dict) -> dict:
        try:
            _call("/api/sync", {"full": False})
        except urllib.error.HTTPError as e:
            if e.code != 409:  # 409 = a sync is already running: just wait for it
                raise
        for _ in range(180):
            st = _call("/api/sync/status")
            if not st["running"]:
                if st.get("ok") is False:
                    raise RuntimeError(f"sync failed: {st.get('error')}")
                return {"summary": st.get("summary"), "last_sync": st.get("last_sync")}
            time.sleep(10)
        raise RuntimeError("sync did not finish in 30 minutes")

    @task(execution_timeout=timedelta(minutes=30))
    def erp_eval(_: dict) -> dict:
        from eval.erp_eval import run  # repo is on PYTHONPATH in the airflow container

        return run(API, f"nightly-{datetime.now():%Y%m%d}")

    @task
    def quality_gate(summary: dict) -> dict:
        bad = []
        if summary["answer_accuracy"] < MIN_ACCURACY:
            bad.append(f"answer_accuracy {summary['answer_accuracy']} < {MIN_ACCURACY}")
        if summary["abstain_accuracy"] < 1.0:
            bad.append(f"abstain_accuracy {summary['abstain_accuracy']} < 1.0")
        if bad:
            raise ValueError("quality gate failed: " + "; ".join(bad))
        return summary

    quality_gate(erp_eval(notion_sync(wait_ready())))


nra_nightly()
