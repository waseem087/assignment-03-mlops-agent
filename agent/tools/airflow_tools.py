"""Airflow 3 tools for the MLOps agent. Authenticated REST API, one approved DAG."""
import os
import re
import time

import httpx

BASE = os.getenv("AIRFLOW_BASE_URL", "http://localhost:8080")
USER = os.getenv("AIRFLOW_USERNAME", "airflow")
PASSWORD = os.getenv("AIRFLOW_PASSWORD", "airflow")
TIMEOUT = 15
APPROVED_DAGS = {"training_pipeline"}
SAFE_ID = re.compile(r"^[A-Za-z0-9_.:+\-]{1,200}$")
_token = {"value": None, "ts": 0.0}


class ToolError(Exception):
    pass


def _auth() -> str:
    if _token["value"] and time.time() - _token["ts"] < 600:
        return _token["value"]
    try:
        r = httpx.post(f"{BASE}/auth/token", timeout=TIMEOUT,
                       json={"username": USER, "password": PASSWORD})
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise ToolError(f"Airflow authentication failed or unreachable: {e}") from e
    _token.update(value=r.json()["access_token"], ts=time.time())
    return _token["value"]


def _api(method: str, path: str, **kw):
    headers = {"Authorization": f"Bearer {_auth()}", **kw.pop("headers", {})}
    try:
        r = httpx.request(method, f"{BASE}/api/v2{path}", headers=headers,
                          timeout=TIMEOUT, **kw)
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise ToolError(f"Airflow API {e.response.status_code}: "
                        f"{e.response.text[:200]}") from e
    except httpx.HTTPError as e:
        raise ToolError(f"Airflow unreachable: {e}") from e
    return r


def _check(value: str, what: str) -> str:
    if not SAFE_ID.match(value):
        raise ToolError(f"invalid {what}: {value!r}")
    return value


def _check_dag(dag_id: str) -> str:
    _check(dag_id, "dag_id")
    return dag_id


def list_dags() -> dict:
    """List available DAGs."""
    d = _api("GET", "/dags").json()
    return {"dags": [{"dag_id": x["dag_id"], "is_paused": x["is_paused"],
                      "approved_for_trigger": x["dag_id"] in APPROVED_DAGS}
                     for x in d["dags"]]}


def get_dag_run_status(dag_id: str = "training_pipeline", limit: int = 5) -> dict:
    """Recent runs of a DAG with state and timing."""
    _check_dag(dag_id)
    limit = max(1, min(int(limit), 20))
    d = _api("GET", f"/dags/{dag_id}/dagRuns",
             params={"order_by": "-start_date", "limit": limit}).json()
    return {"dag_id": dag_id, "runs": [
        {"dag_run_id": r["dag_run_id"], "state": r["state"],
         "start_date": r["start_date"], "end_date": r["end_date"],
         "conf": r.get("conf")} for r in d["dag_runs"]]}


def trigger_training_dag(dag_id: str = "training_pipeline", conf: dict | None = None) -> dict:
    """Trigger an approved training DAG. Only allow-listed DAGs and conf keys."""
    _check_dag(dag_id)
    if dag_id not in APPROVED_DAGS:
        raise ToolError(f"DAG '{dag_id}' is not approved for triggering")
    allowed = {"dataset_version", "min_f1", "n_estimators", "max_depth", "inject_failure"}
    conf = conf or {}
    bad = set(conf) - allowed
    if bad:
        raise ToolError(f"conf keys not allowed: {sorted(bad)}")
    r = _api("POST", f"/dags/{dag_id}/dagRuns",
             json={"logical_date": None, "conf": conf}).json()
    return {"dag_id": dag_id, "dag_run_id": r["dag_run_id"], "state": r["state"],
            "conf": conf}


def _latest_run_id(dag_id: str) -> str:
    runs = get_dag_run_status(dag_id, 1)["runs"]
    if not runs:
        raise ToolError(f"no runs found for {dag_id}")
    return runs[0]["dag_run_id"]


def get_task_instances(dag_id: str = "training_pipeline", dag_run_id: str | None = None) -> dict:
    """Task states for a run (latest run if none given)."""
    _check_dag(dag_id)
    run_id = _check(dag_run_id, "dag_run_id") if dag_run_id else _latest_run_id(dag_id)
    d = _api("GET", f"/dags/{dag_id}/dagRuns/{run_id}/taskInstances").json()
    return {"dag_id": dag_id, "dag_run_id": run_id, "tasks": [
        {"task_id": t["task_id"], "state": t["state"], "try_number": t["try_number"],
         "start_date": t["start_date"], "end_date": t["end_date"],
         "duration": t.get("duration")} for t in d["task_instances"]]}


def identify_failed_tasks(dag_id: str = "training_pipeline",
                          dag_run_id: str | None = None) -> dict:
    """Failed or upstream-failed tasks of a run."""
    d = get_task_instances(dag_id, dag_run_id)
    bad = [t for t in d["tasks"] if t["state"] in ("failed", "upstream_failed")]
    return {"dag_id": dag_id, "dag_run_id": d["dag_run_id"],
            "failed_count": len(bad), "failed_tasks": bad}


def get_task_logs(task_id: str, dag_id: str = "training_pipeline",
                  dag_run_id: str | None = None, try_number: int = 1) -> dict:
    """Tail of a task's log, for failure diagnosis."""
    _check_dag(dag_id)
    _check(task_id, "task_id")
    run_id = _check(dag_run_id, "dag_run_id") if dag_run_id else _latest_run_id(dag_id)
    r = _api("GET", f"/dags/{dag_id}/dagRuns/{run_id}/taskInstances/{task_id}"
                    f"/logs/{max(1, int(try_number))}",
             headers={"Accept": "application/json"})
    try:
        content = r.json().get("content", [])
        parts = []
        for c in content:
            if isinstance(c, dict):
                parts.append(str(c.get("event", "")))
                for ed in c.get("error_detail") or []:
                    parts.append(f"EXCEPTION {ed.get('exc_type')}: {ed.get('exc_value')}")
            else:
                parts.append(str(c))
        text = "\n".join(parts)
    except ValueError:
        text = r.text
    return {"task_id": task_id, "dag_run_id": run_id, "try_number": try_number,
            "log_tail": text[-2500:]}


TOOLS = {
    "list_dags": list_dags,
    "get_dag_run_status": get_dag_run_status,
    "trigger_training_dag": trigger_training_dag,
    "get_task_instances": get_task_instances,
    "identify_failed_tasks": identify_failed_tasks,
    "get_task_logs": get_task_logs,
}


def timed_call(name: str, **kwargs) -> dict:
    start = time.perf_counter()
    try:
        data, status, error = TOOLS[name](**kwargs), "ok", None
    except (ToolError, TypeError, KeyError) as e:
        data, status, error = None, "error", str(e)
    return {"tool": name, "status": status, "error": error, "data": data,
            "duration_ms": round((time.perf_counter() - start) * 1000, 1)}


if __name__ == "__main__":
    import json
    for t, a in [("list_dags", {}), ("get_dag_run_status", {"limit": 2}),
                 ("get_task_instances", {}), ("identify_failed_tasks", {}),
                 ("get_task_logs", {"task_id": "evaluate"})]:
        print(json.dumps(timed_call(t, **a), indent=2)[:1200])
# END
