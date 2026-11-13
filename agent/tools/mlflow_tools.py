"""MLflow tools for the MLOps agent. Read-only, built on MlflowClient."""
import os
import re
import time

os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")

from mlflow.tracking import MlflowClient

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
DEFAULT_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "mlops-agent")
SAFE_NAME = re.compile(r"^[A-Za-z0-9._\- ]{1,100}$")
PROV_KEYS = ["dvc.dataset_version", "dvc.dataset_md5", "git.commit"]


class ToolError(Exception):
    pass


def _client() -> MlflowClient:
    return MlflowClient(tracking_uri=MLFLOW_URI)


def _exp_id(client: MlflowClient, name: str) -> str:
    if not SAFE_NAME.match(name):
        raise ToolError(f"invalid experiment name: {name!r}")
    exp = client.get_experiment_by_name(name)
    if exp is None:
        raise ToolError(f"experiment not found: {name}")
    return exp.experiment_id


def _summary(run) -> dict:
    tags = run.data.tags
    return {
        "run_id": run.info.run_id,
        "run_name": tags.get("mlflow.runName"),
        "status": run.info.status,
        "start_time_ms": run.info.start_time,
        "metrics": dict(run.data.metrics),
        "params": dict(run.data.params),
        "provenance": {k: tags.get(k) for k in PROV_KEYS},
    }


def list_experiments_and_runs(experiment_name: str = DEFAULT_EXPERIMENT,
                              max_runs: int = 10) -> dict:
    """List experiments and the most recent runs of one experiment."""
    c = _client()
    max_runs = max(1, min(int(max_runs), 50))
    try:
        experiments = [{"name": e.name, "experiment_id": e.experiment_id}
                       for e in c.search_experiments()]
        runs = c.search_runs([_exp_id(c, experiment_name)],
                             order_by=["attributes.start_time DESC"],
                             max_results=max_runs)
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"MLflow error: {e}") from e
    return {"experiments": experiments, "experiment": experiment_name,
            "runs": [_summary(r) for r in runs]}


def get_run_details(run_id: str) -> dict:
    """Metrics, params, tags and artifacts of one run."""
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise ToolError("run_id must be a 32-character hex string")
    c = _client()
    try:
        run = c.get_run(run_id)
        artifacts = [a.path for a in c.list_artifacts(run_id)]
    except Exception as e:
        raise ToolError(f"MLflow error: {e}") from e
    d = _summary(run)
    d["artifacts"] = artifacts
    d["artifact_uri"] = run.info.artifact_uri
    return d


def find_best_run(metric: str = "f1", experiment_name: str = DEFAULT_EXPERIMENT,
                  max_runs: int = 50) -> dict:
    """Best run by a metric, with explicit caveats about production readiness."""
    if not re.fullmatch(r"[A-Za-z0-9_.\-]{1,50}", metric):
        raise ToolError(f"invalid metric name: {metric!r}")
    c = _client()
    try:
        runs = c.search_runs([_exp_id(c, experiment_name)],
                             filter_string="attributes.status = 'FINISHED'",
                             order_by=[f"metrics.{metric} DESC"],
                             max_results=max(1, min(int(max_runs), 100)))
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"MLflow error: {e}") from e
    runs = [r for r in runs if metric in r.data.metrics]
    if not runs:
        raise ToolError(f"no finished runs with metric '{metric}'")
    best = runs[0]
    best_val = best.data.metrics[metric]
    caveats = [f"Selected on a single metric ({metric}); this alone does not "
               "make the model production-ready."]
    if not best.data.tags.get("dvc.dataset_md5"):
        caveats.append("No DVC dataset provenance recorded for this run.")
    if len(runs) > 1:
        gap = best_val - runs[1].data.metrics[metric]
        if gap < 0.005:
            caveats.append(f"Margin over runner-up is only {gap:.4f}; "
                           "difference may be noise.")
    ties = [r.info.run_id for r in runs[1:] if r.data.metrics[metric] == best_val]
    if ties:
        caveats.append(f"Tied with other run(s): {ties}")
    caveats.append("Not verified here: held-out/production data performance, "
                   "latency, fairness, drift, or registry approval.")
    return {"metric": metric, "best": _summary(best),
            "ranking": [{"run_id": r.info.run_id, metric: r.data.metrics[metric],
                         "dataset_version": r.data.tags.get("dvc.dataset_version")}
                        for r in runs[:5]],
            "caveats": caveats, "production_ready": "not determined"}


def compare_runs(run_ids: list[str]) -> dict:
    """Side-by-side comparison of 2+ runs."""
    if not isinstance(run_ids, list) or len(run_ids) < 2 or len(run_ids) > 10:
        raise ToolError("provide between 2 and 10 run_ids")
    c = _client()
    rows = []
    for rid in run_ids:
        if not re.fullmatch(r"[0-9a-f]{32}", rid):
            raise ToolError(f"invalid run_id: {rid!r}")
        try:
            rows.append(_summary(c.get_run(rid)))
        except Exception as e:
            raise ToolError(f"MLflow error for {rid}: {e}") from e
    metrics = sorted({m for r in rows for m in r["metrics"]})
    winners = {m: max(rows, key=lambda r: r["metrics"].get(m, float("-inf")))["run_id"]
               for m in metrics}
    return {"runs": rows, "best_per_metric": winners}


def get_registry_info(model_name: str | None = None) -> dict:
    """Registered models and versions, if any."""
    c = _client()
    try:
        if model_name:
            if not SAFE_NAME.match(model_name):
                raise ToolError(f"invalid model name: {model_name!r}")
            versions = c.search_model_versions(f"name='{model_name}'")
            return {"model": model_name, "versions": [
                {"version": v.version, "run_id": v.run_id, "status": v.status,
                 "aliases": list(getattr(v, "aliases", []) or [])} for v in versions]}
        models = c.search_registered_models()
        return {"registered_models": [m.name for m in models], "count": len(models),
                "note": "Registry is empty." if not models else None}
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"MLflow error: {e}") from e


TOOLS = {
    "list_experiments_and_runs": list_experiments_and_runs,
    "get_run_details": get_run_details,
    "find_best_run": find_best_run,
    "compare_runs": compare_runs,
    "get_registry_info": get_registry_info,
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
    for t, a in [("list_experiments_and_runs", {"max_runs": 5}),
                 ("find_best_run", {"metric": "f1"}),
                 ("get_registry_info", {})]:
        print(json.dumps(timed_call(t, **a), indent=2)[:1500])
