"""Agent orchestrator: LLM tool calling with a rule-based fallback and tracing."""
import inspect
import json
import logging
import os
import re
import time
import uuid

from agent.tools import airflow_tools, dvc_tools, kubernetes_tools, mlflow_tools

log = logging.getLogger("agent")
MODEL = os.getenv("AGENT_MODEL", "claude-sonnet-4-5")
MAX_STEPS = 8
SECRET_RE = re.compile(r"(password|token|api[_-]?key|secret)\s*[=:]\s*\S+", re.I)

REGISTRY = {}
for _mod in (dvc_tools, mlflow_tools, airflow_tools, kubernetes_tools):
    for _name in _mod.TOOLS:
        REGISTRY[_name] = _mod

SYSTEM = (
    "You are an MLOps assistant. Answer ONLY from tool results; never invent "
    "operational facts. If a tool fails, say so and state what is unknown. "
    "Separate KNOWN FACTS, UNCERTAINTIES and RECOMMENDED ACTIONS. Never trigger "
    "retraining unless the user explicitly asks. Recommend retraining only with "
    "evidence (stale data, failed quality gate, degradation). Cluster changes "
    "need explicit user confirmation."
)


def redact(text: str) -> str:
    return SECRET_RE.sub(lambda m: m.group(1) + "=***", text)


def brief(obj, limit: int = 1800) -> str:
    return redact(json.dumps(obj, default=str))[:limit]


def summ(res: dict) -> dict:
    err = redact(res["error"]) if res.get("error") else None
    return {"tool": res["tool"], "status": res["status"],
            "duration_ms": res["duration_ms"], "error": err}


def run_tool(name: str, args: dict, request_id: str, confirm: bool = False) -> dict:
    args = dict(args or {})
    if name not in REGISTRY:
        res = {"tool": name, "status": "error", "error": f"unknown tool {name}",
               "data": None, "duration_ms": 0}
    elif name == "trigger_training_dag" and not confirm:
        res = {"tool": name, "status": "ok", "error": None, "duration_ms": 0,
               "data": {"executed": False, "requires_confirmation": True,
                        "message": "Starting a training run needs confirm=true."}}
    else:
        if name == "restart_deployment":
            args["confirm"] = bool(confirm)
        res = REGISTRY[name].timed_call(name, **args)
    log.info(json.dumps({"request_id": request_id, "tool": name,
                         "status": res["status"], "duration_ms": res["duration_ms"]}))
    return res


def _jtype(annotation) -> dict:
    s = str(annotation)
    if s.startswith("list"):
        return {"type": "array", "items": {"type": "string"}}
    for key, jt in (("dict", "object"), ("bool", "boolean"),
                    ("int", "integer"), ("float", "number")):
        if key in s:
            return {"type": jt}
    return {"type": "string"}


def build_specs() -> list:
    specs = []
    for name, mod in REGISTRY.items():
        fn = mod.TOOLS[name]
        props = {p.name: _jtype(p.annotation)
                 for p in inspect.signature(fn).parameters.values()
                 if p.name != "confirm"}
        specs.append({"name": name, "description": (fn.__doc__ or name).strip()[:300],
                      "input_schema": {"type": "object", "properties": props}})
    return specs


def llm_answer(message: str, request_id: str, confirm: bool):
    import anthropic
    c = anthropic.Anthropic(timeout=40)
    specs = build_specs()
    msgs = [{"role": "user", "content": message}]
    used = []
    for _ in range(MAX_STEPS):
        r = c.messages.create(model=MODEL, max_tokens=1500, system=SYSTEM,
                              tools=specs, messages=msgs)
        msgs.append({"role": "assistant", "content": r.content})
        if r.stop_reason != "tool_use":
            return "".join(b.text for b in r.content if b.type == "text"), used
        results = []
        for b in r.content:
            if b.type == "tool_use":
                res = run_tool(b.name, b.input, request_id, confirm)
                used.append(summ(res))
                results.append({"type": "tool_result", "tool_use_id": b.id,
                                "content": brief(res, 6000)})
        msgs.append({"role": "user", "content": results})
    return "Stopped: too many tool steps without a final answer.", used


RUN_ID = re.compile(r"\b[0-9a-f]{32}\b")


def _has(m: str, *words) -> bool:
    return any(w in m for w in words)


def route(msg: str) -> list:
    m, calls = msg.lower(), []
    rid = RUN_ID.search(m)
    if rid and _has(m, "dataset", "version"):
        calls.append(("get_experiment_dataset", {"run_id": rid.group()}))
    elif _has(m, "compare") and "dataset" in m:
        calls.append(("compare_dataset_versions", {"ref_a": "data-v1", "ref_b": "data-v2"}))
    elif "reproduce" in m:
        calls.append(("explain_reproduce_version", {"ref": "data-v1"}))
    elif "changed" in m or ("change" in m and "data" in m):
        calls.append(("detect_data_changes", {}))
    elif "dataset" in m:
        calls += [("list_dvc_datasets", {}), ("get_dataset_version", {})]
    if "best" in m:
        calls.append(("find_best_run", {"metric": "f1"}))
    elif "registry" in m:
        calls.append(("get_registry_info", {}))
    elif _has(m, "experiments", "runs", "mlflow"):
        calls.append(("list_experiments_and_runs", {"max_runs": 5}))
    if _has(m, "trigger", "start training", "run the training"):
        calls.append(("trigger_training_dag", {}))
    elif "fail" in m and _has(m, "pipeline", "dag", "airflow", "task"):
        calls.append(("identify_failed_tasks", {}))
    elif _has(m, "airflow", "dag", "pipeline"):
        calls.append(("get_dag_run_status", {"limit": 3}))
    if "restart" in m:
        calls.append(("restart_deployment", {}))
    elif _has(m, "why", "unavailable", "diagnos", "unhealthy"):
        calls.append(("diagnose_deployment", {}))
    elif "log" in m:
        calls.append(("get_pod_logs", {}))
    elif "service" in m:
        calls.append(("inspect_services", {}))
    elif "version" in m and _has(m, "deploy", "kubernetes"):
        calls.append(("get_deployed_version", {}))
    elif _has(m, "pod", "deploy", "kubernetes", "k8s"):
        calls.append(("list_pods_and_deployments", {}))
    return calls


def render(known, unknown, recs) -> str:
    return "\n".join(["KNOWN FACTS (from tool results):", *(known or ["- none"]),
                      "", "UNCERTAINTIES:", *(unknown or ["- none reported"]),
                      "", "RECOMMENDED ACTIONS:", *(recs or ["- none"])])


def multi_tool(request_id: str, confirm: bool):
    used, known, unknown, recs = [], [], [], []

    def call(name, **a):
        res = run_tool(name, a, request_id, confirm)
        used.append(summ(res))
        if res["status"] != "ok":
            unknown.append(f"{name} unavailable: {redact(str(res['error']))}")
            return None
        return res["data"]

    best, run_md5 = call("find_best_run", metric="f1"), None
    if best:
        b = best["best"]
        prov = b["provenance"]
        run_md5 = prov.get("dvc.dataset_md5")
        known.append(f"Best run by f1: {b['run_id']} ({b['run_name']}), "
                     f"f1={b['metrics'].get('f1', 0):.4f}, dataset "
                     f"{prov.get('dvc.dataset_version')} md5 {run_md5}")
        unknown.extend(best["caveats"])
    cur = call("get_dataset_version")
    if cur:
        known.append(f"Current dataset pointer md5 {cur['md5']} "
                     f"(tag {cur['nearest_tag']}, commit {cur['git_commit']})")
        if run_md5 and run_md5 == cur["md5"]:
            known.append("Best run's training data matches the current dataset pointer.")
        elif run_md5:
            recs.append("Best run's data differs from the current dataset; "
                        "review whether retraining is needed (not started).")
    ver = call("get_deployed_version")
    if ver:
        known.append(f"Deployed image {ver['image']}, "
                     f"{ver['replicas_ready']}/{ver['replicas_desired']} replicas ready")
    diag = call("diagnose_deployment")
    if diag:
        known.append("Kubernetes findings: " + "; ".join(diag["findings"]))
    runs = call("get_dag_run_status", limit=3)
    evidence = []
    if runs:
        states = [r["state"] for r in runs["runs"]]
        known.append(f"Recent Airflow runs (newest first): {states}")
        if states and states[0] == "failed":
            evidence.append("the latest training run failed")
    unknown.append("The deployment does not record which MLflow run its model came from, "
                   "so it cannot be confirmed to be the best run. "
                   "Production performance and drift were not checked.")
    if evidence:
        recs.append("Investigate before retraining: " + "; ".join(evidence) + ".")
    else:
        recs.append("No evidence (freshness policy, failed gate, degradation) "
                    "requires retraining. None was started.")
    return render(known, unknown, recs), used


def rule_answer(message: str, request_id: str, confirm: bool):
    m = message.lower()
    if "best" in m and _has(m, "deploy", "dataset", "retrain"):
        return multi_tool(request_id, confirm)
    calls = route(message)
    if not calls:
        return ("No available tool matches this request, so I have no verified "
                "information to give. Ask about datasets, MLflow runs, Airflow "
                "or Kubernetes."), []
    used, known, unknown, recs = [], [], [], []
    for name, args in calls:
        res = run_tool(name, args, request_id, confirm)
        used.append(summ(res))
        if res["status"] == "ok":
            known.append(f"- {name}: {brief(res['data'], 900)}")
            if name == "find_best_run":
                unknown.extend(f"- {c}" for c in res["data"].get("caveats", []))
            if name == "find_best_run":
                unknown.extend(f"- {c}" for c in res["data"].get("caveats", []))
            if name == "diagnose_deployment" and res["data"].get("warning_events"):
                unknown.append("- Warning events listed above are historical records, "
                               "not necessarily current faults; current state is in findings.")
        else:
            unknown.append(f"- {name} failed: {redact(str(res['error']))}")
    return render(known, unknown, recs), used


def handle(message: str, confirm: bool = False) -> dict:
    request_id, start, mode = uuid.uuid4().hex[:12], time.perf_counter(), "llm"
    key = os.getenv("ANTHROPIC_API_KEY", "")
    try:
        if not key or key.startswith("your-") or os.getenv("AGENT_MODE") == "rules":
            raise RuntimeError("LLM disabled or no API key")
        answer, used = llm_answer(message, request_id, confirm)
    except Exception as e:
        log.warning(json.dumps({"request_id": request_id, "event": "llm_fallback",
                                "reason": redact(str(e))[:200]}))
        mode = "rules"
        answer, used = rule_answer(message, request_id, confirm)
    log.info(json.dumps({"request_id": request_id, "event": "request_done", "mode": mode,
                         "tools": [u["tool"] for u in used],
                         "duration_ms": round((time.perf_counter() - start) * 1000, 1)}))
    return {"request_id": request_id, "answer": answer, "tools_used": used, "mode": mode}
# END
