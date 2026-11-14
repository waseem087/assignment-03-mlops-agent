"""Kubernetes tools for the MLOps agent. Namespace-locked; writes need confirmation."""
import os
import time
from datetime import datetime, timezone

from kubernetes.client.rest import ApiException

from kubernetes import client, config

NAMESPACE = os.getenv("K8S_NAMESPACE", "mlops-agent")
APP_LABEL = "app=inference"
DEPLOYMENT = "inference-api"
_clients = {}


class ToolError(Exception):
    pass


def _api():
    if not _clients:
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()
        _clients["core"] = client.CoreV1Api()
        _clients["apps"] = client.AppsV1Api()
    return _clients["core"], _clients["apps"]


def _guard_ns(namespace: str) -> str:
    if namespace != NAMESPACE:
        raise ToolError(f"namespace '{namespace}' is not approved; only '{NAMESPACE}' is allowed")
    return namespace


def _call(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except ApiException as e:
        raise ToolError(f"Kubernetes API {e.status}: {e.reason}") from e
    except Exception as e:
        raise ToolError(f"Kubernetes unreachable: {type(e).__name__}: {e}") from e


def _pod_info(p) -> dict:
    cs = (p.status.container_statuses or [None])[0]
    waiting = cs.state.waiting.reason if cs and cs.state and cs.state.waiting else None
    last = cs.last_state.terminated if cs and cs.last_state else None
    return {
        "name": p.metadata.name, "phase": p.status.phase,
        "ready": bool(cs and cs.ready), "restarts": cs.restart_count if cs else 0,
        "waiting_reason": waiting,
        "last_terminated": ({"reason": last.reason, "exit_code": last.exit_code}
                            if last else None),
        "image": p.spec.containers[0].image, "node": p.spec.node_name,
    }


def list_pods_and_deployments(namespace: str = NAMESPACE) -> dict:
    _guard_ns(namespace)
    core, apps = _api()
    pods = _call(core.list_namespaced_pod, namespace, label_selector=APP_LABEL).items
    deps = _call(apps.list_namespaced_deployment, namespace).items
    return {"namespace": namespace,
            "deployments": [{"name": d.metadata.name, "desired": d.spec.replicas,
                             "ready": d.status.ready_replicas or 0,
                             "available": d.status.available_replicas or 0} for d in deps],
            "pods": [_pod_info(p) for p in pods]}


def get_pod_logs(pod_name: str | None = None, tail: int = 40, previous: bool = False,
                 namespace: str = NAMESPACE) -> dict:
    _guard_ns(namespace)
    core, _ = _api()
    if not pod_name:
        pods = _call(core.list_namespaced_pod, namespace, label_selector=APP_LABEL).items
        if not pods:
            raise ToolError("no application pods found")
        pod_name = pods[0].metadata.name
    log = _call(core.read_namespaced_pod_log, pod_name, namespace,
                tail_lines=max(1, min(int(tail), 200)), previous=bool(previous))
    return {"pod": pod_name, "previous": previous, "log_tail": log[-3000:]}


def inspect_services(namespace: str = NAMESPACE) -> dict:
    _guard_ns(namespace)
    core, _ = _api()
    out = []
    for s in _call(core.list_namespaced_service, namespace).items:
        ep = _call(core.read_namespaced_endpoints, s.metadata.name, namespace)
        n = sum(len(sub.addresses or []) for sub in (ep.subsets or []))
        out.append({"name": s.metadata.name, "type": s.spec.type,
                    "cluster_ip": s.spec.cluster_ip,
                    "ports": [{"port": p.port, "target": str(p.target_port)}
                              for p in s.spec.ports],
                    "ready_endpoints": n})
    return {"services": out}


def identify_unhealthy_pods(namespace: str = NAMESPACE) -> dict:
    _guard_ns(namespace)
    core, _ = _api()
    pods = _call(core.list_namespaced_pod, namespace, label_selector=APP_LABEL).items
    bad = [_pod_info(p) for p in pods
           if p.status.phase != "Running" or not _pod_info(p)["ready"]
           or _pod_info(p)["restarts"] > 0]
    events = _call(core.list_namespaced_event, namespace).items
    ev = [{"object": e.involved_object.name, "type": e.type, "reason": e.reason,
           "message": (e.message or "")[:200], "count": e.count}
          for e in sorted(events, key=lambda e: e.last_timestamp
                          or e.event_time or datetime.min.replace(tzinfo=timezone.utc))
          if e.type == "Warning"][-10:]
    return {"unhealthy_or_restarted": bad, "warning_events": ev}


def get_deployed_version(namespace: str = NAMESPACE) -> dict:
    _guard_ns(namespace)
    _, apps = _api()
    d = _call(apps.read_namespaced_deployment, DEPLOYMENT, namespace)
    c = d.spec.template.spec.containers[0]
    return {"deployment": DEPLOYMENT, "image": c.image,
            "version_tag": c.image.rsplit(":", 1)[-1],
            "replicas_desired": d.spec.replicas,
            "replicas_ready": d.status.ready_replicas or 0}


def diagnose_deployment(namespace: str = NAMESPACE) -> dict:
    """Collect evidence for 'why is my prediction API unavailable?'."""
    _guard_ns(namespace)
    state = list_pods_and_deployments(namespace)
    svc = inspect_services(namespace)
    health = identify_unhealthy_pods(namespace)
    findings = []
    for d in state["deployments"]:
        if d["ready"] < d["desired"]:
            findings.append(f"Deployment {d['name']}: {d['ready']}/{d['desired']} ready")
    for p in health["unhealthy_or_restarted"]:
        if p["waiting_reason"]:
            findings.append(f"Pod {p['name']} waiting: {p['waiting_reason']}")
        if p["last_terminated"]:
            findings.append(f"Pod {p['name']} last terminated: {p['last_terminated']}")
    for s in svc["services"]:
        if s["ready_endpoints"] == 0:
            findings.append(f"Service {s['name']} has no ready endpoints")
    logs = None
    for p in health["unhealthy_or_restarted"]:
        try:
            logs = get_pod_logs(p["name"], 25, previous=p["restarts"] > 0)
        except ToolError as e:
            logs = {"error": str(e)}
        break
    return {"findings": findings or ["No problems detected from pods, services, events."],
            "deployments": state["deployments"], "services": svc["services"],
            "unhealthy": health["unhealthy_or_restarted"],
            "warning_events": health["warning_events"], "log_evidence": logs}


def restart_deployment(deployment: str = DEPLOYMENT, namespace: str = NAMESPACE,
                       confirm: bool = False) -> dict:
    """Rolling restart. Changes cluster state, so it needs explicit confirmation."""
    _guard_ns(namespace)
    if deployment != DEPLOYMENT:
        raise ToolError(f"deployment '{deployment}' is not approved")
    if confirm is not True:
        return {"executed": False, "requires_confirmation": True,
                "message": f"Rolling restart of {namespace}/{deployment}. "
                           "Re-run with confirm=true to execute."}
    _, apps = _api()
    stamp = datetime.now(timezone.utc).isoformat()
    body = {"spec": {"template": {"metadata": {"annotations": {
        "kubectl.kubernetes.io/restartedAt": stamp}}}}}
    _call(apps.patch_namespaced_deployment, deployment, namespace, body)
    return {"executed": True, "deployment": deployment, "restarted_at": stamp}


TOOLS = {
    "list_pods_and_deployments": list_pods_and_deployments,
    "get_pod_logs": get_pod_logs,
    "inspect_services": inspect_services,
    "identify_unhealthy_pods": identify_unhealthy_pods,
    "get_deployed_version": get_deployed_version,
    "diagnose_deployment": diagnose_deployment,
    "restart_deployment": restart_deployment,
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
    for t, a in [("list_pods_and_deployments", {}), ("get_deployed_version", {}),
                 ("inspect_services", {}), ("diagnose_deployment", {}),
                 ("restart_deployment", {"confirm": False}),
                 ("restart_deployment", {"namespace": "kube-system", "confirm": True})]:
        print(json.dumps(timed_call(t, **a), indent=1)[:900])
# END
