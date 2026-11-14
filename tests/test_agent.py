"""Automated tests: tool selection, failures, safety, logging. No live services."""
import json
import logging

import pytest

from agent import orchestrator as o
from agent.tools import dvc_tools, kubernetes_tools, mlflow_tools


def ok(tool, data):
    return {"tool": tool, "status": "ok", "error": None, "data": data, "duration_ms": 1.0}


def err(tool, msg):
    return {"tool": tool, "status": "error", "error": msg, "data": None, "duration_ms": 1.0}


@pytest.fixture(autouse=True)
def rules_mode(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "rules")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def fake_registry(monkeypatch, responses):
    calls = []

    class Mod:
        TOOLS = {}

        @staticmethod
        def timed_call(name, **kw):
            calls.append((name, kw))
            return responses.get(name, err(name, "not mocked"))

    names = list(responses) + ["restart_deployment", "trigger_training_dag"]
    monkeypatch.setattr(o, "REGISTRY", {n: Mod for n in names})
    return calls


def test_route_selects_dvc_tool_for_run_dataset_question():
    calls = o.route("Which dataset version was used for run " + "a" * 32 + "?")
    assert calls[0][0] == "get_experiment_dataset"


def test_route_selects_mlflow_best_run():
    assert ("find_best_run", {"metric": "f1"}) in o.route("show the best model")


def test_route_selects_kubernetes_diagnosis():
    assert o.route("why is my prediction api unavailable")[0][0] == "diagnose_deployment"


def test_route_unrelated_question_selects_nothing():
    assert o.route("what is the weather") == []


def test_tool_execution_success(monkeypatch):
    fake_registry(monkeypatch, {"get_deployed_version": ok("get_deployed_version",
                                                           {"image": "x:v1"})})
    r = o.handle("which version is deployed on kubernetes")
    assert r["mode"] == "rules"
    assert r["tools_used"][0]["status"] == "ok"
    assert "x:v1" in r["answer"]


def test_agent_does_not_invent_results_when_tool_unavailable(monkeypatch):
    fake_registry(monkeypatch, {"find_best_run": err("find_best_run", "MLflow unreachable")})
    r = o.handle("show the best model")
    assert "UNCERTAINTIES" in r["answer"]
    assert "MLflow unreachable" in r["answer"]
    assert "f1=0." not in r["answer"]


def test_multi_tool_reports_failures_as_unknown(monkeypatch):
    fake_registry(monkeypatch, {
        "find_best_run": err("find_best_run", "timeout"),
        "get_dataset_version": ok("get_dataset_version",
                                  {"md5": "abc", "nearest_tag": "data-v2", "git_commit": "1"}),
        "get_deployed_version": ok("get_deployed_version",
                                   {"image": "i:v1", "replicas_ready": 2,
                                    "replicas_desired": 2}),
        "diagnose_deployment": ok("diagnose_deployment", {"findings": ["fine"]}),
        "get_dag_run_status": err("get_dag_run_status", "airflow down"),
    })
    r = o.handle("Find the best model, dataset version, deployment status, retraining?")
    assert "airflow down" in r["answer"] and "timeout" in r["answer"]
    assert "None was started" in r["answer"]


def test_retraining_never_triggered_automatically(monkeypatch):
    calls = fake_registry(monkeypatch, {
        "find_best_run": err("find_best_run", "x"),
        "get_dataset_version": err("get_dataset_version", "x"),
        "get_deployed_version": err("get_deployed_version", "x"),
        "diagnose_deployment": err("diagnose_deployment", "x"),
        "get_dag_run_status": err("get_dag_run_status", "x")})
    o.handle("Find the best model, dataset version, deployment, is retraining necessary?")
    assert "trigger_training_dag" not in [c[0] for c in calls]




def test_trigger_requires_confirmation(monkeypatch):
    calls = fake_registry(monkeypatch, {})
    res = o.run_tool("trigger_training_dag", {}, "rid", confirm=False)
    assert res["data"]["requires_confirmation"] is True
    assert calls == []


def test_unauthorized_kubernetes_modification_rejected():
    r = kubernetes_tools.timed_call("restart_deployment", namespace="kube-system",
                                    confirm=True)
    assert r["status"] == "error" and "not approved" in r["error"]


def test_restart_without_confirm_does_not_execute():
    r = kubernetes_tools.restart_deployment(confirm=False)
    assert r["executed"] is False and r["requires_confirmation"] is True


def test_dvc_rejects_command_outside_allow_list():
    with pytest.raises(dvc_tools.ToolError):
        dvc_tools._run(["git", "push"])
    with pytest.raises(dvc_tools.ToolError):
        dvc_tools._run(["rm", "-rf", "x"])


def test_dvc_rejects_malicious_ref():
    r = dvc_tools.timed_call("explain_reproduce_version", ref="--upload-pack=evil")
    assert r["status"] == "error"


def test_mlflow_rejects_invalid_run_id():
    r = mlflow_tools.timed_call("get_run_details", run_id="../etc/passwd")
    assert r["status"] == "error"


def test_service_failure_is_reported_not_raised(monkeypatch):
    def boom(*a, **k):
        raise dvc_tools.ToolError("MLflow unreachable")

    monkeypatch.setitem(dvc_tools.TOOLS, "get_experiment_dataset", boom)
    r = dvc_tools.timed_call("get_experiment_dataset", run_id="a" * 32)
    assert r["status"] == "error" and "unreachable" in r["error"]


def test_tool_timeout_is_reported(monkeypatch):
    import subprocess

    def slow(*a, **k):
        raise subprocess.TimeoutExpired(cmd="dvc", timeout=30)

    monkeypatch.setattr(dvc_tools.subprocess, "run", slow)
    r = dvc_tools.timed_call("detect_data_changes")
    assert r["status"] == "error" and "timeout" in r["error"]


def test_secrets_are_redacted():
    assert "hunter2" not in o.redact("password=hunter2 token: abc123")


def test_structured_log_has_tool_status_duration_request_id(monkeypatch, caplog):
    fake_registry(monkeypatch, {"get_deployed_version": ok("get_deployed_version", {})})
    with caplog.at_level(logging.INFO, logger="agent"):
        o.run_tool("get_deployed_version", {}, "req123")
    rec = json.loads(caplog.records[0].message)
    assert rec["request_id"] == "req123" and rec["tool"] == "get_deployed_version"
    assert rec["status"] == "ok" and "duration_ms" in rec


def test_unknown_tool_rejected():
    assert o.run_tool("drop_database", {}, "rid")["status"] == "error"
