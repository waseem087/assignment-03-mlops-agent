"""DVC tools for the MLOps agent. Only allow-listed commands are ever executed."""
import json
import os
import re
import subprocess
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(os.getenv("REPO_ROOT", ".")).resolve()
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
TIMEOUT = 30

# Allow-list: (executable, first argument). Anything else is rejected.
ALLOWED = {
    ("git", "log"), ("git", "show"), ("git", "tag"), ("git", "rev-parse"),
    ("git", "diff"), ("git", "describe"),
    ("dvc", "status"), ("dvc", "diff"), ("dvc", "list"),
}
SAFE_REF = re.compile(r"^[A-Za-z0-9._/\-]{1,100}$")


class ToolError(Exception):
    pass


def _run(cmd: list[str]) -> str:
    if len(cmd) < 2 or (cmd[0], cmd[1]) not in ALLOWED:
        raise ToolError(f"command not allowed: {' '.join(cmd[:2])}")
    try:
        out = subprocess.run(
            cmd, check=False, cwd=REPO_ROOT, capture_output=True, text=True,
            timeout=TIMEOUT
        )
    except subprocess.TimeoutExpired as e:
        raise ToolError(f"timeout after {TIMEOUT}s: {cmd[0]} {cmd[1]}") from e
    except FileNotFoundError as e:
        raise ToolError(f"{cmd[0]} not installed") from e
    if out.returncode != 0:
        raise ToolError(out.stderr.strip() or f"{cmd[0]} failed")
    return out.stdout.strip()


def _check_ref(ref: str) -> str:
    if not SAFE_REF.match(ref) or ref.startswith("-"):
        raise ToolError(f"invalid git ref: {ref!r}")
    return ref


def _pointer_at(ref: str, path: str) -> dict:
    """Parse a .dvc pointer file as it existed at a git ref."""
    text = _run(["git", "show", f"{_check_ref(ref)}:{path}"])
    md5 = re.search(r"md5:\s*([0-9a-f]+)", text)
    size = re.search(r"size:\s*(\d+)", text)
    return {"md5": md5.group(1) if md5 else None,
            "size": int(size.group(1)) if size else None}


def list_dvc_datasets() -> dict:
    """List datasets tracked by DVC (*.dvc pointer files)."""
    datasets = []
    for f in sorted(REPO_ROOT.rglob("*.dvc")):
        if not f.is_file():
            continue
        if {".venv", ".git", ".dvc", "node_modules"} & set(f.relative_to(REPO_ROOT).parts[:-1]):
            continue
        text = f.read_text(encoding="utf-8")
        md5 = re.search(r"md5:\s*([0-9a-f]+)", text)
        size = re.search(r"size:\s*(\d+)", text)
        datasets.append({
            "pointer_file": f.relative_to(REPO_ROOT).as_posix(),
            "md5": md5.group(1) if md5 else None,
            "size_bytes": int(size.group(1)) if size else None,
        })
    return {"datasets": datasets, "count": len(datasets)}


def get_dataset_version(path: str = "data/train.csv.dvc") -> dict:
    """Current md5, Git commit, and nearest tag for a tracked dataset."""
    pointer = REPO_ROOT / path
    if not pointer.exists():
        raise ToolError(f"pointer file not found: {path}")
    text = pointer.read_text(encoding="utf-8")
    md5 = re.search(r"md5:\s*([0-9a-f]+)", text)
    try:
        tag = _run(["git", "describe", "--tags", "--abbrev=0"])
    except ToolError:
        tag = None
    return {
        "pointer_file": path,
        "md5": md5.group(1) if md5 else None,
        "git_commit": _run(["git", "rev-parse", "--short", "HEAD"]),
        "nearest_tag": tag,
    }


def compare_dataset_versions(ref_a: str, ref_b: str,
                             path: str = "data/train.csv.dvc") -> dict:
    """Compare a dataset pointer between two git refs (tags or commits)."""
    a, b = _pointer_at(ref_a, path), _pointer_at(ref_b, path)
    return {
        "ref_a": ref_a, "ref_b": ref_b, "a": a, "b": b,
        "identical": a["md5"] == b["md5"],
        "size_delta_bytes": (b["size"] or 0) - (a["size"] or 0),
    }


def detect_data_changes() -> dict:
    """Run `dvc status` to see whether tracked data differs from its pointer."""
    out = _run(["dvc", "status"])
    clean = "up to date" in out.lower()
    return {"changed": not clean, "dvc_status": out}


def explain_reproduce_version(ref: str, path: str = "data/train.csv.dvc") -> dict:
    """Return the exact commands to reproduce a dataset version. Executes nothing."""
    _check_ref(ref)
    pointer = _pointer_at(ref, path)
    data_path = path.removesuffix(".dvc")
    return {
        "ref": ref,
        "expected_md5": pointer["md5"],
        "steps": [
            f"git checkout {ref} -- {path}",
            f"dvc checkout {path}   # or: dvc pull {path} if not in local cache",
            f"verify: md5 of {data_path} should equal {pointer['md5']}",
        ],
        "note": "Commands are shown, not executed; the agent never changes your files.",
    }


def get_experiment_dataset(run_id: str) -> dict:
    """Read the DVC provenance recorded on an MLflow run (never guessed)."""
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise ToolError("run_id must be a 32-character hex string")
    try:
        r = httpx.get(f"{MLFLOW_URI}/api/2.0/mlflow/runs/get",
                      params={"run_id": run_id}, timeout=TIMEOUT)
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise ToolError(f"MLflow unreachable or run not found: {e}") from e
    tags = {t["key"]: t["value"] for t in r.json()["run"]["data"].get("tags", [])}
    keys = ["dvc.dataset_version", "dvc.dataset_md5", "git.commit", "dvc.dataset_path"]
    prov = {k: tags.get(k) for k in keys}
    if not any(prov.values()):
        return {"run_id": run_id, "provenance": None,
                "note": "No DVC provenance tags recorded on this run."}
    return {"run_id": run_id, "provenance": prov}


TOOLS = {
    "list_dvc_datasets": list_dvc_datasets,
    "get_dataset_version": get_dataset_version,
    "compare_dataset_versions": compare_dataset_versions,
    "detect_data_changes": detect_data_changes,
    "explain_reproduce_version": explain_reproduce_version,
    "get_experiment_dataset": get_experiment_dataset,
}


def timed_call(name: str, **kwargs) -> dict:
    """Execute a tool and wrap the result with status and duration."""
    start = time.perf_counter()
    try:
        data = TOOLS[name](**kwargs)
        status, error = "ok", None
    except (ToolError, TypeError, KeyError) as e:
        data, status, error = None, "error", str(e)
    return {"tool": name, "status": status, "error": error, "data": data,
            "duration_ms": round((time.perf_counter() - start) * 1000, 1)}


if __name__ == "__main__":
    for tool, args in [("list_dvc_datasets", {}), ("get_dataset_version", {}),
                       ("compare_dataset_versions", {"ref_a": "data-v1", "ref_b": "data-v2"}),
                       ("detect_data_changes", {}),
                       ("explain_reproduce_version", {"ref": "data-v1"})]:
        print(json.dumps(timed_call(tool, **args), indent=2))
