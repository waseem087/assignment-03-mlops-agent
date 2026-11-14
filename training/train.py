"""Train a model, log params/metrics/artifacts and DVC provenance to MLflow."""
import os

os.environ.setdefault("MLFLOW_ENABLE_RUN_URL_LOGGING", "false")
import argparse
import hashlib
import os
import subprocess
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

DATA = Path("data/train.csv")
DVC_FILE = Path("data/train.csv.dvc")


def git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unknown"


def dvc_md5() -> str:
    for line in DVC_FILE.read_text().splitlines():
        if line.strip().startswith("- md5:"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--max-depth", type=int, default=8)
    p.add_argument("--experiment", default="mlops-agent")
    p.add_argument("--dataset-version", default=None)
    a = p.parse_args()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(a.experiment)

    df = pd.read_csv(DATA)
    X, y = df.drop(columns=["label"]), df["label"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run() as run:
        model = RandomForestClassifier(
            n_estimators=a.n_estimators, max_depth=a.max_depth, random_state=42
        ).fit(Xtr, ytr)
        pred = model.predict(Xte)
        acc, f1 = accuracy_score(yte, pred), f1_score(yte, pred)

        mlflow.set_tags({
            "dvc.dataset_path": str(DATA),
            "dvc.dataset_md5": dvc_md5(),
            "dvc.dataset_version": a.dataset_version or git("describe", "--tags", "--abbrev=0"),
            "git.commit": git("rev-parse", "--short", "HEAD"),
            "data.sha256": hashlib.sha256(DATA.read_bytes()).hexdigest()[:16],
        })
        mlflow.log_params({"n_estimators": a.n_estimators, "max_depth": a.max_depth,
                           "rows": len(df)})
        mlflow.log_metrics({"accuracy": acc, "f1": f1})

        Path("models").mkdir(exist_ok=True)
        joblib.dump(model, "models/model.joblib")
        mlflow.log_artifact("models/model.joblib")
        print(f"run_id={run.info.run_id} accuracy={acc:.4f} f1={f1:.4f}")


if __name__ == "__main__":
    main()
