"""Stage functions used by the Airflow training DAG."""
import json
import os
import shutil
from pathlib import Path

PROJECT = Path(os.getenv("PROJECT_DIR", "."))
WORK = PROJECT / "work"
DATA = PROJECT / "data" / "train.csv"
POINTER = PROJECT / "data" / "train.csv.dvc"
EXPERIMENT = os.getenv("PIPELINE_EXPERIMENT", "mlops-agent-pipeline")


def _pointer_md5():
    for line in POINTER.read_text(encoding="utf-8").splitlines():
        if "md5:" in line:
            return line.split("md5:")[1].strip()
    return "unknown"


def validate_dataset(min_rows=500):
    import pandas as pd
    if not DATA.exists():
        raise FileNotFoundError(f"dataset missing: {DATA}")
    df = pd.read_csv(DATA)
    if "label" not in df.columns:
        raise ValueError("column 'label' missing")
    if len(df) < min_rows:
        raise ValueError(f"only {len(df)} rows, need {min_rows}")
    if df.isnull().any().any():
        raise ValueError("dataset contains null values")
    if df["label"].nunique() < 2:
        raise ValueError("label has fewer than 2 classes")
    return {"rows": len(df), "dvc_md5": _pointer_md5()}


def preprocess_data():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(DATA)
    tr, te = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    WORK.mkdir(exist_ok=True)
    tr.to_csv(WORK / "train.csv", index=False)
    te.to_csv(WORK / "test.csv", index=False)
    return {"train_rows": len(tr), "test_rows": len(te)}


def train_model(n_estimators=100, max_depth=8):
    import joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    tr = pd.read_csv(WORK / "train.csv")
    m = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                               random_state=42)
    m.fit(tr.drop(columns=["label"]), tr["label"])
    joblib.dump(m, WORK / "model.joblib")
    return {"n_estimators": n_estimators, "max_depth": max_depth}


def evaluate_and_log(dataset_version="unspecified", airflow_run_id=""):
    import joblib
    import mlflow
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score
    model = joblib.load(WORK / "model.joblib")
    te = pd.read_csv(WORK / "test.csv")
    pred = model.predict(te.drop(columns=["label"]))
    acc, f1 = accuracy_score(te["label"], pred), f1_score(te["label"], pred)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run() as run:
        mlflow.set_tags({"dvc.dataset_version": dataset_version,
                         "dvc.dataset_md5": _pointer_md5(),
                         "dvc.dataset_path": "data/train.csv",
                         "airflow.run_id": airflow_run_id, "source": "airflow"})
        mlflow.log_params({"n_estimators": model.n_estimators,
                           "max_depth": model.max_depth})
        mlflow.log_metrics({"accuracy": acc, "f1": f1})
        mlflow.log_artifact(str(WORK / "model.joblib"))
    out = {"mlflow_run_id": run.info.run_id, "accuracy": acc, "f1": f1}
    (WORK / "metrics.json").write_text(json.dumps(out), encoding="utf-8")
    return out


def quality_gate(min_f1=0.85):
    from airflow.exceptions import AirflowFailException
    m = json.loads((WORK / "metrics.json").read_text(encoding="utf-8"))
    if m["f1"] < min_f1:
        raise AirflowFailException(
            f"Quality gate FAILED: f1={m['f1']:.4f} < {min_f1}; model not promoted")
    dest = PROJECT / "models" / "approved"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(WORK / "model.joblib", dest / "model.joblib")
    (dest / "approved.json").write_text(json.dumps(m), encoding="utf-8")
    return {"promoted": True, **m}
# END
