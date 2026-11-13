"""Training pipeline: validate -> preprocess -> train -> evaluate/log -> quality gate."""
import os
import sys
from datetime import datetime, timedelta

from airflow.sdk import dag, get_current_context, task

sys.path.insert(0, os.path.join(os.getenv("PROJECT_DIR", "/opt/airflow/project"), "training"))
import ml_pipeline as mp  # noqa: E402


@dag(dag_id="training_pipeline", schedule=None, catchup=False,
     start_date=datetime(2026, 1, 1), tags=["mlops"],
     default_args={"retries": 2, "retry_delay": timedelta(seconds=10)},
     params={"min_f1": 0.85, "n_estimators": 100, "max_depth": 8,
             "dataset_version": "", "inject_failure": ""})
def training_pipeline():
    @task
    def validate():
        if get_current_context()["params"].get("inject_failure") == "validation":
            raise ValueError("Injected failure: dataset validation")
        return mp.validate_dataset()

    @task
    def preprocess():
        return mp.preprocess_data()

    @task
    def train():
        p = get_current_context()["params"]
        return mp.train_model(int(p["n_estimators"]), int(p["max_depth"]))

    @task
    def evaluate():
        ctx = get_current_context()
        ver = ctx["params"].get("dataset_version") or "unspecified"
        return mp.evaluate_and_log(ver, str(ctx.get("run_id", "")))

    @task(retries=0)
    def gate():
        return mp.quality_gate(float(get_current_context()["params"]["min_f1"]))

    validate() >> preprocess() >> train() >> evaluate() >> gate()


training_pipeline()
# END
