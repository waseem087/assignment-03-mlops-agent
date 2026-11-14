\# Architecture



User -> FastAPI (`/chat`) -> Orchestrator (Claude tool calling, or rule-based fallback)

\-> four tool modules (DVC, MLflow, Airflow, Kubernetes) -> evidence-based answer

(known facts / uncertainties / recommended actions).



\- Provenance: `training/train.py` and the Airflow DAG tag every MLflow run with the DVC md5 and dataset version.

\- Pipeline: validate -> preprocess -> train -> evaluate/log -> quality gate (retries, no promotion on failure).

\- Deployment: FastAPI inference image, 2 replicas, probes, limits, ConfigMap, ClusterIP service.

\- Safety: command allow-list, approved DAG list, namespace guard, confirmation for writes, redacted structured logs.

\- Known limitation: the deployment does not record which MLflow run its model came from.

