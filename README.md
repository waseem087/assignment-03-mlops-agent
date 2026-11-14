\# MLOps AI Agent



An agent that answers natural-language questions about datasets (DVC), experiments (MLflow),

pipelines (Airflow 3) and deployments (Kubernetes), using only real tool results.



\## Setup

&#x20;   python -m venv .venv \&\& source .venv/Scripts/activate   # Git Bash on Windows

&#x20;   pip install -e ".\[dev]"

&#x20;   cp .env.example .env     # add ANTHROPIC\_API\_KEY (optional; rules mode works without)



\## Services

\- MLflow: `mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --serve-artifacts --disable-security-middleware`

\- Airflow: `docker compose -f docker-compose.airflow.yml up -d` (demo login airflow/airflow, local only)

\- Kubernetes (kind): `kind create cluster --name mlops`, `kubectl create namespace mlops-agent`,

&#x20; `docker build -t mlops-inference:v1.0.0 .`, `kind load docker-image mlops-inference:v1.0.0 --name mlops`,

&#x20; `kubectl apply -f kubernetes/`

\- Agent API: `uvicorn agent.main:app --port 9000`



\## Use

&#x20;   curl -X POST localhost:9000/chat -H "Content-Type: application/json" -d '{"message":"Why is my prediction API unavailable?"}'



\## Tests

&#x20;   pytest -q      # all external services are mocked



\## Safety

Allow-listed DVC/git commands, one approved DAG, namespace-locked Kubernetes,

`confirm=true` required for any state change, secrets redacted from logs.

Note: Airflow and kind are not run at the same time on low-RAM machines.

