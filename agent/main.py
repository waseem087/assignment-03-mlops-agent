"""FastAPI interface for the MLOps agent."""
import logging

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(message)s")

from fastapi import FastAPI  # noqa: E402

from agent.orchestrator import handle  # noqa: E402
from agent.schemas import ChatRequest, ChatResponse  # noqa: E402

app = FastAPI(title="MLOps Agent")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    return handle(req.message, req.confirm)
