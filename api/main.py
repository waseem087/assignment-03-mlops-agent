"""FastAPI inference service for the approved model."""
import os
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

APP_VERSION = os.getenv("APP_VERSION", "dev")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/model/model.joblib"))
N_FEATURES = int(os.getenv("N_FEATURES", "10"))

app = FastAPI(title="mlops-inference", version=APP_VERSION)
_model = {"obj": None}


class PredictRequest(BaseModel):
    features: list[float]


@app.on_event("startup")
def load_model() -> None:
    if MODEL_PATH.exists():
        _model["obj"] = joblib.load(MODEL_PATH)


@app.get("/healthz")
def healthz():
    return {"status": "alive"}


@app.get("/readyz")
def readyz():
    if _model["obj"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready"}


@app.get("/version")
def version():
    return {"app_version": APP_VERSION, "model_loaded": _model["obj"] is not None}


@app.post("/predict")
def predict(req: PredictRequest):
    if _model["obj"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    if len(req.features) != N_FEATURES:
        raise HTTPException(status_code=422, detail=f"expected {N_FEATURES} features")
    return {"prediction": int(_model["obj"].predict([req.features])[0])}
# END
