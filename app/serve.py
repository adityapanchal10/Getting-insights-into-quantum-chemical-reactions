"""
FastAPI service serving the QM7 ridge regression model.

Run locally:
    uvicorn app.serve:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health   -> liveness/readiness probe target for Kubernetes
    POST /predict  -> {"H": 4, "C": 1, "N": 0, "O": 0, "S": 0} -> predicted
                       atomization energy in kcal/mol
"""
import os
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "model/champion.joblib"))

app = FastAPI(title="QM7 Atomization Energy Predictor")

_artifact = None  # loaded lazily so /health works even if the model fails to load


class AtomComposition(BaseModel):
    H: int = Field(0, ge=0, description="Hydrogen atom count")
    C: int = Field(0, ge=0, description="Carbon atom count")
    N: int = Field(0, ge=0, description="Nitrogen atom count")
    O: int = Field(0, ge=0, description="Oxygen atom count")
    S: int = Field(0, ge=0, description="Sulfur atom count")


class PredictionResponse(BaseModel):
    atomization_energy_kcal_mol: float
    model_test_mae_kcal_mol: float


def load_model():
    global _artifact
    if _artifact is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"No model artifact at {MODEL_PATH}")
        _artifact = joblib.load(MODEL_PATH)
    return _artifact


@app.on_event("startup")
def _startup():
    try:
        load_model()
    except FileNotFoundError:
        # Don't crash the process — /health reports the problem instead, so
        # Kubernetes gets a clear readiness failure rather than a crash loop
        # with no diagnostic info.
        pass


@app.get("/health")
def health():
    try:
        load_model()
        return {"status": "ok", "model_loaded": True}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
def predict(composition: AtomComposition):
    artifact = load_model()
    counts = np.array(
        [[composition.H, composition.C, composition.N, composition.O, composition.S]],
        dtype=float,
    )
    x_centered = counts - artifact["x_mean"]
    pred_centered = np.dot(x_centered, artifact["weights"])
    prediction = float(pred_centered.flatten()[0] + artifact["t_mean"])
    return PredictionResponse(
        atomization_energy_kcal_mol=prediction,
        model_test_mae_kcal_mol=artifact["test_mae_kcal_mol"],
    )
