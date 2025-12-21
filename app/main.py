import os
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from dotenv import load_dotenv

from app.loaders import load_registry
from app.inference import validate_window, out_of_range_flags, predict_next

load_dotenv()

API_TOKEN = os.getenv("API_TOKEN", "changeme-secret-token")
MODEL_BASE_DIR = os.getenv("MODEL_BASE_DIR", "")
PRELOAD_MODELS = os.getenv("PRELOAD_MODELS", "true").lower() == "true"

app = FastAPI(title="Prediksi Bawang Merah - FastAPI", version="0.1.0")

registry = load_registry(MODEL_BASE_DIR)

if PRELOAD_MODELS:
    preload_summary: Dict[str, float] = {}
    for region, entry in registry.items():
        ms = entry.load()
        preload_summary[region] = ms
    print("Preloaded models (ms):", preload_summary)

class PredictNextRequest(BaseModel):
    region: str
    window: List[float]

class PredictNextResponse(BaseModel):
    region: str
    prediction: float
    model_version_id: str | None = None
    trained_until: str
    window_size: int
    warnings: List[str]
    runtime_ms: float
    meta: Dict[str, Any] | None = None

def auth_check(authorization: str | None):
    if authorization is None:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer" or parts[1] != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/predict/next", response_model=PredictNextResponse)
def predict_next_endpoint(payload: PredictNextRequest, authorization: str | None = Header(default=None)):
    auth_check(authorization)

    region = payload.region.strip()
    if region not in registry:
        raise HTTPException(status_code=404, detail=f"Region '{region}' not found or inactive")

    entry = registry[region]
    ok, msg = validate_window(payload.window, entry.window_size)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)

    if entry.model is None or entry.scaler is None:
        entry.load()

    warnings = out_of_range_flags(payload.window, entry.scaler)
    pred, info = predict_next(payload.window, entry.scaler, entry.model, entry.window_size)

    return PredictNextResponse(
        region=region,
        prediction=pred,
        model_version_id=None,
        trained_until=entry.trained_until,
        window_size=entry.window_size,
        warnings=warnings,
        runtime_ms=info["runtime_ms"],
        meta={"mape_2024": entry.meta.get("mape_2024"), "mape_2025": entry.meta.get("mape_2025")}
    )

@app.get("/health")
def health():
    return {"status": "ok", "regions_loaded": list(registry.keys())}