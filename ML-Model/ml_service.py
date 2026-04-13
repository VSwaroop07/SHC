"""
SHC — ML Anomaly Detection Service (FastAPI)
POST /predict  →  { anomaly: bool, score: float }
GET  /health   →  { status: "ok" }
GET  /info     →  model metadata
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

app = FastAPI(
    title="SHC ML Anomaly Detection",
    description="Isolation Forest anomaly detection for Kubernetes node metrics",
    version="2.0.0",
)

MODEL_PATH  = os.getenv("MODEL_PATH",  "anomaly_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler.pkl")

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

FEATURES = [
    "cpu_usage", "memory_usage", "request_rate", "latency",
    "pod_restarts", "disk_io", "network_errors", "error_rate",
]

class MetricsInput(BaseModel):
    cpu_usage:      float = Field(..., ge=0,  le=100,  description="CPU usage %")
    memory_usage:   float = Field(..., ge=0,  le=100,  description="Memory usage %")
    request_rate:   float = Field(..., ge=0,            description="Requests per second")
    latency:        float = Field(..., ge=0,            description="Latency in ms")
    pod_restarts:   float = Field(..., ge=0,            description="Pod restart count")
    disk_io:        float = Field(..., ge=0,  le=100,  description="Disk I/O utilisation %")
    network_errors: float = Field(..., ge=0,            description="Network errors / min")
    error_rate:     float = Field(..., ge=0,  le=1,    description="Error fraction 0–1")

class PredictionResult(BaseModel):
    anomaly: bool
    score:   float
    label:   str

@app.get("/health")
def health():
    return {"status": "ok", "model": "IsolationForest"}

@app.get("/info")
def info():
    return {
        "algorithm":    "Isolation Forest",
        "n_estimators": model.n_estimators,
        "contamination": model.contamination,
        "features":     FEATURES,
    }

@app.post("/predict", response_model=PredictionResult)
def predict(metrics: MetricsInput):
    try:
        values = np.array([[getattr(metrics, f) for f in FEATURES]])
        scaled = scaler.transform(values)
        prediction = model.predict(scaled)
        score = float(model.decision_function(scaled)[0])
        is_anomaly = bool(prediction[0] == -1)
        return {
            "anomaly": is_anomaly,
            "score":   round(score, 4),
            "label":   "ANOMALY" if is_anomaly else "NORMAL",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
