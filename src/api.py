from pathlib import Path
from typing import Dict

import joblib
import json
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

MODEL_PATH = Path("artifacts/model/heart_disease_pipeline.joblib")
METRICS_PATH = Path("artifacts/model/metrics.json")
DATA_PATH = Path("data/raw/heart_disease_uci.csv")

app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="MLOps Assignment 01 - FastAPI model serving endpoint",
    version="1.0.0",
)

model = None
feature_names = []


class PredictionRequest(BaseModel):
    features: Dict[str, float] = Field(
        ...,
        example={
            "age": 54,
            "sex": 1,
            "cp": 0,
            "trestbps": 130,
            "chol": 246,
            "fbs": 0,
            "restecg": 1,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 1.0,
            "slope": 1,
            "ca": 0,
            "thal": 2,
        },
    )


@app.on_event("startup")
def load_model():
    global model, feature_names
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        feature_names = [c for c in df.columns if c != "target"]


@app.middleware("http")
async def log_requests(request: Request, call_next):
    response = await call_next(request)
    print(f"{request.method} {request.url.path} -> {response.status_code}")
    return response


@app.get("/")
def home():
    return {
        "message": "Heart Disease Risk Prediction API is running",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
    }


@app.get("/model-info")
def model_info():
    return {
        "project": "Heart Disease Risk Prediction",
        "course": "AIMLCZG523 - MLOps",
        "model_path": str(MODEL_PATH),
        "framework": "FastAPI + scikit-learn",
        "prediction_endpoint": "/predict",
    }


@app.get("/features")
def features():
    return {"feature_names": feature_names}


@app.get("/metrics")
def metrics():
    if not METRICS_PATH.exists():
        return {"message": "Metrics file not found. Run python src/train.py first."}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@app.post("/predict")
def predict(payload: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Run training first.")

    missing = [name for name in feature_names if name not in payload.features]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required features: {missing}",
        )

    input_df = pd.DataFrame([payload.features])
    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])
    label = "Heart Disease Risk" if prediction == 1 else "No Heart Disease Risk"

    return {
        "prediction": prediction,
        "prediction_label": label,
        "confidence_probability": probability,
    }


Instrumentator().instrument(app).expose(app)
