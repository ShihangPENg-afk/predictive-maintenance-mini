"""FastAPI inference service for manufacturing quality prediction."""

from __future__ import annotations

import json
import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import PredictRequest, PredictResponse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
SCHEMA_PATH = ARTIFACTS_DIR / "schema.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

LABEL_DISPLAY = {
    "defect": "Defect",
    "normal": "Normal",
}

RISK_THRESHOLDS = {
    "high": 0.7,
    "medium": 0.4,
}

RECOMMENDATIONS = {
    "high": (
        "Immediate inspection recommended. Pause the line if possible and "
        "check temperature, pressure, and vibration readings."
    ),
    "medium": (
        "Elevated defect risk detected. Increase monitoring frequency and "
        "schedule preventive maintenance."
    ),
    "low": "Process appears stable. Continue routine monitoring.",
}


class AppState:
    model: Any = None
    schema: dict[str, Any] = {}
    metrics: dict[str, Any] = {}


state = AppState()


def load_artifacts() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema artifact not found: {SCHEMA_PATH}")

    state.model = joblib.load(MODEL_PATH)
    state.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    if METRICS_PATH.exists():
        state.metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    else:
        state.metrics = {}


def validate_features(features: dict[str, Any]) -> dict[str, float]:
    required = state.schema.get("features", [])
    missing = [name for name in required if name not in features]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "missing_features",
                "message": "Request is missing required feature fields.",
                "missing_fields": missing,
                "required_fields": required,
            },
        )

    numeric_features = set(state.schema.get("numeric_features", []))
    invalid: list[dict[str, str]] = []
    parsed: dict[str, float] = {}

    for name in required:
        value = features[name]
        if name in numeric_features:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                invalid.append(
                    {
                        "field": name,
                        "message": "Expected a numeric value.",
                        "received_type": type(value).__name__,
                    }
                )
                continue
            parsed[name] = float(value)
            if not math.isfinite(parsed[name]):
                invalid.append(
                    {
                        "field": name,
                        "message": "Expected a finite numeric value.",
                        "received_type": type(value).__name__,
                    }
                )
                continue
        else:
            parsed[name] = value

    if invalid:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_feature_values",
                "message": "One or more feature values have invalid types.",
                "invalid_fields": invalid,
            },
        )

    return parsed


def defect_probability(probabilities: dict[str, float] | None) -> float | None:
    if not probabilities:
        return None
    if "defect" in probabilities:
        return probabilities["defect"]
    classes = state.schema.get("classes", [])
    if len(classes) >= 2 and classes[0] == "defect":
        return probabilities.get(classes[0])
    return None


def risk_level_from_probability(defect_prob: float | None, prediction: str) -> str:
    if defect_prob is not None:
        if defect_prob >= RISK_THRESHOLDS["high"]:
            return "high"
        if defect_prob >= RISK_THRESHOLDS["medium"]:
            return "medium"
        return "low"
    return "high" if prediction == "defect" else "low"


def build_recommendation(risk_level: str, prediction: str) -> str:
    if prediction == "defect" and risk_level == "low":
        return RECOMMENDATIONS["medium"]
    return RECOMMENDATIONS[risk_level]


def predict(features: dict[str, float]) -> PredictResponse:
    feature_order = state.schema.get("features", [])
    row = pd.DataFrame([{name: features[name] for name in feature_order}])

    prediction = str(state.model.predict(row)[0])
    prediction_label = LABEL_DISPLAY.get(prediction, prediction.title())

    probabilities: dict[str, float] | None = None
    if hasattr(state.model, "predict_proba"):
        classes = [str(value) for value in state.model.classes_]
        proba = state.model.predict_proba(row)[0]
        probabilities = {
            label: round(float(score), 4) for label, score in zip(classes, proba)
        }

    defect_prob = defect_probability(probabilities)
    risk = risk_level_from_probability(defect_prob, prediction)
    return PredictResponse(
        prediction=prediction,
        prediction_label=prediction_label,
        probabilities=probabilities,
        risk_level=risk,
        recommendation=build_recommendation(risk, prediction),
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="Industrial Health Prediction API",
    description="Manufacturing quality classification inference service.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "model_loaded": str(state.model is not None),
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    return {
        "features": state.schema.get("features", []),
        "classes": state.schema.get("classes", []),
        "metrics": {
            key: state.metrics[key]
            for key in ("accuracy", "f1_macro", "classification_report")
            if key in state.metrics
        },
        "model_type": state.schema.get("model_type"),
    }


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest) -> PredictResponse:
    parsed_features = validate_features(request.features)
    return predict(parsed_features)
