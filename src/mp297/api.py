
import logging
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


# ---------------- LOGGING ----------------

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("mp297.api")


# ---------------- MODEL LOADING ----------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = Path(
    os.getenv("MODEL_PATH", str(PROJECT_ROOT / "models" / "model.joblib"))
)

model = None

try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully from %s", MODEL_PATH)
except Exception:
    logger.exception("Failed to load model from %s", MODEL_PATH)


# ---------------- API ----------------

app = FastAPI(
    title="Cold Storage Spoilage Prediction API",
    description="Predicts spoilage percentage using the trained MP297 model.",
    version="1.0.0",
)


# ---------------- REQUEST SCHEMA ----------------

class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    storage_days: float = Field(ge=0)
    temp_set_c: float
    humidity_pct: float = Field(ge=0, le=100)
    power_cuts_hours: float = Field(ge=0)
    bag_weight_kg: float = Field(gt=0)
    door_openings_per_day: float = Field(ge=0)
    crop: str = Field(min_length=1)
    chamber: str = Field(min_length=1)


# ---------------- HEALTH CHECK ----------------

@app.get("/health")
def health():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available",
        )

    return {
        "status": "healthy",
        "model_loaded": True,
    }


# ---------------- PREDICTION ----------------

@app.post("/predict")
def predict(request: PredictRequest):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available",
        )

    try:
        input_data = pd.DataFrame(
            [request.model_dump()]
        )

        prediction = float(model.predict(input_data)[0])

        logger.info(
            "Prediction successful | crop=%s | chamber=%s | spoilage_pct=%.4f",
            request.crop,
            request.chamber,
            prediction,
        )

        return {
            "predicted_spoilage_pct": round(prediction, 4),
            "crop": request.crop,
            "chamber": request.chamber,
        }

    except Exception:
        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed",
        )
