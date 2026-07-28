"""Prediction routes."""
from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException
from backend.models.schemas import PredictRequest
from prediction.infer import prediction_engine

logger = logging.getLogger("factorymind")

router = APIRouter()


@router.post("/predict")
async def predict_failure(req: PredictRequest):
    """Predict machine failure based on sensor data."""
    try:
        res = prediction_engine.predict(
            air_temp=req.air_temp,
            process_temp=req.process_temp,
            rotational_speed=req.rotational_speed,
            torque=req.torque,
            tool_wear=req.tool_wear
        )
        return res
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
