from __future__ import annotations

import logging
from typing import Any
from prediction.repository import PredictionRepository
from prediction.service import PredictionService

logger = logging.getLogger("factorymind")

class PredictionEngine:
    def __init__(self):
        self.repository = PredictionRepository()
        self.service = PredictionService(self.repository)
        logger.info("Decoupled PredictionEngine initialized with service and repository.")

    def load_model(self):
        # NOP: model training/loading is disabled per user specs
        pass

    def predict(
        self, 
        air_temp: float, 
        process_temp: float, 
        rotational_speed: float, 
        torque: float, 
        tool_wear: float
    ) -> dict[str, Any]:
        """Returns standard decoupled architecture state representing unconnected IoT channels."""
        sensor_values = {
            "air_temperature": air_temp,
            "process_temperature": process_temp,
            "rotational_speed": rotational_speed,
            "torque": torque,
            "tool_wear": tool_wear,
            "vibration": 0.0
        }
        
        result = self.service.get_prediction("M101", sensor_values)
        
        return {
            "failure_probability": 0.0,
            "failure_type": "None",
            "confidence": 0.0,
            "explanation": result["status"],
            "telemetry": sensor_values
        }

prediction_engine = PredictionEngine()
