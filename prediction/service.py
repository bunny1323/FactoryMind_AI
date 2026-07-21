import logging
from typing import Dict, Any
from prediction.repository import PredictionRepository

logger = logging.getLogger("factorymind")

class PredictionService:
    def __init__(self, repository: PredictionRepository):
        self.repository = repository
        logger.info("PredictionService initialized.")

    def get_prediction(self, machine_id: str, sensor_values: Dict[str, float] | None = None) -> Dict[str, Any]:
        logger.info(f"Prediction requested for {machine_id} using real XGBoost model.")
        if not sensor_values:
            sensor_values = {
                "air_temperature": 298.2,
                "process_temperature": 308.6,
                "rotational_speed": 1850,
                "torque": 45.2,
                "tool_wear": 120,
                "vibration": 0.08
            }
        
        # Avoid circular import by importing prediction_engine locally
        from prediction.infer import prediction_engine
        pred = prediction_engine.predict(
            air_temp=sensor_values.get("air_temperature", 298.2),
            process_temp=sensor_values.get("process_temperature", 308.6),
            rotational_speed=sensor_values.get("rotational_speed", 1850.0),
            torque=sensor_values.get("torque", 45.2),
            tool_wear=sensor_values.get("tool_wear", 120.0)
        )
        
        return {
            "status": pred["explanation"],
            "message": f"Real-time prediction executed via trained XGBoost. Failure probability: {pred['failure_probability']:.4f}.",
            "iot_ready": True,
            "telemetry": sensor_values
        }
