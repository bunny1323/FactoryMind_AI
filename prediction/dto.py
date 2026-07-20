from pydantic import BaseModel
from typing import Optional, Dict, Any

class PredictionRequestDTO(BaseModel):
    machine_id: str
    sensor_values: Optional[Dict[str, float]] = None

class PredictionResponseDTO(BaseModel):
    status: str
    message: str
    iot_ready: bool = True
    payload: Optional[Dict[str, Any]] = None
