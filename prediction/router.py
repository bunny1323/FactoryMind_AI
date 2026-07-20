from fastapi import APIRouter, Depends
from prediction.dto import PredictionRequestDTO, PredictionResponseDTO
from prediction.service import PredictionService
from prediction.repository import PredictionRepository

router = APIRouter(prefix="/prediction", tags=["prediction"])

# Dependencies
repo = PredictionRepository()
service = PredictionService(repo)

@router.post("/predict", response_model=PredictionResponseDTO)
async def predict_telemetry(req: PredictionRequestDTO):
    result = service.get_prediction(req.machine_id, req.sensor_values)
    return PredictionResponseDTO(
        status="disconnected",
        message=result["status"],
        iot_ready=True,
        payload=result
    )
