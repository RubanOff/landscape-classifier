from pydantic import BaseModel


class PredictionResponse(BaseModel):
    class_id: int
    class_name: str
    confidence: float
