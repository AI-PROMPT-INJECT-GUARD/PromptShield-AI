from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class PredictRequest(BaseModel):
    prompt: str


class PredictResponse(BaseModel):
    prompt: str
    label: str
    is_injection: bool
    confidence: float
    attack_category: Optional[str] = None
    explanation: Optional[str] = None
    safe_prompt: Optional[str] = None


class HistoryItem(PredictResponse):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True