# app/schemas.py
from pydantic import BaseModel, Field

class ScoreResponse(BaseModel):
    user_id: str
    risk: float
    details: dict

class ConfigResponse(BaseModel):
    ALERT_THRESHOLD: float
    MIN_ALERT_SECONDS: float
    COOLDOWN_SECONDS: float
    SR: int
    FRAME_SEC: float

class ConfigUpdate(BaseModel):
    ALERT_THRESHOLD: float | None = Field(None, ge=0.0, le=1.0)
    MIN_ALERT_SECONDS: float | None = Field(None, ge=0.0)
    COOLDOWN_SECONDS: float | None = Field(None, ge=0.0)
