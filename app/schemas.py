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
    MIN_CALIB_SECONDS: float
    MIN_RMS: float
    MAX_CLIP_RATIO: float
    MIN_SNR_DB: float

class ConfigUpdate(BaseModel):
    ALERT_THRESHOLD: float | None = Field(None, ge=0.0, le=1.0)
    MIN_ALERT_SECONDS: float | None = Field(None, ge=0.0)
    COOLDOWN_SECONDS: float | None = Field(None, ge=0.0)

class CalibValidationResponse(BaseModel):
    ok: bool
    metrics: dict
    failed_checks: list[str] = []
    hints: list[str] = []
