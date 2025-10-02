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
    MIN_SCORE_SECONDS: float
    USE_PREEMPHASIS: bool
    PREEMPHASIS_COEFF: float
    USE_HPF: bool
    HPF_CUTOFF_HZ: float

class ConfigUpdate(BaseModel):
    ALERT_THRESHOLD: float | None = Field(None, ge=0.0, le=1.0)
    MIN_ALERT_SECONDS: float | None = Field(None, ge=0.0)
    COOLDOWN_SECONDS: float | None = Field(None, ge=0.0)
    MIN_SCORE_SECONDS: float | None = Field(None, ge=0.5)
    USE_PREEMPHASIS: bool | None = None
    PREEMPHASIS_COEFF: float | None = Field(None, ge=0.0, le=1.0)
    USE_HPF: bool | None = None
    HPF_CUTOFF_HZ: float | None = Field(None, ge=0.0)
