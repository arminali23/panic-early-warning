# app/config.py
from pydantic import BaseModel, Field

class RuntimeConfig(BaseModel):
    ALERT_THRESHOLD: float = Field(0.72, ge=0.0, le=1.0)
    MIN_ALERT_SECONDS: float = Field(8.0, ge=0.0)
    COOLDOWN_SECONDS: float = Field(120.0, ge=0.0)
    SR: int = 16000
    FRAME_SEC: float = 0.5

# mutable runtime config (in-memory)
CONFIG = RuntimeConfig()
