from pydantic import BaseModel, Field

class RuntimeConfig(BaseModel):
    ALERT_THRESHOLD: float = Field(0.72, ge=0.0, le=1.0)
    MIN_ALERT_SECONDS: float = Field(8.0, ge=0.0)
    COOLDOWN_SECONDS: float = Field(120.0, ge=0.0)
    SR: int = 16000
    FRAME_SEC: float = 0.5

    # Calibration quality thresholds
    MIN_CALIB_SECONDS: float = Field(60.0, ge=1.0)
    MIN_RMS: float = Field(0.005, ge=0.0)          # çok sessiz kayıtları ele
    MAX_CLIP_RATIO: float = Field(0.02, ge=0.0)    # |x|>0.98 örnek oranı
    MIN_SNR_DB: float = Field(12.0)                # kaba SNR eşiği (dB)

CONFIG = RuntimeConfig()
