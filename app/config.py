from pydantic import BaseModel, Field

class RuntimeConfig(BaseModel):
    ALERT_THRESHOLD: float = Field(0.72, ge=0.0, le=1.0)
    MIN_ALERT_SECONDS: float = Field(8.0, ge=0.0)
    COOLDOWN_SECONDS: float = Field(120.0, ge=0.0)
    SR: int = 16000
    FRAME_SEC: float = 0.5

    # Calibration quality
    MIN_CALIB_SECONDS: float = Field(60.0, ge=1.0)
    MIN_RMS: float = Field(0.005, ge=0.0)
    MAX_CLIP_RATIO: float = Field(0.02, ge=0.0)
    MIN_SNR_DB: float = Field(12.0)

    # Scoring quality
    MIN_SCORE_SECONDS: float = Field(5.0, ge=0.5)

    # Preprocessing
    USE_PREEMPHASIS: bool = True
    PREEMPHASIS_COEFF: float = Field(0.97, ge=0.0, le=1.0)
    USE_HPF: bool = True
    HPF_CUTOFF_HZ: float = Field(50.0, ge=0.0)  
