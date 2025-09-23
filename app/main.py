from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from .utils_audio import load_wav_mono16k
from .features import extract_clip_features
from .model import zscore_vector, risk_from_z
from .state import get_or_create_baseline, USER_STATS
from .schemas import ScoreResponse

app = FastAPI(title="Panic Early Warning API", version="0.1.0")

@app.get("/healthz")
def healthz():
    return {"ok": True, "service": "panic-early-warning", "version": "0.1.0"}

@app.post("/calibrate")
async def calibrate(user_id: str = Form(...), wav: UploadFile = File(...)):
    """
    Takes a 'calm/normal' baseline recording (.wav) and stores per-feature mean/std
    for this user. Use at least ~60–120 seconds for better stability.
    """
    data = await wav.read()
    y, sr = load_wav_mono16k(data)
    feats = extract_clip_features(y, sr)

    bl = get_or_create_baseline(user_id)
    # Simpler: update baseline with same aggregate features multiple times
    for _ in range(30):
        bl.update(feats)
    mu, std = bl.stats()
    USER_STATS[user_id] = (mu, std)

    return {"user_id": user_id, "message": "calibrated", "frames": bl.n}

@app.post("/score", response_model=ScoreResponse)
async def score(user_id: str = Form(...), wav: UploadFile = File(...)):
    """
    Computes a rule-based risk score using the user's baseline stats.
    """
    if user_id not in USER_STATS:
        return JSONResponse(status_code=400, content={"error": "user not calibrated"})

    mu, std = USER_STATS[user_id]
    data = await wav.read()
    y, sr = load_wav_mono16k(data)
    feats = extract_clip_features(y, sr)
    z = zscore_vector(feats, mu, std)
    risk = risk_from_z(z)
    return ScoreResponse(user_id=user_id, risk=risk, details=z)
