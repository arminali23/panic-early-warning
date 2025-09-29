from fastapi import Depends
from .deps import require_api_key
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import asyncio, time, json, numpy as np

from .utils_audio import load_wav_mono16k
from .features import extract_clip_features
from .model import zscore_vector, risk_from_z
from .state import get_or_create_baseline, set_user_stats, get_user_stats
from .schemas import ScoreResponse, ConfigResponse, ConfigUpdate
from .config import CONFIG
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import JSONResponse
import asyncio, time, json, numpy as np

from .utils_audio import load_wav_mono16k, calib_metrics
from .features import extract_clip_features
from .model import zscore_vector, risk_from_z
from .state import get_or_create_baseline, set_user_stats, get_user_stats
from .schemas import ScoreResponse, ConfigResponse, ConfigUpdate, CalibValidationResponse
from .config import CONFIG
from .deps import require_api_key  # varsa

app = FastAPI(title="Panic Early Warning API", version="0.2.0")

@app.get("/healthz")
def healthz():
    return {"ok": True, "service": "panic-early-warning", "version": "0.2.0"}

# ---- CONFIG ENDPOINTS ----
@app.get("/config", response_model=ConfigResponse)
def get_config():
    return CONFIG.model_dump()

@app.put("/config", response_model=ConfigResponse, dependencies=[Depends(require_api_key)])
def update_config(update: ConfigUpdate):
    data = update.model_dump(exclude_none=True)
    for k, v in data.items():
        setattr(CONFIG, k, v)
    return CONFIG.model_dump()
# --------------------------

@app.post("/calibrate")
async def calibrate(user_id: str = Form(...), wav: UploadFile = File(...)):
    data = await wav.read()
    y, sr = load_wav_mono16k(data)
    feats = extract_clip_features(y, sr)

    bl = get_or_create_baseline(user_id)
    for _ in range(30):
        bl.update(feats)
    mu, std = bl.stats()
    set_user_stats(user_id, mu, std)
    return {"user_id": user_id, "message": "calibrated", "frames": bl.n}

@app.post("/score", response_model=ScoreResponse)
async def score(user_id: str = Form(...), wav: UploadFile = File(...)):
    if user_id not in USER_STATS:
        return JSONResponse(status_code=400, content={"error": "user not calibrated"})
    pair = get_user_stats(user_id)
    if not pair:
        return JSONResponse(status_code=400, content={"error": "user not calibrated"})
    mu, std = pair
    data = await wav.read()
    y, sr = load_wav_mono16k(data)
    feats = extract_clip_features(y, sr)
    z = zscore_vector(feats, mu, std)
    risk = risk_from_z(z)
    return ScoreResponse(user_id=user_id, risk=risk, details=z)

# ---- STREAM W/ ALERT POLICY ----
@app.websocket("/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    try:
        init_msg = await websocket.receive_text()
        cfg = json.loads(init_msg)
        user_id = cfg.get("user_id")
        if not user_id or user_id not in USER_STATS:
            await websocket.send_text(json.dumps({"error": "user not calibrated"}))
            await websocket.close()
            return

        pair = get_user_stats(user_id)
        if not pair:
            return JSONResponse(status_code=400, content={"error": "user not calibrated"})
        mu, std = pair
        last_alert_time = 0.0
        above_since = None

        while True:
            msg = await websocket.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                x = np.frombuffer(msg["bytes"], dtype=np.float32)
                feats = extract_clip_features(x, sr=CONFIG.SR, frame_sec=CONFIG.FRAME_SEC)
                z = zscore_vector(feats, mu, std)
                risk = float(risk_from_z(z))

                now = time.time()
                in_cooldown = (now - last_alert_time) < CONFIG.COOLDOWN_SECONDS
                alert = False

                if risk >= CONFIG.ALERT_THRESHOLD and not in_cooldown:
                    if above_since is None:
                        above_since = now
                    if (now - above_since) >= CONFIG.MIN_ALERT_SECONDS:
                        alert = True
                        last_alert_time = now
                        above_since = None
                else:
                    above_since = None

                await websocket.send_text(json.dumps({"risk": risk, "alert": alert}))
            else:
                await asyncio.sleep(0.001)
    except WebSocketDisconnect:
        pass
