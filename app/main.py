from fastapi import (
    FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect,
    Depends, Body, Request, Response
)
from fastapi.responses import JSONResponse
import asyncio, time, json
import numpy as np
from .deps import require_api_key
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import PlainTextResponse

from .logging_utils import log_event, dump_logs, new_req_id
from .utils_audio import (
    load_wav_mono16k, calib_metrics, apply_preprocess
)
from .features import extract_clip_features
from .model import zscore_vector, risk_from_z
from .state import get_or_create_baseline, set_user_stats, get_user_stats
from .schemas import (
    ScoreResponse, ConfigResponse, ConfigUpdate, CalibValidationResponse, ModelInfo
)
from .config import CONFIG
from .ml import (
    load_model, unload_model, set_use_model,
    info as model_info, predict_details
)
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from .metrics import REQUESTS, LATENCY, SCORE_SOURCE, STREAM_ALERTS


app = FastAPI(title="Panic Early Warning API", version="0.3.0")


# -------------------------
# Health
# -------------------------
@app.get("/healthz")
def healthz():
    return {"ok": True, "service": "panic-early-warning", "version": "0.3.0"}


# -------------------------
# Config (GET/PUT)
# -------------------------
@app.get("/config", response_model=ConfigResponse)
def get_config():
    return CONFIG.model_dump()


@app.put("/config", response_model=ConfigResponse, dependencies=[Depends(require_api_key)])
def update_config(update: ConfigUpdate):
    data = update.model_dump(exclude_none=True)
    for k, v in data.items():
        setattr(CONFIG, k, v)
    return CONFIG.model_dump()


# -------------------------
# Calibration quality check (no state change)
# -------------------------
@app.post("/validate-calib", response_model=CalibValidationResponse)
async def validate_calib(wav: UploadFile = File(...)):
    data = await wav.read()
    y, sr = load_wav_mono16k(data)
    m = calib_metrics(y, sr)

    failed, hints = [], []

    if m["duration_sec"] < CONFIG.MIN_CALIB_SECONDS:
        failed.append(f"duration<{CONFIG.MIN_CALIB_SECONDS}s")
        hints.append("Record at least 60–120 seconds of calm, steady breathing/speaking.")

    if m["rms"] < CONFIG.MIN_RMS:
        failed.append("too_quiet")
        hints.append("Move closer to the mic or increase input gain; avoid whispering.")

    if m["clip_ratio"] > CONFIG.MAX_CLIP_RATIO:
        failed.append("clipping")
        hints.append("Reduce input gain; avoid shouting or saturating the microphone.")

    if m["snr_db"] < CONFIG.MIN_SNR_DB:
        failed.append("low_snr")
        hints.append("Reduce background noise (fan/traffic), use a quieter room or headset mic.")

    ok = len(failed) == 0
    return CalibValidationResponse(ok=ok, metrics=m, failed_checks=failed, hints=hints)


# -------------------------
# Calibrate (writes baseline)
# -------------------------
@app.post("/calibrate")
async def calibrate(user_id: str = Form(...), wav: UploadFile = File(...)):
    data = await wav.read()
    y, sr = load_wav_mono16k(data)

    # Quality checks on raw signal
    m = calib_metrics(y, sr)
    failed = []
    if m["duration_sec"] < CONFIG.MIN_CALIB_SECONDS:
        failed.append(f"duration<{CONFIG.MIN_CALIB_SECONDS}s")
    if m["rms"] < CONFIG.MIN_RMS:
        failed.append("too_quiet")
    if m["clip_ratio"] > CONFIG.MAX_CLIP_RATIO:
        failed.append("clipping")
    if m["snr_db"] < CONFIG.MIN_SNR_DB:
        failed.append("low_snr")

    if failed:
        return JSONResponse(
            status_code=400,
            content={
                "error": "calibration_failed",
                "metrics": m,
                "failed_checks": failed,
                "hint": "Use /validate-calib to iterate until quality checks pass."
            }
        )

    # Preprocess for feature extraction (keep raw for quality metrics)
    x = apply_preprocess(
        y, sr,
        use_preemph=CONFIG.USE_PREEMPHASIS, pre_c=CONFIG.PREEMPHASIS_COEFF,
        use_hpf=CONFIG.USE_HPF, hpf_cut=CONFIG.HPF_CUTOFF_HZ,
        use_denoise=CONFIG.USE_DENOISE, denoise_strength=CONFIG.DENOISE_STRENGTH
    )

    feats = extract_clip_features(x, sr)
    bl = get_or_create_baseline(user_id)
    # Update baseline stats a few times for stability (same aggregate features)
    for _ in range(30):
        bl.update(feats)
    mu, std = bl.stats()
    set_user_stats(user_id, mu, std)

    return {
        "user_id": user_id,
        "message": "calibrated",
        "frames": bl.n,
        "metrics": m
    }


# -------------------------
# Score (file-based)
# -------------------------
@app.post("/score", response_model=ScoreResponse)
async def score(user_id: str = Form(...), wav: UploadFile = File(...)):
    pair = get_user_stats(user_id)
    if not pair:
        return JSONResponse(status_code=400, content={"error": "user not calibrated"})
    mu, std = pair

    data = await wav.read()
    y, sr = load_wav_mono16k(data)

    # Minimum clip duration
    dur = len(y) / float(sr)
    if dur < CONFIG.MIN_SCORE_SECONDS:
        return JSONResponse(
            status_code=400,
            content={"error": "clip_too_short", "min_seconds": CONFIG.MIN_SCORE_SECONDS, "duration": dur}
        )

    # Preprocess for features
    x = apply_preprocess(
        y, sr,
        use_preemph=CONFIG.USE_PREEMPHASIS, pre_c=CONFIG.PREEMPHASIS_COEFF,
        use_hpf=CONFIG.USE_HPF, hpf_cut=CONFIG.HPF_CUTOFF_HZ,
        use_denoise=CONFIG.USE_DENOISE, denoise_strength=CONFIG.DENOISE_STRENGTH
    )
    feats = extract_clip_features(x, sr)

    # Try ML model → fallback to rule
    md = predict_details(feats)
    if md is not None:
        SCORE_SOURCE.labels(source="model").inc()
        return ScoreResponse(
            user_id=user_id,
            risk=float(md["risk"]),
            details=feats,
            source="model",
            probs=md["probs"]
        )
    else:
        z = zscore_vector(feats, mu, std)
        risk = risk_from_z(z)
        SCORE_SOURCE.labels(source="rule").inc()
        return ScoreResponse(
            user_id=user_id,
            risk=risk,
            details=feats,
            source="rule",
            probs=None
        )


# -------------------------
# Stream (WebSocket) with alert policy
# -------------------------
@app.websocket("/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    try:
        init_msg = await websocket.receive_text()
        cfg = json.loads(init_msg)
        user_id = cfg.get("user_id")

        pair = get_user_stats(user_id) if user_id else None
        if not pair:
            await websocket.send_text(json.dumps({"error": "user not calibrated"}))
            await websocket.close()
            return

        mu, std = pair
        last_alert_time = 0.0
        above_since = None

        while True:
            msg = await websocket.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                # Incoming chunk: float32 PCM @ CONFIG.SR, ~FRAME_SEC seconds
                x = np.frombuffer(msg["bytes"], dtype=np.float32)

                # Preprocess and extract features
                x = apply_preprocess(
                    x, CONFIG.SR,
                    use_preemph=CONFIG.USE_PREEMPHASIS, pre_c=CONFIG.PREEMPHASIS_COEFF,
                    use_hpf=CONFIG.USE_HPF, hpf_cut=CONFIG.HPF_CUTOFF_HZ,
                    use_denoise=CONFIG.USE_DENOISE, denoise_strength=CONFIG.DENOISE_STRENGTH
                )
                feats = extract_clip_features(x, sr=CONFIG.SR, frame_sec=CONFIG.FRAME_SEC)

                # Model → Rule fallback
                md = predict_details(feats)
                if md is not None:
                    risk = float(md["risk"])
                    src = "model"
                else:
                    risk = float(risk_from_z(zscore_vector(feats, mu, std)))
                    src = "rule"

                # Alert policy (hysteresis + cooldown)
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
                        STREAM_ALERTS.inc()
                else:
                    above_since = None

                await websocket.send_text(json.dumps({"risk": risk, "alert": alert, "source": src}))
            else:
                await asyncio.sleep(0.001)
    except WebSocketDisconnect:
        # client disconnected
        pass


# -------------------------
# Model management endpoints
# -------------------------
@app.get("/model/info", response_model=ModelInfo)
def model_get_info():
    return model_info()


@app.post("/model/reload", response_model=ModelInfo)
def model_reload(path: str = Body(..., embed=True)):
    return load_model(path)


@app.post("/model/unload", response_model=ModelInfo)
def model_unload():
    return unload_model()


@app.post("/model/use", response_model=ModelInfo)
def model_use(flag: bool = Body(..., embed=True)):
    return set_use_model(flag)


# -------------------------
# Prometheus metrics
# -------------------------
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    import time as _time
    start = _time.time()
    response = await call_next(request)
    elapsed = _time.time() - start
    endpoint = request.url.path
    LATENCY.labels(endpoint=endpoint, method=request.method).observe(elapsed)
    REQUESTS.labels(endpoint=endpoint, method=request.method, code=str(response.status_code)).inc()
    return response
