from fastapi import WebSocket, WebSocketDisconnect
import asyncio, time, json, numpy as np

from .features import extract_clip_features
from .model import zscore_vector, risk_from_z
from .state import USER_STATS
from .config import ALERT_THRESHOLD, MIN_ALERT_SECONDS, COOLDOWN_SECONDS, SR, FRAME_SEC

@app.websocket("/stream")
async def stream(websocket: WebSocket):
    """
    Protokol:
    1) İlk mesaj (text JSON): {"user_id":"demo"}
    2) Sonra client 0.5 sn'lik float32 PCM (16kHz mono) binary chunk'lar gönderir.
    3) Server her chunk için {"risk": 0.xx, "alert": true/false} döner.
    """
    await websocket.accept()
    try:
        init_msg = await websocket.receive_text()
        cfg = json.loads(init_msg)
        user_id = cfg.get("user_id")

        if not user_id or user_id not in USER_STATS:
            await websocket.send_text(json.dumps({"error": "user not calibrated"}))
            await websocket.close()
            return

        mu, std = USER_STATS[user_id]

        last_alert_time = 0.0
        above_since = None

        while True:
            msg = await websocket.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                x = np.frombuffer(msg["bytes"], dtype=np.float32)
                feats = extract_clip_features(x, sr=SR, frame_sec=FRAME_SEC)
                z = zscore_vector(feats, mu, std)
                risk = float(risk_from_z(z))

                now = time.time()
                in_cooldown = (now - last_alert_time) < COOLDOWN_SECONDS
                alert = False

                if risk >= ALERT_THRESHOLD and not in_cooldown:
                    if above_since is None:
                        above_since = now
                    if (now - above_since) >= MIN_ALERT_SECONDS:
                        alert = True
                        last_alert_time = now
                        above_since = None
                else:
                    above_since = None

                await websocket.send_text(json.dumps({
                    "risk": risk,
                    "alert": alert
                }))

            else:
                await asyncio.sleep(0.001)

    except WebSocketDisconnect:
        pass
