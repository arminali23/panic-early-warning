from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import numpy as np
import json

from .features import extract_clip_features
from .model import zscore_vector, risk_from_z
from .state import USER_STATS, get_or_create_baseline, USER_BASELINES

@app.websocket("/stream")
async def stream(websocket: WebSocket):
    """
    WebSocket protocol:
    - Client first sends a JSON text: {"user_id":"demo"}
    - Server checks if calibrated. If not, sends error and closes.
    - Then client continuously sends binary chunks (float32 PCM 16kHz mono).
    - Server responds with {"risk": value} for each chunk.
    """
    await websocket.accept()
    try:
        # Receive initial config
        msg = await websocket.receive_text()
        cfg = json.loads(msg)
        user_id = cfg.get("user_id")
        if not user_id or user_id not in USER_STATS:
            await websocket.send_text(json.dumps({"error": "user not calibrated"}))
            await websocket.close()
            return
        mu, std = USER_STATS[user_id]

        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"] is not None:
                buf = message["bytes"]
                x = np.frombuffer(buf, dtype=np.float32)
                feats = extract_clip_features(x, sr=16000, frame_sec=0.5)
                z = zscore_vector(feats, mu, std)
                risk = risk_from_z(z)
                await websocket.send_text(json.dumps({"risk": risk}))
            else:
                await asyncio.sleep(0.001)
    except WebSocketDisconnect:
        print("Client disconnected")
