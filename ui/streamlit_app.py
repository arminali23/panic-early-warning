import asyncio
import json
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd
import streamlit as st
import websockets

API_WS = "ws://localhost:8000/stream"
SR = 16000
FRAME_SEC = 0.5
FRAME_SAMPLES = int(SR * FRAME_SEC)

st.set_page_config(page_title="Panic Early Warning", layout="centered")
st.title("Panic Early Warning — Live Risk Monitor")

user_id = st.text_input("User ID (must be calibrated)", value="demo")
start_btn = st.button("Start")
stop_btn = st.button("Stop")

risk_placeholder = st.empty()
alert_placeholder = st.empty()
chart_placeholder = st.empty()

risk_buf = deque(maxlen=200)  # son ~100 sn
time_buf = deque(maxlen=200)

state = {"running": False, "thread": None}

def audio_frames():
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=FRAME_SAMPLES):
        while state["running"]:
            data, _ = sd.rec(FRAME_SAMPLES, samplerate=SR, channels=1, dtype="float32"), sd.wait()
            yield data.flatten()

async def run_ws():
    try:
        async with websockets.connect(API_WS, max_size=None) as ws:
            await ws.send(json.dumps({"user_id": user_id}))
            # background task: sender
            async def sender():
                for x in audio_frames():
                    await ws.send(x.tobytes())
                    await asyncio.sleep(0.0)
            # receiver: update UI buffers
            async def receiver():
                async for msg in ws:
                    try:
                        payload = json.loads(msg)
                        risk = float(payload.get("risk", 0.0))
                        alert = bool(payload.get("alert", False))
                        risk_buf.append(risk)
                        time_buf.append(time.time())
                        # UI updates
                        risk_placeholder.metric("Risk", f"{risk:.2f}")
                        if alert:
                            alert_placeholder.warning("ALERT: sustained high risk — suggest breathing exercise")
                        else:
                            alert_placeholder.info("Monitoring…")
                        chart_placeholder.line_chart(list(risk_buf))
                    except Exception:
                        pass
            await asyncio.gather(sender(), receiver())
    except Exception as e:
        alert_placeholder.error(f"Stream error: {e}")

def run_loop():
    asyncio.run(run_ws())

if start_btn and not state["running"]:
    state["running"] = True
    t = threading.Thread(target=run_loop, daemon=True)
    state["thread"] = t
    t.start()

if stop_btn and state["running"]:
    state["running"] = False
    alert_placeholder.info("Stopped.")
