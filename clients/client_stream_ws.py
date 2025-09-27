import asyncio, json
import websockets
import sounddevice as sd
import numpy as np

SR = 16000
FRAME_SEC = 0.5
FRAME_SAMPLES = int(SR * FRAME_SEC)

async def main():
    uri = "ws://localhost:8000/stream"
    async with websockets.connect(uri, max_size=None) as ws:
        await ws.send(json.dumps({"user_id": "demo"}))

        def frames():
            with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=FRAME_SAMPLES):
                while True:
                    data, _ = sd.rec(FRAME_SAMPLES, samplerate=SR, channels=1, dtype='float32'), sd.wait()
                    yield data.flatten()

        async def sender():
            for x in frames():
                await ws.send(x.tobytes())

        async def receiver():
            async for msg in ws:
                try:
                    payload = json.loads(msg)
                    risk = payload.get("risk")
                    alert = payload.get("alert")
                    if alert:
                        print(f"[ALERT] risk={risk:.2f}  → Suggest: start breathing exercise")
                    else:
                        print(f"risk={risk:.2f}")
                except Exception:
                    print("server:", msg)

        await asyncio.gather(sender(), receiver())

if __name__ == "__main__":
    asyncio.run(main())
