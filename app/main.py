from fastapi import FastAPI

app = FastAPI(title="Panic Early Warning API", version="0.0.1")

@app.get("/healthz")
def healthz():
    return {"ok": True, "service": "panic-early-warning", "version": "0.0.1"}
