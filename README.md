# Panic Early Warning (Audio-based)

A prototype project that aims to detect **early signs of panic attacks** through real-time audio analysis. The system listens to the user’s voice and breathing patterns, extracts acoustic features (e.g., energy, pitch, jitter, breathing irregularities), and computes a **personalized risk score**.  

Disclaimer: This software is **not a medical device** and does not provide medical diagnosis. It is intended for research and prototyping purposes only. Always seek professional help in case of emergency.

---

## Features (Planned Roadmap)

- **MVP**
  - `GET /healthz`: basic health check  
  - Dockerized FastAPI service  

- **Next Iterations**
  - `POST /calibrate`: baseline recording to capture personal “normal” audio profile  
  - `POST /score`: compute anxiety/panic risk from a short `.wav` clip  
  - `WebSocket /stream`: stream 0.5s audio chunks for real-time scoring  
  - Redis-based state sharing for multiple instances  
  - Upgrade from rule-based risk to lightweight ML models (LogReg / LightGBM / 1D-CNN)  

---

## Architecture

- **FastAPI** for API endpoints  
- **Docker** for containerized deployment  
- **In-memory state** for calibration (MVP), later Redis for scalability  
- **Audio feature extraction** (librosa, soundfile, scipy)  
- **Risk scoring**: initially rule-based with z-scores → later ML models  

---

## Getting Started

### 1. Run Locally (without Docker)
```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
