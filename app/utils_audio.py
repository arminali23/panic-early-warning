# app/utils_audio.py
import io
import numpy as np
import soundfile as sf
import librosa

TARGET_SR = 16000

def load_wav_mono16k(file_bytes: bytes):
    y, sr = sf.read(io.BytesIO(file_bytes), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    return y, sr

def calib_metrics(y: np.ndarray, sr: int):
    """Kalibrasyon için basit kalite metrikleri döndürür."""
    dur = len(y) / float(sr)
    rms = float(np.sqrt(np.mean(y**2)) + 1e-12)

    clip_ratio = float(np.mean(np.abs(y) > 0.98))


    abs_y = np.abs(y)
    noise_est = float(np.percentile(abs_y, 20)) + 1e-12
    snr_db = 20.0 * float(np.log10(max(rms, 1e-12) / noise_est))

    return {
        "duration_sec": dur,
        "rms": rms,
        "clip_ratio": clip_ratio,
        "snr_db": snr_db,
    }
