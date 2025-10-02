# app/utils_audio.py
import io
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, lfilter

TARGET_SR = 16000

def load_wav_mono16k(file_bytes: bytes):
    y, sr = sf.read(io.BytesIO(file_bytes), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    return y, sr

def pre_emphasis(y: np.ndarray, coeff: float = 0.97):
    if len(y) < 2:
        return y.astype("float32")
    out = np.empty_like(y, dtype=np.float32)
    out[0] = y[0]
    out[1:] = y[1:] - coeff * y[:-1]
    return out

def highpass(y: np.ndarray, sr: int, cutoff_hz: float = 50.0, order: int = 4):
    if cutoff_hz <= 0.0:
        return y.astype("float32")
    nyq = 0.5 * sr
    b, a = butter(order, cutoff_hz / nyq, btype="high", analog=False)
    return lfilter(b, a, y).astype("float32")

def apply_preprocess(y: np.ndarray, sr: int, use_preemph: bool, pre_c: float, use_hpf: bool, hpf_cut: float):
    x = y.astype("float32")
    if use_hpf:
        x = highpass(x, sr, cutoff_hz=hpf_cut)
    if use_preemph:
        x = pre_emphasis(x, coeff=pre_c)
    return x

def calib_metrics(y: np.ndarray, sr: int):
    dur = len(y) / float(sr)
    rms = float(np.sqrt(np.mean(y**2)) + 1e-12)
    clip_ratio = float(np.mean(np.abs(y) > 0.98))
    abs_y = np.abs(y)
    noise_est = float(np.percentile(abs_y, 20)) + 1e-12
    snr_db = 20.0 * float(np.log10(max(rms, 1e-12) / noise_est))
    return {"duration_sec": dur, "rms": rms, "clip_ratio": clip_ratio, "snr_db": snr_db}
