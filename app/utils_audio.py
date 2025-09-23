# app/utils_audio.py
import io
import numpy as np
import soundfile as sf
import librosa

TARGET_SR = 16000

def load_wav_mono16k(file_bytes: bytes):
    """Reads arbitrary WAV and returns mono float32 at 16kHz."""
    y, sr = sf.read(io.BytesIO(file_bytes), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    return y, sr
