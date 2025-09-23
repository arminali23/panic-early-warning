# app/features.py
import numpy as np
import librosa
from scipy.signal import hilbert, find_peaks

SR = 16000

def extract_clip_features(y: np.ndarray, sr: int = SR, frame_sec: float = 0.5) -> dict:
    """
    Minimal, robust features aggregated over 0.5s frames:
    - RMS, ZCR, spectral centroid/rolloff
    - f0 mean/std (yin)
    - breathing irregularity (envelope peak interval std)
    Returns aggregated stats: *_mean, *_std, *_max
    """
    y = np.asarray(y, dtype=np.float32)
    frame_len = int(sr*frame_sec)
    if len(y) < frame_len:
        return _agg([_frame_feats(y, sr)])

    feats = []
    for i in range(0, len(y) - frame_len + 1, frame_len):
        seg = y[i:i+frame_len]
        feats.append(_frame_feats(seg, sr))
    return _agg(feats)

def _frame_feats(x: np.ndarray, sr: int) -> dict:
    x = np.nan_to_num(x)

    rms = float(np.sqrt(np.mean(x**2)) + 1e-9)
    zcr = float(((x[:-1]*x[1:]) < 0).mean()) if len(x) > 1 else 0.0

    S = np.abs(librosa.stft(x, n_fft=512, hop_length=max(1, len(x)//2))) + 1e-9
    centroid = float(librosa.feature.spectral_centroid(S=S, sr=sr).mean())
    rolloff = float(librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85).mean())

    try:
        f0 = librosa.yin(x, fmin=50, fmax=400, sr=sr,
                         frame_length=min(len(x), 2048),
                         hop_length=max(1, len(x)//4))
        f0_mean = float(np.nanmean(f0)) if np.isfinite(f0).any() else 0.0
        f0_std  = float(np.nanstd(f0))  if np.isfinite(f0).any() else 0.0
    except Exception:
        f0_mean, f0_std = 0.0, 0.0

    env = np.abs(hilbert(x))
    peaks, _ = find_peaks(env, distance=int(0.2*sr))
    breath_irreg = 0.0
    if len(peaks) > 2:
        intervals = np.diff(peaks)/sr
        breath_irreg = float(np.std(intervals))

    return dict(
        rms=rms, zcr=zcr, centroid=centroid, rolloff=rolloff,
        f0_mean=f0_mean, f0_std=f0_std, breath_irreg=breath_irreg
    )

def _agg(frames):
    keys = frames[0].keys()
    out = {}
    for k in keys:
        arr = np.array([f[k] for f in frames], dtype=np.float32)
        out[f"{k}_mean"] = float(arr.mean())
        out[f"{k}_std"]  = float(arr.std())
        out[f"{k}_max"]  = float(arr.max())
    return out
