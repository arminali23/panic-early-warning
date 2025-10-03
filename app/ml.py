# app/ml.py
import joblib
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import numpy as np

@dataclass
class LightModel:
    scaler: Any
    clf: Any
    feature_names: List[str]
    label_map: Dict[int, str]  # e.g., {0:"low",1:"med",2:"high"}

_LOADED: Optional[LightModel] = None
_USE_MODEL: bool = True  # runtime toggle

def load_model(path: str) -> dict:
    global _LOADED
    bundle = joblib.load(path)
    _LOADED = LightModel(
        scaler=bundle["scaler"],
        clf=bundle["clf"],
        feature_names=bundle["feature_names"],
        label_map=bundle.get("label_map", {0:"low",1:"high"})
    )
    return info()

def unload_model() -> dict:
    global _LOADED
    _LOADED = None
    return info()

def set_use_model(flag: bool) -> dict:
    global _USE_MODEL
    _USE_MODEL = bool(flag)
    return info()

def info() -> dict:
    return {
        "loaded": _LOADED is not None,
        "use_model": _USE_MODEL,
        "feature_count": len(_LOADED.feature_names) if _LOADED else 0,
        "classes": list(_LOADED.label_map.values()) if _LOADED else [],
    }

def vectorize_feats(feats: dict, feature_names: List[str]) -> np.ndarray:
    """Order features consistently for the model."""
    x = np.array([float(feats.get(k, 0.0)) for k in feature_names], dtype=np.float32)
    return x.reshape(1, -1)

def predict_proba(feats: dict) -> Optional[float]:
    """Return panic/anxiety risk in [0,1] from model if loaded; else None."""
    if _LOADED is None or not _USE_MODEL:
        return None
    x = vectorize_feats(feats, _LOADED.feature_names)
    xs = _LOADED.scaler.transform(x)
    prob = _LOADED.clf.predict_proba(xs)[0]
    # If binary, take prob of positive class; if multi, take weighted sum (high=1, med=0.5, low=0)
    if len(prob) == 2:
        return float(prob[1])
    # multi-class mapping
    score = 0.0
    for i, p in enumerate(prob):
        name = _LOADED.label_map.get(i, str(i)).lower()
        weight = 1.0 if "high" in name else (0.5 if "med" in name or "moderate" in name else 0.0)
        score += weight * float(p)
    return max(0.0, min(1.0, score))
    
def predict_details(feats: dict) -> Optional[dict]:
    """
    If model is loaded/active: return {"risk": float, "probs": [..], "classes":[..]}
    else: None
    """
    if _LOADED is None or not _USE_MODEL:
        return None
    x = vectorize_feats(feats, _LOADED.feature_names)
    xs = _LOADED.scaler.transform(x)
    probs = _LOADED.clf.predict_proba(xs)[0].tolist()
    classes = [_LOADED.label_map.get(i, str(i)) for i in range(len(probs))]

    # Binary: prob of positive class
    if len(probs) == 2:
        risk = float(probs[1])
    else:
        # multi-class weighted (low=0, med=0.5, high=1)
        risk = 0.0
        for i, p in enumerate(probs):
            name = classes[i].lower()
            w = 1.0 if "high" in name else (0.5 if ("med" in name or "moderate" in name) else 0.0)
            risk += w * float(p)
        risk = max(0.0, min(1.0, risk))
    return {"risk": risk, "probs": probs, "classes": classes}
