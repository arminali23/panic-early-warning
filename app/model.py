# app/model.py
import math

class Baseline:
    """Welford online mean/variance for per-feature baseline."""
    def __init__(self):
        self.n = 0
        self.mu = {}
        self.M2 = {}

    def update(self, feats: dict):
        self.n += 1
        for k, v in feats.items():
            v = float(v)
            if k not in self.mu:
                self.mu[k] = v
                self.M2[k] = 0.0
            else:
                delta = v - self.mu[k]
                self.mu[k] += delta / self.n
                self.M2[k] += delta * (v - self.mu[k])

    def stats(self):
        std = {k: math.sqrt(self.M2[k] / max(1, self.n - 1)) + 1e-9 for k in self.mu}
        return self.mu, std

def zscore_vector(x: dict, mu: dict, std: dict):
    z = {}
    for k, v in x.items():
        m = mu.get(k, 0.0); s = std.get(k, 1.0)
        z[k] = (float(v) - m) / max(1e-9, s)
    return z

def risk_from_z(z: dict) -> float:
    """
    Simple rule-based risk: average positive z-scores of key features,
    then logistic squashing to [0,1].
    """
    sel = []
    for k, v in z.items():
        if any(k.startswith(p) for p in ["rms", "zcr", "centroid", "rolloff", "f0_std", "breath_irreg"]):
            sel.append(max(0.0, min(4.0, float(v))))  # clip outliers
    if not sel:
        return 0.0
    s = sum(sel)/len(sel)
    return 1.0 / (1.0 + math.exp(-(s - 1.0)))
