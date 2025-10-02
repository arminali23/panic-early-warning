# training/train_light_model.py
import argparse, os, json
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report

import soundfile as sf
import librosa

# Reuse feature pipeline from app (copy-paste minimal to avoid package import issues during training)
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.utils_audio import load_wav_mono16k, apply_preprocess
from app.features import extract_clip_features
from app.config import CONFIG  # to keep same preprocessing defaults

LABELS_BIN = {"low":0, "high":1}
LABELS_MULTI = {"low":0, "med":1, "high":2}

def label_encoder(label_series):
    uniq = sorted(label_series.unique())
    if set(uniq) <= {"low","high"}:
        mapping = LABELS_BIN
    else:
        mapping = LABELS_MULTI
    y = label_series.map(mapping).values
    inv = {v:k for k,v in mapping.items()}
    return y, inv

def build_feature_vector(wav_path: str) -> dict:
    with open(wav_path, "rb") as f:
        data = f.read()
    y, sr = load_wav_mono16k(data)
    x = apply_preprocess(
        y, sr,
        use_preemph=CONFIG.USE_PREEMPHASIS, pre_c=CONFIG.PREEMPHASIS_COEFF,
        use_hpf=CONFIG.USE_HPF, hpf_cut=CONFIG.HPF_CUTOFF_HZ
    )
    feats = extract_clip_features(x, sr)
    return feats

def main(args):
    df = pd.read_csv(args.csv)
    feat_rows = []
    for i, row in df.iterrows():
        feats = build_feature_vector(row["path"])
        feats["__label__"] = row["label"]
        feat_rows.append(feats)

    feat_df = pd.DataFrame(feat_rows)
    y, inv_map = label_encoder(feat_df["__label__"])
    X = feat_df.drop(columns=["__label__"])

    feature_names = list(X.columns)
    X = X.values.astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    clf = LogisticRegression(max_iter=200, class_weight="balanced")
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_val_s)
    y_proba = None
    try:
        y_proba = clf.predict_proba(X_val_s)[:, -1]
    except Exception:
        pass

    print("Accuracy:", accuracy_score(y_val, y_pred))
    if y_proba is not None and len(np.unique(y)) == 2:
        print("ROC AUC:", roc_auc_score(y_val, y_proba))
    print("F1(macro):", f1_score(y_val, y_pred, average="macro"))
    print(classification_report(y_val, y_pred, target_names=[inv_map[i] for i in sorted(inv_map)]))

    bundle = {
        "scaler": scaler,
        "clf": clf,
        "feature_names": feature_names,
        "label_map": {i:inv_map[i] for i in inv_map},
        "config_snapshot": CONFIG.model_dump(),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    dump(bundle, args.out)
    print("Saved model to:", args.out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="CSV with columns: path,label")
    p.add_argument("--out", default="models/light_model.joblib", help="Output path for bundled model")
    args = p.parse_args()
    main(args)
