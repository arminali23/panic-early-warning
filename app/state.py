# app/state.py
from typing import Dict, Tuple
from .model import Baseline

USER_BASELINES: Dict[str, Baseline] = {}
USER_STATS: Dict[str, Tuple[dict, dict]] = {}

def get_or_create_baseline(user_id: str) -> Baseline:
    if user_id not in USER_BASELINES:
        USER_BASELINES[user_id] = Baseline()
    return USER_BASELINES[user_id]
