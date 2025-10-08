# app/state.py
import os
import json
from typing import Dict, Tuple
from .model import Baseline

REDIS_URL = os.getenv("REDIS_URL")

USER_BASELINES: Dict[str, Baseline] = {}
USER_STATS: Dict[str, Tuple[dict, dict]] = {}  # in-memory fallback

_redis = None
if REDIS_URL:
    try:
        import redis
        _redis = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        # ping test
        _redis.ping()
    except Exception:
        _redis = None  # fallback to memory

def get_or_create_baseline(user_id: str) -> Baseline:
    # Baseline sadece runtime içinde kullanılıyor; kalıcı istatistik için USER_STATS/Redis
    if user_id not in USER_BASELINES:
        USER_BASELINES[user_id] = Baseline()
    return USER_BASELINES[user_id]

def set_user_stats(user_id: str, mu: dict, std: dict):
    if _redis:
        key = f"user:{user_id}:stats"
        _redis.hset(key, mapping={
            "mu": json.dumps(mu),
            "std": json.dumps(std),
        })
    else:
        USER_STATS[user_id] = (mu, std)

def get_user_stats(user_id: str) -> Tuple[dict, dict] | None:
    if _redis:
        key = f"user:{user_id}:stats"
        if not _redis.exists(key):
            return None
        mu = json.loads(_redis.hget(key, "mu") or "{}")
        std = json.loads(_redis.hget(key, "std") or "{}")
        return (mu, std)
    else:
        return USER_STATS.get(user_id)

def redis_ready() -> bool:
    global _redis
    if _redis is None:
        return False
    try:
        _redis.ping()
        return True
    except Exception:
        return False
