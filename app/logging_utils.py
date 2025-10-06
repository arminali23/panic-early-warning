import json, time, uuid
from collections import deque
from typing import Deque, Dict, Any

MAX_LOGS = 1000  

RECENT_LOGS: Deque[Dict[str, Any]] = deque(maxlen=MAX_LOGS)

def now_ts() -> float:
    return time.time()

def new_req_id() -> str:
    return uuid.uuid4().hex[:12]

def log_event(kind: str, **fields):
    evt = {"ts": now_ts(), "kind": kind}
    evt.update(fields)
    RECENT_LOGS.append(evt)
    return evt

def dump_logs(limit: int = 100):
    if limit <= 0:
        limit = 100
 
    out = list(RECENT_LOGS)[-limit:]
    return out
