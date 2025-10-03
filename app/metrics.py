# app/metrics.py
from prometheus_client import Counter, Histogram, Gauge

REQUESTS = Counter(
    "pew_requests_total", "Total HTTP requests", ["endpoint", "method", "code"]
)
LATENCY = Histogram(
    "pew_request_latency_seconds", "Request latency", ["endpoint", "method"]
)
SCORE_SOURCE = Counter(
    "pew_score_source_total", "Score source counter", ["source"]  # "model" or "rule"
)
STREAM_ALERTS = Counter(
    "pew_stream_alerts_total", "Total stream alerts triggered"
)
