import os
from prometheus_client import Counter, Histogram, Gauge

APP_NAME = os.getenv("APP_NAME", "llm-agent")

REQ_COUNT = Counter(
    f"{APP_NAME}_requests_total",
    "Total requests",
    ["endpoint", "variant", "status"],
)

REQ_LATENCY_MS = Histogram(
    f"{APP_NAME}_request_latency_ms",
    "Request latency in milliseconds",
    ["endpoint", "variant"],
    buckets=(50, 100, 200, 400, 800, 1500, 3000, 5000, 8000, 12000, 20000, 40000),
)

QUEUE_WAIT_MS = Histogram(
    f"{APP_NAME}_queue_wait_ms",
    "Queue wait in milliseconds",
    ["queue", "endpoint", "variant"],   # queue: global/local
    buckets=(0, 50, 100, 200, 400, 800, 1500, 3000, 5000, 8000, 12000),
)

INFLIGHT = Gauge(
    f"{APP_NAME}_inflight",
    "In-flight requests",
    ["endpoint"],
)

TOKENS = Counter(
    f"{APP_NAME}_tokens_total",
    "Token counts",
    ["variant", "kind"],    # kind: input/output/total
)

COST_USD = Counter(
    f"{APP_NAME}_cost_usd_total",
    "Estimated cost in USD",
    ["variant"],
)

