from __future__ import annotations
import os
from redis.asyncio import Redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
METRICS_TTL_SEC = int(os.getenv("METRICS_TTL_SEC", "86400")) # 24h
PREFIX = os.getenv("METRICS_PREFIX", "m:exp")

def _k(day: str, variant: str, field: str) -> str:
    return f"{PREFIX}:{day}:{variant}:{field}"

class Metrics:
    def __init__(self):
        self.redis = Redis.from_url(REDIS_URL, decode_responses=True)

    async def record(self, day: str, variant: str, latency_ms: int, cost_usd: float, ok: bool):
        pipe = self.redis.pipeline()
        pipe.incr(_k(day, variant, "count"), 1)
        pipe.incrby(_k(day, variant, "latency_sum_ms"), int(latency_ms))
        # cost는 float이라 incrbyfloat 사용
        pipe.incrbyfloat(_k(day, variant, "cost_sum_usd"), float(cost_usd))
        if not ok:
            pipe.incr(_k(day, variant, "error_count"), 1)

        # TTL
        for f in ("count", "latency_sum_ms", "cost_sum_usd", "error_count"):
            pipe.expire(_k(day, variant, f), METRICS_TTL_SEC)

        await pipe.execute()

    async def snapshot(self, day: str, variant: str):
        keys = {f: _k(day, variant, f) for f in ("count", "latency_sum_ms", "cost_sum_usd", "error_count")}
        vals = await self.redis.mget(list(keys.values()))

        out = {}
        for (f, k), v in zip(keys.items(), vals):
            out[f] = float(v) if (v and f == "cost_sum_usd") else int(v or 0)

        count = out["count"]
        out["avg_latency_ms"] = (out["latency_sum_ms"] / count) if count else 0
        out["avg_cost_usd"] = (out["cost_sum_usd"] / count) if count else 0
        out["error_rate"] = (out["error_count"] / count) if count else 0
        return out

METRICS = Metrics()
