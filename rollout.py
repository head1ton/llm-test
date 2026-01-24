from __future__ import annotations
import os
from redis.asyncio import Redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
ROLLOUT_KEY = os.getenv("ROLLOUT_KEY", "exp:rollout_v2_pct")

class Rollout:
    def __init__(self):
        self.redis = Redis.from_url(REDIS_URL, decode_responses=True)

    async def get_v2_pct(self, default: int) -> int:
        v = await self.redis.get(ROLLOUT_KEY)
        if v is None:
            return default
        try:
            return max(0, min(100, int(v)))
        except Exception:
            return default

    async def set_v2_pct(self, pct: int):
        pct = max(0, min(100, int(pct)))
        await self.redis.set(ROLLOUT_KEY, str(pct))

ROLLOUT = Rollout()
