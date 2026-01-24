import os
import sys
import asyncio
from datetime import datetime, timezone

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metrics import METRICS
from rollout import ROLLOUT

MIN_V2 = int(os.getenv("GUARD_MIN_V2", "30"))
MAX_ERR = float(os.getenv("GUARD_MAX_ERR", "0.05"))
LATENCY_MULT = float(os.getenv("GUARD_LAT_MULT", "1.5"))
COST_MULT = float(os.getenv("GUARD_COST_MULT", "2.0"))

DEFAULT_ROLLOUT = int(os.getenv("ROLLOUT_V2_PCT", "0"))

async def main():
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    v1 = await METRICS.snapshot(day, "v1")
    v2 = await METRICS.snapshot(day, "v2")

    rollout = await ROLLOUT.get_v2_pct(DEFAULT_ROLLOUT)

    # 샘플 부족하면 아무것도 안함
    if v2["count"] < MIN_V2:
        print({"action": "noop", "reason": "insufficient_samples", "rollout": rollout, "v1": v1, "v2": v2})
        return

    bad = []
    if v2["error_rate"] > MAX_ERR:
        bad.append("error_rate")
    if v1["avg_latency_ms"] > 0 and v2["avg_latency_ms"] > v1["avg_latency_ms"] * LATENCY_MULT:
        bad.append("latency")
    if v1["avg_cost_usd"] > 0 and v2["avg_cost_usd"] > v1["avg_cost_usd"] * COST_MULT:
        bad.append("cost")

    if bad:
        await ROLLOUT.set_v2_pct(0)
        print({"action": "rollback", "bad": bad, "rollout_from": rollout, "rollout_to": 0, "v1": v1, "v2": v2})
    else:
        print({"action": "ok", "rollout": rollout, "v1": v1, "v2": v2})

if __name__ == '__main__':
    asyncio.run(main())
