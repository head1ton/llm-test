import json
import os
import time
from typing import Any, Dict, Optional

LOG_JSON = os.getenv("LOG_JSON", "1") == "1"

COST_IN = float(os.getenv("COST_PER_1K_INPUT_USD", "0.0"))
COST_OUT = float(os.getenv("COST_PER_1K_OUTPUT_USD", "0.0"))

def now_ms() -> int:
    return int(time.time() * 1000)

def estimate_cost_usd(usage: Optional[Dict[str, Any]]) -> float:
    u = normalize_usage(usage or {})
    inp = u.get("input_tokens", 0)
    out = u.get("output_tokens", 0)

    return (inp / 1000.0) * COST_IN + (out / 1000.0) * COST_OUT

def log(event: str, **fields):
    payload = {"ts_ms": now_ms(), "event": event, **fields}
    if LOG_JSON:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(payload)

def normalize_usage(usage: dict | None) -> dict:
    """
    usage 포멧을 통일:
        {"input_tokens": int, "output_tokens": int, "total_tokens": int}
    """
    if not usage:
        return {}

    # 흔한 케이스들 흡수
    inp = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
    out = usage.get("output_tokens") or usage.get("completion_tokens") or 0
    total = usage.get("total_tokens") or (inp + out)

    return {
        "input_tokens": int(inp),
        "output_tokens": int(out),
        "total_tokens": int(total),
    }