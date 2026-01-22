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
    if not usage:
        return 0.0

    # usage 구조는 환경에 따라 다를 수 있어 방어적으로 처리
    # 흔히 input_tokens / output_tokens or prompt_tokens / completion_tokens
    inp = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
    out = usage.get("output_tokens") or usage.get("completion_tokens") or 0

    return (inp / 1000.0) * COST_IN + (out / 1000.0) * COST_OUT

def log(event: str, **fields):
    payload = {"ts_ms": now_ms(), "event": event, **fields}
    if LOG_JSON:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(payload)
