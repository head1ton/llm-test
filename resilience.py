import asyncio
import os
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

LLM_TIMEOUT_SEC = float(os.getenv("LLM_TIMEOUT_SEC", "40"))
MCP_TIMEOUT_SEC = float(os.getenv("MCP_TIMEOUT_SEC", "10"))
RETRY_MAX_ATTEMPTS = int(os.getenv("RETRY_MAX_ATTEMPTS", "2"))
RETRY_BASE_WAIT_SEC = float(os.getenv("RETRY_BASE_WAIT_SEC", "0.5"))

def _retry_policy():
    return retry(
        reraise=True,
        stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
        wait=wait_exponential_jitter(initial=RETRY_BASE_WAIT_SEC, max=3.0),
    )

async def with_timeout(coro, timeout_sec: float):
    return await asyncio.wait_for(coro, timeout=timeout_sec)

# MCP 호출용: timeout + retry
def mcp_retry():
    return _retry_policy()

# LLM 호출용: timeout + retry
def llm_retry():
    return _retry_policy()

