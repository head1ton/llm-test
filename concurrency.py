import asyncio
import os

MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "4"))
QUEUE_TIMEOUT_SEC = float(os.getenv("QUEUE_TIMEOUT_SEC", "10"))

# 전체 서버에서 공유하는 세마포어 (요청 동시 처리 제한)
GLOBAL_SEM = asyncio.Semaphore(MAX_CONCURRENCY)