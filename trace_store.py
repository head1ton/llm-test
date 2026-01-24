from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from redis.asyncio import Redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
TRACE_TTL_SEC = int(os.getenv("TRACE_TTL_SEC", "86400"))        # 24h
TRACE_MAX_EVENTS = int(os.getenv("TRACE_MAX_EVENTS", "2000"))   # 리스트 길이 제한

def _key(request_id: str) -> str:
    return f"trace:{request_id}"

@dataclass
class TraceEvent:
    ts: float
    type: str
    data: Dict[str, Any]

@dataclass
class RequestTrace:
    request_id: str
    events: List[TraceEvent]

class RedisTraceStore:
    def __init__(self):
        self.redis = Redis.from_url(REDIS_URL, decode_responses=True)

    async def start(self, request_id: str) -> None:
        # start는 "키 생성/TTL 설정" 역할
        k = _key(request_id)
        # 이미 있으면 TTL만 갱신 (상황에 따라 유지)
        await self.redis.expire(k, TRACE_TTL_SEC)

    async def add(self, request_id: str, type: str, **data) -> None:
        k = _key(request_id)

        evt = {
            "ts": time.time(),
            "type": type,
            "data": data,
        }
        s = json.dumps(evt, ensure_ascii=False)

        # 이벤트 추가
        await self.redis.rpush(k, s)

        # 길이 제한 (오래 걸리는 스트림 폭주 방지)
        # 최신 TRACE_MAX_EVENTS개만 유지
        await self.redis.ltrim(k, -TRACE_MAX_EVENTS, -1)

        # TTL 갱신 (요청이 길어져도 trace 유지)
        await self.redis.expire(k, TRACE_TTL_SEC)

    async def get(self, request_id: str) -> Optional[RequestTrace]:
        k = _key(request_id)
        items = await self.redis.lrange(k, 0, -1)
        if not items:
            return None

        events: List[TraceEvent] = []
        for it in items:
            try:
                obj = json.loads(it)
                events.append(TraceEvent(ts=obj["ts"], type=obj["type"], data=obj.get("data", {})))
            except Exception:
                # 파싱 실패 항목은 무시 (운영 안전)
                continue

        return RequestTrace(request_id=request_id, events=events)

    async def pop(self, request_id: str) -> Optional[RequestTrace]:
        # 필요 시 사용 (기본은 get만 써도 됨)
        tr = await self.get(request_id)
        if tr:
            await self.redis.delete(_key(request_id))
        return tr

TRACE_STORE = RedisTraceStore()
