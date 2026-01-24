import os
import time
import uuid
import asyncio
from typing import Optional

from redis.asyncio import Redis

# Redis 연결 URL. 환경 변수 REDIS_URL에서 가져오며, 기본값은 로컬 Redis입니다.
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# 전역 최대 동시성 제한 수. 환경 변수 GLOBAL_MAX_CONCURRENCY에서 가져오며, 기본값은 20입니다.
GLOBAL_MAX_CONCURRENCY = int(os.getenv("GLOBAL_MAX_CONCURRENCY", "20"))

# 리스(lease) 시간 설정.
GLOBAL_LEASE_SEC = int(os.getenv("GLOBAL_LEASE_SEC", "120"))

LEASES_KEY = os.getenv("GLOBAL_LEASE_KEY", "llm:global_leases")

# Lua 스크립트: 슬롯 획득
# KEYS[1]: 토큰 리스트 키 (KEY)
# 리스트에서 요소 하나를 꺼내서(LPOP) 반환합니다.
# 리스트가 비어있으면 빈 문자열을 반환합니다.
ACQUIRE_LUA = """
local key = KEYS[1]
local max = tonumber(ARGV[1])
local now = tonumber(ARGV[2])
local lease_sec = tonumber(ARGV[3])
local lease_id = ARGV[4]

-- cleanup expired
redis.call('ZREMRANGEBYSCORE', key, '-inf', now)

local cnt = redis.call('ZCARD', key)
if cnt < max then
    local exp = now + lease_sec
    redis.call('ZADD', key, exp, lease_id)
    return lease_id
end

return ''
"""

# Lua 스크립트: 슬롯 반납
# KEYS[1]: 토큰 리스트 키 (KEY)
# ARGV[1]: 반납할 토큰 값
# 리스트에 토큰을 다시 넣습니다(LPUSH).
RELEASE_LUA = """
local key = KEYS[1]
local lease_id = ARGV[1]
redis.call('ZREM', key, lease_id)
return 1
"""

# Lua: renew (idempotent: lease가 있으면 exp 갱신, 없으면 0)
RENEW_LUA = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local lease_sec = tonumber(ARGV[2])
local lease_id = ARGV[3]

local exists = redis.call('ZSCORE', key, lease_id)
if not exists then
    return 0
end

local exp = now + lease_sec
redis.call('ZADD', key, exp, lease_id)
return 1
"""

class GlobalLimiter:
    """
    Redis를 사용하여 분산 환경에서 전역 동시성을 제어하는 클래스입니다.
    토큰 버킷 알고리즘과 유사하게, Redis 리스트에 토큰을 저장하고 꺼내 쓰는 방식으로 동작합니다.
    """
    def __init__(self):
        # Redis 클라이언트 초기화 (비동기)
        self.redis = Redis.from_url(REDIS_URL, decode_responses=True)

    async def init_once(self):
        # list 채우기 같은 초기화가 필요 없는 구조.
        # 단, 키가 없을 때도 그냥 동작한다.
        # ZSET 기반이므로 별도의 초기화가 필요 없음.
        # 하지만 키가 없으면 redis-cli에서 안 보일 수 있으므로,
        # 명시적으로 키가 존재하는지 체크하거나 로그를 남길 수 있음.
        return
        # pass

    async def _redis_time_epoch(self) -> int:
        """
        Redis Time을 사용해서 서버 간 시간 편차를 최소화.
        """
        t = await self.redis.time() # [seconds, microseconds]
        return int(t[0])

    async def acquire(self, timeout_sec: float = 10.0, lease_sec: Optional[int] = None) -> Optional[str]:
        """
        timeout 내에 lease를 얻으면 lease_id(str) 반환, 실패하면 None
        """
        lease_sec = int(lease_sec or GLOBAL_LEASE_SEC)
        deadline = time.time() + timeout_sec

        while time.time() < deadline:
            now = await self._redis_time_epoch()
            lease_id = f"{now}:{os.getpid()}:{int(time.time() * 1000)}"
            got = await self.redis.eval(
                ACQUIRE_LUA,
                1,
                LEASES_KEY,
                GLOBAL_MAX_CONCURRENCY,
                now,
                lease_sec,
                lease_id,
            )
            if got:
                return got
            # 획득 실패 시 잠시 대기 후 재시도
            await asyncio.sleep(0.05)
        # 타임아웃까지 획득하지 못함
        return None

    async def release(self, lease_id: str) -> None:
        """
        사용이 끝난 슬롯(토큰)을 반납합니다.
        """
        # Lua 스크립트를 실행하여 토큰을 리스트에 다시 넣음
        await self.redis.eval(RELEASE_LUA, 1, LEASES_KEY, lease_id)

    async def renew(self, lease_id: str, lease_sec: Optional[int] = None) -> bool:
        lease_sec = int(lease_sec or GLOBAL_LEASE_SEC)
        now = await self._redis_time_epoch()
        ok = await self.redis.eval(RENEW_LUA, 1, LEASES_KEY, now, lease_sec, lease_id)
        return bool(ok)

# 전역에서 사용할 GlobalLimiter 인스턴스
GLOBAL_LIMITER = GlobalLimiter()
