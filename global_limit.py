import os
import time
import uuid
import asyncio
from redis.asyncio import Redis

# Redis 연결 URL. 환경 변수 REDIS_URL에서 가져오며, 기본값은 로컬 Redis입니다.
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# 전역 최대 동시성 제한 수. 환경 변수 GLOBAL_MAX_CONCURRENCY에서 가져오며, 기본값은 20입니다.
GLOBAL_MAX_CONCURRENCY = int(os.getenv("GLOBAL_MAX_CONCURRENCY", "20"))

# (현재 코드에서는 사용되지 않지만) 리스(lease) 시간 설정.
GLOBAL_LEASE_SEC = int(os.getenv("GLOBAL_LEASE_SEC", "60"))

# Redis에서 사용할 토큰(슬롯) 리스트의 키 이름입니다.
KEY = "llm:global_slots"

# Lua 스크립트: 슬롯 획득
# KEYS[1]: 토큰 리스트 키 (KEY)
# 리스트에서 요소 하나를 꺼내서(LPOP) 반환합니다.
# 리스트가 비어있으면 빈 문자열을 반환합니다.
ACQUIRE_LUA = """
local v = redis.call('LPOP', KEYS[1])
if v then return v else return '' end
"""

# Lua 스크립트: 슬롯 반납
# KEYS[1]: 토큰 리스트 키 (KEY)
# ARGV[1]: 반납할 토큰 값
# 리스트에 토큰을 다시 넣습니다(LPUSH).
RELEASE_LUA = """
redis.call('LPUSH', KEYS[1], ARGV[1])
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
        """
        초기화 메서드.
        Redis에 토큰 리스트가 비어있을 경우, 최대 동시성 개수만큼 토큰을 생성하여 채워넣습니다.
        서버 시작 시 한 번 호출되어야 합니다.
        """
        # 현재 리스트에 있는 토큰 수 확인
        n = await self.redis.llen(KEY)
        if n == 0:
            # 토큰이 하나도 없으면 초기화 진행
            # 0부터 GLOBAL_MAX_CONCURRENCY-1 까지의 숫자를 문자열로 변환하여 토큰 생성
            tokens = [str(i) for i in range(GLOBAL_MAX_CONCURRENCY)]
            # 생성된 토큰들을 Redis 리스트에 한꺼번에 넣음 (LPUSH)
            await self.redis.lpush(KEY, *tokens)

    async def acquire(self, timeout_sec: float = 10.0) -> str | None:
        """
        슬롯(토큰) 획득을 시도합니다.
        
        Args:
            timeout_sec (float): 획득을 시도할 최대 시간(초). 기본값 10초.
            
        Returns:
            str | None: 획득 성공 시 토큰(문자열) 반환, 실패 시 None 반환.
        """
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            # Lua 스크립트를 실행하여 토큰 획득 시도 (원자적 연산)
            token = await self.redis.eval(ACQUIRE_LUA, 1, KEY)
            if token:
                # 토큰을 획득했으면 반환
                return token
            # 획득 실패 시 잠시 대기 후 재시도
            await asyncio.sleep(0.05)
        # 타임아웃까지 획득하지 못함
        return None

    async def release(self, token: str):
        """
        사용이 끝난 슬롯(토큰)을 반납합니다.
        
        Args:
            token (str): 반납할 토큰 값.
        """
        # Lua 스크립트를 실행하여 토큰을 리스트에 다시 넣음
        await self.redis.eval(RELEASE_LUA, 1, KEY, token)

# 전역에서 사용할 GlobalLimiter 인스턴스
GLOBAL_LIMITER = GlobalLimiter()
