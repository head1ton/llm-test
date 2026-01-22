import time
import uuid
from dataclasses import dataclass

@dataclass
class Timer:
    t0: float

    @classmethod
    def start(cls):
        return cls(time.perf_counter())

    def ms(self) -> int:
        return int((time.perf_counter() - self.t0) * 1000)

def ensure_request_id(rid: str | None) -> str:
    return rid or str(uuid.uuid4())
