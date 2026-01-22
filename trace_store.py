from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List
import time

@dataclass
class TraceEvent:
    ts: float
    type: str
    data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RequestTrace:
    request_id: str
    events: List[TraceEvent] = field(default_factory=list)

class TraceStore:
    def __init__(self):
        self._store: Dict[str, RequestTrace] = {}

    def start(self, request_id: str) -> None:
        self._store[request_id] = RequestTrace(request_id=request_id)

    def add(self, request_id: str, type: str, **data):
        tr = self._store.get(request_id)
        if not tr:
            tr = RequestTrace(request_id=request_id)
            self._store[request_id] = tr
        tr.events.append(TraceEvent(ts=time.time(), type=type, data=data))

    def get(self, request_id: str) -> RequestTrace | None:
        return self._store.get(request_id)

    def pop(self, request_id: str) -> RequestTrace | None:
        return self._store.pop(request_id, None)

TRACE_STORE = TraceStore()
