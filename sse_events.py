from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field

# 공통 베이스
class SSEBase(BaseModel):
    type: str
    request_id: str

# -- Queue / Concurrency events --
class StartEvent(SSEBase):
    type: Literal["start"] = "start"

class QueuedGlobalEvent(SSEBase):
    type: Literal["queued_global"] = "queued_global"

class QueuedPingGlobalEvent(SSEBase):
    type: Literal["queued_ping_global"] = "queued_ping_global"
    waited_ms: int = Field(..., ge=0)

class DequeuedGlobalEvent(SSEBase):
    type: Literal["dequeued_global"] = "dequeued_global"
    waited_ms: int = Field(..., ge=0)

class QueuedLocalEvent(SSEBase):
    type: Literal["queued"] = "queued"

class QueuedPingLocalEvent(SSEBase):
    type: Literal["queued_ping"] = "queued_ping"
    waited_ms: int = Field(..., ge=0)

class DequeuedLocalEvent(SSEBase):
    type: Literal["dequeued"] = "dequeued"
    waited_ms: int = Field(..., ge=0)

# --- Router/Stages ---
class RouteEvent(SSEBase):
    type: Literal["route"] = "route"
    topic: Literal["rag", "agent", "mcp", "clarify"]

class StageEvent(SSEBase):
    type: Literal["stage"] = "stage"
    stage: str
    uri: Optional[str] = None
    count: Optional[int] = None

# --- Tools ---
class ToolCallEvent(SSEBase):
    type: Literal["tool_call"] = "tool_call"
    tool_calls: Any
    tool_names: Optional[List[str]] = None

class ToolResultEvent(SSEBase):
    type: Literal["tool_result"] = "tool_result"
    content: str

# --- Tokens / Usage ---
class TokenEvent(SSEBase):
    type: Literal["token"] = "token"
    token: str

class UsageEvent(SSEBase):
    type: Literal["usage"] = "usage"
    usage: Dict[str, Any]
    source: Optional[str] = None

# --- Final / Done ---
class FinalEvent(SSEBase):
    type: Literal["final"] = "final"
    answer: str
    topic: Optional[str] = None
    used_tools: Optional[List[str]] = None
    resource_uri: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    latency_ms: Optional[int] = None

class DoneEvent(SSEBase):
    type: Literal["done"] = "done"

# --- Errors / Cancel ---
class ErrorEvent(SSEBase):
    type: Literal["error"] = "error"
    message: str

class CancelledEvent(SSEBase):
    type: Literal["cancelled"] = "cancelled"
    reason: Optional[str] = None

class ExperimentEvent(SSEBase):
    type: Literal["experiment"] = "experiment"
    variant: str
    model: str
    prompt: str

# Union 타입: 서버/테스트에서 검증할 때 사용
SSEEvent = Union[
    StartEvent,
    ExperimentEvent,
    QueuedGlobalEvent, QueuedPingGlobalEvent, DequeuedGlobalEvent,
    QueuedLocalEvent, QueuedPingLocalEvent, DequeuedLocalEvent,
    RouteEvent, StageEvent,
    ToolCallEvent, ToolResultEvent,
    TokenEvent, UsageEvent,
    FinalEvent, DoneEvent,
    ErrorEvent, CancelledEvent,
]
