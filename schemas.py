from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    request_id: Optional[str] = None

class ChatResponse(BaseModel):
    request_id: str
    answer: str
    topic: Optional[str] = None
    latency_ms: int

    # grounding/trace 메타
    resource_uri: Optional[str] = None
    resource_text: Optional[str] = None     # 운영에서 길면 truncate/요약 추천
    used_tools: Optional[List[str]] = None  # MCP tool names

