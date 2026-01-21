from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., description="대화 메시지 배열. 마지막은 user 권장")
    request_id: Optional[str] = Field(None, description="클라이언트가 주는 요청 ID(선택)")

class ChatResponse(BaseModel):
    request_id: str
    answer: str
    topic: Optional[str] = None
    latency_ms: int
