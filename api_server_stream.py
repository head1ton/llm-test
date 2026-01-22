import json

from fastapi.responses import StreamingResponse
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
# from starlette.requests import Request
import sys
import os

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_mcp_workflow import build_graph
from runner_core import run_agent_events

from schemas import ChatRequest, ChatResponse
from graph_mcp_workflow import mcp_messages_to_chat, router_prompt, router_llm
import asyncio
from utils_obs import Timer, ensure_request_id
from fastapi import FastAPI, HTTPException, Request
from dotenv import load_dotenv

from trace_store import TRACE_STORE

load_dotenv()

app = FastAPI(title="LLM Agent Server (LangGraph + MCP)")

graph_app = build_graph()

# 전역으로 1번만 생성
mcp_client_stream = MultiServerMCPClient(
    {
        "docs": {
            "transport": "stdio",
            "command": "python",
            "args": ["mcp_server_docs.py"],
        }
    }
)

@app.get("/health")
def health():
    return {"ok": True}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    rid = ensure_request_id(req.request_id)
    timer = Timer.start()

    if not req.messages:
        raise HTTPException(status_code=400, detail="messages is empty")

    user_q = req.messages[-1].content

    # /chat도 동일 엔진으로 이벤트를 '수집'해서 최종 결과 생성
    final_evt = None
    async for evt in run_agent_events(
        request_id=rid,
        user_q=user_q,
        mcp_client=mcp_client_stream,   # 전역 MCP client(없으면 새로 하나 전역 생성)
        request=None,   # 비스트리밍이니 disconnect 감지 불필요
    ):
        if evt.get("type") == "final":
            final_evt = evt

    if not final_evt:
        raise HTTPException(status_code=500, detail="No final event produced")

    latency = timer.ms()

    tr = TRACE_STORE.get(rid)
    resource_text = None
    if tr:
        for e in tr.events:
            if e.type == "resource":
                resource_text = e.data.get("text")
                break

    return ChatResponse(
        request_id=rid,
        answer=final_evt.get("answer", ""),
        topic=final_evt.get("topic"),
        latency_ms=latency,
        resource_uri=final_evt.get("resource_uri"),
        resource_text=resource_text,
        used_tools=final_evt.get("used_tools", [])
    )

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    rid = ensure_request_id(req.request_id)
    timer = Timer.start()

    if not req.messages:
        raise HTTPException(status_code=400, detail="messages is empty")

    user_q = req.messages[-1].content

    async def sse(data: dict):
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    async def event_gen():
        async for evt in run_agent_events(
            request_id=rid,
            user_q=user_q,
            mcp_client=mcp_client_stream,
            request=request,
        ):
            # latency는 final 이벤트에서 같이 넣고 싶으면 여기서 주입
            if evt.get("type") == "final":
                evt["latency_ms"] = timer.ms()

            yield await sse(evt)

    return StreamingResponse(event_gen(), media_type="text/event-stream")

@app.get("/trace/{request_id}")
def get_trace(request_id: str):
    tr = TRACE_STORE.get(request_id)
    if not tr:
        return {"request_id": request_id, "events": []}
    return {
        "request_id": request_id,
        "events": [
            {"ts": e.ts, "type": e.type, "data": e.data}
            for e in tr.events
        ]
    }
