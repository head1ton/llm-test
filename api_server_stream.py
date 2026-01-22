import json

from fastapi.responses import StreamingResponse
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from starlette.requests import Request
import sys
import os

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_mcp_workflow import build_graph

from schemas import ChatRequest, ChatResponse
from graph_mcp_workflow import mcp_messages_to_chat, router_prompt, router_llm
import asyncio
from utils_obs import Timer, ensure_request_id
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

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

    try:
        state = {"messages": [m.model_dump() for m in req.messages]}
        out = await graph_app.ainvoke(state)

        answer = out.get("answer")
        if not answer:
            raise RuntimeError("Graph output missing 'answer'.")

        latency = timer.ms()
        topic = out.get("topic")

        print(f"[chat] request_id={rid} latency_ms={latency} topic={topic}")

        return ChatResponse(
            request_id=rid,
            answer=answer,
            topic=topic,
            latency_ms=latency,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    rid = ensure_request_id(req.request_id)
    timer = Timer.start()

    async def sse(data: dict):
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    async def event_gen():
        try:
            yield await sse({"type": "start", "request_id": rid})

            if not req.messages:
                yield await sse({"type": "error", "request_id": rid, "message": "message is empty"})
                return

            user_q = req.messages[-1].content

            # 1) Router
            route = (router_prompt | router_llm).invoke({"q": user_q}).content.strip().lower()
            if route not in ("rag", "agent", "mcp", "clarify"):
                route = "clarify"

            yield await sse({"type": "route", "request_id": rid, "topic": route})

            if route == "clarify":
                msg = "질문을 정확히 잡기 위해 한 가지만 물어볼께. RAG, Agent, MCP 중 어떤 주제를 설명해줄까?"
                yield await sse({"type": "final", "request_id":rid, "answer":msg, "latency_ms": timer.ms()})
                yield await sse({"type": "done", "request_id": rid})
                return

            # 2) MCP Prompt
            prompt_msgs = await mcp_client_stream.get_prompt(
                "docs",
                "explain_concept",
                arguments={"topic": route.upper(), "audience": "beginner"},
            )
            base_messages = mcp_messages_to_chat(prompt_msgs)
            yield await sse({"type": "stage", "request_id": rid, "stage": "prompt_loaded"})

            # 3) MCP Resource
            uri = f"docs://{route}"
            blobs = await mcp_client_stream.get_resources("docs", uris=[uri])
            resource_text = "NO_RESOURCE"
            if blobs:
                blob = blobs[0]
                resource_text = blob.as_string() if hasattr(blob, "as_string") else str(blob)

            yield await sse({"type": "stage", "request_id": rid, "stage": "resource_loaded", "uri": uri})

            # 4) MCP Tools -> LangChain Tools
            tools = await mcp_client_stream.get_tools()
            yield await sse({"type": "stage", "request_id": rid, "stage": "tools_loaded", "count": len(tools)})

            # 5) Streaming 가능한 모델 (usage도 받고 싶으면 stream_usage=True)
            # OpenAI 스트리밍 usage는 기본으로 오지않으므로 stream_usage=True가 필요: contentReference[oaicite:5]{index=5}
            # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True, stream_usage=True)
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", streaming=True)

            agent = create_agent(
                llm,
                tools,
                system_prompt=(
                    "You are a helpful assistant.\n"
                    "Use internal resource text as primary truth.\n"
                    "If needed, call MCP tools.\n"
                    "Answer in Korean with bullets.\n"
                ),
            )

            # 에이전트 입력 메시지 구성
            messages = []
            messages.extend(base_messages)  # iterator해서 들어가고
            messages.append(    # 그냥 그대로 추가하고
                {
                    "role": "user",
                    "content": (
                        f"[Internal Resource: {uri}]\n{resource_text}\n\n"
                        f"[User Question]\n{user_q}"
                    ),
                }
            )

            yield await sse({"type": "stage", "request_id": rid, "stage": "agent_running"})

            # 6) Agent progress stream (툴 호출/결과 포함)
            # LangChain docs: stream_mode="updates"로 에이전트 스탭 업데이트 스트리밍: contentReference[oaicite:6]{index=6}
            final_text_parts = []
            usage = None

            async for update in agent.astream({"messages": messages}, stream_mode="updates"):
                # 클라이언트가 연결 끊었으면 즉시 중단
                if await request.is_disconnected():
                    yield await sse({"type": "cancelled", "request_id": rid})
                    return

                # update는 "어떤 노드가 어떤 메시지를 냈는지" 형태로 들어오는 경우가 많음
                # 여기서는 안전하게 문자열화 + 주요 케이스(툴콜/툴결과/토큰) 분리

                # 보통 update 안에 AIMessage/ToolMessage 등이 포함됨:
                # - AIMessage: tool_calls 포함 가능
                # - ToolMessage: tool 실행 결과
                # - AIMessageChunk: token/usage_metadata 포함 가능

                try:
                    # 가장 흔한 : {"messages": [...]} 형태로 들어오는 업데이트
                    msgs = update.get("messages") if isinstance(update, dict) else None
                    if msgs:
                        for m in msgs:
                            # tool call 감지
                            tool_calls = getattr(m, "tool_calls", None) or getattr(m, "additional_kwargs", {}).get("tool_calls")
                            if tool_calls:
                                yield await sse({"type": "tool_call", "request_id": rid, "tool_calls": tool_calls})
                                continue

                            # tool result 감지 (ToolMessage)
                            m_type = getattr(m, "type", None) or getattr(m, "__class__", type("x", (object,), {})).__name__
                            if "ToolMessage" in str(m_type):
                                yield await sse({"type": "tool_result", "request_id": rid, "content": getattr(m, "content", "")})
                                continue

                            # token/partial text 감지
                            content = getattr(m, "content", None)
                            if content:
                                final_text_parts.append(content)
                                yield await sse({"type": "token", "request_id": rid, "token": content})

                            # usage metadata 감지 (stream_usage=True일 때 chunk로 들어올 수 있음)
                            um = getattr(m, "usage_metadata", None) or getattr(m, "response_metadata", {}).get("usage") if hasattr(m, "response_metadata") else None
                            if um:
                                usage = um
                                yield await sse({"type": "usage", "request_id": rid, "usage": usage})

                    else:
                        # fallback: 그냥 raw 업데이트 내려주기(디버그용)
                        yield await sse({"type": "update", "request_id": rid, "data": str(update)[:2000]})

                except Exception as inner:
                    yield await sse({"type": "debug_error", "request_id": rid, "message": str(inner)})

            final_answer = "".join(final_text_parts).strip()
            yield await sse({
                "type": "final",
                "request_id": rid,
                "answer": final_answer,
                "topic": route,
                "usage": usage,
                "latency_ms": timer.ms(),
            })
            yield await sse({"type": "done", "request_id": rid})

        except Exception as e:
            yield await sse({"type": "error", "request_id": rid, "message": str(e)})

    return StreamingResponse(event_gen(), media_type="text/event-stream")


