import asyncio
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from schemas import ChatRequest, ChatResponse
from utils_obs import Timer, ensure_request_id

from graph_mcp_workflow import build_graph

import json
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

app = FastAPI(title="LLM Agent Server (LangGraph + MCP)")

graph_app = build_graph()

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
async def chat_stream(req: ChatRequest):
    rid = ensure_request_id(req.request_id)
    timer = Timer.start()

    async def event_gen():
        """
        SSE 포멧:
            data: {...}\n\n
        """
        try:
            # 0) 시작 이벤트
            yield f"data: {json.dumps({'type': 'start', 'request_id': rid})}\n\n"

            # 1) 마지막 user 메시지
            if not req.messages:
                yield f"data: {json.dumps({'type': 'error', 'request_id': rid, 'message': 'message is empty'})}\n\n"
                return
            user_q = req.messages[-1].content

            # 2) LangGraph의 Router만 재사용하고 싶지만,
            # 여기서는 스트리밍 경로를 단순화해서 라우팅을 가볍게 수행(LLM router)
            # - graph_mcp_workflow.py의 router_prompt 로직을 그대로 복사해도 됨.
            from graph_mcp_workflow import router_prompt, router_llm, mcp_client, mcp_messages_to_chat

            route = (router_prompt | router_llm).invoke({"q": user_q}).content.strip().lower()
            if route not in ("rag", "agent", "mcp", "clarify"):
                route = "clarify"

            yield f"data: {json.dumps({'type':'route', 'request_id':rid, 'topic':route})}\n\n"

            # 3) clarify면 즉시 종료(스트리밍으로 1회 메시지)
            if route == "clarify":
                msg = "질문을 정확히 잡기 위해 한 가지만 물어볼께. RAG, Agent, MCP 중 어떤 주체를 설명해줄까?"
                yield f"data: {json.dumps({'type':'final', 'request_id':rid, 'answer':msg, 'latency_ms':timer.ms()})}\n\n"
                yield f"data: {json.dumps({'type':'done', 'request_id':rid})}\n\n"
                return

            # 4) MCP Prompt 로드
            prompt_msgs = await mcp_client.get_prompt(
                "docs",
                "explain_concept",
                arguments={"topic": route.upper(), "audience": "beginner"},
            )
            base_messages = mcp_messages_to_chat(prompt_msgs)

            yield f"data: {json.dumps({'type':'stage', 'request_id':rid, 'stage':'prompt_loaded'})}\n\n"

            # 5) MCP Resource 로드 (docs://{topic})
            uri = f"docs://{route}"
            blobs = await mcp_client.get_resources("docs", uris=[uri])
            if not blobs:
                resource_text = "NO_RESOURCE"
            else:
                blob = blobs[0]
                resource_text = blob.as_string() if hasattr(blob, "as_string") else str(blob)

            yield f"data: {json.dumps({'type':'stage', 'request_id':rid, 'stage':'resource_loaded', 'uri':uri})}\n\n"

            # 6) 스트리밍 가능한 LLM로 최종 답변 생성
            #       (툴 루프(create_agent)는 스트리밍/이벤트 전달이 복잡해서 여기서는 제외)
            llm_stream = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", streaming=True)
            # llm_stream = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

            answer_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are a helpful assistant.\n"
                 "Use the provided internal resource as primary truth.\n"
                 "If the resource is NO_RESOURCE, say you don't know and ask what document to use.\n"
                 "Answer in Korean, structured with bullets."),
                ("user",
                 "Internal Resource ({uri}):\n{resource}\n\n"
                 "User Question:\n{question}\n")
            ])

            runnable = answer_prompt | llm_stream

            yield f"data: {json.dumps({'type':'stage', 'request_id':rid, 'stage':'generating'})}\n\n"

            # 토큰 스트리밍
            full = []
            async for chunk in runnable.astream({
                "uri": uri,
                "resource": resource_text,
                "question": user_q
            }):
                # chunk는 AIMessageChunk 형태가 흔함
                token = getattr(chunk, "content", None)
                if token:
                    full.append(token)
                    yield f"data: {json.dumps({'type':'token', 'request_id':rid, 'token':token})}\n\n"

            final_answer = "".join(full).strip()
            yield f"data: {json.dumps({'type':'final', 'request_id':rid, 'answer':final_answer, 'topic':route, 'latency_ms':timer.ms()})}\n\n"
            yield f"data: {json.dumps({'type':'done', 'request_id':rid})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type':'error', 'request_id':rid, 'message':str(e)})}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")
