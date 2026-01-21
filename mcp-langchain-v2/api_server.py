import asyncio
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from schemas import ChatRequest, ChatResponse
from utils_obs import Timer, ensure_request_id

from graph_mcp_workflow import build_graph

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
