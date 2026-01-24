import json
import asyncio
import os
import traceback
import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from pydantic import TypeAdapter
from sse_events import SSEEvent

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_mcp_adapters.client import MultiServerMCPClient

from global_limit import GLOBAL_LIMITER
from graph_mcp_workflow import build_graph
from runner_core import run_agent_events
from schemas import ChatRequest, ChatResponse
from trace_store import TRACE_STORE
from utils_obs import Timer, ensure_request_id

from concurrency import GLOBAL_SEM, QUEUE_TIMEOUT_SEC
from obs_log import log, estimate_cost_usd

from experiments import choose_variant, REGISTRY
from metrics import METRICS
from rollout import ROLLOUT

# from starlette.requests import Request
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

# 로깅 설정 (uvicorn 로그와 통합)
logger = logging.getLogger("uvicorn")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: 서버 시작 시 실행
    logger.info("LIFESPAN: Startup initiated. Initializing GLOBAL_LIMITER...")
    try:
        await GLOBAL_LIMITER.init_once()
        logger.info("LIFESPAN: GLOBAL_LIMITER initialized successfully.")
    except Exception as e:
        logger.error(f"LIFESPAN: Failed to initialize GLOBAL_LIMITER: {e}")
        traceback.print_exc()
    
    yield
    
    # Shutdown: 서버 종료 시 실행
    logger.info("LIFESPAN: Shutdown initiated.")

app = FastAPI(title="LLM Agent Server (LangGraph + MCP)", lifespan=lifespan)

graph_app = build_graph()

QUEUE_PING_SEC = 1.0    # queued 상태에서 heartbeat 간격 (차후 env로 빼자)

GLOBAL_LEASE_RENEW_SEC = os.getenv("GLOBAL_LEASE_RENEW_SEC", "20.0")

VALIDATE_SSE = os.getenv("VALIDATE_SSE", "0") == "1"

MCP_URL = os.getenv("MCP_URL", "http://localhost:9000/mcp")  # http://mcp-docs:9000/mcp

SSE_ADAPTER = TypeAdapter(SSEEvent)

DEFAULT_ROLLOUT_V2_PCT = int(os.getenv("ROLLOUT_V2_PCT", "0"))

EXPERIMENT_ID = os.getenv("EXPERIMENT_ID", "exp-main")

# 전역으로 1번만 생성
mcp_client_stream = MultiServerMCPClient(
    {
        "docs": {
            "transport": "http",
            "url": MCP_URL,
            # 인증/추적 헤더가 필요하면 headers 추가 가능
            # "headers": {"Authorization": "Bearer ..."}
        }
    }
)

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/readiness")
async def readiness():
    # return {"ok": True, "note": "readiness endpoint is live"}
    # try:
    #     # MCP 서버가 응답하는지 확인
    #     blobs = await mcp_client_stream.get_resources("docs", uris=["docs://mcp"])
    #     ok = bool(blobs)
    #     return {"ok": ok}
    # except Exception as e:
    #     return {"ok": False, "error": str(e)}

    try:
        tools = await mcp_client_stream.get_tools()
        return {"ok": True, "mcp_transport": "http", "tools": [t.name for t in tools]}
    except Exception as e:
        # Python 3.11+ : ExceptionGroup(=BaseExceptionGroup) 펼치기
        details = {
            "type": type(e).__name__,
            "message": str(e),
        }

        # ExceptionGroup이면 하위 예외들을 모두 뽑아 보여줌
        if hasattr(e, "exceptions"):
            subs = []
            for sub in e.exceptions:
                subs.append({
                    "type": type(sub).__name__,
                    "message": str(sub),
                    "traceback": "".join(traceback.format_exception(type(sub), sub, sub.__traceback__))[-2000:],
                })
            details["sub_exceptions"] = subs
        else:
            details["traceback"] = "".join(traceback.format_exception(type(e), e, e.__traceback__))[-2000:]

        return {"ok": False, "mcp_transport": "http", "error": details}

async def _select_variant(request: Request) -> tuple[str, object, str | None, int]:
    """
    헤더/롤아웃 비율로 variant 선택.
    반환: (variant_str, variant_cfg, user_id, rollout_pct)
    """
    explicit = request.headers.get("X-EXP-VARIANT") # v1 or v2
    user_id = request.headers.get("X-USER-ID")  # sticky key
    rollout_pct = await ROLLOUT.get_v2_pct(DEFAULT_ROLLOUT_V2_PCT) # Redis 기반
    variant = choose_variant(explicit, rollout_pct, user_id=user_id, experiment_id=EXPERIMENT_ID)
    variant_cfg = REGISTRY[variant]
    return variant, variant_cfg, user_id, rollout_pct

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    rid = ensure_request_id(req.request_id)
    timer = Timer.start()

    if not req.messages:
        raise HTTPException(status_code=400, detail="messages is empty")

    user_q = req.messages[-1].content

    variant, variant_cfg, user_id, rollout_pct = await _select_variant(request)

    # 글로벌 토큰
    global_token = await GLOBAL_LIMITER.acquire(timeout_sec=QUEUE_TIMEOUT_SEC, lease_sec=180)
    if not global_token:
        raise HTTPException(status_code=429, detail="Server busy (global queue timeout)")

    # 큐/동시성 제어 (프로세스 로컬 세마포어)
    try:
        await asyncio.wait_for(GLOBAL_SEM.acquire(), timeout=QUEUE_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        # 글로벌 토큰은 얻었지만 로컬이 막힌 경우 -> 반납
        await GLOBAL_LIMITER.release(global_token)
        raise HTTPException(status_code=429, detail="Server busy (queue timeout)")

    try:
        # /chat도 동일 엔진으로 이벤트를 '수집'해서 최종 결과 생성
        final_evt = None
        async for evt in run_agent_events(
            request_id=rid,
            user_q=user_q,
            mcp_client=mcp_client_stream,   # 전역 MCP client(없으면 새로 하나 전역 생성)
            request=None,   # 비스트리밍이니 disconnect 감지 불필요
            variant_cfg=variant_cfg,
        ):
            if evt.get("type") == "final":
                final_evt = evt

        if not final_evt:
            raise HTTPException(status_code=500, detail="No final event produced")

        latency = timer.ms()
        usage = final_evt.get("usage") or {}
        cost = estimate_cost_usd(usage)

        # trace에서 resource_text 꺼냄
        tr = await TRACE_STORE.get(rid)
        resource_text = None
        if tr:
            for e in tr.events:
                if e.type == "resource":
                    resource_text = e.data.get("text")
                    break

        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        await METRICS.record(day=day, variant=variant, latency_ms=latency, cost_usd=cost, ok=True)

        log(
            "chat_done",
            request_id=rid,
            latency_ms=latency,
            variant=variant,
            model=variant_cfg.model,
            prompt=variant_cfg.prompt_name,
            topic=final_evt.get("topic"),
            used_tools=final_evt.get("used_tools", []),
            usage=usage,
            cost_usd=cost,
        )

        return ChatResponse(
            request_id=rid,
            answer=final_evt.get("answer", ""),
            topic=final_evt.get("topic"),
            latency_ms=latency,
            resource_uri=final_evt.get("resource_uri"),
            resource_text=resource_text,
            used_tools=final_evt.get("used_tools", [])
            # variant=variant,  # schema 확장 시 사용
        )
    finally:
        GLOBAL_SEM.release()
        await GLOBAL_LIMITER.release(global_token)

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    rid = ensure_request_id(req.request_id)
    timer = Timer.start()

    if not req.messages:
        raise HTTPException(status_code=400, detail="messages is empty")

    user_q = req.messages[-1].content

    variant, variant_cfg, user_id, rollout_pct = await _select_variant(request)

    async def sse(data: dict):
        if VALIDATE_SSE:
            # 스키마 검증: 실패하면 예외 발생
            SSE_ADAPTER.validate_python(data)
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    async def acquire_with_queue_events():
        """
        세마포어 즉시 획득 시 queued 없이 진행.
        아니면 queued 이벤트를 보내고 QUEUE_TIMEOUT_SEC 동안 대기
        """
        try:
            GLOBAL_SEM.acquire_nowait()
            return {"queued": False, "start": None}
        except Exception:
            start = asyncio.get_event_loop().time()
            return {"queued": True, "start": start}

    async def event_gen():
        final_evt = None
        acquired = False
        global_token = None

        ok_for_metrics = False
        latency_ms_for_metrics = 0
        cost_usd_for_metrics = 0.0

        try:
            # 시작 이벤트를 먼저 보내면 UX가 좋아짐
            yield await sse({"type": "start", "request_id": rid})
            yield await sse({
                "type": "experiment",
                "request_id": rid,
                "variant": variant,
                "model": variant_cfg.model,
                "prompt": variant_cfg.prompt_name,
            })

            # Global_LIMITER (Redis) 먼저 획득
            start = asyncio.get_event_loop().time()
            global_token = await GLOBAL_LIMITER.acquire(timeout_sec=0.0)

            if global_token:
                yield await sse({"type": "dequeued_global", "request_id": rid, "waited_ms": 0})
            else:
                yield await sse({"type": "queued_global", "request_id": rid})
                last_ping = start

                while True:
                    if await request.is_disconnected():
                        yield await sse({
                            "type": "cancelled",
                            "request_id": rid,
                            "reason": "client_disconnected_in_global_queue"
                        })
                        return

                    now = asyncio.get_event_loop().time()
                    waited = now - start

                    if waited >= QUEUE_TIMEOUT_SEC:
                        yield await sse({
                            "type": "error",
                            "request_id": rid,
                            "message": "Server busy (global queue timeout)"
                        })
                        return

                    if (now - last_ping) >= QUEUE_PING_SEC:
                        yield await sse({
                            "type": "queued_ping_global",
                            "request_id": rid,
                            "waited_ms": int(waited * 1000)
                        })
                        last_ping = now

                    global_token = await GLOBAL_LIMITER.acquire(timeout_sec=0.2)
                    if global_token:
                        yield await sse({
                            "type": "dequeued_global",
                            "request_id": rid,
                            "waited_ms": int(waited * 1000)
                        })
                        break

            # LOCAL 세마포어
            qstate = await acquire_with_queue_events()

            if not qstate["queued"]:
                acquired = True
                yield await sse({"type": "dequeued", "request_id": rid, "waited_ms": 0})
            else:
                # queued 이벤트 먼저 발행
                yield await sse({"type": "queued", "request_id": rid})
                # queued heartbeat + acquire 대기
                start = qstate["start"]
                last_ping = start

                while True:
                    if await request.is_disconnected():
                        yield await sse({"type": "cancelled", "request_id": rid, "reason": "client_disconnected_in_local_queue"})
                        return

                    now = asyncio.get_event_loop().time()
                    waited = now - start

                    if waited >= QUEUE_TIMEOUT_SEC:
                        yield await sse({"type": "error", "request_id": rid, "message": "Server busy (local queue timeout)"})
                        return

                    # 주기적으로 queued_ping 전송 (프론트 UX용)
                    if (now - last_ping) >= QUEUE_PING_SEC:
                        yield await sse({"type": "queued_ping", "request_id": rid, "waited_ms": int(waited * 1000)})
                        last_ping = now

                    # 짧게 acquire 시도
                    try:
                        await asyncio.wait_for(GLOBAL_SEM.acquire(), timeout=0.2)
                        acquired = True
                        yield await sse({"type": "dequeued", "request_id": rid, "waited_ms": int(waited * 1000)})
                        break
                    except asyncio.TimeoutError:
                        continue

            # agent 실행
            last_renew = asyncio.get_event_loop().time()

            async for evt in run_agent_events(
                request_id=rid,
                user_q=user_q,
                mcp_client=mcp_client_stream,
                request=request,
                variant_cfg=variant_cfg,
            ):
                # lease 갱신 (스트리밍이 길어질 때 만료 방지)
                if global_token:
                    now = asyncio.get_event_loop().time()
                    if (now - last_renew) >= float(GLOBAL_LEASE_RENEW_SEC):
                        await GLOBAL_LIMITER.renew(global_token)
                        last_renew = now

                # latency는 final 이벤트에서 같이 넣고 싶으면 여기서 주입
                if evt.get("type") == "final":
                    final_evt = evt
                    evt["latency_ms"] = timer.ms()

                    # metrics/log용 값 확정
                    latency_ms_for_metrics = evt["latency_ms"]
                    usage = evt.get("usage") or {}
                    cost_usd_for_metrics = estimate_cost_usd(usage)
                    ok_for_metrics = True

                yield await sse(evt)

            # final 이벤트가 없고 정상 종료된 경우도 실패로 본다.
            if not ok_for_metrics:
                latency_ms_for_metrics = timer.ms()

        except Exception as e:
            # 스트리밍 중 예외는 error 이벤트로 전달
            yield await sse({"type": "error", "request_id": rid, "message": str(e)})
            latency_ms_for_metrics = timer.ms()

        finally:
            # 반드시 release (클라이언트 중단/예외 발생해도)
            if acquired:
                GLOBAL_SEM.release()

            if global_token:
                await GLOBAL_LIMITER.release(global_token)

            # metrics 기록 (variant 별 집계)
            try:
                day = datetime.now(timezone.utc).strftime("%Y%m%d")
                await METRICS.record(
                    day=day,
                    variant=variant,
                    latency_ms=int(latency_ms_for_metrics or timer.ms()),
                    cost_usd=float(cost_usd_for_metrics or 0.0),
                    ok=bool(ok_for_metrics)
                )
            except Exception:
                # metrics 실패가 본 요청에 영향 주지 않게
                pass

            # 스트리밍도 운영 로그.
            if final_evt:
                usage = final_evt.get("usage") or {}
                cost = estimate_cost_usd(usage)

                log(
                    "chat_stream_done",
                    request_id=rid,
                    variant=variant,
                    model=variant_cfg.model,
                    prompt=variant_cfg.prompt_name,
                    rollout_pct=rollout_pct,
                    user_id_present=bool(user_id),
                    latency_ms=timer.ms(),
                    topic=final_evt.get("topic"),
                    used_tools=final_evt.get("used_tools", []),
                    usage=usage,
                    cost_usd=cost,
                )
            else:
                log(
                    "chat_stream_done_no_final",
                    request_id=rid,
                    variant=variant,
                    model=variant_cfg.model,
                    prompt=variant_cfg.prompt_name,
                    rollout_pct=rollout_pct,
                    user_id_preaent=bool(user_id),
                    latency_ms=timer.ms(),
                )

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/trace/{request_id}")
async def get_trace(request_id: str):
    tr = await TRACE_STORE.get(request_id)
    if not tr:
        return {"request_id": request_id, "events": []}
    return {
        "request_id": request_id,
        "events": [
            {"ts": e.ts, "type": e.type, "data": e.data}
            for e in tr.events
        ]
    }

@app.get("/metrics/experiments")
async def exp_metrics(day: str | None = None):
    if not day:
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
    v1 = await METRICS.snapshot(day, "v1")
    v2 = await METRICS.snapshot(day, "v2")
    return {"day": day, "v1": v1, "v2": v2}
