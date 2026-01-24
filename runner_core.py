import json
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from fastapi import Request
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

from graph_mcp_workflow import mcp_messages_to_chat, router_prompt, router_llm
from trace_store import TRACE_STORE
from resilience import with_timeout, mcp_retry, MCP_TIMEOUT_SEC
from experiments import REGISTRY, VariantConfig

def _extract_tool_names(tool_calls: Any) -> List[str]:
    names: List[str] = []
    if not tool_calls:
        return names

    # OpenAI tool call dict 형태를 가장 우선으로 처리
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            if isinstance(tc, dict):
                fn = tc.get("function", {}) or {}
                nm = fn.get("name")
                if nm:
                    names.append(nm)

    return names

async def run_agent_events(
    *,
    request_id: str,
    user_q: str,
    mcp_client,
    request: Optional[Request] = None,  # 스트리밍에서 disconnect 감지용
    variant_cfg=None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    통합 실행 코어: 이벤트(dict)를 순서대로 yield
    이벤트 타입 예:
        - start, route, stage, tool_call, tool_result, token, usage, final, done, cancelled, error
    """
    if variant_cfg is None:
        variant_cfg = REGISTRY["v1"]

    await TRACE_STORE.start(request_id)
    await TRACE_STORE.add(request_id, "start")

    yield {"type": "start", "request_id": request_id}

    # 1) Router
    route = (router_prompt | router_llm).invoke({"q": user_q}).content.strip().lower()
    if route not in ("rag", "agent", "mcp", "clarify"):
        route = "clarify"

    await TRACE_STORE.add(request_id, "route", topic=route)
    yield {"type": "route", "request_id": request_id, "topic": route}

    if route == "clarify":
        msg = "질문을 정확히 잡기 위해 한 가지만 물어볼께. RAG, Agent, MCP 중 어떤 주제를 설명해줄까?"
        await TRACE_STORE.add(request_id, "final", topic=route, answer=msg, used_tools=[])
        yield {"type": "final", "request_id": request_id, "topic": route, "answer": msg, "used_tools": []}
        yield {"type": "done", "request_id": request_id}
        return

    @mcp_retry()
    async def _load_prompt():
        return await with_timeout(
            mcp_client.get_prompt(
                "docs",
                variant_cfg.prompt_name,
                arguments={"topic": route.upper(), "audience": "beginner"},
            ),
            MCP_TIMEOUT_SEC,
        )

    await TRACE_STORE.add(request_id, "experiment", variant=variant_cfg.variant, model=variant_cfg.model, prompt=variant_cfg.prompt_name)
    yield {"type": "experiment", "request_id": request_id, "model": variant_cfg.model, "prompt": variant_cfg.prompt_name, "variant": variant_cfg.variant}

    # 2) MCP Prompt
    prompt_msgs = await _load_prompt()

    base_messages = mcp_messages_to_chat(prompt_msgs)
    await TRACE_STORE.add(request_id, "stage", stage="prompt_loaded")
    yield {"type": "stage", "request_id": request_id, "stage": "prompt_loaded"}

    @mcp_retry()
    async def _load_resource():
        return await with_timeout(
            mcp_client.get_resources("docs", uris=[uri]),
            MCP_TIMEOUT_SEC,
        )

    # 3) MCP Resource
    uri = f"docs://{route}"
    blobs = await _load_resource()
    if not blobs:
        resource_text = "NO_RESOURCE"
    else:
        blob = blobs[0]
        resource_text = blob.as_string() if hasattr(blob, "as_string") else str(blob)

    await TRACE_STORE.add(request_id, "stage", stage="resource_loaded", uri=uri)
    await TRACE_STORE.add(request_id, "resource", uri=uri, text=resource_text[:4000])
    yield {"type": "stage", "request_idi": request_id, "stage": "resource_loaded", "uri": uri}

    @mcp_retry()
    async def _load_tools():
        return await with_timeout(mcp_client.get_tools(), MCP_TIMEOUT_SEC)

    # 4) MCP Tools 로드
    tools = await _load_tools()

    await TRACE_STORE.add(request_id, "stage", stage="tools_loaded", count=len(tools))
    yield {"type": "stage", "request_id": request_id, "stage": "tools_loaded", "count": len(tools)}

    # 5) Streaming 가능한 LLM + usage 수집(가능한 경우)
    # llm = ChatOpenAI(
    #     model=variant_cfg.model,
    #     temperature=variant_cfg.temperature,
    #     streaming=True,
    #     stream_usage=True,
    # )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        streaming=True,
    )

    agent = create_agent(
        llm,
        tools,
        system_prompt=(
            "You are a helpful assistant.\n"
            "Use internal resource text as primary truth.\n"
            "If needed, call MCP tools.\n"
            "Answer in Korean wiith bullets.\n"
        ),
    )

    messages: List[Dict[str, str]] = []
    messages.extend(base_messages)
    messages.append(
        {
            "role": "user",
            "content": (
                f"[Internal Resource: {uri}]\n{resource_text}\n\n"
                f"[User Question]\n{user_q}"
            ),
        }
    )

    # 6) 에이전트 업데이트 스트림에서 tool_call/tool_result/token/usage를 추출
    final_parts: List[str] = []
    used_tools: List[str] = []
    usage: Optional[dict] = None

    async for update in agent.astream({"messages": messages}, stream_mode="updates"):
        if request is not None and await request.is_disconnected():
            await TRACE_STORE.add(request_id, "cancelled")
            yield {"type": "cancelled", "request_id": request_id}
            return

        # update는 구현/버전별로 구조가 달라서 최대한 방어적으로 처리
        msgs = update.get("messages") if isinstance(update, dict) else None
        if not msgs:
            await TRACE_STORE.add(request_id, "update", raw=str(update)[:1000])
            yield {"type": "update", "request_id": request_id, "data": str(update)[:2000]}
            continue

        for m in msgs:
            # tool call 감지
            tool_calls = getattr(m, "tool_calls", None)
            if tool_calls is None:
                # 일부는 additional_kwargs에 들어감
                ak = getattr(m, "additional_kwargs", {}) or {}
                tool_calls = ak.get("tool_calls")

            if tool_calls:
                names = _extract_tool_names(tool_calls)
                for n in names:
                    if n not in used_tools:
                        used_tools.append(n)

                await TRACE_STORE.add(request_id, "tool_call", tool_calls=tool_calls, tool_names=names)
                yield {"type": "tool_call", "request_id": request_id, "tool_calls": tool_calls, "tool_names": names}
                continue

            # tool_result 감지 (ToolMessage)
            m_type = getattr(m, "type", None) or m.__class__.__name__
            if "ToolMessage" in str(m_type):
                content = getattr(m, "content", "") or ""
                await TRACE_STORE.add(request_id, "tool_result", content=content[:2000])
                yield {"type": "tool_result", "request_id": request_id, "content": content}
                continue

            # token/partial text 감지
            content = getattr(m, "content", None)
            if content:
                final_parts.append(content)
                await TRACE_STORE.add(request_id, "token", token=content)
                yield {"type": "token", "request_id": request_id, "token": content}

            # usage 감지 (가능한 경우)
            um = getattr(m, "usage_metadata", None)
            if um:
                usage = um
                await TRACE_STORE.add(request_id, "usage", usage=usage)
                yield {"type": "usage", "request_id": request_id, "usage": usage}

    final_answer = "".join(final_parts).strip()

    if usage is None:
        # 1) Trace에서 usage 이벤트가 있었으면 그걸 사용
        tr = await TRACE_STORE.get(request_id)
        if tr:
            for ev in reversed(tr.events):
                if ev.type == "usage":
                    usage = ev.data.get("usage")
                    break

    # 2) 그래도 없으면 tiktoken으로 대략 추정
    if usage is None:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("o200k_base")   # 대체로 안전한 기본
            # input은 user_q + resource_text + prompt_msgs를 대략 합쳐 계산(보수적으로)
            approx_in_text = user_q + "\n" + resource_text
            in_tok = len(enc.encode(approx_in_text))
            out_tok = sum(len(enc.encode(p)) for p in final_parts) if final_parts else 0
            usage = {"input_tokens": in_tok, "output_tokens": out_tok, "total_tokens": in_tok + out_tok}
            await TRACE_STORE.add(request_id, "usage", usage=usage, source="tiktoken_estimate")
            yield {"type": "usage", "request_id": request_id, "usage": usage, "source": "tiktoken_estimate"}
        except Exception:
            usage = None

    await TRACE_STORE.add(
        request_id,
        "final",
        topic=route,
        answer=final_answer,
        used_tools=used_tools,
        resource_uri=uri,
        usage=usage or {},
    )
    yield {
        "type": "final",
        "request_id": request_id,
        "topic": route,
        "answer": final_answer,
        "used_tools": used_tools,
        "resource_uri": uri,
        "usage": usage,
    }
    yield {"type": "done", "request_id": request_id}

