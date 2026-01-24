import json
import os
import sys

import pytest
from fastapi.testclient import TestClient
from pydantic import TypeAdapter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import api_server
# from api_server import app
from sse_events import SSEEvent

SSE_ADAPTER = TypeAdapter(SSEEvent)

@pytest.fixture
def client():
    return TestClient(api_server.app)

def test_sse_events_schema(monkeypatch, client):
    """
        목적: /chat/stream이 내보내는 이벤트가 SSEEvent 스키마를 만족하는지 '계약'만 검증한다.
        실제 LLM/MCP 호출은 모킹해서 테스트를 안정화한다.
    """

    async def fake_run_agent_events(*, request_id, user_q, mcp_client, request=None):
        # /chat/stream이 실제로 흘리는 이벤트 흐름을 최소 세트로 재현
        yield {"type": "route", "request_id": request_id, "topic": "mcp"}
        yield {"type": "stage", "request_id": request_id, "stage": "prompt_loaded"}
        yield {"type": "token", "request_id": request_id, "token": "안녕하세요\n"}
        yield {"type": "usage", "request_id": request_id, "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}
        yield {"type": "final", "request_id": request_id, "answer": "• MCP는 ...", "topic": "mcp", "used_tools": ["read_doc"], "resource_uri": "docs://mcp"}
        yield {"type": "done", "request_id": request_id}

    # api_server에서 import한 run_agent_events를 모킹
    monkeypatch.setattr(api_server, "run_agent_events", fake_run_agent_events)

    payload = {
        "messages": [{"role": "user", "content": "MCP가 뭐야? 내부 문서 기반으로 설명해줘."}],
        "request_id": "test-sse-schema-1",
    }

    with client.stream("POST", "/chat/stream", json=payload) as r:
        assert r.status_code == 200

        saw_final = False
        saw_done = False

        for raw in r.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
            if not line.startswith("data: "):
                continue

            evt = json.loads(line[len("data: "):])

            # 스키마 검증
            SSE_ADAPTER.validate_python(evt)

            if evt.get("type") == "final":
                saw_final = True
            if evt.get("type") == "done":
                saw_done = True
                break

        assert saw_final, "final 이벤트를 받지 못했습니다."
        assert saw_done, "done 이벤트를 받지 못했습니다."

def test_sse_stream_has_final(monkeypatch, client):

    async def fake_run_agent_events(*, request_id, user_q, mcp_client, request=None):
        # /chat/stream이 실제로 흘리는 이벤트 흐름을 최소 세트로 재현
        yield {"type": "route", "request_id": request_id, "topic": "mcp"}
        yield {"type": "stage", "request_id": request_id, "stage": "prompt_loaded"}
        yield {"type": "token", "request_id": request_id, "token": "안녕하세요\n"}
        yield {"type": "usage", "request_id": request_id, "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}
        yield {"type": "final", "request_id": request_id, "answer": "• MCP는 ...", "topic": "mcp", "used_tools": ["read_doc"], "resource_uri": "docs://mcp"}
        yield {"type": "done", "request_id": request_id}

    # api_server에서 import한 run_agent_events를 모킹
    monkeypatch.setattr(api_server, "run_agent_events", fake_run_agent_events)

    # client = TestClient(app)

    payload = {
        "messages": [{"role": "user", "content": "MCP가 뭐야? 쉽게 설명해줘."}],
        "request_id": "test-stream-1"
    }

    with client.stream("POST", "/chat/stream", json=payload) as r:
        assert r.status_code == 200
        got_final = False

        for raw in r.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
            if not line.startswith("data: "):
                continue

            evt = json.loads(line[len("data: "):])
            if evt.get("type") == "final":
                got_final = True
                assert "answer" in evt and evt["answer"].strip()
                break

        assert got_final, "No final event received"
