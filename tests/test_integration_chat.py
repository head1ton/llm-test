import os
import sys
import pytest
from fastapi.testclient import TestClient
import api_server

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

pytestmark = pytest.mark.integration

def _enabled():
    return os.getenv("ENABLE_INTEGRATION_TESTS", "0") == "1" and (bool(os.getenv("OPENAI_API_KEY") or bool(os.getenv("GOOGLE_API_KEY"))))

@pytest.mark.skipif(not _enabled(), reason="Integration tests disabled or OPENAI_API_KEY or GOOGLE_API_KEY missing")
def test_chat_integration_smoke():
    client = TestClient(api_server.app)
    payload = {
        "messages": [{"role": "user", "content": "MCP가 뭐야? 내부 문서 기반으로 3줄 요약해줘."}],
        "request_id": "it-chat-1"
    }
    r = client.post("/chat", json=payload, timeout=120)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["answer"].strip()
    assert data.get("topic") in ("mcp", "rag", "agent", "clarify")
