import json
import os
import sys
import pytest
from fastapi.testclient import TestClient

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "mcp_langchain"))

from mcp_langchain.api_server_stream import app

def test_sse_stream_has_final():
    client = TestClient(app)

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

