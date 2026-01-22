import json
import os
import sys
import time
import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api_server_stream import app
from eval_rules import eval_answer_rules

def load_cases(path="tests/golden_cases.jsonl"):
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases

@pytest.mark.parametrize("case", load_cases())
def test_chat_regression(case):
    # time.sleep(10)

    client = TestClient(app)

    payload = {
        "messages": [{"role": "user", "content": case["input"]}],
        "request_id": f"test-{case['id']}"
    }

    r = client.post("/chat", json=payload)
    assert r.status_code == 200, r.text

    data = r.json()
    answer = data["answer"]
    assert isinstance(answer, str) and answer.strip()

    # 룰 기반 평가
    res = eval_answer_rules(answer, case)

    assert res.ok, f"[{case['id']}] score={res.score} reason={res.reasons}\nanswer=\n{answer}\n"
