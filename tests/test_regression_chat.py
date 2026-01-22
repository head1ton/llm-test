import json
import os
import sys
import time
import pytest
from fastapi.testclient import TestClient

from judge_llm import judge_answer

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

ENABLE_LLM_JUDGE = os.getenv("ENABLE_LLM_JUDGE", "0") == "1"

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

    # 1) 룰 기반 평가 (빠르고 안정적)
    res = eval_answer_rules(answer, case)
    assert res.ok, f"[{case['id']}] score={res.score} reason={res.reasons}\nanswer=\n{answer}\n"

    # 2) LLM-as-judge
    if ENABLE_LLM_JUDGE:
        # 내부 리소스는 현재 서버가 docs://{topic} 기반이라면 topic을 이용해 넣을 수 있는데,
        # 여기서는 최소 구현으로 resource를 빈 값으로 두고 평가(정확성/형식 중심)부터 시작.
        question = case["input"]
        resource = "" # or "NO_RESOURCE"
        judge, _usage = judge_answer(question, resource, answer)

        # 게이트 기준
        min_overall = int(os.getenv("JUDGE_MIN_OVERALL", "3"))
        min_accuracy = int(os.getenv("JUDGE_MIN_ACCURACY", "3"))

        assert judge.overall >= min_overall, (
            f"[{case['id']}] JUDGE FAIL overall={judge.overall} < {min_overall}\n"
            f"accuracy={judge.accuracy}, grounding={judge.grounding}, clarify={judge.clarify}, format={judge.format}\n"
            f"reasons={judge.reasons}\nanswer=\n{answer}\n"
        )

        assert judge.accuracy >= min_accuracy, (
            f"[{case['id']}] JUDGE FAIL accuracy={judge.accuracy} < {min_accuracy}\n"
            f"reasons={judge.reasons}\nanswer=\n{answer}\n"
        )


