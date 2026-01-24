import os, json, statistics, time
import requests

BASE = os.getenv("BASE_URL", "http://localhost:8000")
N = int(os.getenv("GATE_N", "10"))
MAX_P95_MS = int(os.getenv("GATE_MAX_P95_MS", "8000"))

def main():
    latencies = []
    errors = 0

    for i in range(N):
        payload = {
            "messages": [{"role": "user", "content": "MCP가 뭐야? 내부 문서 기반으로 3줄 요약해줘."}],
            "request_id": f"gate-{int(time.time())}-{i}"
        }
        t0 = time.time()
        r = requests.post(f"{BASE}/chat", json=payload, timeout=120)
        dt = int((time.time() - t0) * 1000)

        if r.status_code != 200:
            errors += 1
            continue

        data = r.json()
        if not data.get("answer"):
            errors += 1
            continue

        latencies.append(dt)

    if not latencies:
        raise SystemExit("GATE FAIL: no successful responses")

    p95 = statistics.quantiles(latencies, n=20)[18] # 대략 p95
    err_rate = errors / N

    print(json.dumps({
        "n": N,
        "success": len(latencies),
        "errors": errors,
        "error_rate": err_rate,
        "p95_ms": p95
    }, ensure_ascii=False, indent=2))

    if p95 > MAX_P95_MS:
        raise SystemExit(f"GATE FAIL: p95 {p95}ms > {MAX_P95_MS}ms")
    if err_rate > 0.1:  # 초기엔 10%로 두고 점진 강화
        raise SystemExit(f"GATE FAIL: error_rate {err_rate} > 0.1")

if __name__ == '__main__':
    main()