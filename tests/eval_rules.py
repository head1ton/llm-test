import re
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class RuleEvalResult:
    ok: bool
    score: float
    reasons: List[str]

def count_bullets(text: str) -> int:
    # -, *, ·, 1. 같은 흔한 bullet 패턴
    bullet_patterns = [
        r"^\s*[-*·]\s+",
        r"^\s*\d+\.\s+",
    ]
    lines = text.splitlines()
    cnt = 0
    for ln in lines:
        if any(re.search(p, ln) for p in bullet_patterns):
            cnt += 1
    return cnt

def contains_all(text: str, keywords: List[str]) -> List[str]:
    missing = []
    lower = text.lower()
    for k in keywords:
        if k.lower() not in lower:
            missing.append(k)
    return missing

def contains_any(text: str, keywords: List[str]) -> List[str]:
    found = []
    lower = text.lower()
    for k in keywords:
        if k.lower() in lower:
            found.append(k)
    return found

def eval_answer_rules(answer: str, case: Dict[str, Any]) -> RuleEvalResult:
    reasons = []
    score = 1.0

    must_include = case.get("must_include", [])
    must_not = case.get("must_not_include", [])
    min_bullets = int(case.get("min_bullets", 0))

    # 1) 필수 키워드 포함
    missing = contains_all(answer, must_include)
    if missing:
        reasons.append(f"Missing keywords: {missing}")
        score -= 0.4

    # 2) 금지 키워드 포함 여부
    banned_found = contains_any(answer, must_not)
    if banned_found:
        reasons.append(f"Contains banned keywords: {banned_found}")
        score -= 0.4

    # 3) bullet 최소 개수
    bcnt = count_bullets(answer)
    if bcnt < min_bullets:
        reasons.append(f"Too few bullets: {bcnt} < {min_bullets}")
        score -= 0.2

    # 4) 너무 짧은 답변 방지 (보통 품질 낮음)
    if len(answer.strip()) < 40:
        reasons.append("Answer too short")
        score -= 0.2

    score = max(0.0, min(1.0, score))
    ok = score >= 0.6   # 품질 게이트 기준 (원하면 0.7 ~ 0.8로 올려도 됨)
    return RuleEvalResult(ok=ok, score=score, reasons=reasons)


