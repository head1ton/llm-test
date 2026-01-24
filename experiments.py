from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass
from typing import Dict, Literal, Optional

Variant = Literal["v1", "v2"]

@dataclass(frozen=True)
class VariantConfig:
    variant: Variant
    model: str
    prompt_name: str    # MCP prompt 이름 또는 내부 프롬프트 키
    resource_namespace: str     # MCP server name (예: "docs")
    resource_prefix: str        # URI prefix (예: "docs://")
    temperature: float = 0.0

# 여기서 "무엇이 v1/v2인지"를 단일하게 관리
REGISTRY: Dict[Variant, VariantConfig] = {
    "v1": VariantConfig(
        variant="v1",
        # model=os.getenv("MODEL_V1", "gpt-4o-mini"),
        model=os.getenv("MODEL_V1", "gemini-2.5-flash-lite"),
        prompt_name=os.getenv("PROMPT_v1", "explain_concept"),
        resource_namespace="docs",
        resource_prefix="docs://",
        temperature=float(os.getenv("TEMP_v1", "0")),
    ),
    "v2": VariantConfig(
        variant="v2",
        # model=os.getenv("MODEL_V2", "gpt-4o-mini"),
        model=os.getenv("MODEL_V1", "gemini-2.5-flash-lite"),
        prompt_name=os.getenv("PROMPT_v2", "explain_concept_v2"),
        resource_namespace="docs",
        resource_prefix="docs://",
        temperature=float(os.getenv("TEMP_v2", "0")),
    ),
}

def _bucket_0_99(key: str) -> int:
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % 100 # 0~99

def choose_variant(
    explicit: Optional[str],
    rollout_pct_v2: int,
    user_id: Optional[str] = None,
    experiment_id: str = "exp-main",
) -> Variant:
    """
    1) explicit 헤더가 있으면 그대로
    2) user_id가 있으면 sticky bucketing
    3) 없으면 랜덤 rollout
    """
    if explicit in ("v1", "v2"):
        return explicit # type: ignore

    rollout_pct_v2 = max(0, min(100, int(rollout_pct_v2)))

    if user_id:
        b = _bucket_0_99(f"{experiment_id}:{user_id}")
        return "v2" if b < rollout_pct_v2 else "v1"

    # rollout_pct_v2: 0 ~ 100
    r = random.randint(1, 100)
    return "v2" if r <= rollout_pct_v2 else "v1"
