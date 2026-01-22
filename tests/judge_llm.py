import os
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

class JudgeResult(BaseModel):
    # 0 ~ 5 점수
    accuracy: int = Field(..., ge=0, le=5, description="사실/논리적으로 올바른가")
    grounding: int = Field(..., ge=0, le=5, description="주어진 내부 리소스/근거를 잘 활용했는가")
    clarify: int = Field(..., ge=0, le=5, description="명확하고 이해하기 쉬운가")
    format: int = Field(..., ge=0, le=5, description="요구 형식(불릿/구조/언어 등)을 지켰는가")
    overall: int = Field(..., ge=0, le=5, description="좋합 점수")
    reasons: str = Field(..., description="간단한 근거(2~4문장)")

def get_judge_model():
    # 비용/속도 밸런스: mini 추천. 필요하면 상위 모델로.
    # model = os.getenv("JUDGE_MODEL", "gpt-4o-mini")
    # return ChatOpenAI(model=model, temperature=0)
    model = os.getenv("JUDGE_MODEL", "gemini-2.5-flash-lite")
    return ChatGoogleGenerativeAI(model=model)

judge_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict evaluator for an LLM assistant.\n"
     "You will be given a user question, an internal resource text, and the assistant answer.\n"
     "Score each criterion from 0 to 5.\n"
     "Be consistent. If the answer contains hallucinations not supported by the resource. lower accuracy and grounding.\n"
     "If the question is ambiguous and the assistant asks a clarifying question, that can be correct.\n"
     "Return JSON matching the schema strictly."
     ),
    ("user",
     "User question:\n{question}\n\n"
     "Internal resource (may be empty/NO_RESOURCE):\n{resource}\n\n"
     "Assistant answer:\n{answer}\n\n"
     "Scoring rubric:\n"
     "- accuracy: factual/logical correctness\n"
     "- grounding: uses internal resource, no unsupported claims\n"
     "- clarify: clear, easy to understand\n"
     "- format: structured, bullets if appropriate, Korea\n"
     "- overall: overall quality\n"
     )
])

def judge_answer(question: str, resource: str, answer: str) -> tuple[JudgeResult, dict]:
    """
    returns: (JudgeResult, usage_dict)
    usage_dict는 모델/버전에 따라 비어있을 수 있음
    """
    llm = get_judge_model()

    # LangChaini structured output (Pydantic)
    llm_structured = llm.with_structured_output(JudgeResult)

    res: JudgeResult = (judge_prompt | llm_structured).invoke({
        "question": question,
        "resource": resource,
        "answer": answer
    })

    # 토큰 사용량
    # response_metadata/usage_metadata 구조는 모델/버전에 따라 다를 수 있어 방어적으로 처리
    usage = {}
    try:
        # 일부 구현은 llm_structured 호출 결과에 메타데이터가 없을 수 있음
        # 이 경우엔 그냥 빈 dict로 두면 됨
        pass
    except Exception:
        usage = {}

    return res, usage
