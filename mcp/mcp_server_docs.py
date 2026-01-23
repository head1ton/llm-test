import os

from fastmcp import FastMCP

mcp = FastMCP(
    name="DocsServer",
    instructions="Provides tools/resources/prompts for docs."
)

DOCS = {
    "rag": "RAG는 Retrieval-Augmented Generation으로, 외부 지식을 검색해 컨텍스트로 주입하여 환각을 줄이는 방식이다.",
    "agent": "Agent는 LLM이 목표 달성을 위해 도구를 선택/호출하며 다단계로 작업을 수행하는 실행 단위다.",
    "mcp": "MCP는 LLM 앱이 외부 도구/데이터(툴/리소스/프롬프트)를 표준 방식으로 연결하기 위한 프로토콜이다.",
}

# 1) TOOL
@mcp.tool
def read_doc(keyword: str) -> str:
    """키워드(rag|agent|mcp)에 해당하는 내부 문서를 반환한다."""
    key = (keyword or "").strip().lower()
    return DOCS.get(key, "NO_RESULT")

# 2) RESOURCE
@mcp.resource("docs://{keyword}")
def get_doc_resource(keyword: str) -> str:
    """
    docs://rag 처럼 URL로 문서를 읽게 해준다.
    Tool과 달리 '리소스 읽기' 관점의 표준 인터페이스
    """
    key = (keyword or "").strip().lower()
    return DOCS.get(key, "NO_RESULT")

# 3) PROMPT
@mcp.prompt
def explain_concept(topic: str, audience: str = "beginner") -> str:
    """
    주제(topic)를 대상(audience)에 맞춰 설명하게 만드는 프롬프트 템플릿을 제공.
    클라이언트가 이 prompt를 가져다 LLM에게 그대로 넣을 수 있다.
    """
    return (
        f"Explain '{topic}' for a '{audience}' audience.\n"
        f"Rules:\n"
        f"- Use simple examples\n"
        f"- Keep it structured with bullets\n"
        f"- If you reference internal docs, say so\n"
    )

if __name__ == '__main__':
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "9000"))

    mcp.run(transport="http", host=host, port=port)