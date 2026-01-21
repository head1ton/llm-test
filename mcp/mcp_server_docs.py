from fastmcp import FastMCP

mcp = FastMCP(
    name="Docs",
    instructions=(
        "This server provides tiny internal docs."
        "Use read_cod(keyword) where keyword is one of: rag, agent, mcp."
    ),
)

DOCS = {
    "rag": "RAG는 Retrieval-Augmented Generation으로, 외부 지식을 검색해 컨텍스트로 주입하여 환각을 줄이는 방식이다.",
    "agent": "Agent는 LLM이 목표 달성을 위해 도구를 선택/호출하며 다단계로 작업을 수행하는 실행 단위다.",
    "mcp": "MCP는 LLM 앱이 외부 도구/데이터(툴/리소스/프롬프트)를 표준 방식으로 연결하기 위한 프로토콜이다.",
}

@mcp.tool
def read_doc(keyword: str) -> str:
    """키워드에 해당하는 내부 문서를 반환한다. (rag | agent | mcp)"""
    key = (keyword or "").strip().lower()
    return DOCS.get(key, "NO_RESULT")

if __name__ == '__main__':
    mcp.run()
