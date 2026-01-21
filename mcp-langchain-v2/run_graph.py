import asyncio
from graph_mcp_workflow import build_graph

async def main():
    app = build_graph()

    # MCP 설명
    out = await app.ainvoke({"messages": [{"role": "user", "content": "MCP가 뭐야? 쉽게 설명해줘."}]})
    print("\n==== ANSWER 1 ====")
    print(out["answer"])

    # RAG 설명
    out = await app.ainvoke(
        {"messages": [{"role": "user", "content": "RAG가 뭐고 왜 Fine-tuning보다 실무에서 많이 쓰여?"}]})
    print("\n==== ANSWER 2 ====")
    print(out["answer"])

    # 애매한 질문 일 때 -> clarify
    out = await app.ainvoke(
        {"messages": [{"role": "user", "content": "이거 어떻게 해야하지?."}]})
    print("\n==== ANSWER 3 ====")
    print(out["answer"])

if __name__ == '__main__':
    asyncio.run(main())