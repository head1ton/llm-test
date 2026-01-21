import asyncio
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

async def main():
    client = MultiServerMCPClient(
        {
            "docs": {
                "transport": "stdio",
                "command": "python",
                "args": ["mcp_server_docs.py"],
            }
        }
    )

    tools = await client.get_tools()

    agent = create_agent(
        "openai:gpt-4o-mini",
        tools,
        system_prompt=(
            "You are a helpful assistant."
            "When a user asks about RAG/Agent/MCP, call the appropriate MCP tool."
        ),
    )

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "MCP가 뭐야? 내부 문서 기반으로 설명해줘."}]}
    )

    final_msg = result["messages"][-1]
    print(final_msg.content)


if __name__ == '__main__':
    asyncio.run(main())
