import asyncio
import os
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.chat_models import init_chat_model

# os.environ["GOOGLE_API_KEY"] = "key"
model = init_chat_model("google_genai:gemini-2.5-flash-lite")

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
        # "openai:gpt-4o-mini",
        model,
        tools,
        system_prompt=(
            "You are a helpful assistant.\n"
            "When the user asks about RAG/Agent/MCP, call the MCP tool read_doc.\n"
            "Use the tool output as the primary source.\n"
            "If tool returns NO_RESULT, ask which topic to read: rag/agent/mcp.\n"
        ),
    )

    res = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "MCP가 뭐야? 내부 문서 기반으로 설명해줘."}]}
    )
    print(res["messages"][-1].content)

if __name__ == '__main__':
    asyncio.run(main())