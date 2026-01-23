import asyncio
import os
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

async def main():
    url = os.getenv("MCP_URL", "http://localhost:9000/mcp")
    client = MultiServerMCPClient({"docs": {"transport": "http", "url": url}})

    async with client.session("docs") as session:
        tools = await load_mcp_tools(session)
        print("tools:", [t.name for t in tools])

if __name__ == '__main__':
    asyncio.run(main())