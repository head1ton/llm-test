import asyncio
import os
from langchain_mcp_adapters.client import MultiServerMCPClient

MCP_URL = os.getenv("MCP_URL", "http://localhost:9000/mcp")

async def main():
    client = MultiServerMCPClient({"docs": {"transport": "http", "url": MCP_URL}})

    # # prompts
    # try:
    #     prompts = await client.list_prompts()
    # except Exception:
    #     prompts = await client.get_prompts()
    #
    # print("==== PROMPTS ====")
    # for p in prompts:
    #     print(p)

    print("\n=== TRY render explain_concept with empty args ===")
    try:
        r = await client.get_prompt("explain_concept", arguments={})
        print("OK", r)
    except Exception as e:
        print("ERR : ", e)

if __name__ == '__main__':
    asyncio.run(main())
