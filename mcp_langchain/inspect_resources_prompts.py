import asyncio
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
    print("==== TOOLS ====")
    for t in tools:
        print("-", t.name)

    blobs = await client.get_resources("docs", uris=["docs://rag"])
    print("\n==== RESOURCE CONTENT ====")
    for blob in blobs:
        print("URI:", blob.metadata.get("uri"))
        print("MIME:", blob.mimetype)
        print("CONTENT:", blob.as_string())
    #
    #
    prompt = await client.get_prompt("docs", "explain_concept", arguments={"topic": "RAG", "audience": "beginner"})
    print("\n==== PROMPT ====")
    for m in prompt:
        print(f"{m.type}❗️: {m.content}")


if __name__ == '__main__':
    asyncio.run(main())