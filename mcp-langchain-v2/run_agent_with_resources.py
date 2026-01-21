import os
import asyncio
from pathlib import Path
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
# 현재 파일(__file__)의 상위 상위 폴더(프로젝트 루트)에 있는 .env 파일을 지정
dotenv_path = find_dotenv()
load_dotenv(dotenv_path, override=True)

model = init_chat_model("google_genai:gemini-2.5-flash-lite")

def mcp_messages_to_chat(messages):
    """
    langchain_mcp_adapters의 get_prompt()가 반환하는 메시지 객체들을
    create_agent가 받는 {"role":..., "content":...} 형태로 변환
    """
    out = []
    for m in messages:
        # 보통 m.type이 "system"/"user"/"assistant"로 들어오는 케이스가 많음
        role = getattr(m, "type", None) or getattr(m, "role", None) or "system"
        content = getattr(m, "content", None) or getattr(m, "text", None) or str(m)
        out.append({"role": role, "content": content})
    return out

async def main():
    client = MultiServerMCPClient(
        {
            "docs": {
                "transport": "stdio",
                "command": "python",
                "args": ["mcp_server_docs.py"]
            }
        }
    )

    # 1) MCP Tools 로딩
    tools = await client.get_tools()

    # 2) MCP Prompt 템플릿 로딩
    prompt_msgs = await client.get_prompt(
        "docs",
        "explain_concept",
        arguments={"topic": "MCP", "audience": "beginner"},
    )
    base_messages = mcp_messages_to_chat(prompt_msgs)
    print("base_messages: ", base_messages)

    # 3) MCP Resource 로딩
    blobs = await client.get_resources("docs", uris=["docs://mcp"])
    if not blobs:
        resource_text = "NO_RESOURCE"
    else:
        blob = blobs[0]
        resource_text = blob.as_string() if hasattr(blob, "as_string") else str(blob)

    print("resource_text: ", resource_text)

    # 4) Agent 생성
    agent = create_agent(
        model,
        tools,
        system_prompt=(
            "You are a helpful assistant.\n"
            "You may call tools if needed.\n"
            "Prefer using MCP-provided resources as primary truth.\n"
            "If you cannot find enough info, ask a clarifying question.\n"
        ),
    )

    # 5) 최종 실행 메시지 구성
    user_question = "MCP를 한 번에 이해할 수 있게 핵심만 정리해줘. 그리고 RAG랑 Agent랑 관계도 같이 설명해줘."

    messages = []
    messages.extend(base_messages)
    messages.append({
        "role": "user",
        "content": (
            f"[Internal Resource: docs://mcp]\n{resource_text}\n\n"
            f"User question:\n{user_question}"
        )
    })

    # 6) 실행
    res = await agent.ainvoke({"messages": messages})
    print(res)
    print('=====' * 100)
    print(res["messages"][-1].content)
    # print('=====' * 100)
    # print(res["messages"][0].content)
    # print('=====' * 100)
    # print(res["messages"][1].content)
    # print('=====' * 100)
    # print(res["messages"][2].content)

if __name__ == '__main__':
    asyncio.run(main())
