import asyncio
from typing import TypedDict, Literal, List, Optional, Any
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import init_chat_model

dotenv_path = find_dotenv()
load_dotenv(dotenv_path, override=True)

model = init_chat_model("google_genai:gemini-2.5-flash-lite")

mcp_client = MultiServerMCPClient(
    {
        "docs": {
            "transport": "stdio",
            "command": "python",
            "args": ["mcp_server_docs.py"],
        }
    }
)

router_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
answer_llm_name = "google_genai:gemini-2.5-flash-lite"

# State 정의
class GraphState(TypedDict, total=False):
    messages: List[dict]    # [{"role": "user"/"system"/"assistant", "content": "..."}]
    topic: Literal["rag", "agent", "mcp", "clarify"]
    prompt_messages: List[dict]
    resource_text: str
    answer: str

# MCP prompt 메시지 -> LangChain 메시지(dict) 변환
def mcp_messages_to_chat(messages: List[Any]) -> List[dict]:
    out = []
    for m in messages:
        role = getattr(m, "type", None) or getattr(m, "role", None) or "system"
        content = getattr(m, "content", None) or getattr(m, "text", None) or str(m)
        out.append({"role": role, "content": content})
    return out

# 1) Router Node
router_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a router.\n"
     "Decide a topic for internal docs lookup: rag | agent | mcp.\n"
     "If user request is ambiguous and you need clarification, output clarify.\n"
     "Return ONLY one word: rag or agent or mcp or clarify",
     ),
    ("user", "{q}"),
])

async def router_node(state: GraphState) -> GraphState:
    q = state["messages"][-1]["content"]
    route = (router_prompt | router_llm).invoke({"q": q}).content.strip().lower()
    if route not in ("rag", "agent", "mcp", "clarify"):
        route = "clarify"
    return {"topic": route}

# 2) LoadPrompt Node (MCP Prompt)
async def load_prompt_node(state: GraphState) -> GraphState:
    topic = state["topic"]
    if topic == "clarify":
        return {"prompt_messages": []}

    prompt_msgs = await mcp_client.get_prompt(
        "docs",
        "explain_concept",
        arguments={"topic": topic.upper(), "audience": "beginner"},
    )

    return {"prompt_messages": mcp_messages_to_chat(prompt_msgs)}

# 3) LoadResource Node (MCP Resource: docs://{topic})
async def load_resource_node(state: GraphState) -> GraphState:
    topic = state["topic"]
    if topic == "clarify":
        return {"resource_text": ""}

    uri = f"docs://{topic}"
    blobs = await mcp_client.get_resources("docs", uris=[uri])
    if not blobs:
        return {"resource_text": "NO_RESOURCE"}

    blob = blobs[0]
    text = blob.as_string() if hasattr(blob, "as_string") else str(blob)
    return {"resource_text": text}

# 4) Answer Node
async def answer_node(state: GraphState) -> GraphState:
    topic = state["topic"]
    user_q = state["messages"][-1]["content"]

    if topic == "clarify":
        return {"answer": "질문을 정확히 잡기 위해 한 가지만 물어볼께. RAG, Agent, MCP 중 어떤 주제를 설명해줄까?"}

    tools = await mcp_client.get_tools()

    agent = create_agent(
        model,
        tools,
        system_prompt=(
            "You are a helpful assistant.\n"
            "Use the provided internal resource text as primary truth.\n"
            "If you need more internal detail, call MCP Tools.\n"
            "If you cannot answer from internal info, ask a clarifying question.\n"
        ),
    )

    prompt_msgs = state.get("prompt_messages", [])
    resource_text = state.get("resource_text", "")

    messages = []
    messages.extend(prompt_msgs)

    messages.append({
        "role": "user",
        "content": (
            f" [Internal Resource: docs://{topic}]\n{resource_text}\n\n"
            f" [User Question]\n{user_q}\n\n"
            "Answer in Korean, structured with bullets."
        ),
    })

    res = await agent.ainvoke({"messages": messages})
    final = res["messages"][-1].content
    return {"answer": final}

# Graph
def build_graph():
    g = StateGraph(GraphState)
    g.add_node("router", router_node)
    g.add_node("load_prompt", load_prompt_node)
    g.add_node("load_resource", load_resource_node)
    g.add_node("answer", answer_node)

    g.set_entry_point("router")

    g.add_edge("router", "load_prompt")
    g.add_edge("load_prompt", "load_resource")
    g.add_edge("load_resource", "answer")
    g.add_edge("answer", END)

    return g.compile()


