from typing import TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

docs = [
    "RAG는 Retrieval-Augmented Generation의 약자다.",
    "Chunking은 토큰 기준으로 나누는 것이 중요하다.",
    "Query Rewrite는 검색 정확도를 크게 높인다."
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80
)

chunks = splitter.create_documents(docs)

emb = OpenAIEmbeddings()
db = Chroma.from_documents(chunks, emb)
retriever = db.as_retriever(search_kwargs={"k": 6})

# 1) State 정의
class AgentState(TypedDict, total=False):
    question: str
    route: Literal["rag", "calc", "clarify"]
    rag_context: str
    calc_result: str
    answer: str
    attempts: int

# 2) LLM 준비
llm_router = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_answer = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 3) Router Node
router_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a router. Decide the best route for the user question.\n" 
     "Routes: \n" 
     "- rag: needs factual lookup from docs/knowledge\n" 
     "- calc: needs math calculation\n" 
     "- clarify: user request is ambiguous; ask a clarifying question\n" 
     "Return ONLY one word: rag or calc or clarify."),
    ("user", "{question}")])

def router_node(state: AgentState) -> AgentState:
    q = state["question"]
    attempts = state.get("attempts", 0)
    route = (router_prompt | llm_router).invoke({"question": q}).content.strip().lower()

    if route not in ("rag", "calc", "clarify"):
        route = "clarify"

    return {"route": route, "attempts": attempts}

# 4) RAG Node
def rag_node_factory(retriever):
    def rag_node(state: AgentState) -> AgentState:
        q = state["question"]
        docs = retriever.get_relevant_documents(q)
        docs = docs[:4]
        if not docs:
            return {"rag_context": "NO_RESULT"}
        ctx = "\n\n".join([f"[Source {i+1}] {d.page_content}" for i, d in enumerate(docs)])
        return {"rag_context": ctx}
    return rag_node

# 5) Calc Node
calc_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Extract a single math expression from the user question.\n"
     "Return ONLY the expression like: 12*3+4\n"
     "If no clear expression, return: NO_EXPR"),
    ("user", "{question}")
])

def safe_calc(expr: str) -> str:
    allowed = set("0123456789+-*/(). ")
    if any(ch not in allowed for ch in expr):
        return "ERROR: invalid characters"
    try:
        return str(eval(expr, {"__builtins__": {}}))
    except Exception as e:
        return f"ERROR: {e}"

def calc_node(state: AgentState) -> AgentState:
    q = state["question"]
    expr = (calc_prompt | llm_router).invoke({"question": q}).content.strip()
    if expr == "NO_EXPR":
        return {"calc_result": "NO_EXPR"}
    return {"calc_result": safe_calc(expr)}

# 6) Clarify Node
clarify_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Ask ONE short clarifying question to proceed. Keep it concise."),
    ("user", "{question}")
])

def clarify_node(state: AgentState) -> AgentState:
    q = state["question"]
    ask = (clarify_prompt | llm_answer).invoke({"question": q}).content.strip()
    return {"answer": ask}

# 7) Answer Node
answer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You ar a helpful assistent.\n",
     "Use provided context if present. if context says NO_RESULT, say you don't know and ask for more info.\n"
     "If calc_result is present, include it.\n"
     "Keep the final answer clear and structured.\n"),
    ("user",
     "Question: {question}\n\n"
     "Context:\n{rag_context}\n\n"
     "Calc:\n{clac_result}\n\n"
     "Answer:")
])

def answer_node(state: AgentState) -> AgentState:
    q = state["question"]
    ctx = state.get("rag_context", "")
    calc_res = state.get("calc_result", "")

    if ctx == "NO_RESULT":
        return {"answer": "관련 문서를 찾지 못했어. 어떤 문서/도메인을 기준으로 답해야 하는지(예: 사내 위키, 특정 제품 문서) 알려줄래?"}

    out = (answer_prompt | llm_answer).invoke({
        "question": q,
        "rag_context": ctx,
        "calc_result": calc_res
    }).content.strip()
    return {"answer": out}

# 8) 조건 분기 (Edges)
def router_selector(state: AgentState) -> str:
    return state["route"]

def build_graph(retriever):
    graph = StateGraph(AgentState)

    rag_node = rag_node_factory(retriever)

    graph.add_node("router", router_node)
    graph.add_node("rag", rag_node)
    graph.add_node("calc", calc_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges("router", router_selector, {
        "rag": "rag",
        "calc": "calc",
        "clarify": "clarify",
    })

    graph.add_edge("rag", "answer"),
    graph.add_edge("calc", "answer"),
    graph.add_edge("clarify", END)
    graph.add_edge("answer", END)

    return graph.compile()


app = build_graph(retriever)

print(app.invoke({"question": "RAG가 뭐야?"}))
print(app.invoke({"question": "12*3+4 계산해줘"}))
print(app.invoke({"question": "이거 어떻게 해야하는거야?"}))



