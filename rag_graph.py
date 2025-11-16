import warnings
warnings.filterwarnings(
    "ignore",
    message=".*verify_certs=False is insecure.*"
)
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from typing import List, TypedDict

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

from rag_chain import build_rag_chain

# ---- State 定義 ----
class RAGState(TypedDict):
    question: str
    answer: str
    docs: List[Document]


rag_chain = build_rag_chain()


# ---- ノード: RAG 実行 ----
def retrieve_and_answer(state: RAGState, config: RunnableConfig) -> RAGState:
    question = state["question"]
    result = rag_chain.invoke(question, config=config)
    # result は {"answer": str, "docs": List[Document]} という想定
    return {
        "question": question,
        "answer": result["answer"],
        "docs": result["docs"],
    }


# ---- ノード: 出典付き出力整形 ----
def format_output(state: RAGState, config: RunnableConfig) -> RAGState:
    docs = state["docs"]
    answer = state["answer"]

    # 出典リストを整形
    sources_lines = []
    for i, d in enumerate(docs, start=1):
        m = d.metadata
        title = m.get("title", "Unknown")
        url = m.get("url", "")
        label = f"[{i}] {title}" + (f" - {url}" if url else "")
        sources_lines.append(label)

    sources_text = "\n".join(sources_lines)

    formatted_answer = (
        answer
        + "\n\n---\n\n出典:\n"
        + sources_text
    )

    return {
        **state,
        "answer": formatted_answer,
    }


# ---- Graph 構築 ----
def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("format_output", format_output)

    graph.set_entry_point("retrieve_and_answer")
    graph.add_edge("retrieve_and_answer", "format_output")
    graph.add_edge("format_output", END)

    return graph.compile()
