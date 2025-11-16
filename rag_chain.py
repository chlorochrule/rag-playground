import warnings
warnings.filterwarnings(
    "ignore",
    message=".*verify_certs=False is insecure.*"
)
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

OPENSEARCH_URL = "https://localhost:9200"
OPENSEARCH_ADMIN_USER = "admin"
OPENSEARCH_ADMIN_PASSWORD = "yourStrongPassword123!"
INDEX_NAME = "wiki-rag"

def build_rag_chain():
    # VectorStore / Retriever
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = OpenSearchVectorSearch(
        opensearch_url=OPENSEARCH_URL,
        index_name=INDEX_NAME,
        embedding_function=embeddings,
        http_auth=(OPENSEARCH_ADMIN_USER, OPENSEARCH_ADMIN_PASSWORD),
        verify_certs=False
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # LLM (Ollama / llama3.2:1b)
    llm = ChatOllama(
        model="llama3.2:1b",
        temperature=0.1,
    )

    # プロンプト：コンテキスト＋質問から日本語で回答、出典はあとで整形して表示
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "あなたはWikipediaの内容に基づいて丁寧に日本語で回答するアシスタントです。"
                "与えられたコンテキストの範囲でのみ回答し、わからない場合はその旨を伝えてください。",
            ),
            (
                "human",
                "質問: {question}\n\n"
                "=== コンテキスト ===\n{context}\n\n"
                "上記のコンテキストに基づいて、ユーザーの質問に日本語で回答してください。"
            ),
        ]
    )

    def format_docs(docs):
        # LLM に渡すためのテキスト整形
        formatted = []
        for i, d in enumerate(docs, start=1):
            meta = d.metadata
            title = meta.get("title", "")
            url = meta.get("url", "")
            header = f"[{i}] {title} - {url}" if url else f"[{i}] {title}"
            formatted.append(f"{header}\n{d.page_content}")
        return "\n\n".join(formatted)

    # RAG チェーン構築
    rag_chain = (
        {"question": RunnablePassthrough()}
        | RunnableParallel(
            # 並列で context/docs を取得
            docs=lambda x: retriever.invoke(x["question"]),
            question=lambda x: x["question"],
        )
        | (
            # docs を context 文字列に変換しつつプロンプトへ
            lambda x: {
                "question": x["question"],
                "context": format_docs(x["docs"]),
                "docs": x["docs"],
            }
        )
        | (  # プロンプト & LLM
            lambda x: {
                "docs": x["docs"],
                "answer": StrOutputParser().invoke(
                    llm.invoke(prompt.format(question=x["question"], context=x["context"]))
                ),
            }
        )
    )

    return rag_chain
