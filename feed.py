import json
import glob
from pathlib import Path
from typing import List

from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from opensearchpy import OpenSearch

DATA_SOURCE_DIR = "data/wikipedia_ja/*"
OPENSEARCH_URL = "https://localhost:9200"
OPENSEARCH_ADMIN_USER = "admin"
OPENSEARCH_ADMIN_PASSWORD = "yourStrongPassword123!"
INDEX_NAME = "wiki-rag"

def load_wiki_documents(file_path: Path) -> List[Document]:
    docs: List[Document] = []
    with file_path.open("r", encoding="utf-8") as f:
        raw_docs = json.load(f)
        for raw_doc in raw_docs:
            text = raw_doc["text"]
            metadata = {
                "id": raw_doc["id"],
                "title": raw_doc["title"],
                "url": raw_doc["url"]
            }
            docs.append(Document(page_content=text, metadata=metadata))
    return docs

def main():
    # OpenSearchの初期化
    client = OpenSearch(
        hosts=[OPENSEARCH_URL],
        http_auth=(OPENSEARCH_ADMIN_USER, OPENSEARCH_ADMIN_PASSWORD),
        verify_certs=False,
    )
    if client.indices.exists(index=INDEX_NAME):
        client.indices.delete(index=INDEX_NAME)
    index_body = {
        "settings": {
            "index": {
                "knn": True
            }
        },
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "metadata": {"type": "object"},
                "vector_field": {
                    "type": "knn_vector",
                    "dimension": 384,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib"
                    }
                }
            }
        },
    }
    client.indices.create(index=INDEX_NAME, body=index_body)

    # 埋め込みモデル
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # OpenSearch VectorStore を初期化
    vectorstore = OpenSearchVectorSearch(
        opensearch_url=OPENSEARCH_URL,
        index_name=INDEX_NAME,
        embedding_function=embedding_function,
        http_auth=(OPENSEARCH_ADMIN_USER, OPENSEARCH_ADMIN_PASSWORD),
        verify_certs=False,
        bulk_size=10000
    )

    for path in glob.glob(DATA_SOURCE_DIR):
        # Wikipedia ドキュメント読み込み
        raw_docs = load_wiki_documents(Path(path))

        # チャンキング
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", "。", "、", " "],
        )
        chunked_docs = splitter.split_documents(raw_docs)

        # チャンク済みドキュメントを投入
        vectorstore.add_documents(chunked_docs)

        print(f"Indexed {len(chunked_docs)} chunks into index '{INDEX_NAME}'")

if __name__ == "__main__":
    main()
