# rag-playground

RAGアプリケーションのサンプル。

## 仕様

- WikipediaのデータをチャンキングしてRAGインデックスを作成
- LLMにチャットを投げ、その際にコンテキストとしてWikipediaデータのチャンクを与える
- LLMはチャットに対して回答し、参考にしたWikipediaのチャンクをソースとして引用する

## 構成

- RAGインデックス: OpenSearch
- チャンキング: RecursiveCharacterTextSplitter(LangChain)
- Embedding: sentence-transformers/all-MiniLM-L6-v2
- LLM: Llama 3.2 1B

## 実行

```bash
# ollamaのモデル用意
ollama pull llama3.2:1b

# データディレクトリ作成
mkdir data

# https://drive.google.com/file/d/1KbRqykxvNRPkEZrznObf6uvCR1YIXB3A/view?usp=drive_link からwikipediaのデータをダウンロード、wikipedia_jaディレクトリをdataディレクトリにコピー

# OpenSearch, Langfuse起動
docker compose up -d

# localhost:3000にアクセスし、Langfuseのorg, projectを作成、public key, secret keyをexport

export LANGFUSE_HOST="http://localhost:3000"
export LANGFUSE_PUBLIC_KEY="your_public_key"
export LANGFUSE_SECRET_KEY="your_secret_key"


# OpenSearchでRAGインデックス構築
uv run python feed.py

# アプリケーション起動
uv run python main.py
```
