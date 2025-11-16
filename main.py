from langfuse.langchain import CallbackHandler
from rag_graph import build_graph

def main():
    app = build_graph()
    langfuse_handler = CallbackHandler()

    while True:
        q = input("質問を入力してください (exitで終了): ")
        if q.strip().lower() == "exit":
            break

        state = {"question": q, "answer": "", "docs": []}

        result_state = app.invoke(
            state,
            config={
                "callbacks": [langfuse_handler],
                "tags": ["wiki-rag", "local-ollama"],
            },
        )

        print("\n=== 回答 ===\n")
        print(result_state["answer"])
        print("\n========================\n")

if __name__ == "__main__":
    main()
