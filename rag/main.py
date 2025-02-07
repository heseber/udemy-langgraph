from dotenv import load_dotenv

from graph.graph_core import app

load_dotenv()

if __name__ == "__main__":
    print("Running RAG...")
    # question = "How to make a pizza?"
    question = "What is agent memory?"
    # question = "What are technologies for spatial transcriptomics?"
    result = app.invoke(input={"question": question})
    print(result["generation"])
