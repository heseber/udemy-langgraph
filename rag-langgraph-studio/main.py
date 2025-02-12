from pprint import pprint

from dotenv import load_dotenv

from graph.graph_core import app

load_dotenv()

if __name__ == "__main__":
    print("Running RAG...")
    # question = "How to make a pizza?"
    question = "What is agent memory?"
    # question = "What are technologies for spatial transcriptomics?"
    inputs = {"question": question}
    for output in app.stream(inputs, config={"configurable": {"thread_id": "2"}}):
        for key, value in output.items():
            pprint(f"Finished running: {key}:")
    print(value["generation"])
