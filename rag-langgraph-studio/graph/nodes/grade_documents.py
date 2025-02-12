from typing import Any

from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState


def grade_documents(state: GraphState) -> dict[str, Any]:
    """Determines whether the retrieved dcouments are relevant to the question.
    If any document is not relevant, we will set a flag to run a web search.

    Args:
        state (GraphState): The current state of the graph.

    Returns:
        dict[str, Any]: Filtered documents and a flag indicating whether to run a web search.
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    documents = state["documents"]
    question = state["question"]

    filtered_docs = []
    web_search = False
    for doc in documents:
        results = retrieval_grader.invoke({"documents": doc, "question": question})
        if results.binary_score.lower() == "yes":
            print("---GRADE: DOCUMENT IS RELEVANT---")
            filtered_docs.append(doc)
        else:
            print("---GRADE: DOCUMENT IS NOT RELEVANT---")
            web_search = True
            continue

    return {
        "documents": filtered_docs,
        "web_search": web_search,
        "question": question,
    }
