from typing import Any

from graph.chains.generation import generation_chain
from graph.state import GraphState


def generate(state: GraphState) -> dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    generation = generation_chain.invoke({"question": question, "context": documents})
    return {"generation": generation}
