import operator
from typing import Annotated, Sequence, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

load_dotenv()


class State(TypedDict):
    aggregate: Annotated[list, operator.add]
    which: str


class ReturnNodeValue:
    def __init__(self, node_secret: int):
        self._value = node_secret

    def __call__(self, state: State) -> State:
        import time

        time.sleep(1)
        print(f"Adding {self._value} to {state['aggregate']}\n")
        return {"aggregate": [self._value]}


def route_bc_or_cd(state: State) -> Sequence[str]:
    if state["which"] == "cd":
        return ["c", "d"]
    return ["b", "c"]


intermediates = ["b", "c", "d"]

builder = StateGraph(State)
builder.add_node("a", ReturnNodeValue("I'm A"))
builder.add_node("b", ReturnNodeValue("I'm B"))
builder.add_node("c", ReturnNodeValue("I'm C"))
builder.add_node("d", ReturnNodeValue("I'm D"))
builder.add_node("e", ReturnNodeValue("I'm E"))

builder.set_entry_point("a")
builder.add_conditional_edges(
    "a",
    route_bc_or_cd,
    intermediates,
)
for node in intermediates:
    builder.add_edge(node, "e")
builder.add_edge("e", END)

graph = builder.compile()

graph.get_graph().draw_mermaid_png(output_file_path="graph_async_2.png")

if __name__ == "__main__":
    print("Hello, Async Graph!")
    graph.invoke(
        {"aggregate": [], "which": "cd"}, config={"configurable": {"thread_id": "foo"}}
    )
