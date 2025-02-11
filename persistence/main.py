import sqlite3
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    input: str
    user_feedback: str


def step_1(state: State) -> State:
    print("--- Step 1 ---")
    print(f"Input: {state['input']}")
    return {}


def human_feedback(state: State) -> State:
    print("--- Human Feedback ---")
    print(f"User Feedback: {state['user_feedback']}")
    return {}


def step_3(state: State) -> State:
    print("--- Step 3 ---")
    return {}


builder = StateGraph(State)

builder.add_node("step_1", step_1)
builder.add_node("human_feedback", human_feedback)
builder.add_node("step_3", step_3)

builder.add_edge(START, "step_1")
builder.add_edge("step_1", "human_feedback")
builder.add_edge("human_feedback", "step_3")
builder.add_edge("step_3", END)

conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory_saver = SqliteSaver(conn)

graph = builder.compile(checkpointer=memory_saver, interrupt_before=["human_feedback"])

graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

if __name__ == "__main__":
    thread = {"configurable": {"thread_id": "1  "}}
    initial_input = {"input": "Hello, world!"}
    for event in graph.stream(initial_input, thread, stream_mode="values"):
        print(event)
    print(graph.get_state(thread).next)

    user_input = input("Tell me know you want to update the state: ")
    graph.update_state(thread, {"user_feedback": user_input}, as_node="human_feedback")
    print("-- State after human feedback ---")
    print(graph.get_state(thread))
    for event in graph.stream(None, thread, stream_mode="values"):
        print(event)
