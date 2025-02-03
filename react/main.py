from dotenv import load_dotenv
from langchain_core.agents import AgentFinish
from langgraph.graph import END, StateGraph

from nodes import execute_tool, run_agent_reasoning_engine
from state import AgentState

load_dotenv()

AGENT_REASON = "agent_reason"
ACT = "act"


def should_continue(state: AgentState) -> str:
    if isinstance(state["agent_outcome"], AgentFinish):
        return END
    else:
        return ACT


builder = StateGraph(AgentState)
builder.add_node(AGENT_REASON, run_agent_reasoning_engine)
builder.add_node(ACT, execute_tool)

builder.add_conditional_edges(AGENT_REASON, should_continue)
builder.add_edge(ACT, AGENT_REASON)

builder.set_entry_point(AGENT_REASON)

graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

if __name__ == "__main__":
    print("Running ReAct with LangGraph")
    res = graph.invoke(
        {"input": "What is the weather in Berlin, Germany? List it and then Triple it."}
    )
    print(res["agent_outcome"].return_values["output"])
