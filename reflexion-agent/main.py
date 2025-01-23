from dotenv import load_dotenv
from langchain.schema import HumanMessage
from langchain_core.messages import ToolMessage
from langgraph.graph import END, Graph

from chains import first_responder, revisor
from state import State
from tools_executor import execute_tools

load_dotenv()

MAX_ITERATIONS = 2
DRAFT_CREATOR = "draft_creator"
TOOL_EXECUTOR = "tool_executor"
REVISOR = "revisor"

# Create the graph
builder = Graph()

# Add nodes
builder.add_node(DRAFT_CREATOR, first_responder)
builder.add_node(TOOL_EXECUTOR, execute_tools)
builder.add_node(REVISOR, revisor)

# Add edges
builder.add_edge(DRAFT_CREATOR, TOOL_EXECUTOR)
builder.add_edge(TOOL_EXECUTOR, REVISOR)


def event_loop(state: State) -> str:
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state["messages"])
    num_iterations = count_tool_visits
    if num_iterations > MAX_ITERATIONS:
        return END
    return TOOL_EXECUTOR


builder.add_conditional_edges(
    REVISOR,
    event_loop,
    {TOOL_EXECUTOR: TOOL_EXECUTOR, END: END},  # Explicitly define possible destinations
)
builder.set_entry_point(DRAFT_CREATOR)

# Compile the graph
graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")


if __name__ == "__main__":
    print("Hello Reflexion")
    input = HumanMessage(
        content="Write about AI-Powered SOC / autonomous soc problem domain."
        " List startups that do that and raised capital."
    )
    state = State(messages=[input])
    res = graph.invoke(state)
    final_message = res["messages"][-1]
    print(final_message.tool_calls[0]["args"]["answer"])
