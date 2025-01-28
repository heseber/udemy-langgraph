from typing import Callable

from dotenv import load_dotenv
from langchain.chains.base import Chain
from langchain.schema import HumanMessage
from langchain_core.messages import ToolMessage
from langgraph.graph import END, StateGraph

from chains import first_responder, revisor
from search import execute_search
from state import State

load_dotenv()

MAX_ITERATIONS = 2
DRAFT_CREATOR = "draft_creator"
TOOL_EXECUTOR = "tool_executor"
REVISOR = "revisor"


# Create a function that returns a wrapper turning a chain into a node
def create_chain_node(chain: Chain) -> Callable[[State], State]:
    async def wrapper(state: State) -> State:
        messages = state["messages"]
        result = await chain.ainvoke(input={"messages": messages})
        return {"messages": [result]}

    return wrapper


# Create the graph
builder = StateGraph(State)

# Add nodes
builder.add_node(DRAFT_CREATOR, create_chain_node(first_responder))
builder.add_node(TOOL_EXECUTOR, execute_search)
builder.add_node(REVISOR, create_chain_node(revisor))

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
    import asyncio

    async def main():
        print("Thinking ...")
        input = HumanMessage(
            # content="Write about AI-Powered SOC / autonomous soc problem domain."
            # " List startups that do that and raised capital."
            # " Keep a list of startups in the final answer."
            content="Write about the spatial transcriptomics domain."
            " List companies that provide devices for this technology."
            " Provide a short comparison of the technologies from different vendors."
        )
        state = State(messages=[input])
        res = await graph.ainvoke(state)
        final_message = res["messages"][-1]
        print(final_message.tool_calls[0]["args"]["answer"])

    asyncio.run(main())
