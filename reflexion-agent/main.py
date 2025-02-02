from dotenv import load_dotenv
from langchain.schema import HumanMessage
from langgraph.graph import END, StateGraph

from chains import first_responder_chain, revisor_chain
from search import execute_search
from state import State

load_dotenv()

MAX_ITERATIONS = 2
DRAFT_CREATOR = "draft_creator"
TOOL_EXECUTOR = "tool_executor"
REVISOR = "revisor"


async def first_responder_node(state: State) -> State:
    result = await first_responder_chain.ainvoke(input=state)
    return {"messages": [result]}


async def revisor_node(state: State) -> State:
    result = await revisor_chain.ainvoke(input=state)
    return {"messages": [result]}


# Create the graph
builder = StateGraph(State)

# Add nodes
builder.add_node(DRAFT_CREATOR, first_responder_node)
builder.add_node(TOOL_EXECUTOR, execute_search)
builder.add_node(REVISOR, revisor_node)

# Add edges
builder.add_edge(DRAFT_CREATOR, TOOL_EXECUTOR)
builder.add_edge(TOOL_EXECUTOR, REVISOR)


def event_loop(state: State) -> str:
    if state["iteration"] > state["max_iterations"]:
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
            " Make sure you use up-to-date information as of today."
            " List all companies that provide devices for this technology, and list all technologies they offer."
            " Provide a short comparison of the technologies from different vendors."
        )
        state = State(
            messages=[input], references=[], iteration=0, max_iterations=MAX_ITERATIONS
        )
        res = await graph.ainvoke(state)
        final_message = res["messages"][-1]
        print(final_message.tool_calls[0]["args"]["answer"])
        print("""
All References:
While the references section above contains only references from one iteration
of the Reflexion loop (first iteration for OpenAI, last iteration for Anthropic),
the reference list below contains all references that were returned from internet
searches for the queries suggested by the LLM.
""")
        for ref in res["references"]:
            print(f"- [{ref.index}] {ref.url}")

    asyncio.run(main())
