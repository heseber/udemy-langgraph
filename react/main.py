import asyncio
import threading
import warnings

from dotenv import load_dotenv
from langchain.schema import HumanMessage
from langgraph.graph import END, MessagesState, StateGraph

from nodes import run_agent_reasoning, tool_node

load_dotenv()

AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1


def suppress_threading_error():
    """Suppress the Python 3.13 threading cleanup error."""
    try:
        original_del = threading._DeleteDummyThreadOnDel.__del__

        def safe_del(self):
            try:
                original_del(self)
            except (TypeError, AttributeError):
                # Silently ignore the error
                pass

        threading._DeleteDummyThreadOnDel.__del__ = safe_del
    except (AttributeError, TypeError):
        # If the patch fails, continue without it
        pass


def should_continue(state: MessagesState) -> str:
    if not state["messages"][LAST].tool_calls:
        return END
    else:
        return ACT


flow = StateGraph(MessagesState)

flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)

flow.add_conditional_edges(AGENT_REASON, should_continue, {ACT: ACT, END: END})

flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")


async def async_main():
    """Async main function."""
    try:
        print("Hello ReAct LangGraph with Function Calling")
        res = app.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="What is the temperature in Berlin, Germany in Celsius? List it and then Triple it."
                    )
                ]
            }
        )
        print(res["messages"][LAST].content)
    except Exception as e:
        print(f"Error occurred: {e}")


def main():
    # Apply threading error suppression
    suppress_threading_error()

    # Suppress warnings about unclosed threads
    warnings.filterwarnings("ignore", message=".*unclosed.*thread.*")

    try:
        # Run the async main function
        asyncio.run(async_main())
    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    main()
