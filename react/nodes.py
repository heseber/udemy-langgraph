from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from agent_setup import llm, tools

load_dotenv()

SYSTEM_MESSAGE = """
You are a helpful assistant that can use tools to answer questions.

IMPORTANT INSTRUCTIONS:
1. Use the search_weather tool to find current temperature information for locations
2. Use the triple tool to multiply numbers by 3
3. After getting the information you need, provide a clear final answer
4. Do not make unnecessary tool calls once you have the required information
5. Be concise and direct in your responses
"""


def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning node.
    """
    response = llm.invoke(
        [{"role": "system", "content": SYSTEM_MESSAGE}, *state["messages"]]
    )
    return {"messages": [response]}


tool_node = ToolNode(tools)
