import json
import os
from collections import defaultdict

from dotenv import load_dotenv
from langchain.output_parsers import PydanticToolsParser
from langchain.schema import AIMessage, HumanMessage
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import ToolMessage

from schemas import AnswerQuestion
from state import State

load_dotenv()

USE_ANTHROPIC = os.getenv("USE_ANTHROPIC", "true").lower() == "true"

# Create a singleton instance of the Tavily Search
tavily_search = TavilySearchResults()

parser_pydantic = PydanticToolsParser(
    tools=[AnswerQuestion],
    return_id=True,  # This should make it return IDs
    first_tool_only=False,
)


# Anthropic expects a different message format than OpenAI, therefore we
# define an auxiliary function creating a tool message of the required format.
def create_tool_message(tool_call_id: str, result: list) -> ToolMessage:
    if USE_ANTHROPIC:
        formatted_content = {
            "type": "tool_result",
            "output": json.dumps(result),
        }
    else:
        formatted_content = result
    tool_message = ToolMessage(
        content=json.dumps(formatted_content), tool_call_id=tool_call_id
    )
    return tool_message


async def execute_search(state: State) -> State:
    # Transform the tool calls to prepare them for calling Tavily Search
    last_message: AIMessage = state["messages"][-1]
    try:
        parsed_tools: list[AnswerQuestion] = parser_pydantic.invoke(last_message)
        if not parsed_tools:
            raise ValueError("No valid tools were parsed from the message")

        # Validate each tool has search queries
        for tool in parsed_tools:
            if not tool.search_queries:
                raise ValueError(f"Tool {tool.id} has no search queries")
    except Exception as e:
        raise ValueError(f"Failed to parse tools from message: {str(e)}")

    tool_calls = []
    for tool in parsed_tools:
        for query in tool.search_queries:
            tool_calls.append(
                {
                    "args": {"query": query},
                    "type": "tool_call",
                    "id": tool.id,
                    "name": "tavily_search_results_json",
                }
            )
    # Run the Tavily Search
    search_results: list[ToolMessage] = await tavily_search.abatch(tool_calls)
    # Combine results into a single ToolMessage for each tool call id
    results_map = defaultdict(list)
    for message in search_results:
        id = message.tool_call_id
        query = message.artifact["query"]
        content = json.loads(message.content)
        results_map[id].extend(content)
    results = [create_tool_message(id, content) for id, content in results_map.items()]

    return {"messages": results}


if __name__ == "__main__":
    import asyncio

    from langchain.schema import ChatGeneration

    async def main():
        print("Thinking")
        human_message = HumanMessage(content="Answer the questions below.")

        # Format the message as a list of tool invocations
        tool_content = [
            {
                "type": "AnswerQuestion",
                "id": "call_KpYHichFFEmLitHFvFhKy1Ra",
                "args": {
                    "answer": "",
                    "reflection": {"missing": "", "superfluous": ""},
                    "search_queries": [
                        "who won the last french open",
                        "when was Albert Einstein born",
                        "where did Napoleon die",
                    ],
                    "id": "call_KpYHichFFEmLitHFvFhKy1Ra",
                },
            }
        ]

        # Create a proper chat generation
        tool_calls = [
            {
                "name": AnswerQuestion.__name__,
                "args": tool_content[0]["args"],
                "id": tool_content[0]["id"],
                "type": "tool_call",
            }
        ]

        ai_message = AIMessage(
            content=str(tool_content),
            tool_calls=tool_calls,
        )

        # Wrap the message in a ChatGeneration
        chat_generation = ChatGeneration(message=ai_message)

        state = State()
        state["messages"] = [
            human_message,
            chat_generation.message,  # Use the wrapped message
        ]
        result = await execute_search(state)
        print(result)

    asyncio.run(main())
