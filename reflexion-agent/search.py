import json
import os

from dotenv import load_dotenv
from langchain.output_parsers import JsonOutputToolsParser
from langchain.schema import AIMessage, HumanMessage
from langchain_community.tools import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.messages import ToolMessage

from schemas import AnswerQuestion, Reference, ReviseAnswer
from state import State

load_dotenv()

USE_ANTHROPIC = os.getenv("USE_ANTHROPIC", "true").lower() == "true"

# Create a singleton instance of the Tavily Search
tavily_api_wrapper = TavilySearchAPIWrapper()
tavily_search = TavilySearchResults(api_wrapper=tavily_api_wrapper)

# Get llm output parser based on which LLM we're using
parser = JsonOutputToolsParser(tools=[AnswerQuestion, ReviseAnswer])


# Anthropic expects a different message format than OpenAI, therefore we
# define an auxiliary function creating a tool message of the required format.
def create_tool_message(tool_call_id: str, result: list) -> ToolMessage:
    # Convert Reference objects to dictionaries
    if isinstance(result[0], Reference):
        result = [ref.model_dump() for ref in result]

    if USE_ANTHROPIC:
        formatted_content = {
            "type": "tool_response",
            "content": result,
        }
    else:
        formatted_content = result

    return ToolMessage(content=json.dumps(formatted_content), tool_call_id=tool_call_id)


async def execute_search(state: State) -> State:
    """Transform the tool calls to prepare them for calling Tavily Search"""

    # Increment the iteration counter
    state["iteration"] += 1

    # Get the last message
    last_message: AIMessage = state["messages"][-1]

    # Assert that there is exactly one tool call in the last message
    if not last_message.tool_calls:
        raise ValueError("No tool call in response")
    if len(last_message.tool_calls) > 1:
        raise ValueError("Multiple tool calls in response")

    # Parse the tool call
    tool_call = last_message.tool_calls[0]
    tool_call_id = tool_call["id"]

    try:
        parsed_tools = parser.invoke(last_message)
        if not parsed_tools:
            raise ValueError("No valid tools were parsed from the message")
        search_queries = []
        for tool in parsed_tools:
            # First try to get queries directly from args
            queries = tool["args"].get("search_queries", [])
            if not queries:
                # If not found, try to get from reflection
                reflection = tool["args"].get("reflection", {})
                queries = reflection.get("search_queries", [])
            if queries:
                search_queries.extend(queries)
        if not search_queries:
            raise ValueError("No search queries found in tools")
    except Exception as e:
        raise ValueError(f"Failed to parse tools from message: {str(e)}")

    # Create the tool calls for Tavily Search
    tool_calls = []
    for query in search_queries:
        tool_calls.append(
            {
                "args": {"query": query},
                "type": "tool_call",
                "id": tool_call_id,
                "name": "tavily_search_results_json",
            }
        )

    # Run the Tavily Search
    search_results: list[ToolMessage] = await tavily_search.abatch(tool_calls)

    # Parse the search results
    new_references: list[Reference] = []
    for message in search_results:
        # Sometimes a search may fail. We don't retry but just skip that search.
        if "results" not in message.artifact:
            continue
        # Extract the references from the search results
        for result in message.artifact["results"]:
            new_reference = Reference(
                url=result["url"],
                title=result["title"],
                content=result["content"],
                index=len(state["references"]) + 1,
            )
            new_references.append(new_reference)
            state["references"].append(new_reference)

    # Create the tool message for the tool call
    results = create_tool_message(tool_call_id=tool_call_id, result=new_references)
    state["messages"].append(results)
    return state


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

        state = State(
            messages=[
                human_message,
                chat_generation.message,  # Use the wrapped message
            ],
            references=[],
            iteration=0,
            max_iterations=2,
        )

        result = await execute_search(state)
        print(result)

    asyncio.run(main())
