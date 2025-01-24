import json
from collections import defaultdict

from dotenv import load_dotenv
from langchain.schema import AIMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode

from chains import USE_ANTHROPIC, parser
from schemas import AnswerQuestion, Reflection
from state import State

load_dotenv()


tavily_tool = TavilySearchResults()
tavily_tool_node = ToolNode(tools=[tavily_tool])


def create_tool_message(result: dict, tool_call_id: str) -> ToolMessage:
    if USE_ANTHROPIC:
        formatted_content = {
            "type": "tool_result",
            "content": json.dumps(result),
        }
    else:
        formatted_content = json.dumps(result)
    tool_message = ToolMessage(content=formatted_content, tool_call_id=tool_call_id)
    return tool_message


def execute_tools(state: State) -> State:
    # Access messages through the messages attribute of the State object
    last_ai_message: AIMessage = state["messages"][-1]
    parsed_tool_calls = parser.invoke(last_ai_message)

    # The tool calls have a list of search strings as arguments.
    # This needs to be split into separate tool calls for each query.
    ids = []
    tool_calls = []
    for parsed_tool_call in parsed_tool_calls:
        for query in parsed_tool_call["args"]["search_queries"]:
            tool_calls.append(
                {
                    "name": "tavily_search_results_json",
                    "args": {"query": query},
                    "id": parsed_tool_call["id"],
                    "type": "tool_call",
                }
            )
            ids.append(parsed_tool_call["id"])

    # Run Tavily search
    search_results = tavily_tool_node.invoke(
        State(messages=[AIMessage(content="", tool_calls=tool_calls)])
    )

    # Now the search results for the different queries need to be merged
    # into a single ToolMessage.

    # Map each search result to its corresponding ID and query
    results_map = defaultdict(dict)
    for id_, message, tool_call in zip(ids, search_results["messages"], tool_calls):
        results_map[id_][tool_call["args"]["query"]] = message.content

    # Create a single ToolMessage with all results
    all_results = []
    for id_, mapped_result in results_map.items():
        all_results.append({"tool_call_id": id_, "result": mapped_result})

    tool_message = create_tool_message(result=all_results, tool_call_id=ids[0])
    return {"messages": [tool_message]}


if __name__ == "__main__":
    print("Tool Executor Enter")

    human_message = HumanMessage(
        content="Write about AI-Powered SOC / autonomous soc problem domain,"
        " list startups that do that and raise capital."
    )

    answer = AnswerQuestion(
        answer="",
        reflection=Reflection(missing="", superfluous=""),
        search_queries=[
            "AI-powered SOC startups funding",
            "AI SOC problem domain specifics",
            "Technologies used by AI-powered SOC startups",
        ],
        id="call_KpYHichFFEmLitHFvFhKy1Ra",
    )

    state = State()
    state["messages"] = [
        human_message,
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": AnswerQuestion.__name__,
                    "args": answer.model_dump(),
                    "id": "call_KpYHichFFEmLitHFvFhKy1Ra",
                    "type": "tool_call",
                }
            ],
        ),
    ]
    raw_res = execute_tools(state=state)

    raw_res = execute_tools(state)
    print(raw_res)
