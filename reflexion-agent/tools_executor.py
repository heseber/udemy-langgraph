from typing import List

from dotenv import load_dotenv
from langchain.schema import AIMessage, BaseMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode

from chains import parser
from schemas import AnswerQuestion, Reflection

load_dotenv()

tavily_tool = TavilySearchResults()
tavily_tool_node = ToolNode(tools=[tavily_tool])


def execute_tools(state: List[BaseMessage]) -> List[ToolMessage]:
    last_ai_message: AIMessage = state[-1]
    parsed_tool_calls = parser.invoke(last_ai_message)
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
    state = {"messages": [AIMessage(content="", tool_calls=tool_calls)]}
    results = tavily_tool_node.invoke(state)
    pass


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

    raw_res = execute_tools(
        state=[
            human_message,
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": AnswerQuestion.__name__,
                        "args": answer.model_dump(),
                        "id": "call_KpYHichFFEmLitHFvFhKy1Ra",
                    }
                ],
            ),
        ]
    )

    print(raw_res)
