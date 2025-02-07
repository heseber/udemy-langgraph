from typing import Any

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

from graph.state import GraphState

load_dotenv()

tavily_api_wrapper = TavilySearchAPIWrapper()
web_search_tool = TavilySearchResults(api_wrapper=tavily_api_wrapper, max_results=3)


def web_search(state: GraphState) -> dict[str, Any]:
    """Runs a web search and returns the results."""
    print("---RUN WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]
    tavily_results = web_search_tool.invoke(question)
    joined_tavily_results = "\n".join([result["content"] for result in tavily_results])
    web_results = Document(page_content=joined_tavily_results)
    if documents is None:
        documents = [web_results]
    else:
        documents.append(web_results)
    return {"documents": documents}


if __name__ == "__main__":
    web_search(GraphState(question="agent memory", documents=None))
