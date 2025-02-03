import os

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_community.tools import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_openai import ChatOpenAI

load_dotenv()

react_prompt: PromptTemplate = hub.pull("hwchase17/react")


@tool
def triple(num: float) -> float:
    """Calculates the triple of a number.

    Args:
        num (float): number to triple

    Returns:
        float: triple of the number
    """
    return float(num) * 3


tavily_api_wrapper = TavilySearchAPIWrapper()
tavily_search = TavilySearchResults(api_wrapper=tavily_api_wrapper, max_results=1)
tools = [triple, tavily_search]

# Get the LLM, either Anthropic Sonnet 3.5 or OpenAI gpt4-o-mini
USE_ANTHROPIC = os.getenv("USE_ANTHROPIC", "true").lower() == "true"
if USE_ANTHROPIC:
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-latest",
        max_tokens=2048,  # Adjust this number as needed (up to 4096 for Claude 3 Sonnet)
        temperature=0,
    )
else:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

react_agent_runnable = create_react_agent(llm, tools, react_prompt)
