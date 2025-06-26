import os

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()


@tool
def triple(num: float) -> float:
    """Calculates the triple of a number.

    Args:
        num (float): number to triple

    Returns:
        float: triple of the number
    """
    return float(num) * 3


# Configure TavilySearch tool properly
tavily_weather_tool = TavilySearch(
    name="search_weather",
    description="Search for current weather information for any location. Use this to find temperature data.",
    max_results=1,
    time_range="day",
)

tools = [triple, tavily_weather_tool]

# Get the LLM, either Anthropic Sonnet 3.5 or OpenAI gpt4-o-mini
USE_ANTHROPIC = os.getenv("USE_ANTHROPIC", "true").lower() == "true"
if USE_ANTHROPIC:
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,  # Adjust this number as needed (up to 4096 for Claude 3 Sonnet)
        temperature=0,
    )
else:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

llm = llm.bind_tools(tools)
