import os

from langchain import hub
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Get the LLM, either Anthropic Sonnet 3.5 or OpenAI gpt4-o-mini
USE_ANTHROPIC = os.getenv("USE_ANTHROPIC", "true").lower() == "true"
if USE_ANTHROPIC:
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-latest",
        temperature=0,
    )
else:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)


prompt = hub.pull("rlm/rag-prompt")


generation_chain = prompt | llm | StrOutputParser()
