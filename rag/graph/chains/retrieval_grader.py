import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

# Get the LLM, either Anthropic Sonnet 3.5 or OpenAI gpt4-o-mini
USE_ANTHROPIC = os.getenv("USE_ANTHROPIC", "true").lower() == "true"
if USE_ANTHROPIC:
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-latest",
        temperature=0,
    )
else:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)


class GradeDocument(BaseModel):
    """Assesses the relevance of a document to a question."""

    binary_score: str = Field(
        description="Binary score classifying the document whether it is relevant to the question,"
        " allowed values are 'yes' and 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeDocument)

system_message = """
You are a grader assessing the relevance of a retrieved document to a user question.
If the document contains semantic meaning related to the question, grade it as relevant and give a binary score of 'yes'.
Otherwise, grade it as not relevant and give a binary score of 'no'.
"""

user_message = """
Retrieved document:

{documents}

User question: {question}
"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_message),
        ("human", user_message),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader
