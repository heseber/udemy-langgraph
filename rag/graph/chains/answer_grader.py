from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field

from graph.chains.llm import llm


class GradeAnswer(BaseModel):
    """Binare score if the answer addresses the question    ."""

    binary_score: bool = Field(
        description="Binary score classifying if the answer addresses the question or not,"
        "allowed values are 'yes' and 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeAnswer)

system_message = """
You are a grader assessing the relevance of an answer to a question.
If the answer addresses the question, grade it as 'yes'.
Otherwise, grade it as 'no'.
"""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_message),
        ("human", "User question: {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader: RunnableSequence = answer_prompt | structured_llm_grader
