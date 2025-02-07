from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field

from graph.chains.llm import llm


class GradeHallucination(BaseModel):
    """Binare score for hallucination present in generation answer."""

    binary_score: bool = Field(
        description="Binary score classifying the generation answer is grounded in the retrieved documents or not,"
        "allowed values are 'yes' and 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeHallucination)

system_message = """
You are a grader assessing the relevance of a generation answer to the retrieved documents.
If the generation answer is grounded in the retrieved documents, grade it as 'yes'.
Otherwise, grade it as 'no'.
"""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_message),
        (
            "human",
            "Set of documents: \n\n{documents} \n\n LLM generation: {generation}",
        ),
    ]
)

hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader
