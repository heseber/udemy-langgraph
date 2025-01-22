from typing import List

from pydantic import BaseModel, Field


class Reflection(BaseModel):
    missing: str = Field(
        description="A comprehensive description of key information that is missing from the text."
    )
    superfluous: str = Field(
        description="A description of information that is unnecessary or could be removed."
    )


class AnswerQuestion(BaseModel):
    """Answer the question."""

    answer: str = Field(description="~250 word detailed answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    search_queries: List[str] = Field(
        description="Specific search queries to gather missing information.",
        min_length=1,
        max_length=3,
    )


class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question."""

    references: List[str] = Field(
        description="Citations motivating your updated answer."
    )
