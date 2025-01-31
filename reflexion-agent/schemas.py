from pydantic import BaseModel, Field


class Reference(BaseModel):
    url: str = Field(description="The URL of the reference.")
    title: str = Field(description="The title of the reference.")
    content: str = Field(description="The content of the reference.")
    index: int = Field(description="The 1-based index of the reference in the list.")


class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous.")


class AnswerQuestion(BaseModel):
    """Answer the question."""

    answer: str = Field(description="~250 word detailed answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    search_queries: list[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )


class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question."""

    references: list[Reference] = Field(
        description="Citations motivating your updated answer."
    )
