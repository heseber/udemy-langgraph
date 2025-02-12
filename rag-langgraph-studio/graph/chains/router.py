from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from graph.chains.llm import llm


class RouteQuery(BaseModel):
    """Route a user query to the most relavant data source"""

    data_source: Literal["websearch", "vectorstore"] = Field(
        ..., description="The most relevant data source to answer the user query"
    )


router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are and expert at routing a user question to a vectorstore or a websearch.
            The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
            Use the vectorstore for questions on those topics. For all else, use the websearch.
            """,
        ),
        ("user", "{question}"),
    ]
)

question_router = router_prompt | llm.with_structured_output(RouteQuery)
