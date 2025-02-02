import operator
from typing import Annotated

from langgraph.graph import MessagesState

from schemas import Reference


class State(MessagesState):
    references: Annotated[list[Reference], operator.add]
    iteration: Annotated[int, operator.add]
    max_iterations: int
