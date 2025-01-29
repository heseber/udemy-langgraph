from langgraph.graph import MessagesState

from schemas import Reference


class State(MessagesState):
    references: list[Reference]
    iteration: int
    max_iterations: int
