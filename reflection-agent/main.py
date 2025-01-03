from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, MessagesState, StateGraph

from chains import generate_chain, reflect_chain

load_dotenv()

REFLECT = "reflect"
GENERATE = "generate"


class State(MessagesState):
    pass


def generation_node(state: State) -> State:
    ret = generate_chain.invoke({"messages": state["messages"]})
    return State(messages=[ret])


def reflection_node(state: State) -> State:
    ret = reflect_chain.invoke({"messages": state["messages"]})
    return State(messages=[HumanMessage(content=ret.content)])


builder = StateGraph(State)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


def generate_should_continue(state: State):
    if len(state["messages"]) > 6:
        return END
    return REFLECT


def reflect_should_continue(state: State):
    if (
        not state["messages"][-1].content
        or not state["messages"][-1].content.strip()
        or state["messages"][-1].content == "FINAL: No further recommendations."
    ):
        return END
    return GENERATE


builder.add_conditional_edges(GENERATE, generate_should_continue)
builder.add_conditional_edges(REFLECT, reflect_should_continue)

graph = builder.compile()
# print(graph.get_graph().draw_mermaid())
# graph.get_graph().print_ascii()


if __name__ == "__main__":
    print("Thinking...")
    input = HumanMessage(
        content="""
            Make this tweet better: "
                @LangChainAI
                - newly Tool Calling feature is seriously underrated.
                After long wait, it's here- making the implementation of agents across different models with function calling - super easy
                                
                Made a video covering their newes blog post"
        """
    )
    state = State(messages=[input])
    response = graph.invoke(state)
    final_message = [x for x in response["messages"] if isinstance(x, AIMessage)][-1]
    print(final_message.content)
