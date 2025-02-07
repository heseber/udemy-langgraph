from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState

load_dotenv()


def decide_to_generate(state: GraphState) -> str:
    print("---ASSESS GRADED DOCUMENTS---")
    if state["web_search"]:
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO THE QUESTION, INCLUDE WEB SEARCH---"
        )
        return WEBSEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE


def graph() -> StateGraph:
    g = StateGraph(GraphState)

    g.add_node(RETRIEVE, retrieve)
    g.add_node(GENERATE, generate)
    g.add_node(GRADE_DOCUMENTS, grade_documents)
    g.add_node(WEBSEARCH, web_search)

    g.add_edge(RETRIEVE, GRADE_DOCUMENTS)
    g.add_conditional_edges(
        GRADE_DOCUMENTS,
        decide_to_generate,
        {WEBSEARCH: WEBSEARCH, GENERATE: GENERATE},
    )
    g.add_edge(WEBSEARCH, GENERATE)
    g.add_edge(GENERATE, END)

    g.set_entry_point(RETRIEVE)

    return g


app = graph().compile()

if __name__ == "__main__":
    app.get_graph().draw_mermaid_png(output_file_path="graph.png")
