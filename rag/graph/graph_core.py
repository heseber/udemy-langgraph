from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
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


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    docs = state["documents"]
    generation = state["generation"]
    # Check if the generation is a hallucination
    is_fact_based = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    ).binary_score
    if is_fact_based:
        print("---DECISION: ANSWER GROUNDED IN DOCUMENTS---")
        # Check if the generation is useful
        is_useful = answer_grader.invoke(
            {"question": question, "generation": generation}
        ).binary_score
        if is_useful:
            print("---DECISION: ANSWER ADDRESSES THE QUESTION---")
            return "useful"
        else:
            print("---DECISION: ANSWER DOES NOT ADDRESS THE QUESTION---")
            return "not useful"
    else:
        print("---DECISION: ANSWER IS NOT GROUNDED IN DOCUMENTS---")
        return "not supported"


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
    g.add_conditional_edges(
        GENERATE,
        grade_generation_grounded_in_documents_and_question,
        {"not supported": GENERATE, "not useful": WEBSEARCH, "useful": END},
    )

    g.set_entry_point(RETRIEVE)

    return g


app = graph().compile()

if __name__ == "__main__":
    app.get_graph().draw_mermaid_png(output_file_path="graph.png")
