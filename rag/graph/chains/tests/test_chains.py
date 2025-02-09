from pprint import pprint

from dotenv import load_dotenv

from graph.chains.answer_grader import GradeAnswer, answer_grader
from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import GradeHallucination, hallucination_grader
from graph.chains.retrieval_grader import GradeDocument, retrieval_grader
from graph.chains.router import RouteQuery, question_router
from ingestion import retriever

load_dotenv()


def test_retrieval_grader_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content
    res: GradeDocument = retrieval_grader.invoke(
        {"question": question, "documents": doc_txt}
    )
    assert res.binary_score == "yes"


def test_retrieval_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content
    res: GradeDocument = retrieval_grader.invoke(
        {"question": "How to make pancakes?", "documents": doc_txt}
    )
    assert res.binary_score == "no"


def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"question": question, "context": docs})
    pprint(generation)


def test_hallucination_grader_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"question": question, "context": docs})
    res: GradeHallucination = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )
    assert res.binary_score is True


def test_hallucination_grader_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    res: GradeHallucination = hallucination_grader.invoke(
        {
            "documents": docs,
            "generation": "In order to make pizza we need to first start with the dough.",
        }
    )
    assert res.binary_score is False


def test_answer_grader_yes() -> None:
    question = "agent memory"
    res: GradeAnswer = answer_grader.invoke(
        {
            "question": question,
            "generation": "Agent memory is a system that enables AI agents to store,"
            " retrieve, and utilize information from previous interactions, including"
            " facts (semantic memory), experiences (episodic memory), and operational"
            " rules (procedural memory)",
        }
    )
    assert res.binary_score is True


def test_answer_grader_no() -> None:
    question = "agent memory"
    res: GradeAnswer = answer_grader.invoke(
        {
            "question": question,
            "generation": "In order to make pizza we need to first start with the dough.",
        }
    )
    assert res.binary_score is False


def test_question_router_vectorstore() -> None:
    question = "What is agent memory?"
    res: RouteQuery = question_router.invoke({"question": question})
    assert res.data_source == "vectorstore"


def test_question_router_websearch() -> None:
    question = "What is the capital of France?"
    res: RouteQuery = question_router.invoke({"question": question})
    assert res.data_source == "websearch"
