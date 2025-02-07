from pprint import pprint

from dotenv import load_dotenv

from graph.chains.generation import generation_chain
from graph.chains.retrieval_grader import GradeDocument, retrieval_grader
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
