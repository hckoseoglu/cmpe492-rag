"""
RAG Evaluators for LangSmith experiments.

Each evaluator returns two columns via EvaluationResults:
  - "<name>": bool score (pass/fail)
  - "<name>_reason": str — the LLM's reasoning (visible as a separate column)

Usage:
    from evaluators import make_evaluators

    evaluator_fns = make_evaluators(model_name="gpt-4o-mini")
    # evaluator_fns is [correctness, groundedness, relevance, retrieval_relevance]
"""

from langchain.chat_models import init_chat_model
from langsmith.evaluation import EvaluationResult, EvaluationResults
from typing_extensions import Annotated, TypedDict


# ── Grade Schemas ───────────────────────────────────────────────────────────


class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]


class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[
        bool, ..., "Provide the score on whether the answer addresses the question"
    ]


class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[
        bool, ..., "Provide the score on if the answer hallucinates from the documents"
    ]


class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[
        bool,
        ...,
        "True if the retrieved documents are relevant to the question, False otherwise",
    ]


# ── Instructions ────────────────────────────────────────────────────────────

CORRECTNESS_INSTRUCTIONS = """You are a teacher grading a quiz. You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer.
(2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the ground truth answer.

Correctness:
A correctness value of True means that the student's answer meets all of the criteria.
A correctness value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

RELEVANCE_INSTRUCTIONS = """You are a teacher grading a quiz. You will be given a QUESTION and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION

Relevance:
A relevance value of True means that the student's answer meets all of the criteria.
A relevance value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

GROUNDED_INSTRUCTIONS = """You are a teacher grading a quiz. You will be given FACTS and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS.
(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Grounded:
A grounded value of True means that the student's answer meets all of the criteria.
A grounded value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

RETRIEVAL_RELEVANCE_INSTRUCTIONS = """You are a teacher grading a quiz. You will be given a QUESTION and a set of FACTS provided by the student. Here is the grade criteria to follow:
(1) Your goal is to identify FACTS that are completely unrelated to the QUESTION
(2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
(3) It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met

Relevance:
A relevance value of True means that the FACTS contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant.
A relevance value of False means that the FACTS are completely unrelated to the QUESTION.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""


# ── Factory ─────────────────────────────────────────────────────────────────


def make_evaluators(model_name: str = "gpt-4o-mini") -> list:
    """
    Build and return the four evaluator functions.

    Each evaluator returns an EvaluationResults with two entries:
      - The bool score (shown as a column)
      - The reason string (shown as a separate column)

    Args:
        model_name: The OpenAI model to use for grading.

    Returns:
        List of [correctness, groundedness, relevance, retrieval_relevance]
    """

    correctness_llm = init_chat_model(model_name).with_structured_output(
        CorrectnessGrade, method="json_schema", strict=True
    )
    relevance_llm = init_chat_model(model_name).with_structured_output(
        RelevanceGrade, method="json_schema", strict=True
    )
    grounded_llm = init_chat_model(model_name).with_structured_output(
        GroundedGrade, method="json_schema", strict=True
    )
    retrieval_relevance_llm = init_chat_model(model_name).with_structured_output(
        RetrievalRelevanceGrade, method="json_schema", strict=True
    )

    # ── 1. Correctness ──────────────────────────────────────────────────

    def correctness(run, example) -> EvaluationResults:
        """Is the RAG answer factually correct vs the reference?"""
        inputs = example.inputs
        outputs = run.outputs
        reference_outputs = example.outputs

        prompt = (
            f"QUESTION: {inputs['question']}\n"
            f"GROUND TRUTH ANSWER: {reference_outputs['answer']}\n"
            f"STUDENT ANSWER: {outputs['answer']}"
        )
        grade = correctness_llm.invoke(
            [
                {"role": "system", "content": CORRECTNESS_INSTRUCTIONS},
                {"role": "user", "content": prompt},
            ]
        )
        return EvaluationResults(
            results=[
                EvaluationResult(key="correctness", score=grade["correct"]),
                EvaluationResult(key="correctness_reason", value=grade["explanation"]),
            ]
        )

    # ── 2. Relevance ────────────────────────────────────────────────────

    def relevance(run, example) -> EvaluationResults:
        """Does the answer address the user's question?"""
        inputs = example.inputs
        outputs = run.outputs

        prompt = f"QUESTION: {inputs['question']}\nSTUDENT ANSWER: {outputs['answer']}"
        grade = relevance_llm.invoke(
            [
                {"role": "system", "content": RELEVANCE_INSTRUCTIONS},
                {"role": "user", "content": prompt},
            ]
        )
        return EvaluationResults(
            results=[
                EvaluationResult(key="relevance", score=grade["relevant"]),
                EvaluationResult(key="relevance_reason", value=grade["explanation"]),
            ]
        )

    # ── 3. Groundedness ─────────────────────────────────────────────────

    def groundedness(run, example) -> EvaluationResults:
        """Is the answer grounded in the retrieved documents?"""
        outputs = run.outputs

        doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
        prompt = f"FACTS: {doc_string}\nSTUDENT ANSWER: {outputs['answer']}"
        grade = grounded_llm.invoke(
            [
                {"role": "system", "content": GROUNDED_INSTRUCTIONS},
                {"role": "user", "content": prompt},
            ]
        )
        return EvaluationResults(
            results=[
                EvaluationResult(key="groundedness", score=grade["grounded"]),
                EvaluationResult(key="groundedness_reason", value=grade["explanation"]),
            ]
        )

    # ── 4. Retrieval Relevance ──────────────────────────────────────────

    def retrieval_relevance(run, example) -> EvaluationResults:
        """Are the retrieved documents relevant to the question?"""
        inputs = example.inputs
        outputs = run.outputs

        doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
        prompt = f"FACTS: {doc_string}\nQUESTION: {inputs['question']}"
        grade = retrieval_relevance_llm.invoke(
            [
                {"role": "system", "content": RETRIEVAL_RELEVANCE_INSTRUCTIONS},
                {"role": "user", "content": prompt},
            ]
        )
        return EvaluationResults(
            results=[
                EvaluationResult(key="retrieval_relevance", score=grade["relevant"]),
                EvaluationResult(
                    key="retrieval_relevance_reason", value=grade["explanation"]
                ),
            ]
        )

    return [correctness, groundedness, relevance, retrieval_relevance]
