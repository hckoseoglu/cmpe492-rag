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

RETRIEVAL_RELEVANCE_INSTRUCTIONS = (
    RETRIEVAL_RELEVANCE_INSTRUCTIONS
) = """You are an expert evaluator assessing whether retrieved context contains sufficient information to answer a given question.

Your task is to determine if the CONTEXT provides the specific information needed to answer the QUESTION.

Follow these rules strictly:
- Do NOT rely on your own knowledge. Use ONLY what is explicitly written in the CONTEXT.
- Semantic similarity or shared keywords are NOT enough. The CONTEXT must contain actual information that directly addresses what the QUESTION is asking.
- If the QUESTION asks for a definition, the CONTEXT must contain that definition.
- If the QUESTION asks for a list (e.g. "what are the three phases"), the CONTEXT must explicitly mention those items.
- If the QUESTION has multiple parts, the CONTEXT must address ALL parts to be considered relevant.

Evaluation criteria:
- True: The CONTEXT contains specific information that directly answers the QUESTION (or all parts of it). A person reading only the CONTEXT could reasonably answer the QUESTION.
- False: The CONTEXT is on a related topic but does not contain the specific information needed to answer the QUESTION, OR only addresses some parts of a multi-part question.

Here are three examples:

Example 1:
QUESTION: What is the recommended daily protein intake for muscle hypertrophy?
CONTEXT: Protein plays a critical role in muscle repair and growth. Studies suggest that individuals engaged in resistance training should consume between 1.6 to 2.2 grams of protein per kilogram of body weight per day to maximize muscle protein synthesis.
VERDICT: True
REASONING: The question asks for a specific recommendation and the context provides the exact range (1.6-2.2 g/kg/day) with the relevant condition (resistance training for hypertrophy).

Example 2:
QUESTION: What are the benefits of periodization in a training program?
CONTEXT: Periodization involves dividing a training program into distinct phases such as hypertrophy, strength, and peaking. Each phase targets different physiological adaptations through systematic variation in volume and intensity.
VERDICT: False
REASONING: The context explains what periodization IS and describes its structure, but never states its BENEFITS. A reader would understand the concept but could not list the benefits from this context alone.

Example 3:
QUESTION: How does sleep affect recovery and what is the recommended amount for athletes?
CONTEXT: Sleep is essential for athletic recovery. During deep sleep, growth hormone secretion peaks, facilitating tissue repair and muscle growth. Most sports science literature recommends 7 to 9 hours of sleep per night for athletes, with some studies suggesting that extending sleep to 10 hours can further improve reaction time and performance.
VERDICT: True
REASONING: The question has two parts — how sleep affects recovery (growth hormone, tissue repair, muscle growth) and recommended amount (7-9 hours, up to 10). Both parts are directly addressed in the context.


Now evaluate the following. Think step-by-step:
1. Identify what exactly the QUESTION is asking for.
2. Check if each part of the QUESTION is explicitly addressed in the CONTEXT.
3. Determine if someone could answer the QUESTION using ONLY the CONTEXT.
4. Give your final verdict."""


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
