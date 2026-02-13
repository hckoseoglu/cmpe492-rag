"""
RAG Evaluation Script for ACSM Progression Models Paper
Based on: https://docs.langchain.com/langsmith/evaluate-rag-tutorial

This script:
1. Loads and indexes the ACSM PDF into a vector store
2. Builds a RAG pipeline (retrieve + generate)
3. Creates a LangSmith test dataset with Q&A pairs from the paper
4. Defines 4 evaluators: correctness, relevance, groundedness, retrieval relevance
5. Runs the evaluation via LangSmith
"""

import os

os.environ.setdefault("LANGSMITH_TRACING", "true")

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client, traceable
from typing_extensions import Annotated, TypedDict

# ── 1. Indexing & Retrieval ─────────────────────────────────────────────────

PDF_PATHS = ["./math_for_cs.pdf"] 

# Load all PDFs
docs_list = []
for pdf_path in PDF_PATHS:
    loader = PyPDFLoader(pdf_path)
    docs_list.extend(loader.load())

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)

# Build vector store
vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits,
    embedding=OpenAIEmbeddings(),
)

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# ── 2. RAG Generation Pipeline ─────────────────────────────────────────────

llm = ChatOpenAI(model="gpt-4.1", temperature=0)


@traceable()
def rag_bot(question: str) -> dict:
    """RAG pipeline: retrieve relevant chunks then generate an answer."""
    docs = retriever.invoke(question)
    docs_string = "\n\n".join(doc.page_content for doc in docs)

    instructions = f"""You are a helpful assistant who is good at analyzing \
scientific source information and answering questions.
Use ONLY the following source documents to answer the user's questions.
If you cannot find the answer in the source documents, output the following:
"I don't know." Do not use any information that is not contained in the source documents.
Use three sentences maximum and keep the answer concise.

Documents:
{docs_string}"""

    ai_msg = llm.invoke(
        [
            {"role": "system", "content": instructions},
            {"role": "user", "content": question},
        ],
    )
    return {"answer": ai_msg.content, "documents": docs}


# ── 3. Test Dataset ─────────────────────────────────────────────────────────

examples = [
    {
        "inputs": {
            "question": "What loading range is recommended for novice individuals to increase muscular strength?"
        },
        "outputs": {
            "answer": "It is recommended that novice to intermediate individuals train with loads corresponding to 60–70% of 1 RM for 8–12 repetitions."
        },
    },
    {
        "inputs": {
            "question": "How much should the load be increased when an individual can exceed the target repetitions?"
        },
        "outputs": {
            "answer": "A 2–10% increase in load is recommended when the individual can perform the current workload for one to two repetitions over the desired number on two consecutive training sessions."
        },
    },
    {
        "inputs": {
            "question": "What training frequency is recommended for advanced lifters?"
        },
        "outputs": {
            "answer": "It is recommended that advanced lifters train 4–6 days per week."
        },
    },
    {
        "inputs": {
            "question": "What rest period length is recommended for core exercises using heavier loads?"
        },
        "outputs": {
            "answer": "Rest periods of at least 2–3 minutes are recommended for core exercises using heavier loads."
        },
    },
    {
        "inputs": {
            "question": "What is recommended for improving local muscular endurance in novice and intermediate individuals?"
        },
        "outputs": {
            "answer": "It is recommended that relatively light loads be used for 10–15 repetitions with moderate to high volume."
        },
    },
    {
        "inputs": {
            "question": "What loading is recommended for power training with lower body exercises?"
        },
        "outputs": {
            "answer": "Light to moderate loading of 0–60% of 1 RM for lower body exercises performed at a fast/explosive contraction velocity is recommended for power training."
        },
    },
]

client = Client()

dataset_name = "ACSM Progression Models Q&A"

if not client.has_dataset(dataset_name=dataset_name):
    dataset = client.create_dataset(dataset_name=dataset_name)
    client.create_examples(dataset_id=dataset.id, examples=examples)
    print(f"✅ Created dataset '{dataset_name}' with {len(examples)} examples")
else:
    print(f"ℹ️  Dataset '{dataset_name}' already exists — skipping creation")


# ── 4. Evaluators ───────────────────────────────────────────────────────────

# 4a. Correctness — response vs reference answer ............................


class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]


correctness_instructions = """You are a teacher grading a quiz. You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer.
(2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the ground truth answer.

Correctness:
A correctness value of True means that the student's answer meets all of the criteria.
A correctness value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

correctness_llm = ChatOpenAI(model="gpt-4.1", temperature=0).with_structured_output(
    CorrectnessGrade, method="json_schema", strict=True
)


def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """Evaluator: is the RAG answer factually correct vs the reference?"""
    prompt = f"""\
QUESTION: {inputs['question']}
GROUND TRUTH ANSWER: {reference_outputs['answer']}
STUDENT ANSWER: {outputs['answer']}"""
    grade = correctness_llm.invoke(
        [
            {"role": "system", "content": correctness_instructions},
            {"role": "user", "content": prompt},
        ]
    )
    return grade["correct"]


# 4b. Relevance — response vs input ........................................


class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[
        bool, ..., "Provide the score on whether the answer addresses the question"
    ]


relevance_instructions = """You are a teacher grading a quiz. You will be given a QUESTION and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION

Relevance:
A relevance value of True means that the student's answer meets all of the criteria.
A relevance value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

relevance_llm = ChatOpenAI(model="gpt-4.1", temperature=0).with_structured_output(
    RelevanceGrade, method="json_schema", strict=True
)


def relevance(inputs: dict, outputs: dict) -> bool:
    """Evaluator: does the answer address the user's question?"""
    prompt = f"QUESTION: {inputs['question']}\nSTUDENT ANSWER: {outputs['answer']}"
    grade = relevance_llm.invoke(
        [
            {"role": "system", "content": relevance_instructions},
            {"role": "user", "content": prompt},
        ]
    )
    return grade["relevant"]


# 4c. Groundedness — response vs retrieved docs ............................


class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[
        bool, ..., "Provide the score on if the answer hallucinates from the documents"
    ]


grounded_instructions = """You are a teacher grading a quiz. You will be given FACTS and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS.
(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Grounded:
A grounded value of True means that the student's answer meets all of the criteria.
A grounded value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

grounded_llm = ChatOpenAI(model="gpt-4.1", temperature=0).with_structured_output(
    GroundedGrade, method="json_schema", strict=True
)


def groundedness(inputs: dict, outputs: dict) -> bool:
    """Evaluator: is the answer grounded in the retrieved documents?"""
    doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
    prompt = f"FACTS: {doc_string}\nSTUDENT ANSWER: {outputs['answer']}"
    grade = grounded_llm.invoke(
        [
            {"role": "system", "content": grounded_instructions},
            {"role": "user", "content": prompt},
        ]
    )
    return grade["grounded"]


# 4d. Retrieval Relevance — retrieved docs vs input ........................


class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[
        bool,
        ...,
        "True if the retrieved documents are relevant to the question, False otherwise",
    ]


retrieval_relevance_instructions = """You are a teacher grading a quiz. You will be given a QUESTION and a set of FACTS provided by the student. Here is the grade criteria to follow:
(1) Your goal is to identify FACTS that are completely unrelated to the QUESTION
(2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
(3) It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met

Relevance:
A relevance value of True means that the FACTS contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant.
A relevance value of False means that the FACTS are completely unrelated to the QUESTION.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

retrieval_relevance_llm = ChatOpenAI(
    model="gpt-4.1", temperature=0
).with_structured_output(RetrievalRelevanceGrade, method="json_schema", strict=True)


def retrieval_relevance(inputs: dict, outputs: dict) -> bool:
    """Evaluator: are the retrieved documents relevant to the question?"""
    doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
    prompt = f"FACTS: {doc_string}\nQUESTION: {inputs['question']}"
    grade = retrieval_relevance_llm.invoke(
        [
            {"role": "system", "content": retrieval_relevance_instructions},
            {"role": "user", "content": prompt},
        ]
    )
    return grade["relevant"]


# ── 5. Run Evaluation ───────────────────────────────────────────────────────


def target(inputs: dict) -> dict:
    """Wrapper that LangSmith calls for each example in the dataset."""
    return rag_bot(inputs["question"])


if __name__ == "__main__":
    experiment_results = client.evaluate(
        target,
        data=dataset_name,
        evaluators=[correctness, groundedness, relevance, retrieval_relevance],
        experiment_prefix="acsm-rag-eval",
        metadata={
            "version": "v1",
            "model": "gpt-4.1",
            "chunk_size": 500,
            "chunk_overlap": 50,
            "retriever_k": 6,
        },
    )
    print("\n✅ Evaluation complete! View results in LangSmith.")
