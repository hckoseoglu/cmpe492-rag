import json
from dotenv import load_dotenv
from openai import AsyncOpenAI

from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

from ragas import EvaluationDataset, SingleTurnSample
from ragas import experiment
from ragas.metrics.collections import (
    AnswerRelevancy,
    Faithfulness,
    ContextRelevance,
    AnswerAccuracy,
)
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.backends.local_csv import LocalCSVBackend

from pydantic import BaseModel


load_dotenv()

print("=" * 60)
print("Starting RAG Evaluation Script")
print("=" * 60)

# 1. Load the evaluation dataset from JSON
print("\n[1/7] Loading evaluation dataset...")
with open("../test_dataset/dataset.json", "r") as f:
    eval_data = json.load(f)
print(f"✓ Loaded {len(eval_data)} questions from eval_dataset.json")


# 2. Build the RAG pipeline from PDF
print("\n[2/7] Loading and processing PDF document...")
PDF_PATH = "../resources/bodybuilding_anatomy.pdf"  # adjust to your actual PDF path

loader = PyPDFLoader(PDF_PATH)
docs_list = loader.load()
print(f"✓ Loaded {len(docs_list)} pages from PDF")

print("[3/7] Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
)

doc_splits = text_splitter.split_documents(docs_list)
print(f"✓ Created {len(doc_splits)} document chunks")

print("[4/7] Creating vector store (this may take a while)...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)
document_ids = vector_store.add_documents(documents=doc_splits)
print(f"✓ Indexed {len(document_ids)} chunks in vector store")

RETRIEVAL_K = 3

llm = init_chat_model("gpt-4o-mini")

# ---------------------------------------------------------------------------
# 3. Build the agent with dynamic context injection
# ---------------------------------------------------------------------------
SYSTEM_TEMPLATE = """\
Answer the question based only on the following context.
If the context does not cover the question, say so honestly
rather than guessing.

--- Retrieved Context ---
{context}
"""

# Shared state to capture retrieved docs for evaluation
_last_retrieved_docs = []


@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Retrieve the most relevant chunks and inject them as system context."""
    global _last_retrieved_docs
    last_query = request.state["messages"][-1].text

    _last_retrieved_docs = vector_store.similarity_search(last_query, k=RETRIEVAL_K)

    docs_content = "\n\n".join(
        f"[Page {doc.metadata.get('page', '?')}] {doc.page_content}"
        for doc in _last_retrieved_docs
    )

    return SYSTEM_TEMPLATE.format(context=docs_content)


agent = create_agent(llm, tools=[], middleware=[prompt_with_context])


# Build reference text for each item (handle both single and multi-page)
def get_reference_text(item):
    if item.get("is-multi-page") and "reference_texts" in item:
        return " ".join(item["reference_texts"])
    return item.get("reference_text", "")


# 4. Run retrieval + generation for each question, build ragas samples
print("\n[5/7] Running retrieval and generation for each question...")
samples = []
for idx, item in enumerate(eval_data, 1):
    query = item["question"]
    reference = item["ground_truth"]

    print(f"  Processing question {idx}/{len(eval_data)}: {query[:60]}...")
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    response = result["messages"][-1].content
    relevant_docs = _last_retrieved_docs

    samples.append(
        SingleTurnSample(
            user_input=query,
            response=response,
            reference=reference,
            retrieved_contexts=[doc.page_content for doc in relevant_docs],
        )
    )

print(f"✓ Generated {len(samples)} samples for evaluation")

dataset = EvaluationDataset(samples=samples)


# 5. Set up ragas evaluator LLM and embeddings
print("\n[6/7] Setting up evaluation metrics...")
client = AsyncOpenAI()
eval_llm = llm_factory("gpt-4o-mini", client=client)
eval_embeddings = embedding_factory(
    "openai", model="text-embedding-ada-002", client=client
)
print("✓ Evaluator LLM and embeddings configured")


class ExperimentResult(BaseModel):
    answer_relevancy: float
    faithfulness: float
    context_relevance: float
    accuracy: float
    query: str
    rag_response: str = ""
    ground_truth: str = ""
    


# 6. Create experiment function
@experiment(ExperimentResult)
async def run_evaluation(row):

    answer_relevancy = AnswerRelevancy(llm=eval_llm, embeddings=eval_embeddings)
    faithfulness = Faithfulness(llm=eval_llm)
    context_relevance = ContextRelevance(llm=eval_llm)
    accuracy = AnswerAccuracy(llm=eval_llm)

    relevancy_result = await answer_relevancy.ascore(
        user_input=row.user_input, response=row.response
    )

    faithfulness_result = await faithfulness.ascore(
        user_input=row.user_input,
        response=row.response,
        retrieved_contexts=row.retrieved_contexts,
    )

    context_relevance_result = await context_relevance.ascore(
        user_input=row.user_input,
        retrieved_contexts=row.retrieved_contexts,
    )

    accuracy_result = await accuracy.ascore(
        user_input=row.user_input,
        response=row.response,
        reference=row.reference,
    )

    return ExperimentResult(
        answer_relevancy=relevancy_result.value,
        faithfulness=faithfulness_result.value,
        context_relevance=context_relevance_result.value,
        accuracy=accuracy_result.value,
        query=row.user_input,
        rag_response=row.response,
        ground_truth=row.reference,
    )


async def main():
    print("\n[7/7] Running evaluation (this will take several minutes)...")
    print("  Evaluating: Answer Relevancy, Faithfulness, Context Relevance, Accuracy")
    exp_results = await run_evaluation.arun(
        dataset, backend=LocalCSVBackend(root_dir="./evals")
    )
    print("\n" + "=" * 60)
    print("✓ Evaluation Complete!")
    print("=" * 60)
    print(exp_results)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
