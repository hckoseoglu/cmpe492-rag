"""
1. Loads and indexes the Bodybuilding Anatomy PDF into a vector store
2. Builds a RAG pipeline (retrieve + generate)
3. Runs the evaluation via LangSmith
"""

import os

from utils.augment_experiment import augment_experiment

os.environ.setdefault("LANGSMITH_TRACING", "true")

from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client, traceable

from evaluators import make_evaluators
from knowledge_probing.substitution import (
    SUB_LEVEL,
    apply_substitutions,
    reverse_substitutions,
)
from knowledge_probing.save_subs import save_substitution


os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "rag_eval"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

# EXPERIMENT

FOOL = False

#######

# ── Config ──────────────────────────────────────────────────────────────────

PDF_PATH = "../resources/bodybuilding_anatomy.pdf"
OPENAI_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-large"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVAL_K = 2
DATASET_NAME = "Chunking"


# ── 1. Indexing & Retrieval ─────────────────────────────────────────────────

print("[1/4 ] Loading and indexing PDF document...")
loader = PyPDFLoader(PDF_PATH)
docs_list = loader.load()
print(f"  Loaded {len(docs_list)} pages from PDF")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    add_start_index=True,
)

doc_splits = text_splitter.split_documents(docs_list)
print(f"  Split into {len(doc_splits)} chunks")

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
vector_store = InMemoryVectorStore(embeddings)
document_ids = vector_store.add_documents(documents=doc_splits)
print(f"✓ Indexed {len(document_ids)} chunks in vector store")

retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})

# ── 2. RAG Generation Pipeline ─────────────────────────────────────────────

llm = init_chat_model(OPENAI_MODEL)


@traceable(name="llm_generate")
def _llm_generate(messages: list) -> object:
    """Traced LLM invocation step."""
    return llm.invoke(messages)


@traceable()
def rag_bot(question: str) -> dict:
    """RAG pipeline: retrieve relevant chunks then generate an answer."""
    docs = retriever.invoke(question)
    docs_string = "\n\n".join(doc.page_content for doc in docs)

    if FOOL:
        fooled_question, fooled_docs = apply_substitutions(question, docs)
        fooled_docs_string = "\n\n".join(doc.page_content for doc in fooled_docs)
        #save_substitution(question, fooled_question, docs, fooled_docs)

    instructions = f"""You are a helpful assistant who is good at analyzing \
scientific source information and answering questions.
Use ONLY the following source documents to answer the user's questions.
If you cannot find the answer in the source documents, output the following:
"I don't know." Do not use any information that is not contained in the source documents.
Use three sentences maximum and keep the answer concise.

Documents:
{fooled_docs_string if FOOL else docs_string}"""

    ai_msg = _llm_generate(
        [
            {"role": "system", "content": instructions},
            {"role": "user", "content": fooled_question if FOOL else question},
        ]
    )

    clean_answer = reverse_substitutions(ai_msg.content) if FOOL else ai_msg.content
    return {"answer": clean_answer, "documents": docs}


client = Client()

# ── 3. Evaluators ───────────────────────────────────────────────────────────

print("[3/4] Initializing evaluators...")
evaluator_fns = make_evaluators(model_name=OPENAI_MODEL)
print(
    "✓ 4 evaluators ready (correctness, groundedness, relevance, retrieval_relevance)"
)


# ── 4. Run Evaluation ───────────────────────────────────────────────────────


def target(inputs: dict, metadata: dict) -> dict:
    """Wrapper that LangSmith calls for each example in the dataset."""
    result = rag_bot(inputs["question"])
    return augment_experiment(result=result, input=metadata)


if __name__ == "__main__":
    print("[4/4] Running evaluation...")
    experiment_results = client.evaluate(
        target,
        data=DATASET_NAME,
        evaluators=evaluator_fns,
        experiment_prefix="bodybuilding-rag-eval",
        metadata={
            "sub_level": SUB_LEVEL if FOOL else "no obfuscation",
            "model": OPENAI_MODEL,
        },
        num_repetitions=3,
    )
    print("\n[4/4] ✅ Evaluation complete! View results in LangSmith.")
