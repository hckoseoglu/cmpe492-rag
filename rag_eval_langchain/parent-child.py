"""
Parent-Child Chunking RAG Evaluation Pipeline
==============================================

Same pipeline as eval.py but uses parent-child chunking:
  - Parent splitter: SemanticChunker (large, coherent chunks → returned to LLM)
  - Child splitter:  RecursiveCharacterTextSplitter (small chunks → searched)

At query time the child chunks are searched, but the full parent chunks are
returned to the LLM for richer context.
"""

import glob
import os

from utils.augment_experiment import augment_experiment

os.environ.setdefault("LANGSMITH_TRACING", "true")

from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langsmith import Client, traceable
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import openai

from evaluators import make_evaluators
from knowledge_probing.substitution import (
    SUB_LEVEL,
    apply_substitutions,
    reverse_substitutions,
)
from knowledge_probing.save_subs import save_substitution

# Chunking imports
from chunking.chunking import parent_child_chunking


os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "rag_eval"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

# EXPERIMENT

FOOL = False

#######

# ── Config ──────────────────────────────────────────────────────────────────

RESOURCES_DIR = "../resources"
OPENAI_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-large"
RETRIEVAL_K = 16
DATASET_NAME = "Chunking"

# Parent-child specific config
CHILD_CHUNK_SIZE = 300
CHILD_CHUNK_OVERLAP = 45


# ── 1. Indexing & Retrieval ─────────────────────────────────────────────────

print("[1/4] Loading and indexing PDF documents...")
pdf_paths = glob.glob(os.path.join(RESOURCES_DIR, "*.pdf"))
print(f"  Found {len(pdf_paths)} PDF files in '{RESOURCES_DIR}'")
docs_list = []
for pdf_path in pdf_paths:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    filename = os.path.basename(pdf_path)
    for page in pages:
        page.metadata = {"title": filename, "page": page.metadata.get("page", 0)}
    docs_list.extend(pages)
    print(f"  Loaded {len(pages)} pages from {filename}")
print(f"  Total: {len(docs_list)} pages loaded")

# Parent-child chunking: semantic parents + recursive children
retriever = parent_child_chunking(
    docs_list,
    embedding_model=EMBEDDING_MODEL,
    child_chunk_size=CHILD_CHUNK_SIZE,
    child_chunk_overlap=CHILD_CHUNK_OVERLAP,
    search_k=RETRIEVAL_K,
)
print("✓ Parent-child retriever ready")

# ── 2. RAG Generation Pipeline ─────────────────────────────────────────────

llm = init_chat_model(OPENAI_MODEL)


@retry(
    retry=retry_if_exception_type(openai.RateLimitError),
    wait=wait_exponential(multiplier=2, min=4, max=120),
    stop=stop_after_attempt(6),
    reraise=True,
)
@traceable(name="llm_generate")
def _llm_generate(messages: list) -> object:
    """Traced LLM invocation step with retry on rate limits."""
    return llm.invoke(messages)


@traceable()
def rag_bot(question: str) -> dict:
    """RAG pipeline: retrieve relevant parent chunks then generate an answer."""
    docs = retriever.invoke(question)
    docs_string = "\n\n".join(doc.page_content for doc in docs)

    if FOOL:
        fooled_question, fooled_docs = apply_substitutions(question, docs)
        fooled_docs_string = "\n\n".join(doc.page_content for doc in fooled_docs)

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
        evaluators=[evaluator_fns[-1]],  # Only retrieval relevance
        experiment_prefix=f"parent=semantic-child-ch={CHILD_CHUNK_SIZE}-ov={CHILD_CHUNK_OVERLAP}-k={RETRIEVAL_K}",
        max_concurrency=32,
        metadata={
            "sub_level": SUB_LEVEL if FOOL else "no obfuscation",
            "model": OPENAI_MODEL,
            "chunking": "parent-child",
            "parent_splitter": "semantic-perc=95",
            "child_chunk_size": CHILD_CHUNK_SIZE,
            "child_chunk_overlap": CHILD_CHUNK_OVERLAP,
        },
        num_repetitions=1,
    )
    print("\n[4/4] ✅ Evaluation complete! View results in LangSmith.")
