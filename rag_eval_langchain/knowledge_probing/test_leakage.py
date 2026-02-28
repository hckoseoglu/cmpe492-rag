"""
test_leakage.py

LLM-as-Fool: Obfuscation Guessing Experiment (LangSmith)
----------------------------------------------------------
Tests whether the LLM can reverse-engineer the original words behind
obfuscated (nonsense) terms, given both the fooled query AND fooled context.

For each example:
  1. Retrieve relevant documents for the original question.
  2. Apply lexical substitutions to both the query and the documents.
  3. Present the fooled query + fooled context to the LLM and ask it to
     make 3 guesses per obfuscated word.
  4. An evaluator LLM compares the guesses against the original query to
     determine whether the LLM successfully identified the real words.

If the LLM correctly guesses the original word, the substitution failed to
fully block parametric knowledge — this is a detection / leakage event.

Results are recorded in LangSmith under the experiment prefix "leakage".
"""

import json
import os
import re
import sys

# ── Ensure the parent dir is on the path when run from knowledge_probing/ ────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("LANGSMITH_TRACING", "true")

from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client, traceable

from knowledge_probing.sub_map import SOFT_MAP, MEDIUM_MAP, HARD_MAP
from knowledge_probing.substitution import apply_substitutions, SUB_LEVEL

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "rag_eval"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

LEVEL_MAPS = {"soft": SOFT_MAP, "medium": MEDIUM_MAP, "hard": HARD_MAP}
ACTIVE_MAP: dict[str, str] = LEVEL_MAPS[SUB_LEVEL]

# ── Config ───────────────────────────────────────────────────────────────────
OPENAI_MODEL = "gpt-4o-mini"
EVALUATOR_LLM = "gpt-4o-mini"
DATASET_NAME = "Leakage Test Dataset"

PDF_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "../resources/bodybuilding_anatomy.pdf",
)
EMBEDDING_MODEL = "text-embedding-3-large"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVAL_K = 2

# ── Indexing & Retrieval ─────────────────────────────────────────────────────

print("[1/2] Loading and indexing PDF document...")
loader = PyPDFLoader(PDF_PATH)
docs_list = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    add_start_index=True,
)
doc_splits = text_splitter.split_documents(docs_list)

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(documents=doc_splits)
retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})
print(f"  ✓ Indexed {len(doc_splits)} chunks")

llm = init_chat_model(OPENAI_MODEL)
evaluator_llm = init_chat_model(EVALUATOR_LLM)
client = Client()


# ── Helper: find nonsense words present in the fooled query ──────────────────


def find_obfuscated_words(fooled_question: str) -> list[str]:
    """
    Return the de-duplicated list of nonsense words (values from
    ACTIVE_MAP) that actually appear in *fooled_question*.
    """
    nonsense_words = set(ACTIVE_MAP.values())
    found = []
    lower_q = fooled_question.lower()
    for word in sorted(nonsense_words):  # sorted for determinism
        if word.lower() in lower_q:
            found.append(word)
    return found


# ── Step 1: Target — retrieve, substitute, probe ─────────────────────────────


@traceable(name="probe_guesses")
def probe_guesses(
    fooled_question: str,
    fooled_context: str,
    obfuscated_words: list[str],
) -> dict:
    """
    Give the LLM the fooled query + fooled context and the list of
    nonsense words.  Ask it to produce 3 guesses per word.

    Returns dict of the form:
        {"guesses": {"nonsense_word": ["g1", "g2", "g3"], ...}}
    """
    words_str = ", ".join(f'"{w}"' for w in obfuscated_words)
    skeleton = {w: ["guess1", "guess2", "guess3"] for w in obfuscated_words}

    prompt = f"""You are given a question and some context documents.
Some words in the question have been deliberately replaced with made-up
nonsense words to hide their real meaning.

Question:
{fooled_question}

Context documents:
{fooled_context}

The following words in the question are obfuscated (nonsense replacements
for real terms): {words_str}

For each obfuscated word, provide exactly 3 guesses for what the original
real word or phrase might have been. Use the context documents and your
reasoning to inform your guesses.

Respond in this exact JSON format with no extra text:
{json.dumps({"guesses": skeleton}, indent=2)}"""

    response = llm.invoke(
        [
            {
                "role": "system",
                "content": "You are a helpful assistant. Respond only with valid JSON.",
            },
            {"role": "user", "content": prompt},
        ]
    )

    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response.content, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"guesses": {}}


@traceable(name="leakage_target")
def leakage_target(inputs: dict) -> dict:
    """
    Target function called by LangSmith for each dataset example.
    Retrieves docs, applies substitutions, and probes the LLM for guesses.
    """
    question = inputs["question"]

    # Retrieve → substitute
    docs = retriever.invoke(question)
    fooled_question, fooled_docs = apply_substitutions(
        question, docs, sub_map=ACTIVE_MAP
    )
    fooled_context = "\n\n".join(doc.page_content for doc in fooled_docs)

    # Find which nonsense words landed in the query
    obfuscated_words = find_obfuscated_words(fooled_question)

    if not obfuscated_words:
        return {
            "fooled_question": fooled_question,
            "obfuscated_words": [],
            "guesses": {},
            "skipped": True,
        }

    # Probe: ask the LLM for 3 guesses per obfuscated word
    guess_result = probe_guesses(fooled_question, fooled_context, obfuscated_words)

    return {
        "fooled_question": fooled_question,
        "obfuscated_words": obfuscated_words,
        "guesses": guess_result.get("guesses", {}),
        "skipped": False,
    }


# ── Step 2: Evaluator — judge whether guesses are correct ────────────────────


@traceable(name="evaluate_guesses")
def _evaluate_guesses_llm(
    original_question: str,
    fooled_question: str,
    guesses: dict,
) -> dict:
    """LLM judge: checks whether any of the 3 guesses match the real word."""
    prompt = f"""You are an evaluation judge.  A language model was given a
question where certain words were replaced with nonsense words.  It then
tried to guess what the original words were.

Original question (contains the real words):
{original_question}

Obfuscated question (contains the nonsense words):
{fooled_question}

The model's guesses for each obfuscated word:
{json.dumps(guesses, indent=2)}

For each obfuscated word, determine whether ANY of the 3 guesses correctly
identifies the original word **or a close synonym / equivalent**.

Respond in this exact JSON format with no extra text:
{{
  "evaluations": {{
    "<nonsense_word>": {{
      "correct_original": "the actual original word from the original question",
      "best_guess": "which guess (if any) was closest",
      "detected": true or false,
      "reasoning": "brief explanation"
    }}
  }},
  "overall_detected": true or false
}}

Set "overall_detected" to true ONLY if ALL obfuscated words were correctly
guessed; false if any were missed."""

    response = evaluator_llm.invoke([{"role": "user", "content": prompt}])

    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response.content, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"evaluations": {}, "overall_detected": False}


def leakage_evaluator(inputs: dict, outputs: dict) -> dict:
    """
    LangSmith evaluator: returns obfuscation_blocked score (1 = blocked, 0 = detected).
    Skipped examples (no obfuscations in query) return score=None.
    """
    if outputs.get("skipped"):
        return {
            "key": "obfuscation_blocked",
            "score": None,
            "comment": "No obfuscations in query — skipped",
        }

    original_question = inputs["question"]
    fooled_question = outputs["fooled_question"]
    guesses = outputs["guesses"]

    evaluation = _evaluate_guesses_llm(original_question, fooled_question, guesses)
    detected = evaluation.get("overall_detected", False)

    # Build a readable comment with per-word reasoning
    per_word = evaluation.get("evaluations", {})
    comment_lines = []
    for word, ev in per_word.items():
        mark = "DETECTED" if ev.get("detected") else "BLOCKED"
        comment_lines.append(f"{word}: {mark} — {ev.get('reasoning', '')}")

    return {
        "key": "obfuscation_blocked",
        "score": 0 if detected else 1,
        "comment": "\n".join(comment_lines),
    }


# ── Run experiment ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(
        f"[2/2] Running leakage experiment (level={SUB_LEVEL}, {len(ACTIVE_MAP)} substitutions)..."
    )
    experiment_results = client.evaluate(
        leakage_target,
        data=DATASET_NAME,
        evaluators=[leakage_evaluator],
        experiment_prefix="leakage",
        metadata={
            "sub_level": SUB_LEVEL,
            "model": OPENAI_MODEL,
        },
        num_repetitions=5,
    )
    print("\n✅ Leakage experiment complete! View results in LangSmith.")
