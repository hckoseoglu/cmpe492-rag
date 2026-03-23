"""
Reranker degradation debugger.

For every query, retrieves top-K with the embedding model, then reranks.
Detects cases where the reranker demotes a relevant document (gold score=2
or partly-relevant score=1) and prints the doc it swapped places with.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from reranking import load_reranker, rerank_docs

# ── Config ───────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "text-embedding-3-large"
RETRIEVAL_K = 10
RERANKER_MODEL = "BAAI/bge-reranker-large"
RERANKER_TOP_K = 5

_SCRIPT_DIR = os.path.dirname(__file__)
_DATA_DIR = os.path.join(_SCRIPT_DIR, "data")
CORPUS_PATH = os.path.join(_DATA_DIR, "corpus.json")
QUERIES_PATH = os.path.join(_DATA_DIR, "queries.json")
QRELS_PATH = os.path.join(_DATA_DIR, "qrels.json")

OUTPUT_PATH = os.path.join(_SCRIPT_DIR, "reranker_degradations.json")

# ── Helpers ──────────────────────────────────────────────────────────────────

SCORE_LABEL = {2: "GOLD (s=2)", 1: "PARTIAL (s=1)", 0: "distractor (s=0)"}


def doc_preview(doc: Document, qrels: dict[str, int]) -> str:
    did = doc.metadata["id"]
    score = qrels.get(did, 0)
    label = SCORE_LABEL[score]
    title = doc.metadata.get("title", "")
    preview = textwrap.shorten(doc.page_content, width=PREVIEW_CHARS, placeholder="...")
    return f"[{label}] {did} | {title}\n    \"{preview}\""


def rank_of(docs: list[Document], doc_id: str) -> int | None:
    """Return 1-based rank of doc_id in the list, or None if absent."""
    for i, doc in enumerate(docs):
        if doc.metadata["id"] == doc_id:
            return i + 1
    return None


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    # Load corpus
    print("Loading corpus and building vector store...")
    with open(CORPUS_PATH) as f:
        corpus_raw = json.load(f)

    corpus_text: dict[str, str] = {doc["id"]: doc["text"] for doc in corpus_raw}

    documents = [
        Document(
            page_content=doc["text"],
            metadata={"id": doc["id"], "title": doc["title"]},
        )
        for doc in corpus_raw
    ]

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents)
    print(f"  Indexed {len(documents)} documents")

    # Load queries & qrels
    with open(QUERIES_PATH) as f:
        queries_raw = json.load(f)
    with open(QRELS_PATH) as f:
        qrels_raw = json.load(f)

    qrels_lookup: dict[str, dict[str, int]] = {}
    for qrel in qrels_raw:
        qrels_lookup.setdefault(qrel["query-id"], {})[qrel["corpus-id"]] = qrel["score"]

    # Load reranker
    print(f"Loading reranker: {RERANKER_MODEL}...")
    reranker_model = load_reranker(RERANKER_MODEL)

    # Per-query analysis
    degradation_cases: list[dict] = []

    print(f"\nAnalysing {len(queries_raw)} queries...\n")
    for query in queries_raw:
        qid = query["id"]
        q_text = query["text"]
        gold = qrels_lookup.get(qid, {})

        # Embedding ranking (pre-rerank)
        pre_results = vector_store.similarity_search(q_text, k=RETRIEVAL_K)

        # Reranked ranking
        post_results = rerank_docs(q_text, pre_results, top_k=RERANKER_TOP_K, model=reranker_model)

        # Check every relevant doc (score > 0) for rank degradation
        for doc_id, rel_score in gold.items():
            pre_rank = rank_of(pre_results, doc_id)
            post_rank = rank_of(post_results, doc_id)

            # A degradation is: doc was in the reranked window before but
            # either dropped out entirely or moved to a worse rank after reranking
            if pre_rank is None:
                continue  # wasn't retrieved at all — not a reranker issue

            pre_in_window = pre_rank <= RERANKER_TOP_K
            post_in_window = post_rank is not None  # post_results is already top-K

            degraded = False
            if pre_in_window and not post_in_window:
                degraded = True  # reranker pushed it out of the window
            elif pre_in_window and post_in_window and post_rank > pre_rank:
                degraded = True  # reranker moved it to a worse position

            if not degraded:
                continue

            # Find what doc took its place (the doc now at pre_rank in post_results)
            if post_rank is not None:
                # Doc moved down — what moved up to fill its old spot?
                usurper_rank = pre_rank  # the rank it used to hold
                usurper = post_results[usurper_rank - 1] if usurper_rank <= len(post_results) else None
            else:
                # Doc was pushed out entirely — show what now occupies the last slot
                usurper = post_results[-1] if post_results else None

            # Find the original doc object for display
            demoted_doc = next((d for d in pre_results if d.metadata["id"] == doc_id), None)

            # Find usurper's pre-rank for context
            usurper_pre_rank = (
                rank_of(pre_results, usurper.metadata["id"]) if usurper else None
            )

            degradation_cases.append({
                "qid": qid,
                "query": q_text,
                "gold": gold,
                "demoted_id": doc_id,
                "demoted_score": rel_score,
                "pre_rank": pre_rank,
                "post_rank": post_rank,
                "demoted_doc": demoted_doc,
                "usurper": usurper,
                "usurper_pre_rank": usurper_pre_rank,
            })

    # ── Serialise & save ─────────────────────────────────────────────────────

    gold_hits = sum(1 for c in degradation_cases if c["demoted_score"] == 2)
    partial_hits = sum(1 for c in degradation_cases if c["demoted_score"] == 1)

    output = {
        "summary": {
            "reranker_model": RERANKER_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "retrieval_k": RETRIEVAL_K,
            "reranker_top_k": RERANKER_TOP_K,
            "queries_analysed": len(queries_raw),
            "degradation_cases": len(degradation_cases),
            "gold_demotions": gold_hits,
            "partial_demotions": partial_hits,
        },
        "cases": [],
    }

    for case in degradation_cases:
        demoted = case["demoted_doc"]
        usurper = case["usurper"]

        entry = {
            "query_id": case["qid"],
            "query": case["query"],
            "demoted": {
                "doc_id": case["demoted_id"],
                "relevance_score": case["demoted_score"],
                "relevance_label": SCORE_LABEL[case["demoted_score"]],
                "rank_before_rerank": case["pre_rank"],
                "rank_after_rerank": case["post_rank"],
                "pushed_out_of_window": case["post_rank"] is None,
                "text": demoted.page_content if demoted else None,
                "title": demoted.metadata.get("title") if demoted else None,
            },
            "promoted_in_its_place": None,
        }

        if usurper:
            usurper_id = usurper.metadata["id"]
            usurper_score = case["gold"].get(usurper_id, 0)
            entry["promoted_in_its_place"] = {
                "doc_id": usurper_id,
                "relevance_score": usurper_score,
                "relevance_label": SCORE_LABEL[usurper_score],
                "rank_before_rerank": case["usurper_pre_rank"],
                "text": usurper.page_content,
                "title": usurper.metadata.get("title"),
            }

        output["cases"].append(entry)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Queries analysed   : {len(queries_raw)}")
    print(f"  Degradation cases  : {len(degradation_cases)}")
    print(f"    Gold demotions   : {gold_hits}")
    print(f"    Partial demotions: {partial_hits}")
    print(f"\n  Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
