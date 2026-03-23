"""
Token length analysis for query-document pairs.

Uses the tokenizer from BAAI/bge-reranker-large (XLM-RoBERTa) to measure
how many tokens each query+document pair produces — exactly as the reranker
sees them. Flags pairs that exceed the 512-token limit and saves full results
to token_lengths.json.
"""

import json
import os

from transformers import AutoTokenizer

# ── Config ───────────────────────────────────────────────────────────────────

RERANKER_MODEL = "BAAI/bge-reranker-large"
TOKEN_LIMIT = 512

_SCRIPT_DIR = os.path.dirname(__file__)
_DATA_DIR = os.path.join(_SCRIPT_DIR, "data")
CORPUS_PATH = os.path.join(_DATA_DIR, "corpus.json")
QUERIES_PATH = os.path.join(_DATA_DIR, "queries.json")
QRELS_PATH = os.path.join(_DATA_DIR, "qrels.json")
OUTPUT_PATH = os.path.join(_SCRIPT_DIR, "token_lengths.json")


def main():
    # Load data
    with open(CORPUS_PATH) as f:
        corpus_raw = json.load(f)
    with open(QUERIES_PATH) as f:
        queries_raw = json.load(f)
    with open(QRELS_PATH) as f:
        qrels_raw = json.load(f)

    corpus_by_id = {doc["id"]: doc for doc in corpus_raw}
    queries_by_id = {q["id"]: q for q in queries_raw}

    # Build qrels lookup: query_id -> {doc_id: score}
    qrels_lookup: dict[str, dict[str, int]] = {}
    for qrel in qrels_raw:
        qrels_lookup.setdefault(qrel["query-id"], {})[qrel["corpus-id"]] = qrel["score"]

    # Load tokenizer
    print(f"Loading tokenizer from {RERANKER_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
    print(f"  Max model length : {tokenizer.model_max_length}")

    # Measure every query × every corpus doc (the full retrieval pool)
    print(f"\nTokenizing {len(queries_raw)} queries × {len(corpus_raw)} docs...")

    results = []
    over_limit = []

    query_stats: dict[str, list[int]] = {}  # query_id -> list of token counts

    for query in queries_raw:
        qid = query["id"]
        q_text = query["text"]
        query_stats[qid] = []

        for doc in corpus_raw:
            did = doc["id"]
            d_text = doc["text"]

            # Replicate exactly how CrossEncoder tokenizes: query + doc as a pair
            tokens = tokenizer(
                q_text,
                d_text,
                truncation=False,  # no truncation — we want the real length
                return_tensors=None,
            )
            n_tokens = len(tokens["input_ids"])

            relevance_score = qrels_lookup.get(qid, {}).get(did, 0)

            entry = {
                "query_id": qid,
                "query": q_text,
                "doc_id": did,
                "doc_title": doc["title"],
                "relevance_score": relevance_score,
                "token_count": n_tokens,
                "exceeds_limit": n_tokens > TOKEN_LIMIT,
            }
            results.append(entry)
            query_stats[qid].append(n_tokens)

            if n_tokens > TOKEN_LIMIT:
                over_limit.append(entry)

    # Aggregate stats
    all_counts = [r["token_count"] for r in results]
    relevant_counts = [r["token_count"] for r in results if r["relevance_score"] > 0]
    distractor_counts = [r["token_count"] for r in results if r["relevance_score"] == 0]

    def stats(counts: list[int]) -> dict:
        if not counts:
            return {}
        return {
            "min": min(counts),
            "max": max(counts),
            "mean": round(sum(counts) / len(counts), 1),
            "median": sorted(counts)[len(counts) // 2],
            "over_512": sum(1 for c in counts if c > TOKEN_LIMIT),
            "pct_over_512": round(100 * sum(1 for c in counts if c > TOKEN_LIMIT) / len(counts), 2),
        }

    summary = {
        "reranker_model": RERANKER_MODEL,
        "token_limit": TOKEN_LIMIT,
        "total_pairs": len(results),
        "pairs_over_limit": len(over_limit),
        "pct_over_limit": round(100 * len(over_limit) / len(results), 2),
        "all_pairs": stats(all_counts),
        "relevant_pairs_only": stats(relevant_counts),
        "distractor_pairs_only": stats(distractor_counts),
    }

    output = {
        "summary": summary,
        "over_limit_pairs": sorted(over_limit, key=lambda x: x["token_count"], reverse=True),
        "all_pairs": sorted(results, key=lambda x: x["token_count"], reverse=True),
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print("\n" + "=" * 55)
    print(f"  Token Length Report — {RERANKER_MODEL}")
    print("=" * 55)
    print(f"  Total pairs analysed : {summary['total_pairs']}")
    print(f"  Pairs over {TOKEN_LIMIT} tokens : {summary['pairs_over_limit']} ({summary['pct_over_limit']}%)")
    print()
    print(f"  {'':25} {'min':>5} {'mean':>6} {'max':>5} {'>512':>5}")
    print(f"  {'─'*25} {'─'*5} {'─'*6} {'─'*5} {'─'*5}")
    for label, s in [
        ("All pairs", summary["all_pairs"]),
        ("Relevant pairs (s>0)", summary["relevant_pairs_only"]),
        ("Distractor pairs (s=0)", summary["distractor_pairs_only"]),
    ]:
        if s:
            print(f"  {label:25} {s['min']:>5} {s['mean']:>6} {s['max']:>5} {s['over_512']:>5}")
    print("=" * 55)
    print(f"\n  Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
