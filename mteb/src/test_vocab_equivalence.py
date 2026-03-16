"""
Vocabulary Semantic Equivalence Test

Tests whether text-embedding-3-large already treats fitness abbreviations
and lay terms as semantically equivalent to their canonical forms.

Approach: for each vocab term pair (informal → canonical), search the corpus
for chunks that contain the canonical term, sample up to SAMPLE_SIZE of them,
then create an "informal version" of each chunk by substituting the canonical
term with the informal term. Embed both versions and compute cosine similarity.

High similarity (≥ 0.95) → model is insensitive to the substitution; transformation adds no signal.
Low similarity (< 0.90) → substitution meaningfully shifts the embedding; transformation helps.

Terms with zero corpus matches are reported separately.
"""

import json
import random
import re
import sys
from pathlib import Path

import numpy as np
from openai import OpenAI

# Allow running from repo root or from src/
sys.path.insert(0, str(Path(__file__).parent))
from custom_embedder import _embed, DEFAULT_MODEL


VOCAB_PATH = Path(__file__).parent.parent / "vocab" / "fitness_vocab.json"
CORPUS_PATH = Path(__file__).parent.parent / "data" / "original_corpus.json"
RESULTS_PATH = Path(__file__).parent.parent / "results" / "vocab_equivalence_test.json"

SAMPLE_SIZE = 20
RANDOM_SEED = 42

SIMILARITY_THRESHOLD_EQUIVALENT = 0.95
SIMILARITY_THRESHOLD_LOW = 0.90


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def find_matching_chunks(
    corpus: list[dict], canonical: str, sample_size: int, rng: random.Random
) -> list[dict]:
    """Return up to sample_size corpus chunks that contain the canonical term."""
    pattern = re.compile(re.escape(canonical), re.IGNORECASE)
    matches = [
        c
        for c in corpus
        if pattern.search(c.get("text", "") + " " + c.get("title", ""))
    ]
    if len(matches) > sample_size:
        matches = rng.sample(matches, sample_size)
    return matches


def substitute(text: str, canonical: str, informal: str) -> str:
    """Replace canonical term with informal term (case-insensitive, whole-word)."""
    pattern = re.compile(re.escape(canonical), re.IGNORECASE)
    return pattern.sub(informal, text)


def chunk_text(chunk: dict) -> str:
    title = chunk.get("title", "")
    text = chunk.get("text", "")
    return (title + ". " + text).strip() if title else text


def evaluate_vocab_section(
    client: OpenAI,
    section_name: str,
    vocab: dict[str, str],
    corpus: list[dict],
    rng: random.Random,
) -> tuple[list[dict], list[str]]:
    """
    Returns:
      results       — list of dicts for terms that had corpus matches
      no_match_terms — list of term strings with zero corpus matches
    """
    results: list[dict] = []
    no_match_terms: list[str] = []

    all_sentences: list[str] = []
    # (term, canonical, chunk_id, idx_canonical, idx_informal)
    pair_map: list[tuple[str, str, str, int, int]] = []

    for term, canonical in vocab.items():
        matching = find_matching_chunks(corpus, canonical, SAMPLE_SIZE, rng)
        if not matching:
            no_match_terms.append(term)
            continue
        for chunk in matching:
            canonical_sent = chunk_text(chunk)
            informal_sent = substitute(canonical_sent, canonical, term)
            idx_c = len(all_sentences)
            all_sentences.append(canonical_sent)
            idx_i = len(all_sentences)
            all_sentences.append(informal_sent)
            pair_map.append((term, canonical, chunk["id"], idx_c, idx_i))

    if not all_sentences:
        return results, no_match_terms

    print(
        f"  Embedding {len(all_sentences)} sentences for "
        f"{len(vocab) - len(no_match_terms)} matched terms "
        f"({len(no_match_terms)} skipped — no corpus match)…"
    )
    embeddings = _embed(client, all_sentences, DEFAULT_MODEL)

    # Group by term
    term_sims: dict[str, list[float]] = {}
    term_per_chunk: dict[str, list[dict]] = {}
    for term, canonical, chunk_id, idx_c, idx_i in pair_map:
        sim = cosine_similarity(embeddings[idx_c], embeddings[idx_i])
        term_sims.setdefault(term, []).append(sim)
        term_per_chunk.setdefault(term, []).append(
            {
                "chunk_id": chunk_id,
                "similarity": round(sim, 6),
            }
        )

    canonical_map = dict(vocab)
    for term, sims in term_sims.items():
        mean_sim = float(np.mean(sims))
        results.append(
            {
                "term": term,
                "canonical": canonical_map[term],
                "chunks_sampled": len(sims),
                "mean_similarity": round(mean_sim, 6),
                "min_similarity": round(float(np.min(sims)), 6),
                "max_similarity": round(float(np.max(sims)), 6),
                "per_chunk": term_per_chunk[term],
            }
        )

    return results, no_match_terms


def print_section_table(title: str, results: list[dict], no_match: list[str]) -> None:
    print(f"\n=== {title} ({len(results)} matched, {len(no_match)} skipped) ===")
    print(f"{'Term':<20} {'Canonical':<40} {'N':>4} {'Similarity':>10}")
    print("-" * 78)
    for r in sorted(results, key=lambda x: x["mean_similarity"]):
        term_display = r["term"][:18]
        canonical_display = r["canonical"][:38]
        flag = (
            "  *** LOW ***" if r["mean_similarity"] < SIMILARITY_THRESHOLD_LOW else ""
        )
        print(
            f"{term_display:<20} {canonical_display:<40} "
            f"{r['chunks_sampled']:>4} {r['mean_similarity']:>10.4f}{flag}"
        )
    if no_match:
        print(f"\nNo corpus match (skipped): {', '.join(no_match)}")
    if results:
        sims = [r["mean_similarity"] for r in results]
        print(
            f"\nMean: {np.mean(sims):.4f}   Min: {np.min(sims):.4f}   Max: {np.max(sims):.4f}"
        )


def interpret(label: str, mean_sim: float) -> str:
    if mean_sim >= SIMILARITY_THRESHOLD_EQUIVALENT:
        verdict = "EQUIVALENT"
    elif mean_sim >= SIMILARITY_THRESHOLD_LOW:
        verdict = "MOSTLY EQUIVALENT (some pairs may benefit from transformation)"
    else:
        verdict = "DISTINCT (transformations likely to help)"
    return f"{label}: model treats them as {verdict} (mean={mean_sim:.4f}, threshold={SIMILARITY_THRESHOLD_EQUIVALENT})"


def main() -> None:
    vocab = json.loads(VOCAB_PATH.read_text())
    lay_to_canonical: dict[str, str] = vocab["lay_to_canonical"]
    abbreviations: dict[str, str] = vocab["abbreviations"]
    corpus: list[dict] = json.loads(CORPUS_PATH.read_text())

    rng = random.Random(RANDOM_SEED)
    client = OpenAI()

    print("=== Vocabulary Semantic Equivalence Test (Corpus-Based) ===")
    print(f"Model:       {DEFAULT_MODEL}")
    print(f"Corpus size: {len(corpus)} chunks")
    print(f"Sample size: up to {SAMPLE_SIZE} chunks per term")
    print(f"Lay terms: {len(lay_to_canonical)}   Abbreviations: {len(abbreviations)}\n")

    print("--- Abbreviations ---")
    abbrev_results, abbrev_no_match = evaluate_vocab_section(
        client, "Abbreviations", abbreviations, corpus, rng
    )

    print("\n--- Lay Terms ---")
    lay_results, lay_no_match = evaluate_vocab_section(
        client, "Lay Terms", lay_to_canonical, corpus, rng
    )

    print_section_table("Abbreviations", abbrev_results, abbrev_no_match)
    print_section_table("Lay Terms", lay_results, lay_no_match)

    abbrev_mean = (
        float(np.mean([r["mean_similarity"] for r in abbrev_results]))
        if abbrev_results
        else 0.0
    )
    lay_mean = (
        float(np.mean([r["mean_similarity"] for r in lay_results]))
        if lay_results
        else 0.0
    )

    low_pairs = [
        r
        for r in abbrev_results + lay_results
        if r["mean_similarity"] < SIMILARITY_THRESHOLD_LOW
    ]

    print("\n=== Interpretation ===")
    if abbrev_results:
        print(interpret("Abbreviations", abbrev_mean))
    if lay_results:
        print(interpret("Lay terms    ", lay_mean))

    if low_pairs:
        print(
            f"\nLow-similarity pairs (< {SIMILARITY_THRESHOLD_LOW}) — transformation would help:"
        )
        for r in sorted(low_pairs, key=lambda x: x["mean_similarity"]):
            print(
                f"  {r['term']!r:20s} → {r['canonical']!r}  (sim={r['mean_similarity']:.4f})"
            )
    else:
        print(
            f"\nNo pairs below {SIMILARITY_THRESHOLD_LOW} — model handles all terms well."
        )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "model": DEFAULT_MODEL,
        "corpus_size": len(corpus),
        "sample_size": SAMPLE_SIZE,
        "random_seed": RANDOM_SEED,
        "thresholds": {
            "equivalent": SIMILARITY_THRESHOLD_EQUIVALENT,
            "low": SIMILARITY_THRESHOLD_LOW,
        },
        "abbreviations": abbrev_results,
        "abbreviations_no_match": abbrev_no_match,
        "lay_to_canonical": lay_results,
        "lay_to_canonical_no_match": lay_no_match,
        "summary": {
            "abbrev_mean": round(abbrev_mean, 6),
            "lay_mean": round(lay_mean, 6),
            "abbrev_matched": len(abbrev_results),
            "abbrev_skipped": len(abbrev_no_match),
            "lay_matched": len(lay_results),
            "lay_skipped": len(lay_no_match),
            "low_similarity_pairs": [r["term"] for r in low_pairs],
        },
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
