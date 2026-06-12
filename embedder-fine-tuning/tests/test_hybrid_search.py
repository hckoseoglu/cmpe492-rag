"""Synthetic-corpus tests for the hybrid search pipeline.

These tests exercise BM25, RRF fusion, source-chunk exclusion, and the corpus
loader without downloading the real BGE-M3 model — `MockDense` produces scores
from a hand-coded paraphrase map, which is enough to verify that the dense
half of the fuser is wired in correctly.

Usage:
    cd dataset-generation
    python -m tests.test_hybrid_search
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.bm25_index import BM25Index  # noqa: E402
from retrieval.corpus import load_corpus  # noqa: E402
from retrieval.hybrid import rrf_fuse, hybrid_search  # noqa: E402


GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"


def _check(name: str, cond: bool, detail: str = ""):
    tag = f"{GREEN}PASS{RESET}" if cond else f"{RED}FAIL{RESET}"
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))
    return cond


CORPUS = [
    # (id, content)
    ("c00", "The squat targets the quadriceps, glutes, and hamstrings."),
    ("c01", "The deadlift targets the posterior chain including the hamstrings, glutes, and erector spinae."),
    ("c02", "Hypertrophy training emphasizes moderate loads and moderate volume to maximize muscle protein synthesis."),
    ("c03", "Performing 3-6 sets of 6-12 repetitions at 67-85% 1RM is a common hypertrophy prescription for advanced lifters."),
    ("c04", "Leg press primarily develops the quadriceps with secondary involvement from the glutes."),
    ("c05", "Lactate threshold occurs at 50-60% VO2 max in untrained individuals and 70-80% in trained endurance athletes."),
    ("c06", "Caffeine taken 30-60 minutes before exercise at 3-6 mg/kg improves endurance performance."),
    ("c07", "Depth jumps from 30-50 cm boxes are recommended for advanced athletes; over 75 cm increases ground contact time."),
    ("c08", "Plyometric training improves rate of force development and reactive strength."),
    ("c09", "Compound exercises like squats, deadlifts, and presses recruit multiple muscle groups simultaneously."),
]


# Hand-coded paraphrase map: query keyword → list of corpus_idx ranked by "semantic similarity"
# This stands in for the dense-embedder behaviour so we can test fusion without downloading bge-m3.
PARAPHRASE_MAP = {
    "muscle growth set rep prescription": [3, 2, 9, 0, 1],
    "what muscles does the squat work": [0, 9, 4, 1, 3],
    "depth jump box height": [7, 8, 3, 9, 0],
}


class MockDense:
    """Stand-in for DenseIndex.rank used only in tests."""

    def __init__(self, paraphrase_map: dict[str, list[int]]):
        self.paraphrase_map = paraphrase_map

    def rank(self, query: str, top_n: int) -> list[tuple[int, float]]:
        ranking = self.paraphrase_map.get(query, [])
        # Synthetic descending scores.
        return [(idx, 1.0 - 0.05 * r) for r, idx in enumerate(ranking[:top_n])]


def test_corpus_loader():
    print("\nTEST: corpus loader filters skipped chunks")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        chunks_path = td / "demo.jsonl"
        skipped_path = td / "_skipped.jsonl"
        chunks_path.write_text(
            '{"id":"c00","content":"foo","summary":"f"}\n'
            '{"id":"c01","content":"bar","summary":"b"}\n'
            '{"id":"c02","content":"baz","summary":"z"}\n'
        )
        skipped_path.write_text(
            '{"chunk_id":"c01","skip_reason":"x","summary":"b","source_file":"demo.jsonl"}\n'
            '{"chunk_id":"c02","skip_reason":"x","summary":"z","source_file":"OTHER.jsonl"}\n'
        )
        corpus = load_corpus(chunks_path, skipped_path)
        ok = True
        ok &= _check("c01 dropped (own file)", "c01" not in corpus.id_to_idx)
        ok &= _check("c02 retained (different source_file)", "c02" in corpus.id_to_idx)
        ok &= _check("c00 retained", "c00" in corpus.id_to_idx)
        ok &= _check("len == 2", len(corpus) == 2, f"got {len(corpus)}")
        return ok


def test_bm25_exact_keyword():
    print("\nTEST: BM25 ranks exact-keyword match first")
    contents = [c for _, c in CORPUS]
    bm25 = BM25Index(contents)
    ranking = bm25.rank("depth jump box height advanced athletes", top_n=3)
    top_idx = ranking[0][0]
    return _check("c07 ranked first for depth-jump query", top_idx == 7, f"got idx={top_idx}")


def test_dense_paraphrase():
    print("\nTEST: Mock dense ranks the paraphrase-best chunk first")
    dense = MockDense(PARAPHRASE_MAP)
    ranking = dense.rank("muscle growth set rep prescription", top_n=3)
    top_idx = ranking[0][0]
    return _check("c03 ranked first for paraphrase query", top_idx == 3, f"got idx={top_idx}")


def test_rrf_consensus_wins():
    print("\nTEST: RRF places the doc that wins both halves at rank 1")
    bm25_top = [(3, 5.0), (2, 4.0), (9, 3.0), (0, 1.0)]
    dense_top = [(3, 0.9), (2, 0.8), (1, 0.7), (9, 0.6)]
    fused = rrf_fuse(bm25_top, dense_top, k=60)
    top_idx = fused[0][0]
    return _check("c03 wins consensus", top_idx == 3, f"got idx={top_idx}")


def test_hybrid_excludes_source():
    print("\nTEST: hybrid_search excludes the source chunk from results")
    contents = [c for _, c in CORPUS]
    bm25 = BM25Index(contents)
    dense = MockDense(PARAPHRASE_MAP)
    # Source = c07 (the depth-jump chunk that BM25 will rank first).
    results = hybrid_search(
        query="depth jump box height",
        bm25=bm25,
        dense=dense,
        pool_size=10,
        top_k=5,
        rrf_k=60,
        exclude_idx=7,
    )
    indices = [idx for idx, _, _ in results]
    return _check(
        "source idx 7 not in fused results",
        7 not in indices,
        f"got indices={indices}",
    )


def test_hybrid_top_k_respected():
    print("\nTEST: hybrid_search respects top_k")
    contents = [c for _, c in CORPUS]
    bm25 = BM25Index(contents)
    dense = MockDense(PARAPHRASE_MAP)
    results = hybrid_search(
        query="depth jump box height",
        bm25=bm25,
        dense=dense,
        pool_size=10,
        top_k=3,
        rrf_k=60,
        exclude_idx=None,
    )
    return _check("len(results) == 3", len(results) == 3, f"got {len(results)}")


def main():
    tests = [
        test_corpus_loader,
        test_bm25_exact_keyword,
        test_dense_paraphrase,
        test_rrf_consensus_wins,
        test_hybrid_excludes_source,
        test_hybrid_top_k_respected,
    ]
    results = [t() for t in tests]
    passed = sum(1 for r in results if r)
    total = len(results)
    color = GREEN if passed == total else RED
    print("\n" + "=" * 60)
    print(f"  {color}{passed}/{total}{RESET} passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
