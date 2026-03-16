"""
Add hard-negative distractor chunks to an existing corpus.

Reads the existing queries and corpus from data/, generates 3 distractor
chunks per query, appends them to the corpus, and rewrites corpus.json.
Then rebuilds the MTEB dataset.

Usage:
    python src/add_distractors.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_dataset import generate_all_distractors, save_json

DATA_DIR = Path(__file__).parent.parent / "data"


def main():
    queries = json.loads((DATA_DIR / "queries.json").read_text())
    corpus = json.loads((DATA_DIR / "corpus.json").read_text())

    # Remove any previously generated distractors (re-runnable)
    gold_corpus = [doc for doc in corpus if not doc["id"].startswith("distractor_")]
    print(f"Loaded {len(queries)} queries, {len(gold_corpus)} gold chunks")

    if len(gold_corpus) != len(queries):
        print(f"WARNING: {len(gold_corpus)} gold chunks != {len(queries)} queries")

    print("\nGenerating hard-negative distractor chunks...")
    distractor_docs = generate_all_distractors(queries, gold_corpus, num_distractors=3)

    full_corpus = gold_corpus + distractor_docs
    save_json(full_corpus, DATA_DIR / "corpus.json")

    print(f"\nCorpus updated:")
    print(f"  Gold chunks  : {len(gold_corpus)}")
    print(f"  Distractors  : {len(distractor_docs)}")
    print(f"  Total corpus : {len(full_corpus)}")
    print(f"\nNext steps:")
    print(f"  1. python src/build_mteb_dataset.py   # rebuild MTEB format")
    print(f"  2. python src/run_benchmark.py         # re-run evaluation")


if __name__ == "__main__":
    main()
