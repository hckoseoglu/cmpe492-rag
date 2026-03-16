"""
Rename distractor document IDs from `distractor_XXXX_Y` to sequential
`doc_0101`, `doc_0102`, ... and save a mapping file for traceability.

Usage:
    python src/rename_distractors.py
"""

import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

DISTRACTOR_RE = re.compile(r"^distractor_(\d+)_(\d+)$")


def main():
    corpus_path = DATA_DIR / "corpus.json"
    corpus = json.loads(corpus_path.read_text())

    gold_docs = []
    distractors = []

    for doc in corpus:
        m = DISTRACTOR_RE.match(doc["id"])
        if m:
            distractors.append((int(m.group(1)), int(m.group(2)), doc))
        else:
            gold_docs.append(doc)

    if not distractors:
        print("No distractor documents found in corpus.json — nothing to rename.")
        return

    # Sort by source doc number, then distractor index
    distractors.sort(key=lambda x: (x[0], x[1]))

    # Assign new sequential IDs starting from doc_0101
    mapping = {}
    next_id = len(gold_docs) + 1  # 101 if 100 gold docs

    for source_num, distractor_idx, doc in distractors:
        new_id = f"doc_{next_id:04d}"
        source_doc = f"doc_{source_num:04d}"
        source_query = f"q_{source_num:04d}"

        mapping[new_id] = {
            "source_query": source_query,
            "source_doc": source_doc,
            "distractor_index": distractor_idx,
        }

        doc["id"] = new_id
        next_id += 1

    # Save updated corpus
    full_corpus = gold_docs + [doc for _, _, doc in distractors]
    with open(corpus_path, "w") as f:
        json.dump(full_corpus, f, indent=2)

    # Save mapping
    mapping_path = DATA_DIR / "distractor_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"Renamed {len(distractors)} distractors → doc_{len(gold_docs) + 1:04d} to doc_{next_id - 1:04d}")
    print(f"Gold docs: {len(gold_docs)}")
    print(f"Total corpus: {len(full_corpus)}")
    print(f"Mapping saved to: {mapping_path}")
    print(f"\nNext steps:")
    print(f"  1. python src/build_mteb_dataset.py")
    print(f"  2. python src/run_benchmark.py")


if __name__ == "__main__":
    main()
