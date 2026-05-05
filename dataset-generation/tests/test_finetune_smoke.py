"""Smoke test for the finetune package — no LLM calls, no model download.

Asserts:
  1. The NO_DUPLICATES batch sampler distributes a query's k rows across k
     different batches (the core invariant the whole design rests on).
  2. The dataset/split helpers produce coherent train and test partitions on
     a synthetic JudgeRecord set.
  3. recall_at_k / ndcg_at_k behave correctly on edge cases.

The end-to-end model-fitting smoke is gated behind FINETUNE_SMOKE_FULL=1 so
this test stays fast in CI but the real round-trip is one env-var away.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finetune.dataset import (
    JudgeRecord,
    assert_dedup_feasibility,
    explode_to_triplets,
    load_judge_records,
)
from finetune.metrics import ndcg_at_k, recall_at_k
from finetune.split import train_test_split_by_chunk_id


def _make_record(query: str, source_id: str, pos_ids, hn_ids, chunk_contents):
    return JudgeRecord(
        query=query,
        style="formal",
        source_chunk_id=source_id,
        source_file="test.jsonl",
        positive_ids=list(pos_ids),
        hard_negative_ids=list(hn_ids),
        chunk_contents=chunk_contents,
    )


def test_explode_carries_query_id_and_skips_empty_hn():
    chunks = {f"c{i}": f"content_{i}" for i in range(10)}
    records = [
        _make_record("q1", "c0", ["c0", "c1"], ["c2", "c3"], chunks),  # 2x2 = 4 rows
        _make_record("q2", "c4", ["c4"], [], chunks),                   # dropped: no HN
        _make_record("q3", "c5", ["c5"], ["c6"], chunks),               # 1x1 = 1 row
    ]
    rows = explode_to_triplets(records)
    assert len(rows) == 5, f"expected 5 rows, got {len(rows)}"
    q1_rows = [r for r in rows if r.query_id == "q1"]
    assert len(q1_rows) == 4
    assert all(r.anchor == "q1" for r in q1_rows)
    pos_ids = {r.positive_chunk_id for r in q1_rows}
    hn_ids = {r.hard_negative_chunk_id for r in q1_rows}
    assert pos_ids == {"c0", "c1"}
    assert hn_ids == {"c2", "c3"}
    print("test_explode_carries_query_id_and_skips_empty_hn: OK")


def test_split_disjoint_queries():
    chunks = {f"c{i}": f"content_{i}" for i in range(50)}
    records = []
    for i in range(20):
        sid = f"c{i}"
        records.append(
            _make_record(f"q{i}", sid, [sid], [f"c{20 + i}", f"c{30 + i}"], chunks)
        )
    rows = explode_to_triplets(records)
    train_rows, test_queries = train_test_split_by_chunk_id(
        records, rows, test_frac=0.2, seed=42
    )
    train_queries = {r.query_id for r in train_rows}
    test_query_strs = {tq.query for tq in test_queries}
    assert not (train_queries & test_query_strs), "query-level leakage detected"
    assert len(test_queries) == 4  # 20 * 0.2
    assert len(train_queries) == 16
    # Every test query has its source_chunk_id in its relevant set
    for tq in test_queries:
        assert tq.source_chunk_id in tq.relevant_chunk_ids
    print("test_split_disjoint_queries: OK")


def test_metrics_basic():
    # query has 2 relevant chunks; retrieved list has them at ranks 1 and 3
    retrieved = ["good1", "wrong1", "good2", "wrong2", "wrong3"]
    relevant = {"good1", "good2"}
    assert recall_at_k(retrieved, relevant, 1) == 0.5
    assert recall_at_k(retrieved, relevant, 5) == 1.0
    n10 = ndcg_at_k(retrieved, relevant, 10)
    assert 0.0 < n10 < 1.0, f"unexpected ndcg {n10}"
    # all-or-nothing degenerate cases
    assert recall_at_k([], relevant, 5) == 0.0
    assert ndcg_at_k(retrieved, set(), 10) == 0.0
    print("test_metrics_basic: OK")


def test_no_duplicates_invariant():
    """Synthesize triplets where one query has 4 rows; pass through the actual
    NO_DUPLICATES sampler and confirm no batch contains two same-anchor rows.
    """
    from datasets import Dataset
    from sentence_transformers.sentence_transformer.training_args import (
        BatchSamplers,
        SentenceTransformerTrainingArguments,
    )

    rows = []
    # query A appears 4 times (k=2 positives × m=2 HNs)
    for p in ("Pa", "Pb"):
        for n in ("Na", "Nb"):
            rows.append({"anchor": "queryA", "positive": p, "negative": n})
    # five other queries with 1 row each
    for i in range(5):
        rows.append({"anchor": f"queryB{i}", "positive": f"Pb{i}", "negative": f"Nb{i}"})

    ds = Dataset.from_list(rows)
    args = SentenceTransformerTrainingArguments(
        output_dir=tempfile.mkdtemp(),
        per_device_train_batch_size=4,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        report_to=[],
        num_train_epochs=1,
    )
    # Recreate the dataloader the trainer would build
    from torch.utils.data import DataLoader
    from sentence_transformers.base.sampler import NoDuplicatesBatchSampler

    sampler = NoDuplicatesBatchSampler(
        dataset=ds,
        batch_size=4,
        drop_last=False,
        valid_label_columns=[],
    )
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=lambda x: x)
    seen_violation = False
    for batch in loader:
        anchors = [row["anchor"] for row in batch]
        if len(anchors) != len(set(anchors)):
            seen_violation = True
            print(f"  VIOLATION: batch has duplicates: {anchors}")
    assert not seen_violation, "NO_DUPLICATES sampler emitted same-anchor rows in one batch"
    print("test_no_duplicates_invariant: OK")


def test_dedup_feasibility_warning(capsys=None):
    chunks = {f"c{i}": f"content_{i}" for i in range(30)}
    # one query with 20 rows, batch size 4 → 5 batches, query needs 20 → starvation
    big_record = _make_record(
        "big_q", "c0", [f"c{i}" for i in range(4)], [f"c{i}" for i in range(4, 9)], chunks
    )
    # several small records to give us batches
    smalls = [_make_record(f"sq{i}", f"c{20+i}", [f"c{20+i}"], [f"c{25+i}"], chunks) for i in range(5)]
    rows = explode_to_triplets([big_record, *smalls])
    # Just ensure it doesn't crash; warning is logged at WARN level.
    assert_dedup_feasibility(rows, batch_size=4)
    print("test_dedup_feasibility_warning: OK")


def test_load_judge_records_roundtrip():
    """End-to-end load: write a synthetic chunks/<f>.jsonl + triplets/<f>.jsonl
    pair, point load_judge_records at them, verify content is hydrated.
    """
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        chunks_dir = td / "chunks"
        triplets_dir = td / "triplets"
        chunks_dir.mkdir()
        triplets_dir.mkdir()
        with open(chunks_dir / "book.jsonl", "w") as f:
            for i in range(5):
                f.write(json.dumps({"id": f"k{i}", "content": f"body_{i}"}) + "\n")
        with open(triplets_dir / "book.jsonl", "w") as f:
            f.write(json.dumps({
                "query": "what is k0?",
                "style": "formal",
                "source_chunk_id": "k0",
                "positives": ["k0", "k1"],
                "hard_negatives": ["k2", "k3"],
                "irrelevants": [],
            }) + "\n")
        # Should ignore these:
        (triplets_dir / "book.triplets.jsonl").write_text("")
        (triplets_dir / "book.judge_debug.jsonl").write_text("")

        records = load_judge_records(triplets_dir, chunks_dir)
        assert len(records) == 1
        rec = records[0]
        assert rec.positives() == [("k0", "body_0"), ("k1", "body_1")]
        assert rec.hard_negatives() == [("k2", "body_2"), ("k3", "body_3")]
    print("test_load_judge_records_roundtrip: OK")


def main():
    test_explode_carries_query_id_and_skips_empty_hn()
    test_split_disjoint_queries()
    test_metrics_basic()
    test_load_judge_records_roundtrip()
    test_dedup_feasibility_warning()
    test_no_duplicates_invariant()

    if os.environ.get("FINETUNE_SMOKE_FULL") == "1":
        # Trigger the actual model fit on M1 with 50 rows; takes a few minutes.
        import subprocess
        subprocess.check_call([sys.executable, "-m", "finetune.train", "--smoke"])
        print("FINETUNE_SMOKE_FULL: model fit OK")

    print("\nall finetune smoke tests passed")


if __name__ == "__main__":
    main()
