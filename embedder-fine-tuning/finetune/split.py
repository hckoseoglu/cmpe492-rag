"""Train/test split by source_chunk_id.

Each query is generated from exactly one source chunk (Step 2), so partitioning
the unique source_chunk_id set is equivalent to partitioning queries — no query
can land in both splits. We use a seeded RNG and split deterministically.
"""

import logging
import random
from dataclasses import dataclass

from finetune.dataset import JudgeRecord, TripletRow

logger = logging.getLogger(__name__)


@dataclass
class TestQuery:
    query: str
    style: str
    source_chunk_id: str
    source_file: str
    relevant_chunk_ids: set[str]


def train_test_split_by_chunk_id(
    records: list[JudgeRecord],
    triplet_rows: list[TripletRow],
    test_frac: float = 0.2,
    seed: int = 42,
) -> tuple[list[TripletRow], list[TestQuery]]:
    """Split into (train_rows, test_queries).

    Train rows: TripletRows whose source_chunk_id is in the train set.
    Test queries: one per (source_file, query, style) record on the test side,
    with relevant_chunk_ids = {source} ∪ judge-positives.
    """
    chunk_ids = sorted({rec.source_chunk_id for rec in records})
    rng = random.Random(seed)
    rng.shuffle(chunk_ids)

    n_test = max(1, int(round(len(chunk_ids) * test_frac)))
    test_chunk_ids = set(chunk_ids[-n_test:])
    train_chunk_ids = set(chunk_ids[:-n_test])

    train_rows = [r for r in triplet_rows if r.source_chunk_id in train_chunk_ids]

    test_queries: list[TestQuery] = []
    for rec in records:
        if rec.source_chunk_id not in test_chunk_ids:
            continue
        relevant = {rec.source_chunk_id, *rec.positive_ids}
        test_queries.append(
            TestQuery(
                query=rec.query,
                style=rec.style,
                source_chunk_id=rec.source_chunk_id,
                source_file=rec.source_file,
                relevant_chunk_ids=relevant,
            )
        )

    train_queries = {r.query_id for r in train_rows}
    mean_relevant = (
        sum(len(t.relevant_chunk_ids) for t in test_queries) / max(1, len(test_queries))
    )
    logger.info(
        f"split: {len(train_chunk_ids)} train chunks / {len(test_chunk_ids)} test chunks"
    )
    logger.info(
        f"split: {len(train_rows)} train rows over {len(train_queries)} unique queries; "
        f"{len(test_queries)} test queries (mean |relevant|={mean_relevant:.2f})"
    )

    # Check by source_chunk_id (the structural guarantee), not query text — two
    # independent chunks can coincidentally produce identical question strings.
    test_source_ids = {q.source_chunk_id for q in test_queries}
    train_source_ids = {r.source_chunk_id for r in train_rows}
    overlap = test_source_ids & train_source_ids
    if overlap:
        raise RuntimeError(
            f"chunk-level leakage: {len(overlap)} source chunks present in both splits"
        )

    return train_rows, test_queries
