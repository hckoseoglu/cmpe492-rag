"""Retrieval metrics for multi-relevant binary ground truth.

Each query has a set of relevant chunk_ids (size >= 1). Given a ranked list of
retrieved chunk_ids (length >= max k of interest), we compute:

  Recall@k = |retrieved[:k] ∩ relevant| / |relevant|
  NDCG@k   with binary gains over the relevant set, ideal DCG = sum of
            1/log2(i+1) for i=1..min(k, |relevant|)

Bootstrap is a simple percentile bootstrap over per-query metric values.
"""

import math
import random


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for cid in retrieved[:k] if cid in relevant)
    return hits / len(relevant)


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = 0.0
    for i, cid in enumerate(retrieved[:k], start=1):
        if cid in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def macro_average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def bootstrap_ci(
    values: list[float], n_resamples: int = 1000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(n_resamples * (alpha / 2))]
    hi = means[int(n_resamples * (1 - alpha / 2))]
    return (lo, hi)
