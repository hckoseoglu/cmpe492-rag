from retrieval.bm25_index import BM25Index
from retrieval.dense_index import DenseIndex


def rrf_fuse(
    bm25_ranking: list[tuple[int, float]],
    dense_ranking: list[tuple[int, float]],
    k: int = 60,
) -> list[tuple[int, float, dict]]:
    """Reciprocal-rank fusion. Returns [(idx, rrf_score, meta)] sorted by score desc.

    meta carries each index's per-list rank (1-based) so callers can show provenance:
        {"bm25_rank": int|None, "dense_rank": int|None}
    """
    bm25_ranks = {idx: r for r, (idx, _) in enumerate(bm25_ranking, start=1)}
    dense_ranks = {idx: r for r, (idx, _) in enumerate(dense_ranking, start=1)}

    all_indices = set(bm25_ranks) | set(dense_ranks)
    scored: list[tuple[int, float, dict]] = []
    for idx in all_indices:
        score = 0.0
        if idx in bm25_ranks:
            score += 1.0 / (k + bm25_ranks[idx])
        if idx in dense_ranks:
            score += 1.0 / (k + dense_ranks[idx])
        scored.append(
            (
                idx,
                score,
                {
                    "bm25_rank": bm25_ranks.get(idx),
                    "dense_rank": dense_ranks.get(idx),
                },
            )
        )
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def hybrid_search(
    query: str,
    bm25: BM25Index,
    dense: DenseIndex,
    pool_size: int,
    top_k: int,
    rrf_k: int,
    exclude_idx: int | None = None,
) -> list[tuple[int, float, dict]]:
    """Run BM25 + dense search, fuse with RRF, optionally drop one corpus index,
    return top_k. Indices are positions in the shared corpus arrays."""
    bm25_top = bm25.rank(query, pool_size)
    dense_top = dense.rank(query, pool_size)
    fused = rrf_fuse(bm25_top, dense_top, k=rrf_k)
    if exclude_idx is not None:
        fused = [item for item in fused if item[0] != exclude_idx]
    return fused[:top_k]
