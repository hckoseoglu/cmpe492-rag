"""Remove false hard negatives from Step-4 triplet records.

A mined `hard_negative` that is (near-)identical to one of the query's positives
is not a negative at all — it is a paraphrase of the answer. Training MNRL on it
pushes the query *away* from a chunk that actually answers it, which is exactly
what dragged the first fine-tuning run below baseline.

This step compares each `hard_negative` against *every* positive of the same
query (source chunk + judge positives) and drops it when it looks like a
duplicate by EITHER measure:

  - lexical : Jaccard over alphanumeric, lowercased, >2-char token sets >= jaccard_thresh
  - semantic: BAAI/bge-m3 cosine (L2-normalised) >= cosine_thresh

`positives` are never touched, so the downstream eval ground truth
(`relevant = {source} ∪ positives`) is unaffected. Cleaned per-query records are
written to `config.filtered_triplets_dir/<book>.jsonl` (originals untouched), and
a `<book>.filter_stats.json` records how many were removed out of how many.

    python -m finetune.filter_negatives                 # all books in triplets/
    python -m finetune.filter_negatives --jaccard-thresh 0.7 --cosine-thresh 0.9
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Make sibling imports work under `python -m finetune.filter_negatives`
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config  # noqa: E402
from finetune.dataset import _load_chunk_contents, _read_jsonl  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def tokenize(text: str) -> set[str]:
    """Lowercase, alphanumeric, drop tokens of <=2 chars. Matches the probe
    tokenisation used during the contamination investigation."""
    cleaned = "".join(c.lower() if c.isalnum() else " " for c in text)
    return {tok for tok in cleaned.split() if len(tok) > 2}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def filter_record_hard_negatives(
    positives: list[tuple[str, str]],
    hard_negatives: list[tuple[str, str]],
    jaccard_thresh: float,
    cosine_thresh: float,
    embeddings: dict | None = None,
) -> tuple[list[str], list[dict]]:
    """Decide which hard negatives to keep for one query.

    A hard negative is removed if, against ANY positive, its Jaccard
    >= jaccard_thresh OR (when embeddings are supplied) its cosine
    >= cosine_thresh.

    Returns (kept_hardneg_ids, removed) where `removed` is a list of dicts
    carrying the triggering positive and both scores (for stats / human review).
    """
    pos_tokens = [(pid, tokenize(content)) for pid, content in positives]

    kept_ids: list[str] = []
    removed: list[dict] = []

    for hn_id, hn_content in hard_negatives:
        hn_tokens = tokenize(hn_content)
        best_jac, best_jac_pid = 0.0, None
        best_cos, best_cos_pid = 0.0, None

        for pid, p_tokens in pos_tokens:
            j = jaccard(hn_tokens, p_tokens)
            if j > best_jac:
                best_jac, best_jac_pid = j, pid

        if embeddings is not None and hn_id in embeddings:
            hn_vec = embeddings[hn_id]
            for pid, _ in positives:
                p_vec = embeddings.get(pid)
                if p_vec is None:
                    continue
                c = float(hn_vec @ p_vec)  # both L2-normalised => cosine
                if c > best_cos:
                    best_cos, best_cos_pid = c, pid

        lexical_hit = best_jac >= jaccard_thresh
        semantic_hit = embeddings is not None and best_cos >= cosine_thresh

        if lexical_hit or semantic_hit:
            removed.append({
                "hardneg_id": hn_id,
                "trigger_positive_id": best_jac_pid if lexical_hit else best_cos_pid,
                "jaccard": round(best_jac, 4),
                "cosine": round(best_cos, 4),
                "lexical_hit": lexical_hit,
                "semantic_hit": semantic_hit,
            })
        else:
            kept_ids.append(hn_id)

    return kept_ids, removed


def _embed_chunks(chunk_ids: list[str], chunk_contents: dict[str, str], config: Config) -> dict:
    """Embed the referenced chunks once with bge-m3, L2-normalised, returning
    {chunk_id: np.ndarray}. Cosine then reduces to a dot product."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    from retrieval.dense_index import _resolve_device

    ids = [cid for cid in chunk_ids if chunk_contents.get(cid)]
    texts = [chunk_contents[cid] for cid in ids]
    device = _resolve_device(config.embedder_device)
    logger.info(f"embedding {len(ids)} referenced chunks with {config.embedder_model} on {device}")
    model = SentenceTransformer(config.embedder_model, device=device)
    vecs = model.encode(
        texts,
        batch_size=config.embedder_batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32, copy=False)
    return {cid: vecs[i] for i, cid in enumerate(ids)}


def filter_book(
    triplets_path: Path,
    chunks_path: Path,
    out_path: Path,
    jaccard_thresh: float,
    cosine_thresh: float,
    use_semantic: bool,
    config: Config,
) -> dict:
    """Filter one book's per-query records; write cleaned file + return stats."""
    chunk_contents = _load_chunk_contents(chunks_path)
    records = list(_read_jsonl(triplets_path))

    # Collect every chunk referenced as a positive or hard_negative, embed once.
    embeddings = None
    if use_semantic:
        referenced: set[str] = set()
        for r in records:
            referenced.update(r.get("positives", []))
            referenced.update(r.get("hard_negatives", []))
        embeddings = _embed_chunks(sorted(referenced), chunk_contents, config)

    stats = {
        "book": triplets_path.name,
        "thresholds": {"jaccard": jaccard_thresh, "cosine": cosine_thresh if use_semantic else None},
        "semantic_enabled": use_semantic,
        "records_processed": 0,
        "hard_negatives_before": 0,
        "hard_negatives_removed": 0,
        "hard_negatives_remaining": 0,
        "removed_by_lexical_only": 0,
        "removed_by_semantic_only": 0,
        "removed_by_both": 0,
        "records_emptied": 0,
        "top_removed": [],
    }
    all_removed: list[dict] = []

    with open(out_path, "w") as f:
        for r in records:
            stats["records_processed"] += 1
            positives = [(cid, chunk_contents[cid]) for cid in r.get("positives", []) if chunk_contents.get(cid)]
            hard_negs = [(cid, chunk_contents[cid]) for cid in r.get("hard_negatives", []) if chunk_contents.get(cid)]
            stats["hard_negatives_before"] += len(r.get("hard_negatives", []))

            kept_ids, removed = filter_record_hard_negatives(
                positives, hard_negs, jaccard_thresh, cosine_thresh, embeddings
            )

            # Keep ids whose content was missing (couldn't be judged) untouched —
            # conservative: only drop things we actively flagged.
            judged_ids = {cid for cid, _ in hard_negs}
            unjudged = [cid for cid in r.get("hard_negatives", []) if cid not in judged_ids]
            new_hard_negs = kept_ids + unjudged

            stats["hard_negatives_removed"] += len(removed)
            stats["hard_negatives_remaining"] += len(new_hard_negs)
            if r.get("hard_negatives") and not new_hard_negs:
                stats["records_emptied"] += 1
            for rem in removed:
                if rem["lexical_hit"] and rem["semantic_hit"]:
                    stats["removed_by_both"] += 1
                elif rem["lexical_hit"]:
                    stats["removed_by_lexical_only"] += 1
                else:
                    stats["removed_by_semantic_only"] += 1
                all_removed.append({"query": r.get("query", ""), **rem})

            out_record = dict(r)
            out_record["hard_negatives"] = new_hard_negs
            f.write(json.dumps(out_record) + "\n")

    before = stats["hard_negatives_before"]
    stats["pct_removed"] = round(100 * stats["hard_negatives_removed"] / before, 2) if before else 0.0
    all_removed.sort(key=lambda d: max(d["jaccard"], d["cosine"]), reverse=True)
    stats["top_removed"] = all_removed[:20]
    return stats


def main():
    parser = argparse.ArgumentParser(description="Drop hard negatives that duplicate a positive")
    parser.add_argument("--triplets-dir", type=str, default=None, help="Default: config.triplets_dir")
    parser.add_argument("--chunks-dir", type=str, default=None, help="Default: config.output_dir")
    parser.add_argument("--out-dir", type=str, default=None, help="Default: config.filtered_triplets_dir")
    parser.add_argument("--jaccard-thresh", type=float, default=0.7)
    parser.add_argument("--cosine-thresh", type=float, default=0.9)
    parser.add_argument("--no-semantic", action="store_true",
                        help="Lexical (Jaccard) filter only — skips the bge-m3 embedding pass.")
    args = parser.parse_args()

    config = Config()
    triplets_dir = Path(args.triplets_dir) if args.triplets_dir else config.triplets_dir
    chunks_dir = Path(args.chunks_dir) if args.chunks_dir else config.output_dir
    out_dir = Path(args.out_dir) if args.out_dir else config.filtered_triplets_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    use_semantic = not args.no_semantic

    books = [
        p for p in sorted(triplets_dir.glob("*.jsonl"))
        if not (p.name.endswith(".triplets.jsonl") or p.name.endswith(".judge_debug.jsonl")
                or p.name.endswith(".filter_stats.json"))
    ]
    if not books:
        logger.error(f"no per-query triplet files in {triplets_dir}")
        sys.exit(1)

    for triplets_path in books:
        chunks_path = chunks_dir / triplets_path.name
        if not chunks_path.exists():
            logger.warning(f"no matching chunks file for {triplets_path.name} — skipping")
            continue
        out_path = out_dir / triplets_path.name
        stats = filter_book(
            triplets_path, chunks_path, out_path,
            args.jaccard_thresh, args.cosine_thresh, use_semantic, config,
        )
        stats_path = out_dir / f"{triplets_path.stem}.filter_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(
            f"{stats['book']}: removed {stats['hard_negatives_removed']}/"
            f"{stats['hard_negatives_before']} hard negatives ({stats['pct_removed']}%) "
            f"[lexical-only {stats['removed_by_lexical_only']}, "
            f"semantic-only {stats['removed_by_semantic_only']}, "
            f"both {stats['removed_by_both']}]; "
            f"{stats['records_emptied']} records left with zero hard negatives; "
            f"wrote {out_path.name} + {stats_path.name}"
        )


if __name__ == "__main__":
    main()
