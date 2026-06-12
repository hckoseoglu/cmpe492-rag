"""Step 3 — Hybrid Search / Hard-Negative Candidate Mining.

For each (chunk_id, question, style) row in pairs/<file>.jsonl, run a global
hybrid search (BM25 + BGE-M3 dense, fused via RRF) over the chunk corpus,
exclude the source chunk, and emit the top-K candidates to candidates/<file>.jsonl.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from checkpoint import get_checkpoint_path, load_checkpoint, save_checkpoint
from config import Config
from retrieval.bm25_index import BM25Index
from retrieval.corpus import load_corpus
from retrieval.dense_index import DenseIndex
from retrieval.hybrid import hybrid_search

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def append_jsonl(path: Path, record: dict):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def iter_pairs(pairs_path: Path):
    with open(pairs_path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def process(
    chunks_path: Path,
    pairs_path: Path,
    config: Config,
    top_k: int,
    device: str,
    limit: int | None,
    resume: bool,
):
    skipped_path = config.pairs_dir / "_skipped.jsonl"
    corpus = load_corpus(chunks_path, skipped_path)

    if len(corpus) == 0:
        logger.error(f"corpus is empty after filtering — aborting")
        sys.exit(1)

    logger.info("building BM25 index...")
    bm25 = BM25Index(corpus.contents)

    logger.info("building dense index (BGE-M3)...")
    dense = DenseIndex(
        contents=corpus.contents,
        ids=corpus.ids,
        model_name=config.embedder_model,
        device=device,
        batch_size=config.embedder_batch_size,
        cache_dir=config.cache_dir,
        cache_tag=chunks_path.stem,
    )

    out_path = config.candidates_dir / chunks_path.name
    ckpt_path = get_checkpoint_path(config, chunks_path.name, "candidates")
    state = load_checkpoint(ckpt_path) if resume else None
    processed = set(state.get("processed_keys", [])) if state else set()
    if processed:
        logger.info(f"resuming: {len(processed)} queries already done")

    n_seen = 0
    n_done = 0
    for row in iter_pairs(pairs_path):
        if limit is not None and n_done >= limit:
            break
        chunk_id = row["chunk_id"]
        style = row["style"]
        question = row["question"]
        key = f"{chunk_id}::{style}"
        n_seen += 1

        if key in processed:
            continue

        if chunk_id not in corpus.id_to_idx:
            logger.warning(f"  {key}: source chunk not in corpus (likely skipped) — emitting anyway")
            exclude_idx = None
        else:
            exclude_idx = corpus.id_to_idx[chunk_id]

        results = hybrid_search(
            query=question,
            bm25=bm25,
            dense=dense,
            pool_size=config.candidate_pool_size,
            top_k=top_k,
            rrf_k=config.rrf_k,
            exclude_idx=exclude_idx,
        )

        candidates = []
        for idx, score, meta in results:
            candidates.append(
                {
                    "chunk_id": corpus.ids[idx],
                    "content": corpus.contents[idx],
                    "summary": corpus.summaries[idx],
                    "bm25_rank": meta["bm25_rank"],
                    "dense_rank": meta["dense_rank"],
                    "rrf_score": round(score, 6),
                }
            )

        append_jsonl(
            out_path,
            {
                "chunk_id": chunk_id,
                "question": question,
                "style": style,
                "source_chunk_id": chunk_id,
                "candidates": candidates,
            },
        )
        processed.add(key)
        save_checkpoint(ckpt_path, {"processed_keys": sorted(processed)})
        n_done += 1
        logger.info(f"  {key}: {len(candidates)} candidates")

    logger.info(
        f"done. queries_this_run={n_done}, total_processed={len(processed)}, scanned={n_seen}"
    )


def main():
    parser = argparse.ArgumentParser(description="Hybrid search / hard-negative candidate mining (Step 3)")
    parser.add_argument(
        "--chunks-file",
        type=str,
        required=True,
        help="Chunks JSONL filename in ./chunks/ (the corpus to search over)",
    )
    parser.add_argument(
        "--pairs-file",
        type=str,
        default=None,
        help="Pairs JSONL filename in ./pairs/ (defaults to chunks-file basename)",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Candidates per query (default: config.top_k=5)")
    parser.add_argument("--limit", type=int, help="Cap queries processed (smoke testing)")
    parser.add_argument("--resume", action="store_true", help="Skip queries already in checkpoint")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device for the embedder (default: auto)",
    )
    args = parser.parse_args()

    config = Config()
    chunks_path = config.output_dir / args.chunks_file
    pairs_name = args.pairs_file or args.chunks_file
    pairs_path = config.pairs_dir / pairs_name

    if not chunks_path.exists():
        logger.error(f"chunks file not found: {chunks_path}")
        sys.exit(1)
    if not pairs_path.exists():
        logger.error(f"pairs file not found: {pairs_path}")
        sys.exit(1)

    top_k = args.top_k if args.top_k is not None else config.top_k

    process(
        chunks_path=chunks_path,
        pairs_path=pairs_path,
        config=config,
        top_k=top_k,
        device=args.device,
        limit=args.limit,
        resume=args.resume,
    )


if __name__ == "__main__":
    print("Starting hybrid search...")
    main()
