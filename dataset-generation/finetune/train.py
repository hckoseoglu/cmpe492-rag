"""Fine-tune BAAI/bge-m3 on judge-labeled triplets with MNRL.

Pipeline:
  1. Load per-query records from triplets/*.jsonl, hydrate content from chunks/.
  2. Explode to (anchor, positive, negative) rows — k positives × m hard_negs per query.
  3. Split by source_chunk_id (queries are 1:1 with source chunks).
  4. Train with MultipleNegativesRankingLoss + BatchSamplers.NO_DUPLICATES so
     same-query rows never share a batch (avoids in-batch false negatives).
  5. Save final model under checkpoints/bge-m3-finetuned-<timestamp>/ and
     refresh the bge-m3-finetuned-latest symlink.

Smoke mode (`--smoke`): caps to 50 rows, 1 epoch, batch 8, forces device=mps.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Make `finetune.*` importable when running as `python -m finetune.train`
# from inside the dataset-generation directory.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config  # noqa: E402
from finetune.dataset import (  # noqa: E402
    assert_dedup_feasibility,
    explode_to_triplets,
    load_judge_records,
)
from finetune.split import train_test_split_by_chunk_id  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _refresh_symlink(target_dir: Path, link_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(target_dir.name)
    logger.info(f"updated symlink {link_path.name} -> {target_dir.name}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BGE-M3 with MNRL on judge triplets")
    parser.add_argument("--triplets-dir", type=str, default=None, help="Default: <config.triplets_dir>")
    parser.add_argument("--chunks-dir", type=str, default=None, help="Default: <config.output_dir>")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Default: checkpoints/bge-m3-finetuned-<timestamp>")
    parser.add_argument("--base-model", type=str, default="BAAI/bge-m3")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--smoke", action="store_true",
                        help="1 epoch, batch 8, device=mps, capped to 50 rows of real data")
    parser.add_argument("--synthetic-smoke", action="store_true",
                        help="Like --smoke but synthesizes 50 rows in-memory; no real triplet files needed. "
                             "Use this on M1 before any real GCP run has produced triplets.")
    args = parser.parse_args()

    config = Config()
    triplets_dir = Path(args.triplets_dir) if args.triplets_dir else config.triplets_dir
    chunks_dir = Path(args.chunks_dir) if args.chunks_dir else config.output_dir

    if args.smoke or args.synthetic_smoke:
        args.epochs = 1
        args.batch_size = 8
        args.device = "mps"
        mode = "SYNTHETIC SMOKE" if args.synthetic_smoke else "SMOKE"
        logger.info(f"{mode}: epochs=1 batch=8 device=mps row-cap=50")

    device = _resolve_device(args.device)
    logger.info(f"device resolved to {device}")

    if args.synthetic_smoke:
        from finetune.dataset import TripletRow
        train_rows = [
            TripletRow(
                anchor=f"What does chunk {i} say?",
                positive=f"Chunk {i} discusses topic A in detail and provides specifics.",
                negative=f"Chunk {i} mentions topic B but does not address topic A.",
                query_id=f"q{i}",
                source_chunk_id=f"c{i}",
                positive_chunk_id=f"c{i}",
                hard_negative_chunk_id=f"c{i + 1000}",
                style="formal",
                source_file="synthetic.jsonl",
            )
            for i in range(50)
        ]
        test_queries = []
        logger.info(f"synthetic smoke: generated {len(train_rows)} synthetic rows")
    else:
        records = load_judge_records(triplets_dir, chunks_dir)
        if not records:
            logger.error(f"no judge records found in {triplets_dir} — aborting")
            sys.exit(1)

        triplet_rows = explode_to_triplets(records)
        if not triplet_rows:
            logger.error("zero triplet rows after explosion — nothing to train on")
            sys.exit(1)

        train_rows, test_queries = train_test_split_by_chunk_id(
            records, triplet_rows, test_frac=args.test_frac, seed=args.seed
        )

        if args.smoke:
            train_rows = train_rows[:50]
            logger.info(f"smoke: capped train rows to {len(train_rows)}")

    if not train_rows:
        logger.error("zero train rows after split — aborting")
        sys.exit(1)

    assert_dedup_feasibility(train_rows, args.batch_size)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        if args.synthetic_smoke:
            suffix = "synthetic-smoke"
        elif args.smoke:
            suffix = "smoke"
        else:
            suffix = timestamp
        output_dir = config.checkpoint_dir / f"bge-m3-finetuned-{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"output dir: {output_dir}")

    # Persist the test split so evaluate.py uses the exact same queries
    test_split_path = output_dir / "test_queries.jsonl"
    with open(test_split_path, "w") as f:
        for tq in test_queries:
            f.write(json.dumps({
                "query": tq.query,
                "style": tq.style,
                "source_chunk_id": tq.source_chunk_id,
                "source_file": tq.source_file,
                "relevant_chunk_ids": sorted(tq.relevant_chunk_ids),
            }) + "\n")
    logger.info(f"wrote {len(test_queries)} test queries to {test_split_path.name}")

    # Heavy imports deferred until we know we have data
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
    from sentence_transformers.sentence_transformer.training_args import (
        BatchSamplers,
        SentenceTransformerTrainingArguments,
    )

    ds = Dataset.from_list([
        {"anchor": r.anchor, "positive": r.positive, "negative": r.negative}
        for r in train_rows
    ])

    model = SentenceTransformer(args.base_model, device=device)
    if args.max_seq_length:
        model.max_seq_length = args.max_seq_length
    loss = MultipleNegativesRankingLoss(model)

    fp16 = (device == "cuda")  # mps + fp16 is unstable; cpu doesn't need it
    train_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        fp16=fp16,
        bf16=False,
        logging_steps=max(1, len(train_rows) // (args.batch_size * 10)),
        save_strategy="epoch",
        save_total_limit=2,
        seed=args.seed,
        dataloader_drop_last=False,
        report_to=[],  # no W&B etc.
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        loss=loss,
    )
    logger.info(
        f"starting training: rows={len(train_rows)} epochs={args.epochs} "
        f"batch={args.batch_size} lr={args.lr} sampler=NO_DUPLICATES"
    )
    trainer.train()

    final_dir = output_dir / "final"
    model.save(str(final_dir))
    logger.info(f"saved final model to {final_dir}")

    # Stable alias for evaluate.py — but only for non-smoke runs.
    if not (args.smoke or args.synthetic_smoke):
        latest = config.checkpoint_dir / "bge-m3-finetuned-latest"
        _refresh_symlink(output_dir, latest)


if __name__ == "__main__":
    main()
