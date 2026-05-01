import argparse
import json
import logging
import sys
from pathlib import Path

from checkpoint import get_checkpoint_path, load_checkpoint, save_checkpoint
from config import Config
from grouper import group_propositions
from llm_client import LLMClient
from pdf_loader import batch_pages, load_pdf
from propositions import extract_propositions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def process_pdf(pdf_path: Path, config: Config, llm: LLMClient) -> bool:
    """Process a single PDF through the full pipeline. Returns True if successful."""
    pdf_name = pdf_path.name
    stem = pdf_path.stem

    # Check completion checkpoint
    done_ckpt = get_checkpoint_path(config, pdf_name, "done")
    if done_ckpt.exists():
        logger.info(f"Skipping '{pdf_name}' (already completed)")
        return True

    # Step 1: Load PDF
    pages = load_pdf(pdf_path, config)
    if not pages:
        return False

    # Step 2: Extract propositions (with checkpoint)
    props_ckpt = get_checkpoint_path(config, pdf_name, "propositions")
    cached_props = load_checkpoint(props_ckpt)

    if cached_props is not None:
        all_propositions = cached_props
        logger.info(f"'{pdf_name}': loaded {len(all_propositions)} propositions from checkpoint")
    else:
        batches = batch_pages(pages, config.max_chars_per_batch, config.pages_per_batch)
        all_propositions = []

        for i, batch in enumerate(batches):
            batch_text = "\n\n".join(p["text"] for p in batch)
            props = extract_propositions(llm, batch_text)
            all_propositions.extend(props)

            logger.info(
                f"'{pdf_name}': batch {i + 1}/{len(batches)}, "
                f"{len(props)} new propositions, {len(all_propositions)} total"
            )

        save_checkpoint(props_ckpt, all_propositions)
        logger.info(f"'{pdf_name}': saved {len(all_propositions)} propositions to checkpoint")

    if not all_propositions:
        logger.warning(f"'{pdf_name}': no propositions extracted, skipping")
        return False

    # Step 3: Group propositions into chunks
    logger.info(f"'{pdf_name}': grouping {len(all_propositions)} propositions into chunks...")
    chunks = group_propositions(llm, all_propositions, config.max_propositions_per_group)

    if not chunks:
        logger.warning(f"'{pdf_name}': grouping produced no chunks")
        return False

    # Step 4: Assign IDs and write JSONL
    output_path = config.output_dir / f"{stem}.jsonl"
    with open(output_path, "w") as f:
        for i, chunk in enumerate(chunks):
            record = {
                "id": f"{stem}_{i:04d}",
                "content": chunk["chunk_content"],
                "summary": chunk["summary"],
            }
            f.write(json.dumps(record) + "\n")

    logger.info(f"'{pdf_name}': wrote {len(chunks)} chunks to {output_path}")

    # Step 5: Mark as done
    save_checkpoint(done_ckpt, {"chunks": len(chunks), "propositions": len(all_propositions)})
    return True


def main():
    parser = argparse.ArgumentParser(description="Agentic chunking pipeline for fitness PDFs")
    parser.add_argument("--resume", action="store_true", help="Skip PDFs with completion checkpoints")
    parser.add_argument("--pdf", type=str, help="Process a single PDF by filename")
    args = parser.parse_args()

    config = Config()
    llm = LLMClient(config)

    if args.pdf:
        pdf_path = config.pdf_dir / args.pdf
        if not pdf_path.exists():
            logger.error(f"PDF not found: {pdf_path}")
            sys.exit(1)
        process_pdf(pdf_path, config, llm)
        return

    # Process all PDFs
    pdf_files = sorted(config.pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDFs found in {config.pdf_dir}")
        sys.exit(1)

    logger.info(f"Found {len(pdf_files)} PDFs in {config.pdf_dir}")

    completed = 0
    for i, pdf_path in enumerate(pdf_files):
        logger.info(f"\n--- PDF {i + 1}/{len(pdf_files)}: {pdf_path.name} ---")

        if args.resume:
            done_ckpt = get_checkpoint_path(config, pdf_path.name, "done")
            if done_ckpt.exists():
                logger.info(f"Skipping '{pdf_path.name}' (--resume, already completed)")
                completed += 1
                continue

        if process_pdf(pdf_path, config, llm):
            completed += 1

    logger.info(f"\nDone: {completed}/{len(pdf_files)} PDFs processed successfully")


if __name__ == "__main__":
    print("Starting chunker...")
    main()
