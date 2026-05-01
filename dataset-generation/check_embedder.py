"""Standalone smoke test: download (if needed) and load BAAI/bge-m3, embed a few
random strings, print timings + output shape.

Verbose by design — the HF download has no progress bar by default and looks
like a hang. This script enables tqdm + transformers info logging so you can
see bytes moving.

Usage:
    cd dataset-generation
    python check_embedder.py
    python check_embedder.py --device cpu          # force CPU
    python check_embedder.py --model BAAI/bge-m3   # change model
"""

import argparse
import logging
import os
import time

# Surface tqdm + transformers progress instead of silent waits.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "info")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("check_embedder")


def resolve_device(requested: str) -> str:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    device = resolve_device(args.device)
    logger.info(f"using device={device}, model={args.model}")

    logger.info("importing sentence_transformers (first import can take a few seconds)...")
    t0 = time.time()
    from sentence_transformers import SentenceTransformer

    logger.info(f"  imported in {time.time() - t0:.1f}s")

    logger.info("loading model — first run will download ~2.3GB; subsequent runs are instant")
    logger.info("  (watch ~/.cache/huggingface/hub/models--BAAI--bge-m3/ if you want to verify progress)")
    t0 = time.time()
    model = SentenceTransformer(args.model, device=device)
    logger.info(f"  model loaded in {time.time() - t0:.1f}s")

    sentences = [
        "The squat targets the quadriceps, glutes, and hamstrings.",
        "how many sets and reps should i do for muscle growth?",
        "Caffeine taken 30-60 minutes before exercise improves endurance.",
    ]
    logger.info(f"encoding {len(sentences)} sentences...")
    t0 = time.time()
    emb = model.encode(
        sentences,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    logger.info(f"  encoded in {time.time() - t0:.2f}s, shape={emb.shape}, dtype={emb.dtype}")

    # Quick sanity: cosine similarity between sentence 0 and 1 (related) vs 0 and 2 (unrelated).
    import numpy as np

    sim_01 = float(np.dot(emb[0], emb[1]))
    sim_02 = float(np.dot(emb[0], emb[2]))
    logger.info(f"cosine(sq-question, reps-question) = {sim_01:.4f}")
    logger.info(f"cosine(sq-question, caffeine)      = {sim_02:.4f}")
    logger.info("OK — embedder is working." if sim_01 > sim_02 else "WARN — sims look off but model loaded")


if __name__ == "__main__":
    main()
