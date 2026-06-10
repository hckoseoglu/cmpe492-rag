"""Build the data.json that powers the interactive retrieval demo.

For each of the three fine-tuned retrievers (at its best epoch), this:
  1. loads the book-scoped NCSA corpus (same skip filter as evaluate.py),
  2. embeds the corpus with the off-the-shelf BASELINE and the FINE-TUNED model,
  3. for every held-out test query, records the rank-1 chunk each model returns,
     plus the rank the *correct* (relevant) chunk lands at,
  4. mines the queries where fine-tuning flipped rank-1 from wrong -> right,
  5. writes:
       website/_all_wins.json  — every rank-1 flip (full detail, for curation)
       website/data.json       — curated demo payload consumed by index.html

Run:  source ~/.venvs/.rag/bin/activate && python website/build_data.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # dataset-generation/
sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402
from retrieval.corpus import load_corpus  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"
CACHE.mkdir(exist_ok=True)

SEQ = 512
TOP_K_STORE = 5  # how many ranks to keep per query for display
BOOK = "NCSA_Essentials_of_ Strength_Training_and_Conditioning.jsonl"

MODELS = [
    {
        "tag": "bgesmall",
        "label": "BGE-small-en-v1.5",
        "hf": "BAAI/bge-small-en-v1.5",
        "params": "33.4M",
        "epoch": 1,
    },
    {
        "tag": "minilm",
        "label": "all-MiniLM-L6-v2",
        "hf": "sentence-transformers/all-MiniLM-L6-v2",
        "params": "22.7M",
        "epoch": 1,
    },
    {
        "tag": "micro",
        "label": "bge-micro-v2",
        "hf": "TaylorAI/bge-micro-v2",
        "params": "17.4M",
        "epoch": 2,
    },
]


def resolve_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def load_test_queries(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                row["relevant_chunk_ids"] = set(row["relevant_chunk_ids"])
                rows.append(row)
    return rows


def encode_corpus(model, contents: list[str], cache_key: str) -> np.ndarray:
    npy = CACHE / f"{cache_key}.npy"
    if npy.exists():
        return np.load(npy)
    emb = model.encode(
        contents,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    np.save(npy, emb)
    return emb


def main():
    from sentence_transformers import SentenceTransformer

    device = resolve_device()
    print(f"device: {device}")

    config = Config()
    chunks_path = config.output_dir / BOOK
    skipped_path = config.pairs_dir / "_skipped.jsonl"
    corpus = load_corpus(chunks_path, skipped_path)
    ids = corpus.ids
    id_to_idx = {cid: i for i, cid in enumerate(ids)}
    print(f"corpus: {len(ids)} chunks")

    all_wins = {}            # tag -> list of win records (full detail)
    model_meta = {}          # tag -> header info + metrics
    per_model_examples = {}  # tag -> all per-query records (compact)

    for m in MODELS:
        tag = m["tag"]
        run_dir = ROOT / "checkpoints" / f"exp-{tag}"
        ft_path = run_dir / f"epoch-{m['epoch']}"
        test_queries = load_test_queries(run_dir / "test_queries.jsonl")
        queries = [tq["query"] for tq in test_queries]
        print(f"\n=== {tag}: {len(test_queries)} queries, ft={ft_path.name} ===")

        # baseline + fine-tuned doc embeddings
        variants = {}
        for vname, model_path in (("baseline", m["hf"]), ("finetuned", str(ft_path))):
            st = SentenceTransformer(model_path, device=device)
            st.max_seq_length = SEQ
            doc_emb = encode_corpus(st, corpus.contents, f"{tag}_{vname}_docs")
            q_emb = st.encode(
                queries, batch_size=64, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=True,
            ).astype(np.float32)
            variants[vname] = (doc_emb, q_emb)
            del st

        b_docs, b_q = variants["baseline"]
        f_docs, f_q = variants["finetuned"]

        def rank_info(doc_emb, q_vec, rel_ids):
            scores = doc_emb @ q_vec
            order = np.argsort(-scores)
            top = [
                {"chunk_id": ids[i], "score": float(scores[i])}
                for i in order[:TOP_K_STORE]
            ]
            pos = {ids[order[r]]: r + 1 for r in range(len(order))}
            gold_rank = min((pos[c] for c in rel_ids if c in pos), default=None)
            return top, gold_rank

        records = []
        for qi, tq in enumerate(test_queries):
            rel = tq["relevant_chunk_ids"]
            b_top, b_gold = rank_info(b_docs, b_q[qi], rel)
            f_top, f_gold = rank_info(f_docs, f_q[qi], rel)
            b_ok = b_top[0]["chunk_id"] in rel
            f_ok = f_top[0]["chunk_id"] in rel
            records.append({
                "query": tq["query"],
                "style": tq.get("style", "unknown"),
                "source_chunk_id": tq["source_chunk_id"],
                "relevant_chunk_ids": sorted(rel),
                "baseline": {"top": b_top, "gold_rank": b_gold, "rank1_ok": b_ok},
                "finetuned": {"top": f_top, "gold_rank": f_gold, "rank1_ok": f_ok},
            })
        per_model_examples[tag] = records

        # wins = baseline rank-1 wrong, fine-tuned rank-1 right
        wins = [r for r in records if (not r["baseline"]["rank1_ok"]) and r["finetuned"]["rank1_ok"]]
        print(f"  rank-1 flips (wrong->right): {len(wins)} / {len(records)}")
        all_wins[tag] = wins

        comp = json.load(open(run_dir / "results" / f"{tag}_epoch{m['epoch']}.json"))
        model_meta[tag] = {
            "label": m["label"], "hf": m["hf"], "params": m["params"],
            "best_epoch": m["epoch"],
            "metrics": {
                "baseline": {k: comp["baseline"]["overall"][k] for k in ("recall@1", "recall@5", "recall@10", "ndcg@10")},
                "finetuned": {k: comp["finetuned"]["overall"][k] for k in ("recall@1", "recall@5", "recall@10", "ndcg@10")},
            },
        }

    # hydrate chunk contents for everything referenced by any win
    needed = set()
    for tag, wins in all_wins.items():
        for r in wins:
            needed.update(r["relevant_chunk_ids"])
            for v in ("baseline", "finetuned"):
                for t in r[v]["top"]:
                    needed.add(t["chunk_id"])
    chunk_text = {
        cid: {"content": corpus.contents[id_to_idx[cid]], "summary": corpus.summaries[id_to_idx[cid]]}
        for cid in needed if cid in id_to_idx
    }

    json.dump(
        {"models": model_meta, "wins": all_wins, "chunks": chunk_text},
        open(HERE / "_all_wins.json", "w"), indent=2,
    )
    print(f"\nwrote {HERE/'_all_wins.json'} "
          f"({sum(len(w) for w in all_wins.values())} total wins, {len(chunk_text)} chunks)")


if __name__ == "__main__":
    main()
