"""Compare a fine-tuned BGE-M3 retriever against the off-the-shelf baseline.

Builds the corpus from chunks/*.jsonl (the same skip filter Step 3 uses), then
encodes every chunk with each model variant, retrieves top-k for each test
query, and reports Recall@{1,5,10} + NDCG@10. Multi-relevant ground truth:
relevant_chunk_ids = {source_chunk} ∪ judge-positives, persisted by train.py
into <run-dir>/test_queries.jsonl.

Outputs:
  <run-dir>/results/comparison.json   structured numbers + bootstrap CIs
  printed table on stdout
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Make sibling imports work under `python -m finetune.evaluate`
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config  # noqa: E402
from finetune.metrics import bootstrap_ci, macro_average, ndcg_at_k, recall_at_k  # noqa: E402
from retrieval.corpus import Corpus, load_corpus  # noqa: E402
from retrieval.dense_index import DenseIndex  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


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


def build_combined_corpus(chunks_dir: Path, skipped_path: Path) -> Corpus:
    """Concatenate all chunks/*.jsonl into a single Corpus.

    Step-3 hybrid search runs per book, but evaluation should pool the full
    corpus so we measure cross-book retrieval too. We re-use load_corpus's skip
    filter per file then concat.
    """
    all_ids: list[str] = []
    all_contents: list[str] = []
    all_summaries: list[str] = []
    seen: set[str] = set()
    for chunks_path in sorted(chunks_dir.glob("*.jsonl")):
        c = load_corpus(chunks_path, skipped_path)
        for cid, content, summary in zip(c.ids, c.contents, c.summaries):
            if cid in seen:
                continue
            seen.add(cid)
            all_ids.append(cid)
            all_contents.append(content)
            all_summaries.append(summary)
    logger.info(f"combined corpus: {len(all_ids)} chunks across {len(list(chunks_dir.glob('*.jsonl')))} files")
    return Corpus(ids=all_ids, contents=all_contents, summaries=all_summaries)


def evaluate_variant(
    label: str,
    model_name: str,
    corpus: Corpus,
    test_queries: list[dict],
    config: Config,
    device: str,
    top_k: int,
) -> dict:
    logger.info(f"\n=== evaluating variant '{label}' (model={model_name}) ===")
    index = DenseIndex(
        contents=corpus.contents,
        ids=corpus.ids,
        model_name=model_name,
        device=device,
        batch_size=config.embedder_batch_size,
        cache_dir=config.cache_dir,
        cache_tag=f"eval_combined__{label}",
    )

    per_q_recall = {1: [], 5: [], 10: []}
    per_q_ndcg10: list[float] = []
    per_style: dict[str, dict] = {}

    for tq in test_queries:
        ranked = index.rank(tq["query"], top_n=top_k)
        retrieved_ids = [corpus.ids[i] for i, _ in ranked]
        rel = tq["relevant_chunk_ids"]
        r1 = recall_at_k(retrieved_ids, rel, 1)
        r5 = recall_at_k(retrieved_ids, rel, 5)
        r10 = recall_at_k(retrieved_ids, rel, 10)
        n10 = ndcg_at_k(retrieved_ids, rel, 10)
        per_q_recall[1].append(r1)
        per_q_recall[5].append(r5)
        per_q_recall[10].append(r10)
        per_q_ndcg10.append(n10)
        style = tq.get("style", "unknown")
        bucket = per_style.setdefault(style, {1: [], 5: [], 10: [], "ndcg10": []})
        bucket[1].append(r1)
        bucket[5].append(r5)
        bucket[10].append(r10)
        bucket["ndcg10"].append(n10)

    overall = {
        "recall@1": macro_average(per_q_recall[1]),
        "recall@5": macro_average(per_q_recall[5]),
        "recall@10": macro_average(per_q_recall[10]),
        "ndcg@10": macro_average(per_q_ndcg10),
        "recall@1_ci": bootstrap_ci(per_q_recall[1]),
        "recall@5_ci": bootstrap_ci(per_q_recall[5]),
        "recall@10_ci": bootstrap_ci(per_q_recall[10]),
        "ndcg@10_ci": bootstrap_ci(per_q_ndcg10),
    }
    by_style = {
        s: {
            "recall@1": macro_average(b[1]),
            "recall@5": macro_average(b[5]),
            "recall@10": macro_average(b[10]),
            "ndcg@10": macro_average(b["ndcg10"]),
            "n": len(b[1]),
        }
        for s, b in per_style.items()
    }
    return {"overall": overall, "by_style": by_style, "n_queries": len(test_queries)}


def _fmt_ci(ci: tuple) -> str:
    lo, hi = ci
    return f"[{lo:.3f}, {hi:.3f}]"


def print_table(baseline: dict, finetuned: dict | None) -> None:
    rows = ["recall@1", "recall@5", "recall@10", "ndcg@10"]
    print()
    print(f"{'metric':<12} {'baseline':>10} {'baseline CI':>20}", end="")
    if finetuned is not None:
        print(f" {'finetuned':>10} {'finetuned CI':>20} {'delta':>10}")
    else:
        print()
    for m in rows:
        b = baseline["overall"][m]
        b_ci = baseline["overall"][f"{m}_ci"]
        print(f"{m:<12} {b:>10.3f} {_fmt_ci(b_ci):>20}", end="")
        if finetuned is not None:
            f = finetuned["overall"][m]
            f_ci = finetuned["overall"][f"{m}_ci"]
            d = f - b
            print(f" {f:>10.3f} {_fmt_ci(f_ci):>20} {d:+10.3f}")
        else:
            print()
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline vs fine-tuned BGE-M3 retriever")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Path to a finetune output dir (must contain test_queries.jsonl)")
    parser.add_argument("--baseline-model", type=str, default="BAAI/bge-m3")
    parser.add_argument("--finetuned-path", type=str, default=None,
                        help="Path to fine-tuned model dir; default = <run-dir>/final")
    parser.add_argument("--only", choices=["baseline", "finetuned", "both"], default="both")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--chunks-dir", type=str, default=None)
    args = parser.parse_args()

    config = Config()
    chunks_dir = Path(args.chunks_dir) if args.chunks_dir else config.output_dir
    skipped_path = config.pairs_dir / "_skipped.jsonl"

    run_dir = Path(args.run_dir)
    test_path = run_dir / "test_queries.jsonl"
    if not test_path.exists():
        logger.error(f"test_queries.jsonl not found in {run_dir}; run train.py first")
        sys.exit(1)
    test_queries = load_test_queries(test_path)
    logger.info(f"loaded {len(test_queries)} test queries from {test_path}")

    corpus = build_combined_corpus(chunks_dir, skipped_path)

    finetuned_path = Path(args.finetuned_path) if args.finetuned_path else run_dir / "final"

    results: dict = {}
    if args.only in ("baseline", "both"):
        results["baseline"] = evaluate_variant(
            "baseline", args.baseline_model, corpus, test_queries, config, args.device, args.top_k
        )
    if args.only in ("finetuned", "both"):
        if not finetuned_path.exists():
            logger.error(f"fine-tuned model not found at {finetuned_path}")
            sys.exit(1)
        results["finetuned"] = evaluate_variant(
            "finetuned", str(finetuned_path), corpus, test_queries, config, args.device, args.top_k
        )

    if "baseline" in results and "finetuned" in results:
        results["delta"] = {
            m: results["finetuned"]["overall"][m] - results["baseline"]["overall"][m]
            for m in ("recall@1", "recall@5", "recall@10", "ndcg@10")
        }

    out_dir = run_dir / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "comparison.json"

    def _to_jsonable(obj):
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, dict):
            return {k: _to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_jsonable(x) for x in obj]
        return obj

    with open(out_path, "w") as f:
        json.dump(_to_jsonable(results), f, indent=2)
    logger.info(f"wrote {out_path}")

    print_table(
        results.get("baseline", {"overall": {f"{m}": 0.0 for m in ("recall@1", "recall@5", "recall@10", "ndcg@10")}}),
        results.get("finetuned"),
    )


if __name__ == "__main__":
    main()
