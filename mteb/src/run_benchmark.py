"""
Run the fitness RAG MTEB benchmark.

Evaluates two embedders against a custom local retrieval task:
  - FitnessRAGEmbedder: applies lay-term and abbreviation transformations to queries
  - BaselineEmbedder: same model, no transformations

Prints a side-by-side comparison of retrieval metrics.
"""

import sys
from pathlib import Path

import mteb
import numpy as np
from datasets import load_from_disk
from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData

sys.path.insert(0, str(Path(__file__).parent))
from custom_embedder import BaselineEmbedder, FitnessRAGEmbedder

DATASET_DIR = Path(__file__).parent.parent / "dataset"
VOCAB_PATH = Path(__file__).parent.parent / "vocab" / "fitness_vocab.json"
RESULTS_DIR = Path(__file__).parent.parent / "results"


# ---------------------------------------------------------------------------
# Custom MTEB task — loads from local Arrow files instead of HuggingFace Hub
# ---------------------------------------------------------------------------

class FitnessRetrievalTask(AbsTaskRetrieval):
    metadata = mteb.TaskMetadata(
        name="FitnessRAGRetrieval",
        dataset={"path": "fitness-rag-local", "revision": "local"},
        description="Custom fitness RAG retrieval benchmark with query transformations",
        reference=None,
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        corpus_ds = load_from_disk(str(DATASET_DIR / "corpus"))["test"]
        queries_ds = load_from_disk(str(DATASET_DIR / "queries"))["test"]
        qrels_ds = load_from_disk(str(DATASET_DIR / "qrels"))["test"]

        # Build relevant_docs: dict[str, dict[str, int]]
        relevant_docs: dict[str, dict[str, int]] = {}
        for row in qrels_ds:
            qid = row["query-id"]
            did = row["corpus-id"]
            score = int(row["score"])
            relevant_docs.setdefault(qid, {})[did] = score

        # Filter queries to only those present in qrels
        qid_set = set(relevant_docs.keys())
        queries_ds = queries_ds.filter(lambda x: x["id"] in qid_set)

        self.dataset["default"]["test"] = RetrievalSplitData(
            corpus=corpus_ds,
            queries=queries_ds,
            relevant_docs=relevant_docs,
            top_ranked=None,
        )
        self.data_loaded = True


# ---------------------------------------------------------------------------
# SearchProtocol wrappers — bridge our list-based embedders to MTEB 2.x API
# ---------------------------------------------------------------------------

class FitnessSearchModel:
    """
    Wraps any embedder with encode_queries / encode_corpus methods to satisfy
    the MTEB 2.x SearchProtocol (index + search).

    Cosine similarity is computed in-memory via normalised dot product.
    OpenAI embeddings are already unit-normalised, so this is equivalent.
    """

    mteb_model_meta = None  # required by SearchProtocol's runtime isinstance check

    def __init__(self, embedder):
        self.embedder = embedder
        self._corpus_embeddings: np.ndarray | None = None
        self._corpus_ids: list[str] | None = None

    def index(self, corpus, *, task_metadata, hf_split, hf_subset,
              encode_kwargs, num_proc=None):
        docs = [{"title": row.get("title", ""), "text": row.get("text", "")}
                for row in corpus]
        self._corpus_ids = list(corpus["id"])
        embeddings = self.embedder.encode_corpus(docs)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self._corpus_embeddings = embeddings / np.maximum(norms, 1e-9)

    def search(self, queries, *, task_metadata, hf_split, hf_subset,
               top_k, encode_kwargs, top_ranked=None, num_proc=None):
        query_texts = list(queries["text"])
        query_ids = list(queries["id"])

        q_emb = self.embedder.encode_queries(query_texts)
        norms = np.linalg.norm(q_emb, axis=1, keepdims=True)
        q_emb = q_emb / np.maximum(norms, 1e-9)

        # scores: (num_queries, num_docs)
        scores = q_emb @ self._corpus_embeddings.T

        results: dict[str, dict[str, float]] = {}
        for i, qid in enumerate(query_ids):
            top_idx = np.argsort(-scores[i])[:top_k]
            results[qid] = {
                self._corpus_ids[j]: float(scores[i, j])
                for j in top_idx
            }
        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_score(results, metric: str) -> float:
    try:
        return results[0].scores["test"][0][metric]
    except (KeyError, IndexError):
        return 0.0


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_evaluation():
    task = FitnessRetrievalTask()

    print("\n=== Running: Custom Embedder (with transformations) ===")
    custom_model = FitnessSearchModel(
        FitnessRAGEmbedder(vocab_path=str(VOCAB_PATH))
    )
    evaluation = mteb.MTEB(tasks=[task])
    custom_results = evaluation.run(
        custom_model,
        output_folder=str(RESULTS_DIR / "custom_embedder"),
        overwrite_results=True,
    )

    # Reset so the second run reloads / re-indexes cleanly
    task.data_loaded = False

    print("\n=== Running: Baseline Embedder (no transformations) ===")
    baseline_model = FitnessSearchModel(BaselineEmbedder())
    evaluation = mteb.MTEB(tasks=[task])
    baseline_results = evaluation.run(
        baseline_model,
        output_folder=str(RESULTS_DIR / "baseline_embedder"),
        overwrite_results=True,
    )

    # Comparison table
    print("\n=== Results Comparison ===")
    print(f"{'Metric':<20} {'Custom':>10} {'Baseline':>10} {'Delta':>10}")
    print("-" * 52)
    for metric in ["ndcg_at_10", "ndcg_at_5", "recall_at_10", "precision_at_10"]:
        c = extract_score(custom_results, metric)
        b = extract_score(baseline_results, metric)
        print(f"{metric:<20} {c:>10.4f} {b:>10.4f} {c - b:>+10.4f}")


if __name__ == "__main__":
    run_evaluation()
