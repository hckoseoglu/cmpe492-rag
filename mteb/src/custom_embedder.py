"""
Custom MTEB-compatible embedders for the fitness RAG benchmark.

FitnessRAGEmbedder: applies vocabulary transformations (lay terms → canonical,
abbreviations → full forms) to queries before encoding with OpenAI text-embedding-3-large.

BaselineEmbedder: same model, no transformations — used for comparison.
"""

import json
import re
from pathlib import Path

import numpy as np
from openai import OpenAI

DEFAULT_VOCAB_PATH = Path(__file__).parent.parent / "vocab" / "fitness_vocab.json"
DEFAULT_MODEL = "text-embedding-3-large"
_BATCH_SIZE = 100  # OpenAI embeddings API limit per request


def _embed(client: OpenAI, texts: list[str], model: str) -> np.ndarray:
    """Call OpenAI embeddings API in batches and return stacked numpy array."""
    all_embeddings = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        response = client.embeddings.create(input=batch, model=model)
        all_embeddings.extend(e.embedding for e in response.data)
    return np.array(all_embeddings, dtype=np.float32)


class FitnessRAGEmbedder:
    """
    Custom MTEB-compatible embedder that:
    1. Applies lay term → canonical term transformations to queries
    2. Applies abbreviation expansion to queries
    3. Encodes using OpenAI text-embedding-3-large

    Documents are encoded WITHOUT transformations (they use canonical terms already).
    Transformations only apply to queries (encode_queries method).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        vocab_path: str = str(DEFAULT_VOCAB_PATH),
    ):
        self.model_name = model_name
        self.client = OpenAI()
        self.vocab = json.loads(Path(vocab_path).read_text())
        self.lay_to_canonical: dict[str, str] = self.vocab["lay_to_canonical"]
        self.abbreviations: dict[str, str] = self.vocab["abbreviations"]
        self._build_patterns()

    def _build_patterns(self) -> None:
        """Pre-compile regex patterns for all vocabulary terms."""
        # Abbreviations: match whole word, case-sensitive
        self.abbrev_pattern: dict[str, re.Pattern] = {
            abbr: re.compile(r"\b" + re.escape(abbr) + r"\b")
            for abbr in self.abbreviations
        }
        # Lay terms: match whole word, case-insensitive
        self.lay_pattern: dict[str, re.Pattern] = {
            term: re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
            for term in self.lay_to_canonical
        }

    def transform_query(self, query: str) -> str:
        """Apply vocabulary transformations to a single query."""
        # 1. Expand abbreviations first (before lay term matching)
        for abbr, full_form in self.abbreviations.items():
            query = self.abbrev_pattern[abbr].sub(full_form, query)

        # 2. Convert lay terms to canonical terms
        for lay_term, canonical in self.lay_to_canonical.items():
            query = self.lay_pattern[lay_term].sub(canonical, query)

        return query

    def encode_queries(
        self, queries: list[str], batch_size: int = _BATCH_SIZE, **kwargs
    ) -> np.ndarray:
        """
        MTEB calls this for query encoding.
        Applies transformations before encoding.
        """
        transformed = [self.transform_query(q) for q in queries]
        return _embed(self.client, transformed, self.model_name)

    def encode_corpus(
        self, corpus: list[dict], batch_size: int = _BATCH_SIZE, **kwargs
    ) -> np.ndarray:
        """
        MTEB calls this for document encoding.
        No transformations applied — documents already use canonical terms.
        """
        texts = [
            (doc.get("title", "") + " " + doc.get("text", "")).strip()
            for doc in corpus
        ]
        return _embed(self.client, texts, self.model_name)

    def encode(
        self, sentences: list[str], batch_size: int = _BATCH_SIZE, **kwargs
    ) -> np.ndarray:
        """
        Fallback encode method required by MTEB interface.
        Used for non-retrieval tasks. No transformations applied.
        """
        return _embed(self.client, sentences, self.model_name)


class BaselineEmbedder:
    """Same model, no query transformations. Used as baseline for comparison."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.client = OpenAI()

    def encode_queries(
        self, queries: list[str], batch_size: int = _BATCH_SIZE, **kwargs
    ) -> np.ndarray:
        return _embed(self.client, queries, self.model_name)

    def encode_corpus(
        self, corpus: list[dict], batch_size: int = _BATCH_SIZE, **kwargs
    ) -> np.ndarray:
        texts = [
            (doc.get("title", "") + " " + doc.get("text", "")).strip()
            for doc in corpus
        ]
        return _embed(self.client, texts, self.model_name)

    def encode(
        self, sentences: list[str], batch_size: int = _BATCH_SIZE, **kwargs
    ) -> np.ndarray:
        return _embed(self.client, sentences, self.model_name)
