"""
Token-cost tracking wrapper around OpenAIEmbeddings.

Counts tokens via tiktoken before each embed call, logs per-call and
cumulative usage, and writes a CSV row for each call.

Usage:
    embeddings = TrackedEmbeddings(model="text-embedding-3-large", log_path="embed_log.csv")
    # use exactly like OpenAIEmbeddings
"""

import csv
import os
from datetime import datetime

import tiktoken
from langchain_openai import OpenAIEmbeddings

# Pricing per 1M tokens (as of early 2025)
PRICE_PER_1M = {
    "text-embedding-3-large": 0.13,
    "text-embedding-3-small": 0.02,
    "text-embedding-ada-002": 0.10,
}

DEFAULT_PRICE = 0.13


class TrackedEmbeddings(OpenAIEmbeddings):
    """
    Drop-in replacement for OpenAIEmbeddings that logs token usage and cost.
    """

    log_path: str = "embed_log.csv"
    stage: str = "unknown"
    _total_tokens: int = 0
    _total_cost: float = 0.0
    _enc: object = None

    def model_post_init(self, __context):
        super().model_post_init(__context)
        # cl100k_base is used for all text-embedding-3-* and ada-002 models
        object.__setattr__(self, "_enc", tiktoken.get_encoding("cl100k_base"))
        object.__setattr__(self, "_total_tokens", 0)
        object.__setattr__(self, "_total_cost", 0.0)
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["timestamp", "stage", "call", "num_texts", "tokens", "cost_usd",
                     "cumulative_tokens", "cumulative_cost_usd"]
                )

    def _count_tokens(self, texts: list[str]) -> int:
        return sum(len(self._enc.encode(t)) for t in texts)

    def _price_per_1m(self) -> float:
        return PRICE_PER_1M.get(self.model, DEFAULT_PRICE)

    def _log(self, call: str, num_texts: int, tokens: int):
        cost = tokens / 1_000_000 * self._price_per_1m()
        object.__setattr__(self, "_total_tokens", self._total_tokens + tokens)
        object.__setattr__(self, "_total_cost", self._total_cost + cost)
        print(
            f"[EmbeddingTracker] {call}: {num_texts} texts, {tokens:,} tokens, "
            f"${cost:.4f}  "
            f"(cumulative: {self._total_tokens:,} tokens, ${self._total_cost:.4f})"
        )
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now().strftime("%H:%M:%S"),
                self.stage,
                call,
                num_texts,
                tokens,
                f"{cost:.6f}",
                self._total_tokens,
                f"{self._total_cost:.6f}",
            ])

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        tokens = self._count_tokens(texts)
        self._log("embed_documents", len(texts), tokens)
        return super().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        tokens = self._count_tokens([text])
        self._log("embed_query", 1, tokens)
        return super().embed_query(text)

    def summary(self):
        print(
            f"\n[EmbeddingTracker] ── Summary ──────────────────────────\n"
            f"  Total tokens : {self._total_tokens:,}\n"
            f"  Total cost   : ${self._total_cost:.4f}\n"
            f"  Model        : {self.model} (${self._price_per_1m()}/1M tokens)\n"
            f"────────────────────────────────────────────────────────"
        )
