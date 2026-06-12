import re

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase + alphanumeric tokens. No stemming, no stopword removal — keep it
    simple and predictable so behaviour matches what users see in their queries."""
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, contents: list[str]):
        self.tokenized = [tokenize(c) for c in contents]
        self.bm25 = BM25Okapi(self.tokenized)
        self.size = len(contents)

    def rank(self, query: str, top_n: int) -> list[tuple[int, float]]:
        """Return [(corpus_idx, score)] sorted by descending BM25 score, len <= top_n."""
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        n = min(top_n, len(scores))
        # Argsort the top-n scores. For ~4k docs, full sort is fine.
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [(i, float(scores[i])) for i in order]
