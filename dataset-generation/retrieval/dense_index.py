import hashlib
import json
import logging
from pathlib import Path

import numpy as np

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


def _fingerprint(model_name: str, ids: list[str], contents: list[str]) -> str:
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    h.update(b"\x00")
    for cid, content in zip(ids, contents):
        h.update(cid.encode("utf-8"))
        h.update(b"\x01")
        h.update(content.encode("utf-8"))
        h.update(b"\x02")
    return h.hexdigest()[:16]


class DenseIndex:
    """Dense embedding index over a corpus.

    Embeddings are L2-normalized so cosine similarity == dot product. The
    embedding matrix is cached to disk keyed by a hash of (model_name, ids,
    contents); unchanged corpora are re-loaded instantly across runs.
    """

    def __init__(
        self,
        contents: list[str],
        ids: list[str],
        model_name: str,
        device: str = "auto",
        batch_size: int = 32,
        cache_dir: Path | None = None,
        cache_tag: str | None = None,
    ):
        self.model_name = model_name
        self.device = _resolve_device(device)
        self.batch_size = batch_size

        if cache_dir is not None and cache_tag is not None:
            cache_dir = Path(cache_dir) / "embeddings"
            cache_dir.mkdir(parents=True, exist_ok=True)
            fp = _fingerprint(model_name, ids, contents)
            self.cache_npy = cache_dir / f"{cache_tag}.{fp}.npy"
            self.cache_meta = cache_dir / f"{cache_tag}.{fp}.json"
        else:
            self.cache_npy = None
            self.cache_meta = None

        if self.cache_npy is not None and self.cache_npy.exists():
            logger.info(f"dense: loading cached embeddings from {self.cache_npy.name}")
            self.embeddings = np.load(self.cache_npy)
            self._model = None  # lazy-loaded for query encoding
        else:
            self._model = self._load_model()
            logger.info(
                f"dense: encoding {len(contents)} docs on {self.device} "
                f"(batch={batch_size}, model={model_name})"
            )
            self.embeddings = self._encode(contents)
            if self.cache_npy is not None:
                np.save(self.cache_npy, self.embeddings)
                with open(self.cache_meta, "w") as f:
                    json.dump(
                        {"model": model_name, "n_docs": len(contents), "dim": int(self.embeddings.shape[1])},
                        f,
                    )
                logger.info(f"dense: cached embeddings to {self.cache_npy.name}")

    def _load_model(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.model_name, device=self.device)

    def _encode(self, texts: list[str]) -> np.ndarray:
        emb = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return emb.astype(np.float32, copy=False)

    def rank(self, query: str, top_n: int) -> list[tuple[int, float]]:
        if self._model is None:
            self._model = self._load_model()
        q_emb = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32, copy=False)
        scores = (self.embeddings @ q_emb[0])  # cosine, since both normalized
        n = min(top_n, scores.shape[0])
        # argpartition then sort the top-n window — O(N) + O(n log n)
        if n < scores.shape[0]:
            part = np.argpartition(-scores, n - 1)[:n]
            order = part[np.argsort(-scores[part])]
        else:
            order = np.argsort(-scores)
        return [(int(i), float(scores[i])) for i in order]
