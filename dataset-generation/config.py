import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # Paths
    pdf_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "resources")
    output_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "chunks")
    pairs_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "pairs")
    candidates_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "candidates")
    triplets_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "triplets")
    cache_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / ".cache")
    checkpoint_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "checkpoints")

    # LLM settings (defaults to Ollama; override via env vars for vLLM on GCP)
    base_url: str = field(default_factory=lambda: os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"))
    model: str = field(default_factory=lambda: os.environ.get("LLM_MODEL", "gemma2:9b"))
    api_key: str = field(default_factory=lambda: os.environ.get("LLM_API_KEY", "ollama"))
    temperature: float = 0.1
    is_gemma: bool = field(default_factory=lambda: os.environ.get("LLM_IS_GEMMA", "false").lower() == "true")

    # Processing params
    pages_per_batch: int = 3
    max_chars_per_batch: int = 6000
    max_propositions_per_group: int = 40
    min_page_chars: int = 50
    min_pdf_chars: int = 1000

    # Pair generation
    max_pair_retries: int = 2

    # Hybrid search / hard-negative mining
    embedder_model: str = field(default_factory=lambda: os.environ.get("EMBEDDER_MODEL", "BAAI/bge-m3"))
    embedder_device: str = field(default_factory=lambda: os.environ.get("EMBEDDER_DEVICE", "auto"))
    embedder_batch_size: int = 32
    top_k: int = 5
    rrf_k: int = 60
    candidate_pool_size: int = 50  # per-list pool size before RRF fusion

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pairs_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        self.triplets_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
