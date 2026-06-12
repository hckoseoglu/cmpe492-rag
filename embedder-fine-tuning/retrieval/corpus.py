import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Corpus:
    ids: list[str]
    contents: list[str]
    summaries: list[str]
    id_to_idx: dict[str, int] = field(init=False)

    def __post_init__(self):
        self.id_to_idx = {cid: i for i, cid in enumerate(self.ids)}

    def __len__(self) -> int:
        return len(self.ids)

    def get(self, chunk_id: str) -> dict:
        idx = self.id_to_idx[chunk_id]
        return {
            "id": self.ids[idx],
            "content": self.contents[idx],
            "summary": self.summaries[idx],
        }


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_corpus(chunks_path: Path, skipped_path: Path | None = None) -> Corpus:
    """Load chunks JSONL into a Corpus, dropping any IDs marked as skipped for this file.

    skipped_path is treated as optional — if it doesn't exist, no filtering happens.
    Only entries whose source_file matches the chunks_path filename are dropped, so
    the global _skipped.jsonl can be shared across books safely.
    """
    rows = _read_jsonl(chunks_path)
    logger.info(f"corpus: loaded {len(rows)} raw chunks from {chunks_path.name}")

    skip_ids: set[str] = set()
    if skipped_path is not None and skipped_path.exists():
        for row in _read_jsonl(skipped_path):
            if row.get("source_file") == chunks_path.name:
                skip_ids.add(row["chunk_id"])
        logger.info(f"corpus: {len(skip_ids)} chunks marked skipped, will be excluded")

    ids: list[str] = []
    contents: list[str] = []
    summaries: list[str] = []
    for row in rows:
        cid = row["id"]
        if cid in skip_ids:
            continue
        ids.append(cid)
        contents.append(row.get("content", ""))
        summaries.append(row.get("summary", ""))

    logger.info(f"corpus: {len(ids)} chunks retained after skip filtering")
    return Corpus(ids=ids, contents=contents, summaries=summaries)
