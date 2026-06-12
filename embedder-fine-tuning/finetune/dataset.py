"""Load judge-labeled per-query records and explode them into MNRL training rows.

The Step-4 output `triplets/<book>.jsonl` stores positives / hard_negatives as
*chunk_id lists*. We hydrate content from `chunks/<book>.jsonl` (the same file
the judge itself reads) so each row carries the actual text the loss needs.

A judge record with k positives (source_chunk + judge-positives) and m
hard_negatives expands to k*m training rows. All rows for a given query share
the same anchor text, which is what `BatchSamplers.NO_DUPLICATES` keys on to
prevent multi-positive rows of the same query from co-occurring in a batch.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class JudgeRecord:
    query: str
    style: str
    source_chunk_id: str
    source_file: str
    positive_ids: list[str]
    hard_negative_ids: list[str]
    chunk_contents: dict[str, str]

    def positives(self) -> list[tuple[str, str]]:
        out = []
        for cid in self.positive_ids:
            content = self.chunk_contents.get(cid)
            if content:
                out.append((cid, content))
        return out

    def hard_negatives(self) -> list[tuple[str, str]]:
        out = []
        for cid in self.hard_negative_ids:
            content = self.chunk_contents.get(cid)
            if content:
                out.append((cid, content))
        return out


@dataclass
class TripletRow:
    anchor: str
    positive: str
    negative: str
    query_id: str
    source_chunk_id: str
    positive_chunk_id: str
    hard_negative_chunk_id: str
    style: str
    source_file: str


def _read_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_chunk_contents(chunks_path: Path) -> dict[str, str]:
    contents: dict[str, str] = {}
    for row in _read_jsonl(chunks_path):
        contents[row["id"]] = row.get("content", "")
    return contents


def load_judge_records(triplets_dir: Path, chunks_dir: Path) -> list[JudgeRecord]:
    """Load every per-query record from `triplets/*.jsonl`, hydrating content
    from the matching `chunks/<book>.jsonl`.

    Skips:
      - the exploded `*.triplets.jsonl` files (already-flattened, missing chunk_ids)
      - the `*.judge_debug.jsonl` files (per-call debug logs)
    """
    records: list[JudgeRecord] = []
    files_seen = 0
    for triplets_path in sorted(triplets_dir.glob("*.jsonl")):
        name = triplets_path.name
        if name.endswith(".triplets.jsonl") or name.endswith(".judge_debug.jsonl"):
            continue
        chunks_path = chunks_dir / name
        if not chunks_path.exists():
            logger.warning(f"no matching chunks file for {name} — skipping")
            continue
        chunk_contents = _load_chunk_contents(chunks_path)
        files_seen += 1

        n_in_file = 0
        for row in _read_jsonl(triplets_path):
            records.append(
                JudgeRecord(
                    query=row["query"],
                    style=row["style"],
                    source_chunk_id=row["source_chunk_id"],
                    source_file=name,
                    positive_ids=list(row.get("positives", [])),
                    hard_negative_ids=list(row.get("hard_negatives", [])),
                    chunk_contents=chunk_contents,
                )
            )
            n_in_file += 1
        logger.info(f"  {name}: {n_in_file} judge records")

    logger.info(f"loaded {len(records)} judge records across {files_seen} files")
    return records


def explode_to_triplets(records: list[JudgeRecord]) -> list[TripletRow]:
    """Cartesian product of (positive, hard_negative) per query record.

    Source chunk is treated as a positive (it's already in record.positive_ids
    by the judge convention). Records with zero hydratable hard_negatives are
    dropped — MNRL needs an explicit negative; in-batch only would inject
    silent zeros into the negative-text column.
    """
    rows: list[TripletRow] = []
    n_dropped_no_hn = 0
    n_dropped_no_pos = 0

    for rec in records:
        positives = rec.positives()
        hard_negs = rec.hard_negatives()
        if not positives:
            n_dropped_no_pos += 1
            continue
        if not hard_negs:
            n_dropped_no_hn += 1
            continue
        for pos_id, pos_content in positives:
            for hn_id, hn_content in hard_negs:
                rows.append(
                    TripletRow(
                        anchor=rec.query,
                        positive=pos_content,
                        negative=hn_content,
                        query_id=rec.query,
                        source_chunk_id=rec.source_chunk_id,
                        positive_chunk_id=pos_id,
                        hard_negative_chunk_id=hn_id,
                        style=rec.style,
                        source_file=rec.source_file,
                    )
                )

    logger.info(
        f"exploded to {len(rows)} triplet rows "
        f"(dropped {n_dropped_no_hn} records with no hard_negatives, "
        f"{n_dropped_no_pos} with no positives)"
    )
    return rows


def assert_dedup_feasibility(rows: list[TripletRow], batch_size: int) -> None:
    """The NO_DUPLICATES sampler defers same-anchor rows to later batches. If a
    query produces more rows than there are batches in an epoch, its tail rows
    starve. Warn early rather than discover it mid-training.
    """
    if not rows:
        return
    num_batches = max(1, len(rows) // batch_size)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.query_id] = counts.get(r.query_id, 0) + 1
    max_per_query = max(counts.values())
    if max_per_query > num_batches:
        logger.warning(
            f"some queries have more rows ({max_per_query}) than batches per epoch "
            f"({num_batches}); their tail rows will starve under NO_DUPLICATES"
        )
    else:
        logger.info(
            f"dedup feasibility OK: max_rows_per_query={max_per_query} <= "
            f"num_batches_per_epoch={num_batches}"
        )
