"""Step 4 — Expert Judge Validation.

For each (query, candidate) in candidates/<file>.jsonl, ask the judge LLM
whether the candidate fully answers the query. Source chunks are always
treated as positives without a judge call. Output:

  triplets/<file>.jsonl          — one record per query with positive/negative IDs
  triplets/<file>.triplets.jsonl — exploded (query, positive, hard_negative) rows
                                    ready for MultipleNegativesRankingLoss
"""

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path

from checkpoint import get_checkpoint_path, load_checkpoint, save_checkpoint
from config import Config
from llm_client import LLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["positive", "hard_negative"]},
        "reason": {"type": "string"},
    },
    "required": ["label", "reason"],
    "additionalProperties": False,
}


JUDGE_SYSTEM_PROMPT = """\
You are a strict relevance grader for a fitness/strength-training retrieval dataset.

You will be given:
  - QUERY:     a user's fitness question
  - CANDIDATE: a chunk of text retrieved from a sports-science book

Decide whether the CANDIDATE chunk fully and definitively answers the QUERY.

Output JSON: {"label": "positive" | "hard_negative", "reason": "..."}.

Definitions:
  - "positive"      — the chunk contains the COMPLETE answer to the query, sufficient
                      on its own. A reader with only this chunk in front of them must
                      be able to give a full answer with no outside knowledge and no
                      filling in of missing details.
  - "hard_negative" — the chunk is topically related, surface-similar, or partially
                      overlapping with the query, but does NOT fully answer it.
                      Topical adjacency without the answer is a hard_negative.

Important guidance:
  - Partial answers are hard_negative, not positive. If the query asks for sets AND
    reps AND rest, and the chunk gives only sets and reps, that is hard_negative.
  - Definition-of-X chunks for a topic mentioned in the query are usually
    hard_negative if the query asks for prescriptions, comparisons, or numbers.
  - Surface keyword overlap is NOT enough. The chunk must answer the actual question.

Keep "reason" short and specific (one sentence). Return JSON only — no commentary.

────────────────────────────────────────
EXAMPLES

[positive — chunk fully answers the query]
QUERY: "how much caffeine should i take before a workout and how long beforehand?"
CANDIDATE: "Caffeine taken 30-60 minutes before exercise at a dose of 3-6 mg per kg of body weight has been shown to improve endurance performance and reduce perceived exertion."
OUTPUT: {"label": "positive", "reason": "Chunk gives both timing (30-60 min) and dose (3-6 mg/kg), fully answering the query."}

[hard_negative — partial overlap, missing the prescriptive numbers]
QUERY: "What set, repetition, and rest-interval ranges are recommended for hypertrophy in advanced lifters?"
CANDIDATE: "Hypertrophy training emphasizes moderate loads and moderate volume to maximize muscle protein synthesis and metabolic stress."
OUTPUT: {"label": "hard_negative", "reason": "Topically about hypertrophy training but gives no specific set, rep, or rest numbers."}

[hard_negative — topical look-alike with no answer]
QUERY: "What box height range is typically recommended for depth jumps performed by advanced athletes?"
CANDIDATE: "Depth jumps are a plyometric exercise in which the athlete steps off a raised platform and immediately rebounds upward upon landing."
OUTPUT: {"label": "hard_negative", "reason": "Defines depth jumps but does not specify box height ranges."}

[positive — chunk contains the prescriptive numbers asked for]
QUERY: "At approximately what percentage of VO2 max does the lactate threshold occur in untrained vs trained endurance athletes?"
CANDIDATE: "The lactate threshold is the exercise intensity at which blood lactate begins to accumulate above resting levels, typically occurring at 50-60% of VO2 max in untrained individuals and 70-80% in well-trained endurance athletes."
OUTPUT: {"label": "positive", "reason": "Chunk provides both percentage ranges (50-60% untrained, 70-80% trained) the query asks for."}
"""


JUDGE_USER_TEMPLATE = """\
QUERY: {query}

CANDIDATE:
\"\"\"
{content}
\"\"\"

Return JSON matching the schema."""


def judge_candidate(llm: LLMClient, query: str, candidate_content: str) -> dict:
    user_prompt = JUDGE_USER_TEMPLATE.format(query=query, content=candidate_content)
    return llm.chat_structured(JUDGE_SYSTEM_PROMPT, user_prompt, JUDGE_SCHEMA)


def append_jsonl(path: Path, record: dict):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def iter_candidates(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def process(
    candidates_path: Path,
    config: Config,
    llm: LLMClient,
    limit: int | None,
    resume: bool,
):
    out_records_path = config.triplets_dir / candidates_path.name
    out_triplets_path = config.triplets_dir / f"{candidates_path.stem}.triplets.jsonl"
    ckpt_path = get_checkpoint_path(config, candidates_path.name, "triplets")

    state = load_checkpoint(ckpt_path) if resume else None
    processed = set(state.get("processed_keys", [])) if state else set()
    if processed:
        logger.info(f"resuming: {len(processed)} queries already labeled")

    n_done = 0
    for row in iter_candidates(candidates_path):
        if limit is not None and n_done >= limit:
            break
        chunk_id = row["chunk_id"]
        style = row["style"]
        question = row["question"]
        source_chunk_id = row["source_chunk_id"]
        key = f"{chunk_id}::{style}"

        if key in processed:
            continue

        # Source chunk is always a positive — never sent to the judge.
        positives_meta: list[dict] = [{"chunk_id": source_chunk_id, "content": None, "judge_reason": "source_chunk"}]
        # Look up the source content from the candidates file if not present.
        # (Hybrid search excludes it from candidates — we'll fetch it lazily by id below.)

        positive_ids: list[str] = [source_chunk_id]
        hard_negative_ids: list[str] = []
        positive_contents: dict[str, str] = {}
        hard_negative_contents: dict[str, str] = {}

        for cand in row["candidates"]:
            verdict = judge_candidate(llm, question, cand["content"])
            label = verdict.get("label", "hard_negative")
            if label == "positive":
                positive_ids.append(cand["chunk_id"])
                positive_contents[cand["chunk_id"]] = cand["content"]
                positives_meta.append(
                    {
                        "chunk_id": cand["chunk_id"],
                        "content": cand["content"],
                        "judge_reason": verdict.get("reason", ""),
                    }
                )
            else:
                hard_negative_ids.append(cand["chunk_id"])
                hard_negative_contents[cand["chunk_id"]] = cand["content"]

        # Per-query record
        append_jsonl(
            out_records_path,
            {
                "query": question,
                "style": style,
                "source_chunk_id": source_chunk_id,
                "positives": positive_ids,
                "hard_negatives": hard_negative_ids,
            },
        )

        # Exploded triplets — for the source positive we don't have its content in
        # the candidates row (it was excluded by Step 3). We still emit triplets
        # using only the judge-promoted positives, which DO have content here.
        # The source positive id is preserved in the per-query record above; the
        # downstream trainer can re-hydrate it from the chunks file by id.
        for pos_id, neg_id in itertools.product(positive_ids, hard_negative_ids):
            pos_content = positive_contents.get(pos_id)
            neg_content = hard_negative_contents.get(neg_id)
            if pos_content is None or neg_content is None:
                # source-chunk content isn't in this file; skip — the per-query
                # record carries the id so it can be re-joined later.
                continue
            append_jsonl(
                out_triplets_path,
                {
                    "query": question,
                    "positive": pos_content,
                    "hard_negative": neg_content,
                },
            )

        processed.add(key)
        save_checkpoint(ckpt_path, {"processed_keys": sorted(processed)})
        n_done += 1
        logger.info(
            f"  {key}: positives={len(positive_ids)} hard_negatives={len(hard_negative_ids)}"
        )

    logger.info(f"done. queries_this_run={n_done}, total_processed={len(processed)}")


def main():
    parser = argparse.ArgumentParser(description="Negative-candidate judge labelling (Step 4)")
    parser.add_argument(
        "--candidates-file",
        type=str,
        required=True,
        help="Candidates JSONL filename in ./candidates/ (output of hybrid_search.py)",
    )
    parser.add_argument("--limit", type=int, help="Cap queries processed (smoke testing)")
    parser.add_argument("--resume", action="store_true", help="Skip queries already in checkpoint")
    args = parser.parse_args()

    config = Config()
    candidates_path = config.candidates_dir / args.candidates_file
    if not candidates_path.exists():
        logger.error(f"candidates file not found: {candidates_path}")
        sys.exit(1)

    llm = LLMClient(config)
    process(candidates_path, config, llm, args.limit, args.resume)


if __name__ == "__main__":
    print("Starting negative judge...")
    main()
