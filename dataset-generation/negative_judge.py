"""Step 4 — Expert Judge Validation.

For each (query, candidate) in candidates/<file>.jsonl, ask the judge LLM
whether the candidate fully answers the query. Source chunks are always
treated as positives without a judge call. Output:

  triplets/<file>.jsonl              — one record per query with positive/negative IDs
  triplets/<file>.triplets.jsonl     — exploded (query, positive, hard_negative) rows
                                        ready for MultipleNegativesRankingLoss
  triplets/<file>.judge_debug.jsonl  — one row per judge call (query, candidate,
                                        label, reason) for prompt-tuning review
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


# NOTE: key order in `properties` and `required` is significant under strict
# JSON-schema structured outputs — the model emits keys in this order. We put
# `reason` first so the model writes its justification BEFORE committing to a
# label (chain-of-thought before verdict).
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "label": {
            "type": "string",
            "enum": ["positive", "hard_negative", "irrelevant"],
        },
    },
    "required": ["reason", "label"],
    "additionalProperties": False,
}


JUDGE_SYSTEM_PROMPT = """\
You are a strict relevance grader for a fitness/strength-training retrieval dataset.

You will be given:
  - QUERY:     a user's fitness question
  - CANDIDATE: a chunk of text retrieved from a sports-science book

Decide whether the CANDIDATE chunk answers the QUERY in full, is a topical
look-alike that does not answer it, or is plainly off-topic.

OUTPUT FORMAT — IMPORTANT
You MUST output JSON with the keys IN THIS ORDER:
  1. "reason"  — write this FIRST. Quote or paraphrase the specific phrase(s) in
                 CANDIDATE that you are basing the verdict on, and explain how
                 they do or do not satisfy QUERY. One or two sentences max. This
                 is your reasoning step — do it before deciding the label.
  2. "label"   — one of "positive" | "hard_negative" | "irrelevant", chosen
                 AFTER and CONSISTENT WITH the reason you just wrote.

Definitions:
  - "positive"      — CANDIDATE contains a COMPLETE answer to QUERY, sufficient on
                      its own. A reader given only CANDIDATE could answer QUERY
                      fully, with no outside knowledge.
  - "hard_negative" — CANDIDATE does NOT actually answer QUERY, BUT it shares
                      meaningful surface keyword overlap or topical / semantic
                      similarity with it. The classic shape: CANDIDATE gives a
                      perfectly good answer to a DIFFERENT but adjacent question
                      on the same topic — same exercise / nutrient / system, but
                      different goal, different scope, different population, or a
                      different facet entirely. It "looks like it could be
                      relevant" because of shared vocabulary, but reading it does
                      not answer what QUERY actually asked.
  - "irrelevant"    — CANDIDATE neither answers QUERY nor shares meaningful
                      surface keywords / topical / semantic similarity with it.

Decision order:
  1. Does CANDIDATE fully answer QUERY (every sub-part of the question)?
     → "positive".
  2. Else, is CANDIDATE topically/semantically related or sharing meaningful
     keywords with QUERY? → "hard_negative".
  3. Else → "irrelevant".

Important guidance:
  - GROUND your verdict in the actual text of CANDIDATE. If you claim CANDIDATE
    "doesn't specify X", that claim must be checkable against the quoted text in
    your reason. If CANDIDATE literally states X, you cannot label hard_negative
    on the grounds that X is missing.
  - Qualitative answers can be valid. If QUERY asks for an intensity range and
    CANDIDATE says "moderate loads" — and "moderate" IS the standard sports-
    science answer to that question — CANDIDATE is positive, not hard_negative.
    Don't penalise correct domain-typical phrasing for lacking numbers.
  - Definition-of-X chunks for a topic mentioned in QUERY are usually
    hard_negative when QUERY asks for prescriptions, recommendations, or
    comparisons — defining the topic is not answering the question about it.
  - Surface keyword overlap alone is NOT enough to be positive. The classic
    hard_negative answers a NEIGHBOURING question (e.g. rest interval for max
    strength when QUERY asks about hypertrophy; caffeine effects when QUERY
    asks about caffeine dose; box height for beginners when QUERY asks about
    advanced athletes) — same vocabulary, wrong answer.
  - A chunk on a different exercise, different physiological system, or different
    nutrient with no shared terminology is irrelevant, NOT hard_negative.
  - The CONVERSE also holds: a chunk that discusses the SAME topic as QUERY but
    only at a qualitative, mechanistic, or structural level — without the specific
    prescription QUERY asks for — is hard_negative, NOT irrelevant. Topical
    overlap with QUERY's subject matter is sufficient for hard_negative; you do
    NOT need a number, range, or percentage in CANDIDATE for the topical-overlap
    test to pass. "Doesn't give the answer asked for" is hard_negative if the
    topic matches; "doesn't even share the topic" is irrelevant.
  - Before finalising "hard_negative", re-read your reason. 
    If your reason contains phrases like "doesn't specify", "doesn't mention", "fails to address", 
    verify by re-reading CANDIDATE that the missing element is genuinely absent 
    AND was genuinely required by QUERY (not invented by you). 
    If CANDIDATE's text covers it — even briefly — the label is positive.

Keep "reason" short and specific (one or two sentences). Return JSON only — no
commentary outside the JSON object.

────────────────────────────────────────
EXAMPLES

[positive — CANDIDATE fully answers QUERY]
QUERY: "how much caffeine should i take before a workout and how long beforehand?"
CANDIDATE: "Taking 3 to 6 milligrams of caffeine per kilogram of body weight roughly half an hour to an hour before exercise improves endurance and reduces the sense of effort."
OUTPUT: {"reason": "Candidate states '3 to 6 mg/kg' (dose) and 'half an hour to an hour before exercise' (timing), covering both sub-parts of the query.", "label": "positive"}

[positive — qualitative but domain-correct answer]
QUERY: "What loading intensity is typically recommended for hypertrophy training?"
CANDIDATE: "Hypertrophy training is most commonly performed with moderate loads, sufficient to reach mechanical failure within a moderate repetition range."
OUTPUT: {"reason": "'Moderate loads' is the standard sports-science answer to a loading-intensity question for hypertrophy; the candidate directly answers what was asked.", "label": "positive"}

[hard_negative — same keywords, different training goal]
QUERY: "What rest-interval range is typically recommended between sets when training for hypertrophy?"
CANDIDATE: "When training for maximal strength with loads at or above 85% of 1RM, rest intervals of 3 to 5 minutes between sets are recommended to allow phosphocreatine resynthesis."
OUTPUT: {"reason": "Candidate gives a complete rest-interval prescription, but for maximal-strength training, not for hypertrophy as the query asks — same vocabulary, neighbouring goal.", "label": "hard_negative"}

[hard_negative — same keywords, different population/scope]
QUERY: "What box height range is typically recommended for depth jumps performed by advanced athletes?"
CANDIDATE: "Beginners learning depth jumps should start with low boxes around 12 to 18 inches to develop landing mechanics before progressing higher."
OUTPUT: {"reason": "Candidate prescribes box heights for beginners, but the query asks about advanced athletes — same exercise and same dimension, wrong population.", "label": "hard_negative"}

[hard_negative — same nutrient, different sub-question]
QUERY: "How long before exercise should caffeine be ingested to maximise its ergogenic effect?"
CANDIDATE: "Caffeine acts primarily as a central nervous system stimulant, blunting adenosine receptor activity and thereby lowering perceived exertion during exercise."
OUTPUT: {"reason": "Candidate explains caffeine's mechanism of action but does not address pre-exercise timing — same nutrient, different question.", "label": "hard_negative"}

[hard_negative — qualitative discussion of the topic with no prescriptive answer]
QUERY: "How frequently should an advanced powerlifter incorporate a deload week, and to what extent should training volume be reduced during it?"
CANDIDATE: "Periodic reductions in training stress allow the central nervous system and connective tissues to recover from accumulated fatigue, supporting continued long-term progress."
OUTPUT: {"reason": "Candidate is on-topic for deloads (explains why they exist) but states neither a frequency nor a volume-reduction range — topical overlap without the prescription QUERY asks for. NOT irrelevant: a retriever could plausibly surface this for the query.", "label": "hard_negative"}

[hard_negative — same topic, structural/organizational answer instead of the asked dimension]
QUERY: "Approximately how many minutes of general aerobic warm-up are recommended before a heavy resistance training session?"
CANDIDATE: "An effective warm-up generally proceeds from a general aerobic component, to dynamic mobility drills, to specific movement preparation with submaximal loads of the lifts to be trained."
OUTPUT: {"reason": "Candidate describes the sequence of warm-up components but gives no duration for the aerobic portion — same warm-up topic, different facet (structure, not minutes). NOT irrelevant: clear topical overlap.", "label": "hard_negative"}

[positive — CANDIDATE provides the prescriptive numbers asked for]
QUERY: "At approximately what percentage of VO2 max does the lactate threshold occur in untrained vs trained endurance athletes?"
CANDIDATE: "Blood lactate begins to accumulate above resting levels at roughly 50 to 60 percent of VO2 max in untrained individuals and at 70 to 80 percent in well-trained endurance athletes."
OUTPUT: {"reason": "Candidate gives both percentage ranges ('50 to 60 percent' for untrained, '70 to 80 percent' for trained), covering both sides of the query.", "label": "positive"}

[irrelevant — different topic, no surface or semantic overlap]
QUERY: "What box height range is typically recommended for depth jumps performed by advanced athletes?"
CANDIDATE: "Cardiac stroke volume rises with submaximal aerobic exercise and plateaus near 40-60% of VO2 max in untrained adults due to limits on ventricular filling time."
OUTPUT: {"reason": "Candidate is about cardiac stroke volume during aerobic exercise — no mention of depth jumps, box heights, or plyometrics.", "label": "irrelevant"}

[irrelevant — different domain, no shared terms]
QUERY: "How many grams of carbohydrate per kg of bodyweight should an endurance athlete consume on heavy training days?"
CANDIDATE: "A pronated grip on the bench press places the forearms in roughly vertical alignment over the elbows at the bottom of the descent, reducing shear at the wrist."
OUTPUT: {"reason": "Candidate describes bench-press grip biomechanics; no mention of carbohydrate, nutrition, or endurance.", "label": "irrelevant"}
"""


JUDGE_USER_TEMPLATE = """\
QUERY: {query}

CANDIDATE:
\"\"\"
{candidate_content}
\"\"\"

Return JSON matching the schema. Remember: write "reason" FIRST (quoting the
specific candidate phrase you are evaluating), then write "label"."""


def judge_candidate(llm: LLMClient, query: str, candidate_content: str) -> dict:
    user_prompt = JUDGE_USER_TEMPLATE.format(
        query=query,
        candidate_content=candidate_content,
    )
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


def load_chunk_contents(chunks_path: Path) -> dict[str, str]:
    """Load id → content for every chunk in the source chunks JSONL.

    Used to hydrate source-positive content so it contributes to the exploded
    triplets file. The judge itself does NOT see this content — it only sees
    QUERY and CANDIDATE.
    """
    contents: dict[str, str] = {}
    with open(chunks_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            contents[row["id"]] = row["content"]
    return contents


def process(
    candidates_path: Path,
    config: Config,
    llm: LLMClient,
    limit: int | None,
    resume: bool,
):
    out_records_path = config.triplets_dir / candidates_path.name
    out_triplets_path = config.triplets_dir / f"{candidates_path.stem}.triplets.jsonl"
    out_debug_path = config.triplets_dir / f"{candidates_path.stem}.judge_debug.jsonl"
    ckpt_path = get_checkpoint_path(config, candidates_path.name, "triplets")

    chunks_path = config.output_dir / candidates_path.name
    if chunks_path.exists():
        chunk_contents = load_chunk_contents(chunks_path)
        logger.info(
            f"loaded {len(chunk_contents)} chunks for source-positive triplet hydration"
        )
    else:
        chunk_contents = {}
        logger.warning(
            f"chunks file not found at {chunks_path} — source-positive triplets "
            f"will be skipped (judge-promoted positives still contribute)."
        )

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

        # Source chunk is always treated as a positive — never sent to the judge.
        # Its content is hydrated from the chunks file (if available) so it can
        # contribute to the exploded triplets file.
        source_content = chunk_contents.get(source_chunk_id)
        positives_meta: list[dict] = [
            {
                "chunk_id": source_chunk_id,
                "content": source_content,
                "judge_reason": "source_chunk",
            }
        ]

        positive_ids: list[str] = [source_chunk_id]
        hard_negative_ids: list[str] = []
        irrelevant_ids: list[str] = []
        positive_contents: dict[str, str] = {}
        if source_content is not None:
            positive_contents[source_chunk_id] = source_content
        hard_negative_contents: dict[str, str] = {}
        irrelevant_contents: dict[str, str] = {}

        for cand in row["candidates"]:
            verdict = judge_candidate(llm, question, cand["content"])
            # Conservative default: treat malformed labels as irrelevant so
            # noisy chunks never silently leak into the MNRL training set.
            label = verdict.get("label", "irrelevant")
            reason = verdict.get("reason", "")
            append_jsonl(
                out_debug_path,
                {
                    "query": question,
                    "style": style,
                    "source_chunk_id": source_chunk_id,
                    # source_content is recorded for the human reviewer doing
                    # prompt iteration; the judge itself never sees it.
                    "source_content": source_content,
                    "candidate_chunk_id": cand["chunk_id"],
                    "candidate_content": cand["content"],
                    "label": label,
                    "reason": reason,
                    "raw_verdict": verdict,
                },
            )
            if label == "positive":
                positive_ids.append(cand["chunk_id"])
                positive_contents[cand["chunk_id"]] = cand["content"]
                positives_meta.append(
                    {
                        "chunk_id": cand["chunk_id"],
                        "content": cand["content"],
                        "judge_reason": reason,
                    }
                )
            elif label == "hard_negative":
                hard_negative_ids.append(cand["chunk_id"])
                hard_negative_contents[cand["chunk_id"]] = cand["content"]
            else:
                irrelevant_ids.append(cand["chunk_id"])
                irrelevant_contents[cand["chunk_id"]] = cand["content"]

        # Per-query record
        append_jsonl(
            out_records_path,
            {
                "query": question,
                "style": style,
                "source_chunk_id": source_chunk_id,
                "positives": positive_ids,
                "hard_negatives": hard_negative_ids,
                "irrelevants": irrelevant_ids,
            },
        )

        # Exploded triplets — source content is now loaded from the chunks file,
        # so the source positive contributes triplets too (no longer deferred to
        # downstream re-hydration).
        for pos_id, neg_id in itertools.product(positive_ids, hard_negative_ids):
            pos_content = positive_contents.get(pos_id)
            neg_content = hard_negative_contents.get(neg_id)
            if pos_content is None or neg_content is None:
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
            f"  {key}: positives={len(positive_ids)} hard_negatives={len(hard_negative_ids)} "
            f"irrelevants={len(irrelevant_ids)}"
        )

    logger.info(f"done. queries_this_run={n_done}, total_processed={len(processed)}")


def main():
    parser = argparse.ArgumentParser(
        description="Negative-candidate judge labelling (Step 4)"
    )
    parser.add_argument(
        "--candidates-file",
        type=str,
        required=True,
        help="Candidates JSONL filename in ./candidates/ (output of hybrid_search.py)",
    )
    parser.add_argument(
        "--limit", type=int, help="Cap queries processed (smoke testing)"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Skip queries already in checkpoint"
    )
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
