import argparse
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


GENERATOR_SCHEMA = {
    "type": "object",
    "properties": {
        "is_content": {"type": "boolean"},
        "skip_reason": {"type": "string"},
        "question": {"type": "string"},
    },
    "required": ["is_content", "skip_reason", "question"],
    "additionalProperties": False,
}

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "valid": {"type": "boolean"},
        "failure_reason": {"type": "string"},
    },
    "required": ["valid", "failure_reason"],
    "additionalProperties": False,
}

VALID_SKIP_REASONS = {
    "table_of_contents",
    "author_credentials",
    "copyright_notice",
    "references",
    "figure_caption",
    "other_metadata",
}

GENERATOR_SYSTEM_PROMPT = """\
You are a synthetic-question generator for a fitness/strength-training retrieval dataset.
You will be given one CHUNK of text (with a short SUMMARY) extracted from a sports-science book.

Your job has TWO parts:

PART A — Decide whether the chunk contains substantive content.
Substantive content = information about exercise science, anatomy, physiology, biomechanics,
training programming, nutrition for athletes, technique, or related topics that a personal
trainer or athlete would actually want to look up.

NON-substantive content includes:
  - "table_of_contents": chapter/section listings, page-number indexes
  - "author_credentials": author bios, credentials, dedications, acknowledgements
  - "copyright_notice": publisher info, ISBNs, edition history, legal text
  - "references": pure bibliography / citation lists with no explanation
  - "figure_caption": isolated figure or table captions with no surrounding explanation
  - "other_metadata": page headers/footers, foreword fluff, anything else with no real content

If the chunk is NON-substantive, set is_content=false, set skip_reason to one of the labels
above, and set question to "".

PART B — If the chunk IS substantive, generate ONE question in the requested STYLE.

Hard requirements for the question:
  1. The chunk MUST fully and definitively answer the question. A reader with only this
     chunk in front of them must be able to give a complete answer — no outside knowledge,
     no partial answers.
  2. The question must be SPECIFIC to this chunk. Avoid generic questions like
     "what is strength training?" that thousands of other chunks could also answer.
  3. NEVER name or reference any of the following in the question, even if the chunk
     mentions them:
       - Institutions or organizations (e.g., NSCA, ACSM, NASM, IOC, university names)
       - People — researchers, authors, coaches, athletes (e.g., "Schoenfeld", "Zatsiorsky")
       - Specific studies, papers, journals, or experiments (no "in a 2010 study...",
         no "according to research by...", no journal names)
       - Specific books or resources (no "according to this textbook", no book titles)
     Ask about the underlying concept directly. The question is simulating a real user
     query; a user does not say "according to the NSCA" — they just ask the question.
     If the only meaningful answerable content of the chunk is "X said Y", that chunk
     is not a good source — set is_content=false with skip_reason="other_metadata".
  4. USER PERSONA — The question must come from the perspective of a real user (athlete,
     trainee, personal trainer) asking a fitness question. The user has NO IDEA the
     answer will come from a particular book, chapter, edition, or document. Therefore
     the question must NEVER:
       - Refer to the source itself: "this book", "this chapter", "this section",
         "this text", "this manual", "this guide", "this version"
       - Refer to editions: "the third edition", "the new edition", "the updated edition"
       - Refer to chapters by number or position: "chapter 9", "chapters 9 and 10",
         "the next chapter", "the previous section"
       - Use meta-framing about the source's structure or content coverage:
         "what's new in...", "what's been updated", "what topics are covered in...",
         "what areas have been expanded", "what does the book/chapter discuss"
       - Reference authors of the source: "the author(s)", "the editors"
     If the chunk is itself purely a meta-description of a book's structure (e.g., "this
     third edition adds new chapters on nutrition", "chapters 9 and 10 cover X") and
     contains no concrete fitness information, set is_content=false with
     skip_reason="other_metadata".
  5. The question must match the requested STYLE exactly:
     - "formal": complete sentence, third person, technical vocabulary, textbook tone.
       Example: "What set and repetition ranges are generally recommended for hypertrophy
       training in advanced lifters?"
     - "informal": conversational, first/second person, casual gym-goer voice. Lowercase
       is fine. Example: "how many sets and reps should i do if i'm trying to bulk up?"

If you are given RETRY FEEDBACK from a previous rejected attempt, produce a DIFFERENT
question that addresses the rejection reason — do not just rephrase.

Return JSON only — no commentary.

────────────────────────────────────────
EXAMPLES

[Example 1: substantive chunk, formal style — chunk names the NSCA but the question must NOT]
CHUNK: "The NSCA recommends 3-6 sets of 6-12 repetitions at 67-85% of 1RM for hypertrophy
training in advanced lifters, with 30-90 seconds of rest between sets."
SUMMARY: Recommended hypertrophy prescription for advanced lifters.
STYLE: formal
OUTPUT: {"is_content": true, "skip_reason": "", "question": "What set, repetition, intensity, and rest-interval ranges are generally recommended for hypertrophy training in advanced lifters?"}

[Example 2: same chunk, informal style — also no institution name]
STYLE: informal
OUTPUT: {"is_content": true, "skip_reason": "", "question": "how many sets, reps, and how much rest should an advanced lifter do for muscle growth?"}

[Example 3: chunk that cites a specific researcher — ask about the concept, not the person]
CHUNK: "Schoenfeld (2010) demonstrated that performing squats below parallel produced
approximately 7% greater glute activation than parallel squats."
SUMMARY: Squat depth and glute activation.
STYLE: formal
OUTPUT: {"is_content": true, "skip_reason": "", "question": "Approximately how much greater is glute activation when squats are performed below parallel compared to parallel squats?"}

[Example 4: table-of-contents chunk]
CHUNK: "Chapter 1: Introduction ... 1\\nChapter 2: Anatomy ... 17\\nChapter 3: Programming ... 45"
SUMMARY: Table of contents listing.
STYLE: formal
OUTPUT: {"is_content": false, "skip_reason": "table_of_contents", "question": ""}

[Example 5: author credentials chunk]
CHUNK: "Dr. Jane Smith holds a PhD in Exercise Physiology from the University of X and has
served on the editorial board of the Journal of Strength and Conditioning Research."
SUMMARY: Author bio for Dr. Jane Smith.
STYLE: informal
OUTPUT: {"is_content": false, "skip_reason": "author_credentials", "question": ""}

[Example 6: book/edition meta-description chunk — pure description of the source itself]
CHUNK: "This third edition of Essentials of Strength Training and Conditioning has been
updated with new content on nutrition for athletes and performance-enhancing substances.
Chapters 9 and 10 have been expanded with revised programming guidelines."
SUMMARY: Description of new content added in the third edition.
STYLE: formal
OUTPUT: {"is_content": false, "skip_reason": "other_metadata", "question": ""}

[Example 7: chunk discusses real content but uses meta-framing — extract the concept]
CHUNK: "Chapter 9 explains that for hypertrophy, training to within 1-2 reps of failure
across most working sets produces greater muscle growth than stopping further from failure."
SUMMARY: Proximity-to-failure for hypertrophy training.
STYLE: informal
OUTPUT: {"is_content": true, "skip_reason": "", "question": "how close to failure should i be training my sets if i want to maximize muscle growth?"}
"""

GENERATOR_USER_TEMPLATE = """\
CHUNK:
\"\"\"
{content}
\"\"\"

SUMMARY: {summary}

STYLE: {style}
{retry_block}
Return JSON matching the schema."""

RETRY_BLOCK_TEMPLATE = """
RETRY FEEDBACK — your previous attempt was REJECTED.
Previous question: "{prior_question}"
Reason for rejection: {judge_reason}

Produce a DIFFERENT question that fixes this issue.
"""

JUDGE_SYSTEM_PROMPT = """\
You are a strict judge for a synthetic-question dataset used to fine-tune a retriever.

You will be given:
  - CHUNK: the source text
  - QUESTION: a candidate question
  - STYLE: the requested style ("formal" or "informal")

Decide if the question is VALID. A question is VALID only if ALL FIVE checks pass:

  1. ANSWERABILITY — The chunk fully and definitively answers the question. The reader
     should not need any outside knowledge, and the chunk must contain the WHOLE answer
     (not just part of it).
  2. SPECIFICITY — The question targets information unique to this chunk. Reject overly
     generic questions that thousands of other fitness chunks could also answer.
  3. NO ATTRIBUTION — The question must NOT name or reference any institution
     (e.g., NSCA, ACSM, NASM), person (researcher, author, coach, athlete), specific
     study/paper/journal, or specific book/resource. Phrases like "according to the NSCA",
     "Schoenfeld found that", "in this textbook", or "in a 2010 study" are all REJECTIONS.
     The question must ask about the underlying concept directly, as a real user would.
  4. USER PERSONA — The question must come from a real user (athlete, trainee, trainer)
     who has NO IDEA the answer comes from a particular book, chapter, or edition.
     Reject any question that betrays awareness of the source. Forbidden phrasings:
       - "this book", "this chapter", "this section", "this text", "this manual"
       - "the third edition", "the new edition", "the updated edition"
       - "chapter 9", "chapters 9 and 10", "the next chapter"
       - "what's new in...", "what's been updated", "what topics are covered in...",
         "what areas have been expanded", "what does the book/chapter discuss"
       - "the author(s)", "the editors"
     The question must ask about the underlying fitness concept directly.
  5. STYLE MATCH —
       formal:   complete sentence, third person, technical vocabulary, textbook tone
       informal: conversational, first/second person, casual gym-goer voice (lowercase ok)

If the question fails ANY check, set valid=false and write a SHORT, ACTIONABLE
failure_reason that names which check failed and why. The reason will be fed back to
the generator, so be specific.

If valid=true, set failure_reason to "".

Return JSON only.

────────────────────────────────────────
EXAMPLES

[Valid]
CHUNK: "The squat targets the quadriceps, glutes, and hamstrings."
QUESTION: "Which three muscle groups does the squat primarily target?"
STYLE: formal
OUTPUT: {"valid": true, "failure_reason": ""}

[Answerability fail — needs outside knowledge]
CHUNK: "The squat targets the quadriceps, glutes, and hamstrings."
QUESTION: "What is the optimal squat depth for glute activation?"
STYLE: formal
OUTPUT: {"valid": false, "failure_reason": "ANSWERABILITY: the chunk does not mention squat depth or glute activation percentages, so the question cannot be answered from the chunk alone."}

[Specificity fail — too generic]
CHUNK: "The squat targets the quadriceps, glutes, and hamstrings."
QUESTION: "What is a compound exercise?"
STYLE: formal
OUTPUT: {"valid": false, "failure_reason": "SPECIFICITY: the question is generic and could be answered by countless other chunks; it does not target information unique to this chunk."}

[Attribution fail — names an institution]
CHUNK: "The NSCA recommends 3-6 sets of 6-12 reps for hypertrophy in advanced lifters."
QUESTION: "According to the NSCA, what set and repetition ranges are recommended for hypertrophy in advanced lifters?"
STYLE: formal
OUTPUT: {"valid": false, "failure_reason": "NO_ATTRIBUTION: the question references the NSCA by name. Ask about the underlying concept directly, e.g. 'What set and repetition ranges are generally recommended for hypertrophy in advanced lifters?'"}

[Attribution fail — names a researcher]
CHUNK: "Schoenfeld (2010) showed that squats below parallel produced ~7% greater glute activation."
QUESTION: "What did Schoenfeld find about squat depth and glute activation?"
STYLE: informal
OUTPUT: {"valid": false, "failure_reason": "NO_ATTRIBUTION: the question names the researcher 'Schoenfeld'. Rephrase to ask about the underlying finding without attribution, e.g. 'how much more glute activation do you get from squatting below parallel?'"}

[User persona fail — references the book/edition]
CHUNK: "This third edition has been updated with new content on nutrition for athletes and on performance-enhancing substances."
QUESTION: "What new content on nutrition and performance enhancers has been added in the third edition?"
STYLE: formal
OUTPUT: {"valid": false, "failure_reason": "USER_PERSONA: the question references 'the third edition', which a real user wouldn't say. The chunk is itself meta-description of a book — there is no underlying fitness concept to ask about, so it should have been skipped at generation time."}

[User persona fail — references chapters by number]
CHUNK: "Chapters 9 and 10 cover periodization and program design for hypertrophy."
QUESTION: "What topics are covered in chapters 9 and 10?"
STYLE: informal
OUTPUT: {"valid": false, "failure_reason": "USER_PERSONA: the question asks about 'chapters 9 and 10', which betrays awareness of the source. A real user would just ask 'what does periodization and program design for hypertrophy involve?' — or this chunk should be skipped if it has no concrete content beyond a TOC entry."}

[Style fail — formal question when informal was requested]
CHUNK: "Hypertrophy is best stimulated with 6-12 reps per set in the 67-85% 1RM range."
QUESTION: "What repetition and intensity ranges are most effective for stimulating hypertrophy?"
STYLE: informal
OUTPUT: {"valid": false, "failure_reason": "STYLE: the question is in formal textbook tone (complete sentence, third person, technical vocabulary), but informal style was requested — should sound like a casual gym-goer using lowercase and first/second person."}
"""

JUDGE_USER_TEMPLATE = """\
CHUNK:
\"\"\"
{content}
\"\"\"

QUESTION: {question}
STYLE: {style}

Return JSON matching the schema."""


def generate_question(
    llm: LLMClient,
    chunk: dict,
    style: str,
    prior_question: str | None = None,
    judge_reason: str | None = None,
) -> dict:
    """Call generator. Returns dict with keys is_content, skip_reason, question."""
    retry_block = ""
    if prior_question is not None and judge_reason is not None:
        retry_block = RETRY_BLOCK_TEMPLATE.format(
            prior_question=prior_question, judge_reason=judge_reason
        )

    user_prompt = GENERATOR_USER_TEMPLATE.format(
        content=chunk["content"],
        summary=chunk["summary"],
        style=style,
        retry_block=retry_block,
    )

    return llm.chat_structured(GENERATOR_SYSTEM_PROMPT, user_prompt, GENERATOR_SCHEMA)


def judge_question(llm: LLMClient, chunk: dict, question: str, style: str) -> dict:
    """Call judge. Returns dict with keys valid, failure_reason."""
    user_prompt = JUDGE_USER_TEMPLATE.format(
        content=chunk["content"], question=question, style=style
    )
    return llm.chat_structured(JUDGE_SYSTEM_PROMPT, user_prompt, JUDGE_SCHEMA)


def process_style(
    llm: LLMClient,
    chunk: dict,
    style: str,
    max_retries: int,
) -> tuple[str, dict]:
    """Run generator → judge loop for one style.

    Returns ("ok", {question, attempts}) on success,
            ("skip", {skip_reason}) if generator marked the chunk non-content,
            ("fail", {attempts: [{question, judge_reason}, ...]}) if all retries failed.
    """
    attempts_log = []
    prior_question = None
    prior_reason = None

    for attempt in range(max_retries + 1):
        gen = generate_question(llm, chunk, style, prior_question, prior_reason)

        if not gen["is_content"]:
            reason = gen["skip_reason"] or "other_metadata"
            if reason not in VALID_SKIP_REASONS:
                reason = "other_metadata"
            return "skip", {"skip_reason": reason}

        question = gen["question"].strip()
        if not question:
            prior_question = ""
            prior_reason = "Generator returned an empty question. Produce a non-empty question."
            attempts_log.append({"question": "", "judge_reason": prior_reason})
            continue

        verdict = judge_question(llm, chunk, question, style)

        if verdict["valid"]:
            return "ok", {"question": question, "attempts": attempt + 1}

        prior_question = question
        prior_reason = verdict["failure_reason"] or "unspecified"
        attempts_log.append({"question": question, "judge_reason": prior_reason})

    return "fail", {"attempts": attempts_log}


def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def append_jsonl(path: Path, record: dict):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def process_chunks_file(
    chunks_path: Path,
    config: Config,
    llm: LLMClient,
    styles: list[str],
    resume: bool,
    limit: int | None,
):
    stem = chunks_path.stem
    chunks = load_chunks(chunks_path)
    logger.info(f"'{chunks_path.name}': loaded {len(chunks)} chunks")

    pairs_path = config.pairs_dir / chunks_path.name
    skipped_path = config.pairs_dir / "_skipped.jsonl"
    failed_path = config.pairs_dir / "_failed.jsonl"
    ckpt_path = get_checkpoint_path(config, chunks_path.name, "pairs")

    state = load_checkpoint(ckpt_path) if resume else None
    processed_ids = set(state.get("processed_chunk_ids", [])) if state else set()
    if processed_ids:
        logger.info(f"'{chunks_path.name}': resuming, {len(processed_ids)} chunks already processed")

    processed_count = 0
    for chunk in chunks:
        if limit is not None and processed_count >= limit:
            break

        chunk_id = chunk["id"]
        if chunk_id in processed_ids:
            continue

        logger.info(f"  chunk {chunk_id}")

        chunk_skipped = False
        for style in styles:
            verdict, payload = process_style(llm, chunk, style, config.max_pair_retries)

            if verdict == "skip":
                append_jsonl(
                    skipped_path,
                    {
                        "chunk_id": chunk_id,
                        "skip_reason": payload["skip_reason"],
                        "summary": chunk.get("summary", ""),
                        "source_file": chunks_path.name,
                    },
                )
                logger.info(f"    [skip] {payload['skip_reason']}")
                chunk_skipped = True
                break  # skip is chunk-level — don't try the other style

            if verdict == "ok":
                append_jsonl(
                    pairs_path,
                    {
                        "chunk_id": chunk_id,
                        "question": payload["question"],
                        "style": style,
                        "attempts": payload["attempts"],
                    },
                )
                logger.info(f"    [ok:{style}] attempts={payload['attempts']}")
            else:  # "fail"
                append_jsonl(
                    failed_path,
                    {
                        "chunk_id": chunk_id,
                        "style": style,
                        "summary": chunk.get("summary", ""),
                        "source_file": chunks_path.name,
                        "attempts": payload["attempts"],
                    },
                )
                logger.warning(f"    [fail:{style}] dropped after {len(payload['attempts'])} attempts")

        processed_ids.add(chunk_id)
        save_checkpoint(ckpt_path, {"processed_chunk_ids": sorted(processed_ids)})
        processed_count += 1

    logger.info(
        f"'{chunks_path.name}': finished. processed_this_run={processed_count}, "
        f"total_processed={len(processed_ids)}/{len(chunks)}"
    )


def main():
    parser = argparse.ArgumentParser(description="Positive pair generation (Step 2)")
    parser.add_argument(
        "--chunks-file",
        type=str,
        help="Single chunks JSONL filename in ./chunks/ (default: process all)",
    )
    parser.add_argument("--resume", action="store_true", help="Skip chunks already in checkpoint")
    parser.add_argument(
        "--style",
        choices=["both", "formal", "informal"],
        default="both",
        help="Which style(s) to generate (default: both)",
    )
    parser.add_argument("--limit", type=int, help="Cap chunks processed per file (smoke testing)")
    args = parser.parse_args()

    config = Config()
    llm = LLMClient(config)

    styles = ["formal", "informal"] if args.style == "both" else [args.style]

    if args.chunks_file:
        chunks_path = config.output_dir / args.chunks_file
        if not chunks_path.exists():
            logger.error(f"Chunks file not found: {chunks_path}")
            sys.exit(1)
        process_chunks_file(chunks_path, config, llm, styles, args.resume, args.limit)
        return

    chunks_files = sorted(config.output_dir.glob("*.jsonl"))
    if not chunks_files:
        logger.error(f"No chunks JSONL files in {config.output_dir}")
        sys.exit(1)

    logger.info(f"Found {len(chunks_files)} chunks files in {config.output_dir}")
    for i, chunks_path in enumerate(chunks_files):
        logger.info(f"\n--- File {i + 1}/{len(chunks_files)}: {chunks_path.name} ---")
        process_chunks_file(chunks_path, config, llm, styles, args.resume, args.limit)


if __name__ == "__main__":
    print("Starting pair generator...")
    main()
