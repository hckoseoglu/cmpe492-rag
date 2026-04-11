import logging

from llm_client import LLMClient

logger = logging.getLogger(__name__)

GROUPER_SCHEMA = {
    "type": "object",
    "properties": {
        "chunks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chunk_content": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": ["chunk_content", "summary"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["chunks"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
You are a text organization engine. Given a list of atomic propositions about fitness \
and exercise science, group them into thematic chunks.

Rules:
1. Each chunk should contain 3-15 closely related propositions.
2. For each chunk, synthesize the propositions into a coherent paragraph (chunk_content) \
that reads naturally and preserves all factual details.
3. For each chunk, write a 1-2 sentence summary that captures the main topic.
4. Every proposition must appear in exactly one chunk — do not drop or duplicate any.
5. Group by semantic similarity: same exercise, same muscle group, same training principle, etc.

Return a JSON array of objects, each with "chunk_content" (string) and "summary" (string). \
No commentary — only the JSON array.

Example input:
1. The squat is a compound exercise that primarily targets the quadriceps, glutes, and hamstrings.
2. A typical squat prescription for strength development is 3-5 sets of 5-8 reps at 70-85% of 1RM.
3. The deadlift is a posterior chain exercise targeting the erector spinae, glutes, and hamstrings.
4. Conventional deadlifts place greater demand on the erector spinae compared to sumo deadlifts.
5. Progressive overload is the gradual increase of stress placed on the body during training.

Example output:
[
  {
    "chunk_content": "The squat is a compound exercise that primarily targets the quadriceps, \
glutes, and hamstrings. A typical squat prescription for strength development is 3-5 sets of \
5-8 reps at 70-85% of 1RM.",
    "summary": "The squat targets the quadriceps, glutes, and hamstrings, typically prescribed \
at 3-5 sets of 5-8 reps for strength."
  },
  {
    "chunk_content": "The deadlift is a posterior chain exercise targeting the erector spinae, \
glutes, and hamstrings. Conventional deadlifts place greater demand on the erector spinae \
compared to sumo deadlifts.",
    "summary": "The deadlift targets the posterior chain, with conventional and sumo variations \
differing in erector spinae demand."
  },
  {
    "chunk_content": "Progressive overload is the gradual increase of stress placed on the body \
during training.",
    "summary": "Progressive overload involves gradually increasing training stress over time."
  }
]"""

USER_PROMPT_TEMPLATE = """\
Group the following {count} propositions into thematic chunks:

{propositions}

Return a JSON array of objects with "chunk_content" and "summary" fields."""


def group_propositions(
    llm: LLMClient,
    propositions: list[str],
    max_per_call: int = 40,
) -> list[dict]:
    """Group propositions into thematic chunks with summaries.

    For large sets, uses a sliding window with overlap and deduplication.
    """
    if len(propositions) <= max_per_call:
        return _group_single_call(llm, propositions)

    # Sliding window for large proposition sets
    overlap = 5
    all_chunks = []
    start = 0
    total_windows = (len(propositions) - overlap) // (max_per_call - overlap) + 1

    window_num = 0
    while start < len(propositions):
        end = min(start + max_per_call, len(propositions))
        window = propositions[start:end]
        window_num += 1

        chunks = _group_single_call(llm, window)

        for chunk in chunks:
            all_chunks.append(chunk)

        logger.info(
            f"window {window_num}/{total_windows}: "
            f"props {start+1}-{end}, {len(chunks)} chunks produced, "
            f"{len(all_chunks)} total chunks so far"
        )

        start += max_per_call - overlap

    return all_chunks


def _group_single_call(llm: LLMClient, propositions: list[str]) -> list[dict]:
    """Group a single batch of propositions via one LLM call."""
    numbered = "\n".join(f"{i+1}. {p}" for i, p in enumerate(propositions))
    user_prompt = USER_PROMPT_TEMPLATE.format(
        count=len(propositions),
        propositions=numbered,
    )

    try:
        result = llm.chat_structured(SYSTEM_PROMPT, user_prompt, GROUPER_SCHEMA)
    except Exception:
        logger.error(f"Failed to group {len(propositions)} propositions")
        return []

    chunks = result.get("chunks", [])

    valid_chunks = []
    for item in chunks:
        if isinstance(item, dict) and "chunk_content" in item and "summary" in item:
            valid_chunks.append({
                "chunk_content": item["chunk_content"],
                "summary": item["summary"],
            })

    return valid_chunks
