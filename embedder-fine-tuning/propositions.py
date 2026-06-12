import logging

from llm_client import LLMClient

logger = logging.getLogger(__name__)

PROPOSITIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "propositions": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["propositions"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
You are a precise text decomposition engine. Your task is to break down the given \
text into atomic, self-contained factual propositions.

Rules:
1. Each proposition must be a single, complete factual statement.
2. Each proposition must be understandable WITHOUT any surrounding context. \
Use full names and references (e.g., "The barbell bench press targets the pectoralis major" \
NOT "This exercise targets it").
3. Preserve specific numbers, percentages, ranges, and citations.
4. Do NOT include: table of contents entries, page headers/footers, copyright notices, \
figure/table captions that lack substantive content, or bibliographic references.
5. If the text contains no extractable factual content, return an empty array.

Return a JSON array of strings. No commentary, no explanations — only the JSON array.

Example input:
"The squat is a compound exercise. It primarily targets the quadriceps, glutes, \
and hamstrings. Research by Schoenfeld (2010) showed that deep squats (below parallel) \
produced 7% greater glute activation than parallel squats. A typical prescription is \
3-5 sets of 5-8 reps at 70-85% of 1RM for strength development."

Example output:
[
  "The squat is a compound exercise that primarily targets the quadriceps, glutes, and hamstrings.",
  "Research by Schoenfeld (2010) showed that deep squats performed below parallel produced 7% greater glute activation than parallel squats.",
  "A typical squat prescription for strength development is 3-5 sets of 5-8 reps at 70-85% of 1RM."
]"""

USER_PROMPT_TEMPLATE = """\
Decompose the following text into atomic, self-contained factual propositions:

---
{text}
---

Return a JSON array of proposition strings."""


def extract_propositions(llm: LLMClient, batch_text: str) -> list[str]:
    """Extract atomic propositions from a batch of page text."""
    user_prompt = USER_PROMPT_TEMPLATE.format(text=batch_text)

    try:
        result = llm.chat_structured(SYSTEM_PROMPT, user_prompt, PROPOSITIONS_SCHEMA)
    except Exception:
        logger.error("Failed to extract propositions from batch")
        return []

    raw = result.get("propositions", [])
    propositions = [p for p in raw if isinstance(p, str) and len(p.strip()) > 10]
    return propositions
