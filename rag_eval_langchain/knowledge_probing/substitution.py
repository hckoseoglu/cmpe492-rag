"""
substitution.py

Implements the LLM-as-Fool substitution logic:
- Replaces domain keywords in the query and retrieved documents with random
  nonsense words so the LLM cannot rely on parametric (base) knowledge.
- Provides a reverse function to restore original terms before evaluation.

Substitution maps are defined in sub_map.py at three intensity levels:
    SOFT_MAP   – muscles + critical query terms
    MEDIUM_MAP – adds exercises, equipment, movement vocabulary
    HARD_MAP   – maximal coverage including anatomy, directions, jargon

Change the active map by setting SUB_LEVEL below.
"""

from copy import deepcopy

from langchain_core.documents import Document

from knowledge_probing.sub_map import SOFT_MAP, MEDIUM_MAP, HARD_MAP

# ── Select active substitution level ─────────────────────────────────────────
# Change this to MEDIUM_MAP or HARD_MAP to increase obfuscation intensity.
SUB_LEVEL = "hard"

_LEVEL_MAPS = {
    "soft": SOFT_MAP,
    "medium": MEDIUM_MAP,
    "hard": HARD_MAP,
}

SUBSTITUTION_MAP: dict[str, str] = _LEVEL_MAPS[SUB_LEVEL]

# Reverse map is derived automatically — no need to maintain separately
REVERSE_MAP: dict[str, str] = {v: k for k, v in SUBSTITUTION_MAP.items()}


# ── Core Replacement Logic ───────────────────────────────────────────────────


def _replace_all(text: str, substitution_map: dict[str, str]) -> str:
    """
    Replace all occurrences of keys in substitution_map within text.
    Matches are case-insensitive; replacement preserves the nonsense word as-is.
    Longer keys are matched first to avoid partial replacements
    (e.g. 'biceps brachii' before 'biceps').
    """
    # Sort by length descending so multi-word phrases are matched first
    sorted_keys = sorted(substitution_map.keys(), key=len, reverse=True)

    for key in sorted_keys:
        # Simple case-insensitive replacement
        import re

        pattern = re.compile(re.escape(key), re.IGNORECASE)
        text = pattern.sub(substitution_map[key], text)

    return text


def apply_substitutions(
    question: str,
    docs: list[Document],
    sub_map: dict[str, str] | None = None,
) -> tuple[str, list[Document]]:
    """
    Apply keyword substitutions to the query and retrieved documents.

    Call this AFTER retrieval and BEFORE prompt construction so that:
      - The LLM cannot rely on parametric knowledge for substituted terms.
      - Retrieved context is the only possible source of correct answers.

    Args:
        question: The original user query.
        docs:     Retrieved LangChain Document objects.
        sub_map:  Optional custom substitution map.  Falls back to
                  SUBSTITUTION_MAP (set by SUB_LEVEL) when None.

    Returns:
        A tuple of (substituted_question, substituted_docs).
        Original documents are not mutated.
    """
    active = sub_map if sub_map is not None else SUBSTITUTION_MAP
    substituted_question = _replace_all(question, active)

    substituted_docs = []
    for doc in docs:
        new_doc = deepcopy(doc)
        new_doc.page_content = _replace_all(doc.page_content, active)
        substituted_docs.append(new_doc)

    return substituted_question, substituted_docs


def reverse_substitutions(text: str) -> str:
    """
    Restore original domain terms in the LLM's answer before evaluation.

    Call this AFTER the LLM generates its answer and BEFORE passing
    the answer to evaluators that compare against gold labels.

    Args:
        text: The LLM-generated answer containing substituted terms.

    Returns:
        The answer with nonsense words replaced by original domain terms.
    """
    return _replace_all(text, REVERSE_MAP)
