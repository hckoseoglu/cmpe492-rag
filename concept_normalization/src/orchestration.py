"""
Concept Normalization Module — Orchestration Layer

Loads vocabulary.json, calls each submodule in order, passes outputs
between them, and produces the final normalized queries + transformation log.

Usage:
    from src.orchestration import ConceptNormalizationModule
    module = ConceptNormalizationModule("vocabulary.json", llm_client)
    result = module.normalize("What are the benefits of RT and HIIT for VO2max?")
"""

import json
import time
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field, asdict


@dataclass
class TransformationLog:
    original_query: str = ""
    abbreviation_step: dict = field(default_factory=dict)
    coordination_step: dict = field(default_factory=dict)
    term_variation_step: list = field(default_factory=list)
    final_normalized_queries: list = field(default_factory=list)
    total_latency_ms: float = 0.0


class ConceptNormalizationModule:
    """
    Orchestrates the three normalization submodules:
      1. Abbreviation Expansion
      2. Coordination Resolution
      3. Term Variation Handling

    Each submodule is an LLM call with a system prompt that has
    the vocabulary JSON injected at the marked placeholders.
    """

    def __init__(
        self,
        vocab_path: str,
        llm_client: Any,
        model: str = "claude-sonnet-4-20250514",
        prompts_dir: str = "prompts",
    ):
        self.llm = llm_client
        self.model = model

        # Load vocabulary
        with open(vocab_path, "r") as f:
            self.vocabulary = json.load(f)

        # Build system prompts with vocabulary injected
        prompts = Path(prompts_dir)
        self.abbrev_prompt = self._inject_vocab(
            (prompts / "abbreviation_expansion.txt").read_text(),
            abbreviations_only=True,
        )
        self.coord_prompt = self._inject_vocab(
            (prompts / "coordination_resolution.txt").read_text(),
        )
        self.variation_prompt = self._inject_vocab(
            (prompts / "term_variation.txt").read_text(),
        )

    def _inject_vocab(self, template: str, abbreviations_only: bool = False) -> str:
        """Replace vocabulary placeholders in a prompt template."""
        if abbreviations_only:
            return template.replace(
                '{vocabulary_json["abbreviations"]}',
                json.dumps(self.vocabulary["abbreviations"], indent=2),
            )
        return template.replace(
            "{full_vocabulary_json}",
            json.dumps(self.vocabulary, indent=2),
        )

    def _call_llm(self, system_prompt: str, user_msg: str) -> dict:
        """Call LLM and parse JSON response. Retries once on parse failure."""
        for attempt in range(2):
            response = self.llm.messages.create(
                model=self.model,
                max_tokens=2048,
                system=system_prompt,
                messages=[{"role": "user", "content": user_msg}],
                temperature=0.0,
            )
            text = response.content[0].text.strip()

            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                text = text.rsplit("```", 1)[0]

            try:
                return json.loads(text)
            except json.JSONDecodeError:
                if attempt == 0:
                    # Retry with explicit JSON reminder
                    user_msg = (
                        user_msg
                        + "\n\nIMPORTANT: Respond ONLY with valid JSON. "
                        "No markdown, no explanation."
                    )
                else:
                    raise

    def normalize(self, query: str) -> dict:
        """
        Run the full normalization pipeline.

        Returns:
            {
                "normalized_queries": ["..."],
                "transformation_log": { ... }
            }
        """
        log = TransformationLog(original_query=query)
        start = time.time()

        # --- Step 1: Abbreviation Expansion ---
        try:
            abbrev_result = self._call_llm(self.abbrev_prompt, query)
            expanded_query = abbrev_result.get("expanded_query", query)
            log.abbreviation_step = abbrev_result
        except Exception as e:
            expanded_query = query
            log.abbreviation_step = {"error": str(e), "fallback": "original preserved"}

        # --- Step 2: Coordination Resolution ---
        try:
            coord_result = self._call_llm(self.coord_prompt, expanded_query)
            sub_queries = coord_result.get("sub_queries", [expanded_query])
            log.coordination_step = coord_result
        except Exception as e:
            sub_queries = [expanded_query]
            log.coordination_step = {"error": str(e), "fallback": "original preserved"}

        # --- Step 3: Term Variation Handling (per sub-query) ---
        normalized_queries = []
        for sq in sub_queries:
            try:
                var_result = self._call_llm(self.variation_prompt, sq)
                normalized_queries.append(var_result.get("normalized_query", sq))
                log.term_variation_step.append(var_result)
            except Exception as e:
                normalized_queries.append(sq)
                log.term_variation_step.append(
                    {"error": str(e), "fallback": "original preserved", "input": sq}
                )

        log.final_normalized_queries = normalized_queries
        log.total_latency_ms = (time.time() - start) * 1000

        return {
            "normalized_queries": normalized_queries,
            "transformation_log": asdict(log),
        }


# --- CLI entry point for testing ---
if __name__ == "__main__":
    import sys

    # Requires: pip install anthropic
    from anthropic import Anthropic

    vocab_path = sys.argv[1] if len(sys.argv) > 1 else "vocabulary.json"
    query = sys.argv[2] if len(sys.argv) > 2 else "What are the benefits of RT and HIIT for VO2max?"

    client = Anthropic()
    module = ConceptNormalizationModule(vocab_path=vocab_path, llm_client=client)
    result = module.normalize(query)
    print(json.dumps(result, indent=2))
