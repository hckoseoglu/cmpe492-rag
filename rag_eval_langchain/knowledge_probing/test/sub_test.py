import pytest
from langchain_core.documents import Document

import sys
import os


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from substitution import (
    _replace_all,
    apply_substitutions,
    reverse_substitutions,
    SUBSTITUTION_MAP,
    REVERSE_MAP,
)



class TestSubstitutionLogic:
    def test_replace_all_basic(self):
        """Test simple substitution."""
        text = "I tore my hamstring playing soccer."
        expected = "I tore my veldrak playing soccer."
        assert _replace_all(text, SUBSTITUTION_MAP) == expected

    def test_replace_all_case_insensitive(self):
        """Test that capitalization does not break the matching."""
        text = "BICEPS, Triceps, and Quads"
        expected = "kaskush, blorbex, and trimfaz"
        assert _replace_all(text, SUBSTITUTION_MAP) == expected

    def test_replace_all_longest_match_first(self):
        """
        Test that longer phrases are substituted before shorter ones.
        'biceps brachii' should become 'kaskush', rather than 'kaskush brachii'.
        """
        text = "The biceps brachii and latissimus dorsi."
        expected = "The kaskush and fomblur."
        assert _replace_all(text, SUBSTITUTION_MAP) == expected

    def test_apply_substitutions(self):
        """Test the public API for questions and documents."""
        question = "What is the best exercise for the gluteus?"

        docs = [
            Document(
                page_content="Squats target the glutes and quadriceps.",
                metadata={"source": "book"},
            ),
            Document(
                page_content="Don't forget your CALVES.", metadata={"source": "article"}
            ),
        ]

        sub_q, sub_docs = apply_substitutions(question, docs)

        # Check question
        assert sub_q == "What is the best exercise for the snorvik?"

        # Check documents
        assert len(sub_docs) == 2
        assert sub_docs[0].page_content == "Squats target the snorvik and trimfaz."
        assert sub_docs[0].metadata == {"source": "book"}
        assert sub_docs[1].page_content == "Don't forget your grelvak."

        # Ensure deepcopy worked (originals are untouched)
        assert docs[0].page_content == "Squats target the glutes and quadriceps."

    def test_reverse_substitutions(self):
        """
        Test that nonsense words are reverted.
        Note: Because SUBSTITUTION_MAP has a many-to-one relationship,
        REVERSE_MAP relies on Python dictionary insertion order (keeping the last key).
        """
        text = "Train your snorvik and kaskush."

        # We fetch the expected terms directly from the map to prevent brittle tests,
        # since "kaskush" maps to both "biceps" and "biceps brachii" in the original map.
        expected_snorvik = REVERSE_MAP["snorvik"]
        expected_kaskush = REVERSE_MAP["kaskush"]

        expected_text = f"Train your {expected_snorvik} and {expected_kaskush}."
        assert reverse_substitutions(text) == expected_text

    def test_no_matches_found(self):
        """Test text that contains no substitution keywords."""
        text = "This text has nothing to do with anatomy."
        assert _replace_all(text, SUBSTITUTION_MAP) == text


if __name__ == "__main__":
    import pytest

    # This tells pytest to run this exact file with verbose (-v) output
    pytest.main(["-v", __file__])
