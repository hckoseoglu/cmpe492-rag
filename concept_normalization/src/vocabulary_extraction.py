"""
Vocabulary Extraction Module — One-Time Preprocessing

Extracts text from the 9 knowledge-base PDFs, sends each through an LLM
to extract structured domain vocabulary, merges and deduplicates across
all PDFs, and saves the result as vocabulary.json.

Usage:
    python src/vocabulary_extraction.py /path/to/pdf/dir vocabulary.json

Requirements:
    pip install pdfplumber tiktoken langchain-openai
"""

import json
import sys
from pathlib import Path
from typing import Any

import pdfplumber
import tiktoken
from langchain.chat_models import init_chat_model
from typing_extensions import Annotated, TypedDict


MAX_TOKENS_PER_CHUNK = 12000  # Leave room for prompt + output


VOCAB_EXTRACTION_PROMPT = """
You are a domain vocabulary extractor for exercise science.
Given the following text from an exercise science textbook or research paper,
extract ALL domain-specific terminology and organize it into the categories below.

## CATEGORIES

1. exercise_types: Named exercise modalities, training methods, and movement patterns
   (e.g., resistance training, plyometrics, isometric exercise)

2. physiological_concepts: Physiological processes, adaptations, biomarkers, and outcomes
   (e.g., muscle hypertrophy, VO2max, lactate threshold, myofibrillar protein synthesis)

3. training_variables: Prescriptive parameters and programming concepts
   (e.g., repetition maximum, training volume, rest interval, progressive overload, periodization)

4. anatomy: Muscles, joints, body regions, and anatomical structures
   (e.g., quadriceps, glenohumeral joint, lumbar spine, rotator cuff)

5. conditions_populations: Health conditions, injuries, special populations
   (e.g., type 2 diabetes, sarcopenia, postmenopausal women, anterior cruciate ligament tear)

6. equipment: Training equipment and modalities
   (e.g., barbell, resistance band, isokinetic dynamometer)

7. abbreviations: Map abbreviation to full form
   (e.g., RT -> resistance training, HIIT -> high-intensity interval training)

8. lay_to_technical: Map common lay terms to their technical equivalents
   (e.g., toning -> muscular endurance training, bulking -> hypertrophy-focused training)

## RULES
- Include BOTH singular and plural forms where applicable
- Include common spelling variations
- For each term, use its most canonical form as it appears in the text
- Only extract terms that actually appear in the text
- For abbreviations, only include those explicitly defined or used in the text

## TEXT TO ANALYZE
<<<TEXT_START>>>
{text_chunk}
<<<TEXT_END>>>
""".strip()


# ── Structured Output Schemas ─────────────────────────────────────────


class AbbreviationEntry(TypedDict):
    abbreviation: Annotated[str, ..., "The abbreviation as it appears in the text (e.g. RT, HIIT)"]
    expansion: Annotated[str, ..., "The full form of the abbreviation (e.g. resistance training)"]


class LayTermEntry(TypedDict):
    lay_term: Annotated[str, ..., "The common lay term (e.g. toning, bulking)"]
    technical_term: Annotated[str, ..., "The technical equivalent (e.g. muscular endurance training)"]


class VocabularySchema(TypedDict):
    exercise_types: Annotated[list[str], ..., "Named exercise modalities, training methods, and movement patterns"]
    physiological_concepts: Annotated[list[str], ..., "Physiological processes, adaptations, biomarkers, and outcomes"]
    training_variables: Annotated[list[str], ..., "Prescriptive parameters and programming concepts"]
    anatomy: Annotated[list[str], ..., "Muscles, joints, body regions, and anatomical structures"]
    conditions_populations: Annotated[list[str], ..., "Health conditions, injuries, special populations"]
    equipment: Annotated[list[str], ..., "Training equipment and modalities"]
    abbreviations: Annotated[list[AbbreviationEntry], ..., "Abbreviations mapped to their full forms"]
    lay_to_technical: Annotated[list[LayTermEntry], ..., "Common lay terms mapped to technical equivalents"]


# ── PDF Text Extraction ──────────────────────────────────────────────


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a single PDF using pdfplumber."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages)


def extract_all_pdfs(pdf_dir: str) -> dict[str, str]:
    """Extract text from all PDFs in a directory."""
    results = {}
    pdf_files = sorted(Path(pdf_dir).glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {pdf_dir}")

    for pdf_path in pdf_files:
        print(f"Extracting: {pdf_path.name}")
        try:
            text = extract_text_from_pdf(str(pdf_path))
            results[pdf_path.name] = text
            print(f"  -> {len(text):,} characters")
        except Exception as e:
            print(f"  -> ERROR: {e}")
            results[pdf_path.name] = ""
    return results


# ── Text Chunking ────────────────────────────────────────────────────


def chunk_text(text: str, max_tokens: int = MAX_TOKENS_PER_CHUNK) -> list[str]:
    """Split text into token-limited chunks at paragraph boundaries.

    Single paragraphs exceeding max_tokens are hard-split at sentence
    boundaries to avoid exceeding the API limit.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = len(enc.encode(para))

        # Oversized single paragraph: split at sentences
        if para_tokens > max_tokens:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk.clear()
                current_tokens = 0
            sentences = para.replace(". ", ".\n").split("\n")
            sub_chunk: list[str] = []
            sub_tokens = 0
            for sent in sentences:
                sent_tokens = len(enc.encode(sent))
                if sub_tokens + sent_tokens > max_tokens and sub_chunk:
                    chunks.append(" ".join(sub_chunk))
                    sub_chunk = []
                    sub_tokens = 0
                sub_chunk.append(sent)
                sub_tokens += sent_tokens
            if sub_chunk:
                chunks.append(" ".join(sub_chunk))
            continue

        if current_tokens + para_tokens > max_tokens and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_tokens = 0
        current_chunk.append(para)
        current_tokens += para_tokens

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    return chunks


# ── LLM Vocabulary Extraction ───────────────────────────────────────


def extract_vocab_from_text(text: str, model: str) -> dict:
    """Extract vocabulary from a single PDF's text, chunking if needed."""
    llm = init_chat_model(model).with_structured_output(
        VocabularySchema, method="json_schema", strict=True
    )
    chunks = chunk_text(text)
    all_vocabs = []

    for i, chunk in enumerate(chunks):
        print(f"  Processing chunk {i + 1}/{len(chunks)}")
        chunk_prompt = VOCAB_EXTRACTION_PROMPT.replace("{text_chunk}", chunk)

        response = llm.invoke([{"role": "user", "content": chunk_prompt}])

        # Convert list-of-entry format back to dict for downstream merging
        vocab = dict(response)
        vocab["abbreviations"] = {
            entry["abbreviation"]: entry["expansion"]
            for entry in response.get("abbreviations", [])
        }
        vocab["lay_to_technical"] = {
            entry["lay_term"]: entry["technical_term"]
            for entry in response.get("lay_to_technical", [])
        }
        all_vocabs.append(vocab)

    if not all_vocabs:
        return _empty_vocab()

    return _merge_vocab_list(all_vocabs)


# ── Merging & Deduplication ──────────────────────────────────────────

LIST_CATEGORIES = [
    "exercise_types",
    "physiological_concepts",
    "training_variables",
    "anatomy",
    "conditions_populations",
    "equipment",
]


def _empty_vocab() -> dict:
    result = {cat: [] for cat in LIST_CATEGORIES}
    result["abbreviations"] = {}
    result["lay_to_technical"] = {}
    return result


def _merge_vocab_list(vocab_list: list[dict]) -> dict:
    """Merge multiple vocabulary dicts (from chunks of one PDF or across PDFs)."""
    merged_sets = {cat: set() for cat in LIST_CATEGORIES}
    merged_abbrevs: dict[str, str] = {}
    merged_lay: dict[str, str] = {}

    for vocab in vocab_list:
        for cat in LIST_CATEGORIES:
            merged_sets[cat].update(vocab.get(cat, []))
        merged_abbrevs.update(vocab.get("abbreviations", {}))
        # Prefer longer technical mapping (same conflict-resolution as abbreviations)
        for lay_term, technical in vocab.get("lay_to_technical", {}).items():
            if lay_term not in merged_lay or len(technical) > len(merged_lay[lay_term]):
                merged_lay[lay_term] = technical

    result = {cat: sorted(merged_sets[cat]) for cat in LIST_CATEGORIES}
    result["abbreviations"] = merged_abbrevs
    result["lay_to_technical"] = merged_lay
    return result


def merge_across_pdfs(per_pdf_vocabs: dict[str, dict]) -> dict:
    """Merge vocabulary from all PDFs with cross-PDF deduplication."""
    merged = _merge_vocab_list(list(per_pdf_vocabs.values()))

    # Case-insensitive dedup for list categories
    for cat in LIST_CATEGORIES:
        seen: dict[str, str] = {}
        deduped = []
        for term in merged[cat]:
            lower = term.lower().strip()
            if lower not in seen:
                seen[lower] = term
                deduped.append(term)
        merged[cat] = sorted(deduped, key=lambda x: x.lower())

    # Normalize abbreviation keys to uppercase, prefer longer expansions
    abbrevs = merged["abbreviations"]
    normalized: dict[str, str] = {}
    for abbr, expansion in abbrevs.items():
        upper = abbr.upper()
        if upper in normalized:
            if len(expansion) > len(normalized[upper]):
                normalized[upper] = expansion
        else:
            normalized[upper] = expansion
    merged["abbreviations"] = dict(sorted(normalized.items()))

    # Normalize lay_to_technical keys to lowercase, sort for stable output
    lay = merged["lay_to_technical"]
    merged["lay_to_technical"] = dict(sorted(
        {k.lower().strip(): v for k, v in lay.items()}.items()
    ))

    return merged


# ── Full Pipeline ────────────────────────────────────────────────────


def run_full_extraction(
    pdf_dir: str,
    output_path: str,
    model: str = "gpt-5-mini",
) -> dict:
    """
    Full extraction pipeline: PDFs -> vocabulary.json

    Args:
        pdf_dir: Directory containing the 9 knowledge-base PDFs
        output_path: Where to save the merged vocabulary JSON
        model: Model to use for extraction
    """
    # Step 1: Extract text from all PDFs
    print("=" * 60)
    print("STEP 1: Extracting text from PDFs")
    print("=" * 60)
    pdf_texts = extract_all_pdfs(pdf_dir)

    # Step 2: Extract vocabulary from each PDF via LLM
    print("\n" + "=" * 60)
    print("STEP 2: Extracting vocabulary via LLM")
    print("=" * 60)
    per_pdf_vocabs = {}
    for pdf_name, text in pdf_texts.items():
        if not text:
            print(f"Skipping {pdf_name} (no text extracted)")
            continue
        print(f"\nProcessing: {pdf_name}")
        vocab = extract_vocab_from_text(text, model)
        per_pdf_vocabs[pdf_name] = vocab

    # Step 3: Merge and deduplicate across all PDFs
    print("\n" + "=" * 60)
    print("STEP 3: Merging and deduplicating")
    print("=" * 60)
    final_vocab = merge_across_pdfs(per_pdf_vocabs)

    # Step 4: Save
    with open(output_path, "w") as f:
        json.dump(final_vocab, f, indent=2, ensure_ascii=False)

    print(f"\nSaved vocabulary to {output_path}")
    for cat in LIST_CATEGORIES:
        print(f"  {cat}: {len(final_vocab[cat])} terms")
    print(f"  abbreviations: {len(final_vocab['abbreviations'])} entries")
    print(f"  lay_to_technical: {len(final_vocab['lay_to_technical'])} entries")

    return final_vocab


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: python src/vocabulary_extraction.py [pdf_dir] [output_json] [model]")
        print("Defaults: pdf_dir=../resources, output_json=vocabulary.json, model=gpt-5-mini")
        sys.exit(0)

    pdf_dir = sys.argv[1] if len(sys.argv) > 1 else "../resources"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "vocabulary.json"
    model = sys.argv[3] if len(sys.argv) > 3 else "gpt-5-mini"

    run_full_extraction(pdf_dir, output_path, model)
