"""
substitution_recorder.py

Records substituted (fooled) queries and documents to numbered files.
Call save_substitution() from your rag_bot after apply_substitutions().

Output structure:
    substitutions/
        sub1.txt
        sub2.txt
        ...
"""

import os

from langchain_core.documents import Document

OUTPUT_DIR = "substitutions"


def save_substitution(
    original_question: str,
    fooled_question: str,
    original_docs: list[Document],
    fooled_docs: list[Document],
) -> None:
    """
    Save original and substituted query/docs to the next numbered file.

    Args:
        original_question: The original user query.
        fooled_question:   The substituted query string.
        original_docs:     List of original retrieved LangChain Document objects.
        fooled_docs:       List of substituted LangChain Document objects.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Determine next ordinal number
    existing = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("sub") and f.endswith(".txt")]
    next_index = len(existing) + 1

    lines = []
    lines.append(f"ORIGINAL QUERY:\n{original_question}")
    lines.append(f"\nSUBSTITUTED QUERY:\n{fooled_question}")

    for i, (original_doc, fooled_doc) in enumerate(zip(original_docs, fooled_docs), start=1):
        lines.append(f"\n{'─' * 60}")
        lines.append(f"DOCUMENT {i} — ORIGINAL:")
        lines.append(original_doc.page_content)
        lines.append(f"\nDOCUMENT {i} — SUBSTITUTED:")
        lines.append(fooled_doc.page_content)

    output_path = os.path.join(OUTPUT_DIR, f"sub{next_index}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✓ Saved {output_path}")