import logging
from pathlib import Path

from pypdf import PdfReader

from config import Config

logger = logging.getLogger(__name__)


def load_pdf(path: Path, config: Config) -> list[dict]:
    """Extract text from a PDF, returning a list of page dicts.

    Skips pages with fewer than min_page_chars characters.
    Returns empty list (with warning) if total text is below min_pdf_chars.
    """
    reader = PdfReader(path)
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if len(text) < config.min_page_chars:
            continue
        pages.append({
            "page": i + 1,
            "text": text,
            "source": path.name,
        })

    total_chars = sum(len(p["text"]) for p in pages)
    if total_chars < config.min_pdf_chars:
        logger.warning(
            f"Skipping '{path.name}': only {total_chars} chars extracted "
            f"(likely scanned/image-only PDF)"
        )
        return []

    logger.info(f"Loaded '{path.name}': {len(pages)} pages, {total_chars:,} chars")
    return pages


def batch_pages(
    pages: list[dict],
    max_chars: int,
    max_pages: int,
) -> list[list[dict]]:
    """Group consecutive pages into LLM-sized batches.

    Each batch contains at most max_pages pages and max_chars total characters.
    """
    batches = []
    current_batch = []
    current_chars = 0

    for page in pages:
        page_chars = len(page["text"])

        if current_batch and (
            len(current_batch) >= max_pages
            or current_chars + page_chars > max_chars
        ):
            batches.append(current_batch)
            current_batch = []
            current_chars = 0

        current_batch.append(page)
        current_chars += page_chars

    if current_batch:
        batches.append(current_batch)

    return batches
