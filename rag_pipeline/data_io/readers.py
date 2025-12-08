from __future__ import annotations

import logging
from pathlib import Path

from pypdf import PdfReader
from unstructured.partition.pdf import partition_pdf

logger = logging.getLogger(__name__)


def read_txt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return path.read_text(encoding="cp949", errors="ignore")


def _read_pdf_with_unstructured(pdf_path: Path) -> str:
    elements = partition_pdf(
        filename=str(pdf_path),
        strategy="hi_res",
        infer_table_structure=True,
    )
    texts: list[str] = []
    for el in elements:
        text = (el.text or "").strip()
        if text:
            texts.append(text)
    return "\n\n".join(texts).strip()


def _read_pdf_with_pypdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        extracted = page.extract_text() or ""
        pages.append(extracted.strip())
    return "\n\n".join([p for p in pages if p]).strip()


def read_pdf(path: Path) -> str:
    """Read a PDF as text using Unstructured with PyPDF fallback."""
    try:
        parsed = _read_pdf_with_unstructured(path)
        if parsed:
            return parsed
    except Exception as exc:  # pragma: no cover - logging side effect
        logger.warning("Unstructured parsing failed for %s: %s", path, exc)

    logger.info("Falling back to PyPDF parsing for %s", path)
    return _read_pdf_with_pypdf(path)
