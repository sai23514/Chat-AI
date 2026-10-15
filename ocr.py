"""OCR: read text from scanned PDF pages and pictures, on your own computer.

Uses RapidOCR (PaddleOCR models run with ONNX Runtime). No API key, works offline.
rag.py calls these only when a page has no real text of its own.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger("document-chatbot.ocr")

OCR_ENABLED = os.getenv("OCR_ENABLED", "true").lower() == "true"
OCR_MIN_CHARS = int(os.getenv("OCR_MIN_CHARS", "20"))    # fewer chars than this = scanned page
OCR_MAX_PAGES = int(os.getenv("OCR_MAX_PAGES", "50"))    # stop huge scans from taking forever
OCR_DPI = int(os.getenv("OCR_DPI", "200"))               # higher = sharper but slower
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

_engine = None


def engine():
    """Load the OCR models once (the first call takes a few seconds)."""
    global _engine
    if _engine is None:
        from rapidocr import RapidOCR
        _engine = RapidOCR()
    return _engine


def read_image(image) -> str:
    """image can be a file path or a PIL image. Returns the text, one line per row."""
    if not isinstance(image, (str, Path)):
        image = np.array(image.convert("RGB"))
    else:
        image = str(image)
    result = engine()(image)
    return "\n".join(result.txts or [])


def read_pdf_page(path: Path, page_index: int) -> str:
    """Turn one PDF page into a picture, then OCR it."""
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(Path(path).read_bytes())  # bytes: avoids Windows/OneDrive path problems
    try:
        picture = pdf[page_index].render(scale=OCR_DPI / 72).to_pil()
    finally:
        pdf.close()
    return read_image(picture)