"""Multimodal ingestion and redaction package for section 4.2.

Intended contents include PDF/DOCX/HTML->text+offsets extraction, OCR, and
image/DICOM redaction.
"""

from __future__ import annotations

from .ocr import OcrResult, available_ocr_engines, run_doctr_ocr
from .ocr import ocr as run_ocr

__all__ = [
    "OcrResult",
    "available_ocr_engines",
    "run_ocr",
    "run_doctr_ocr",
]
