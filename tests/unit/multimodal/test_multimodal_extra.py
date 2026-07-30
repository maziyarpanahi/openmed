"""Tests for the [multimodal] optional-dependency extra and import isolation.

The multimodal foundation must not pull heavy ingestion dependencies
(pdfplumber/python-docx/Pillow) into the base ``import openmed`` path, and
``redact_document`` must raise a clear, actionable error naming the extra when
those dependencies are absent.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from openmed.multimodal import redact_document
from openmed.multimodal.exceptions import MissingDependencyError

# Distribution package -> import module name for optional multimodal/OCR deps.
HEAVY_MODULES = (
    "pdfplumber",
    "docx",
    "PIL",
    "doctr",
    "piexif",
    "pikepdf",
    "pydicom",
    "pytesseract",
    "easyocr",
    "paddleocr",
    "markdown_it",
)


def _import_then_check(import_line: str) -> set[str]:
    """Import something in a clean subprocess and report which heavy modules loaded."""
    code = (
        f"import sys\n{import_line}\n"
        f"loaded = [m for m in {HEAVY_MODULES!r} if m in sys.modules]\n"
        "print(','.join(loaded))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    return {m for m in result.stdout.strip().split(",") if m}


def test_importing_openmed_does_not_load_heavy_multimodal_deps():
    assert _import_then_check("import openmed") == set()


def test_importing_openmed_multimodal_does_not_load_heavy_deps():
    # The foundation module itself must defer heavy imports to the ingesters.
    assert _import_then_check("import openmed.multimodal") == set()


def test_redact_document_raises_named_error_when_extra_missing(monkeypatch):
    import openmed.multimodal.base as base

    monkeypatch.setattr(
        base, "_missing_multimodal_dependencies", lambda: ["pdfplumber"]
    )
    with pytest.raises(MissingDependencyError) as excinfo:
        redact_document("note.pdf")

    message = str(excinfo.value)
    assert "multimodal" in message
    assert "pdfplumber" in message


def test_missing_dependency_error_is_importerror():
    assert issubclass(MissingDependencyError, ImportError)
