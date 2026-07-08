"""Synthetic redacted-PDF fixtures for verify_pdf tests.

Builds minimal, fully synthetic single-page PDFs (no real PHI) from raw content
streams so tests do not depend on a PDF-writing library. Text is placed with an
absolute text matrix and redaction boxes are drawn as filled rectangles, which
``pdfplumber`` reports under ``page.rects`` with ``fill=True``.

Three canonical variants back the fidelity checks:

* ``original`` — the source page with the name present and no box.
* ``clean_redaction`` — the name removed from the text layer AND a box drawn.
* ``leaky_redaction`` — a box drawn but the name still selectable underneath.
"""

from __future__ import annotations

# The synthetic name lives only in these fixtures; it is not real PHI.
_NAME_LINE = "Patient John Doe"
_SCRUBBED_LINE = "Patient"
_SECOND_LINE = "MRN 12345"

# Generous box (PDF bottom-up coords) covering the "John Doe" glyphs at y=720.
_REDACTION_RECT = (110.0, 708.0, 95.0, 22.0)


def _text_block(lines: list[tuple[float, float, str]]) -> bytes:
    parts = [b"BT\n/F1 12 Tf\n"]
    for x, y, text in lines:
        parts.append(f"1 0 0 1 {x} {y} Tm\n".encode("ascii"))
        parts.append(b"(" + text.encode("ascii") + b") Tj\n")
    parts.append(b"ET\n")
    return b"".join(parts)


def _rect_fill(rect: tuple[float, float, float, float]) -> bytes:
    x, y, w, h = rect
    return f"0 0 0 rg\n{x} {y} {w} {h} re\nf\n".encode("ascii")


def build_pdf(content_stream: bytes) -> bytes:
    """Assemble a minimal one-page PDF around ``content_stream``."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "
        + str(len(content_stream)).encode("ascii")
        + b" >>\nstream\n"
        + content_stream
        + b"endstream",
    ]

    payload = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(payload))
        payload.extend(f"{index} 0 obj\n".encode("ascii"))
        payload.extend(obj)
        payload.extend(b"\nendobj\n")

    xref_offset = len(payload)
    payload.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    payload.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        payload.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    payload.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(payload)


def original_pdf_bytes() -> bytes:
    """Source page: the synthetic name is present and no box is drawn."""
    return build_pdf(
        _text_block([(72.0, 720.0, _NAME_LINE), (72.0, 700.0, _SECOND_LINE)])
    )


def clean_redaction_pdf_bytes() -> bytes:
    """Correct redaction: the name is removed AND a box is drawn over it."""
    return build_pdf(
        _rect_fill(_REDACTION_RECT)
        + _text_block([(72.0, 720.0, _SCRUBBED_LINE), (72.0, 700.0, _SECOND_LINE)])
    )


def leaky_redaction_pdf_bytes() -> bytes:
    """Leaky redaction: a box is drawn but the name is still selectable."""
    return build_pdf(
        _rect_fill(_REDACTION_RECT)
        + _text_block([(72.0, 720.0, _NAME_LINE), (72.0, 700.0, _SECOND_LINE)])
    )


__all__ = [
    "build_pdf",
    "clean_redaction_pdf_bytes",
    "leaky_redaction_pdf_bytes",
    "original_pdf_bytes",
]
