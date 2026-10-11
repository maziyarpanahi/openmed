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


def build_pdf_pages(
    content_streams: list[bytes],
    *,
    page_sizes: list[tuple[float, float]] | None = None,
) -> bytes:
    """Assemble a minimal synthetic PDF around one or more content streams."""
    if not content_streams:
        raise ValueError("At least one content stream is required")
    sizes = page_sizes or [(612.0, 792.0)] * len(content_streams)
    if len(sizes) != len(content_streams):
        raise ValueError("page_sizes must match content_streams")

    page_ids = tuple(range(3, 3 + len(content_streams)))
    font_id = 3 + len(content_streams)
    first_stream_id = font_id + 1
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("ascii"),
    ]
    for index, (width, height) in enumerate(sizes):
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width:g} "
                f"{height:g}] /Resources << /Font << /F1 {font_id} 0 R >> >> "
                f"/Contents {first_stream_id + index} 0 R >>"
            ).encode("ascii")
        )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for content_stream in content_streams:
        objects.append(
            b"<< /Length "
            + str(len(content_stream)).encode("ascii")
            + b" >>\nstream\n"
            + content_stream
            + b"endstream"
        )

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


def build_pdf(content_stream: bytes) -> bytes:
    """Assemble a minimal one-page PDF around ``content_stream``."""
    return build_pdf_pages([content_stream])


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


def moved_leak_pdf_bytes() -> bytes:
    """Leaky redaction whose source name was moved below the requested box."""
    return build_pdf(
        _rect_fill(_REDACTION_RECT)
        + _text_block(
            [
                (72.0, 720.0, _SCRUBBED_LINE),
                (72.0, 700.0, _SECOND_LINE),
                (72.0, 680.0, "John Doe"),
            ]
        )
    )


def separated_moved_leak_pdf_bytes() -> bytes:
    """Leaky redaction with selected words separated in the output layer."""
    return build_pdf(
        _rect_fill(_REDACTION_RECT)
        + _text_block(
            [
                (72.0, 720.0, _SCRUBBED_LINE),
                (72.0, 700.0, _SECOND_LINE),
                (72.0, 680.0, "John SYNTHETIC Doe"),
            ]
        )
    )


def duplicate_token_original_pdf_bytes() -> bytes:
    """Source with one selected name and one unrelated repeated first name."""
    return build_pdf(
        _text_block(
            [
                (72.0, 720.0, _NAME_LINE),
                (72.0, 700.0, _SECOND_LINE),
                (72.0, 680.0, "Clinician John Smith"),
            ]
        )
    )


def duplicate_token_clean_redaction_pdf_bytes() -> bytes:
    """Correct redaction preserving an unrelated identical source word."""
    return build_pdf(
        _rect_fill(_REDACTION_RECT)
        + _text_block(
            [
                (72.0, 720.0, _SCRUBBED_LINE),
                (72.0, 700.0, _SECOND_LINE),
                (72.0, 680.0, "Clinician John Smith"),
            ]
        )
    )


def shifted_non_phi_pdf_bytes() -> bytes:
    """Same redaction but with a non-PHI line shifted as a fidelity regression."""
    return build_pdf(
        _rect_fill(_REDACTION_RECT)
        + _text_block([(72.0, 720.0, _SCRUBBED_LINE), (180.0, 700.0, _SECOND_LINE)])
    )


def multipage_pdf_bytes() -> bytes:
    """Two-page source with different page sizes and synthetic text only."""
    return build_pdf_pages(
        [
            _text_block([(72.0, 720.0, _NAME_LINE), (72.0, 700.0, _SECOND_LINE)]),
            _text_block([(48.0, 540.0, "Follow up notes remain stable")]),
        ],
        page_sizes=[(612.0, 792.0), (420.0, 595.0)],
    )


def annotated_pdf_bytes(
    subtype: str = "FreeText",
    *,
    contents: str = "SYNTHETIC_CONTENTS_3726",
    appearance_text: str = "SYNTHETIC_AP_3726",
    state_dictionary: bool = False,
    malformed: str | None = None,
    rotation: int = 0,
    matrix: tuple[float, ...] | None = None,
    cropbox: tuple[float, ...] | None = None,
) -> bytes:
    """Build an annotation with independent synthetic Contents and appearance."""
    from io import BytesIO

    import pikepdf

    with pikepdf.open(BytesIO(original_pdf_bytes())) as pdf:
        page = pdf.pages[0]
        page.Rotate = rotation
        if cropbox is not None:
            page.CropBox = pikepdf.Array(cropbox)
        appearance = pdf.make_stream(
            b"BT /F1 12 Tf 1 0 0 1 4 14 Tm ("
            + appearance_text.encode("ascii")
            + b") Tj ET"
        )
        appearance.Type = pikepdf.Name("/XObject")
        appearance.Subtype = pikepdf.Name("/Form")
        appearance.BBox = pikepdf.Array([0, 0, 228, 40])
        appearance.Resources = page.Resources
        if matrix is not None:
            appearance.Matrix = pikepdf.Array(matrix)
        annotation = pdf.make_indirect(
            pikepdf.Dictionary(
                Type=pikepdf.Name("/Annot"),
                Subtype=pikepdf.Name("/" + subtype),
                Rect=pikepdf.Array([72, 540, 300, 580]),
                Contents=contents,
                AP=pikepdf.Dictionary(N=appearance),
                F=4,
            )
        )
        if state_dictionary:
            annotation.AP.N = pikepdf.Dictionary(On=appearance)
            annotation.AS = pikepdf.Name("/On")
        if subtype == "Widget":
            annotation.FT = pikepdf.Name("/Tx")
            annotation.T = "synthetic-field"
            annotation.V = contents
            pdf.Root.AcroForm = pikepdf.Dictionary(
                Fields=pikepdf.Array([annotation]), NeedAppearances=False
            )
        if malformed == "bbox":
            del appearance["/BBox"]
        elif malformed == "matrix":
            appearance.Matrix = pikepdf.Array([0, 0, 0, 0, 0, 0])
        elif malformed == "state":
            annotation.AP.N = pikepdf.Dictionary(On=appearance)
            annotation.AS = pikepdf.Name("/Missing")
        elif malformed == "image":
            image = pdf.make_stream(b"\x00\x00\x00")
            image.Type = pikepdf.Name("/XObject")
            image.Subtype = pikepdf.Name("/Image")
            image.Width = image.Height = 1
            image.ColorSpace = pikepdf.Name("/DeviceRGB")
            image.BitsPerComponent = 8
            appearance.Resources = pikepdf.Dictionary(
                XObject=pikepdf.Dictionary(Im0=image)
            )
            appearance.write(b"q 20 0 0 20 0 0 cm /Im0 Do Q")
        elif malformed == "vector":
            appearance.write(appearance.read_bytes() + b" 0 0 m 12 0 l 6 12 l h f")
        elif malformed == "clipped":
            annotation.Rect = pikepdf.Array([-72, 540, 156, 580])
        elif malformed == "overflow":
            appearance.write(b"BT /F1 12 Tf 1 0 0 1 -20 14 Tm (OUTSIDE) Tj ET")
        page.Annots = pikepdf.Array([annotation])
        output = BytesIO()
        pdf.save(output)
        return output.getvalue()


__all__ = [
    "annotated_pdf_bytes",
    "build_pdf",
    "build_pdf_pages",
    "clean_redaction_pdf_bytes",
    "duplicate_token_clean_redaction_pdf_bytes",
    "duplicate_token_original_pdf_bytes",
    "leaky_redaction_pdf_bytes",
    "moved_leak_pdf_bytes",
    "multipage_pdf_bytes",
    "original_pdf_bytes",
    "separated_moved_leak_pdf_bytes",
    "shifted_non_phi_pdf_bytes",
]
