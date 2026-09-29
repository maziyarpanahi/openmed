"""Synthetic malformed PDF and DICOM header fixtures for multimodal preflight.

Every fixture is a tiny, deterministic, fully synthetic byte string assembled
from constants only: no randomness, no timestamps, no real PHI, no clinical
pixels, and no third-party dependency. Payloads are at most a few hundred bytes
so they can be inlined into preflight tests without touching the filesystem.

The fixtures exist to pin the *pre-decode* boundaries that already refuse
malformed document and medical-image headers:

* PDF payloads are consumed by ``openmed.multimodal.pdf_geometry.read_pdf_geometry``,
  the bounded, dependency-free header and page-tree reader. Each fixture fails
  with one stable reason code from ``PDF_REASON_CODES``.
* DICOM payloads are consumed by ``openmed.multimodal.preflight.preflight_asset``,
  whose media-type check reads at most
  ``openmed.multimodal.media_type.MAX_MEDIA_TYPE_PREFIX_BYTES`` (132) leading
  bytes and matches the ``DICM`` magic at offset 128. No byte-level DICOM
  structure reader exists in this repository, so the DICOM fixtures document
  the bounded magic boundary plus the preflight media-type verdict; they never
  reach ``pydicom``.

``MALFORMED_HEADER_CASES`` maps each corruption class (truncated, oversized,
cyclic, inconsistent, unsupported) for each modality to the boundary that
refuses it and the reason that boundary reports, so tests and documentation
cannot drift from the implemented behavior.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final

CORRUPTION_CLASSES: Final[tuple[str, ...]] = (
    "truncated",
    "oversized",
    "cyclic",
    "inconsistent",
    "unsupported",
)

MODALITIES: Final[tuple[str, ...]] = ("pdf", "dicom")

PDF_BOUNDARY: Final[str] = "pdf_geometry"
DICOM_BOUNDARY: Final[str] = "media_type_preflight"

# Largest payload any fixture may carry; keeps the fixtures tiny by construction.
MAX_FIXTURE_BYTES: Final[int] = 4096

# Byte markers that must never appear in a fixture payload.
PHI_MARKERS: Final[tuple[bytes, ...]] = (
    b"Patient",
    b"MRN",
    b"DOE^",
    b"1977",
    b"OpenMed Synthetic",
)

_DICOM_PREAMBLE_BYTES: Final[int] = 128
_DICOM_MAGIC: Final[bytes] = b"DICM"
_DICOM_FOREIGN_MAGIC: Final[bytes] = b"DICX"

# Synthetic PDF building blocks (version string, object bodies, trailer).
_PDF_HEADER: Final[bytes] = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n"
_PDF_CATALOG: Final[str] = "<< /Type /Catalog /Pages 2 0 R >>"
_PDF_SINGLE_PAGE_TREE: Final[str] = "<< /Type /Pages /Kids [3 0 R] /Count 1 >>"
_PDF_SINGLE_PAGE: Final[str] = (
    "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 8 8] /Contents 4 0 R >>"
)
_PDF_EMPTY_CONTENT: Final[str] = "<< /Length 0 >>\nstream\n\nendstream"
_PDF_CYCLIC_PAGE_TREE: Final[str] = "<< /Type /Pages /Kids [2 0 R] /Count 1 >>"
_PDF_UNBALANCED_PAGE_TREE: Final[str] = "<< /Type /Pages /Kids [3 0 R] /Count 3 >>"


@dataclass(frozen=True, slots=True)
class MalformedHeaderCase:
    """One synthetic malformed-header fixture and its pre-decode verdict.

    Attributes:
        name: Stable kebab-case identifier used as the test id.
        modality: ``"pdf"`` or ``"dicom"``.
        corruption: One of :data:`CORRUPTION_CLASSES`.
        payload: The synthetic bytes handed to the preflight boundary.
        boundary: Boundary that refuses the payload (``pdf_geometry`` or
            ``media_type_preflight``).
        expected_reason: Reason the boundary reports: a full ``PDF_REASON_CODES``
            member for the PDF boundary, or the ``media_type`` reason code
            (``"unknown"`` or ``"mismatch"``) for the DICOM boundary.
        expected_detected: Media type the bounded sniffer reports, or ``None``
            when the payload is not a supported type.
        rejected: ``True`` when the boundary rejects the payload outright;
            ``False`` when it reports the reason on a review verdict.
        declared_media_type: Media type the manifest declares when the DICOM
            boundary is exercised; empty for PDF fixtures.
        declared_fields: Manifest count fields paired with ``declared_media_type``.
        reader_limits: Keyword limits passed to ``read_pdf_geometry`` for the
            cases that exercise the reader's bounded byte/object budget.
        notes: One-line explanation of the corruption, reused by the docs page.
    """

    name: str
    modality: str
    corruption: str
    payload: bytes
    boundary: str
    expected_reason: str
    expected_detected: str | None = None
    rejected: bool = True
    declared_media_type: str = ""
    declared_fields: tuple[tuple[str, int], ...] = ()
    reader_limits: tuple[tuple[str, int], ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if self.modality not in MODALITIES:
            raise ValueError("fixture modality is unsupported")
        if self.corruption not in CORRUPTION_CLASSES:
            raise ValueError("fixture corruption class is unsupported")
        if self.boundary not in (PDF_BOUNDARY, DICOM_BOUNDARY):
            raise ValueError("fixture boundary is unsupported")
        if not isinstance(self.payload, bytes) or not self.payload:
            raise ValueError("fixture payload must be non-empty bytes")
        if len(self.payload) > MAX_FIXTURE_BYTES:
            raise ValueError("fixture payload exceeds the tiny-fixture budget")
        if (self.boundary == DICOM_BOUNDARY) != bool(self.declared_media_type):
            raise ValueError("only DICOM fixtures declare a media type")

    def limits(self) -> dict[str, int]:
        """Return the reader limits as keyword arguments."""
        return dict(self.reader_limits)

    def declaration(self) -> dict[str, object]:
        """Return the manifest fields used to exercise the DICOM boundary."""
        return {"media_type": self.declared_media_type, **dict(self.declared_fields)}

    def sha256(self) -> str:
        """Return the hex digest that pins this payload's bytes."""
        return hashlib.sha256(self.payload).hexdigest()


def _classic_pdf(objects: dict[int, str], *, trailer: str = "") -> bytes:
    """Assemble a synthetic PDF with a classic cross-reference table."""
    output = bytearray(_PDF_HEADER)
    offsets: dict[int, int] = {}
    for number in sorted(objects):
        offsets[number] = len(output)
        output += f"{number} 0 obj\n{objects[number]}\nendobj\n".encode("latin-1")
    size = max(objects) + 1
    xref = len(output)
    output += f"xref\n0 {size}\n0000000000 65535 f \n".encode("ascii")
    for number in range(1, size):
        if number in offsets:
            output += f"{offsets[number]:010d} 00000 n \n".encode("ascii")
        else:
            output += b"0000000000 65535 f \n"
    output += (
        f"trailer\n<< /Size {size} /Root 1 0 R {trailer}>>\nstartxref\n{xref}\n%%EOF\n"
    ).encode("latin-1")
    return bytes(output)


def _single_page_pdf() -> bytes:
    """Return the smallest valid one-page PDF the geometry reader accepts."""
    return _classic_pdf(
        {
            1: _PDF_CATALOG,
            2: _PDF_SINGLE_PAGE_TREE,
            3: _PDF_SINGLE_PAGE,
            4: _PDF_EMPTY_CONTENT,
        }
    )


def pdf_truncated_header() -> bytes:
    """Return a PDF cut off inside its first indirect object, before the xref."""
    return _PDF_HEADER + b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"


def pdf_object_overflow() -> bytes:
    """Return a valid PDF that exceeds a small explicit object budget."""
    objects = dict(
        {
            1: _PDF_CATALOG,
            2: _PDF_SINGLE_PAGE_TREE,
            3: _PDF_SINGLE_PAGE,
            4: _PDF_EMPTY_CONTENT,
        }
    )
    for number in range(5, 13):
        objects[number] = f"<< /Synthetic {number} >>"
    return _classic_pdf(objects)


def pdf_cyclic_page_tree() -> bytes:
    """Return a PDF whose root page-tree node lists itself as its only child."""
    return _classic_pdf({1: _PDF_CATALOG, 2: _PDF_CYCLIC_PAGE_TREE})


def pdf_inconsistent_page_count() -> bytes:
    """Return a PDF whose declared ``/Count`` disagrees with its page tree."""
    return _classic_pdf(
        {
            1: _PDF_CATALOG,
            2: _PDF_UNBALANCED_PAGE_TREE,
            3: _PDF_SINGLE_PAGE,
            4: _PDF_EMPTY_CONTENT,
        }
    )


def pdf_unsupported_encryption() -> bytes:
    """Return a PDF whose trailer declares an encryption dictionary."""
    return _classic_pdf(
        {
            1: _PDF_CATALOG,
            2: _PDF_SINGLE_PAGE_TREE,
            3: _PDF_SINGLE_PAGE,
            4: _PDF_EMPTY_CONTENT,
            5: "<< /Filter /Standard /V 1 >>",
        },
        trailer="/Encrypt 5 0 R ",
    )


def _dicom_payload(magic: bytes, *, preamble_bytes: int = 128) -> bytes:
    """Return a synthetic preamble plus magic with a zero-filled data set."""
    if preamble_bytes < _DICOM_PREAMBLE_BYTES:
        raise ValueError("preamble must fit the DICOM preamble size")
    return b"\x00" * preamble_bytes + magic + b"\x00" * 8


def dicom_truncated_magic() -> bytes:
    """Return a DICOM payload cut off inside the 128-byte preamble."""
    return b"\x00" * 128 + b"DIC"


def dicom_oversized_preamble() -> bytes:
    """Return a DICOM payload whose magic sits past the bounded prefix."""
    return _dicom_payload(_DICOM_MAGIC, preamble_bytes=192)


def dicom_cyclic_preamble() -> bytes:
    """Return a DICOM payload whose 128-byte preamble block repeats."""
    return (b"\x00" * 128) * 2 + _DICOM_MAGIC


def dicom_inconsistent_magic() -> bytes:
    """Return a well-formed DICOM magic that a mismatch declaration contradicts."""
    return _dicom_payload(_DICOM_MAGIC)


def dicom_unsupported_magic() -> bytes:
    """Return a DICOM payload whose magic is a foreign four-byte signature."""
    return _dicom_payload(_DICOM_FOREIGN_MAGIC)


MALFORMED_HEADER_CASES: Final[tuple[MalformedHeaderCase, ...]] = (
    MalformedHeaderCase(
        name="pdf-header-truncated",
        modality="pdf",
        corruption="truncated",
        payload=pdf_truncated_header(),
        boundary=PDF_BOUNDARY,
        expected_reason="pdf_catalog_missing",
        notes="Header present but the file stops before its catalog, xref, and trailer.",
    ),
    MalformedHeaderCase(
        name="pdf-header-oversized",
        modality="pdf",
        corruption="oversized",
        payload=pdf_object_overflow(),
        boundary=PDF_BOUNDARY,
        expected_reason="pdf_object_limit",
        reader_limits=(("max_objects", 4),),
        notes="A twelve-object file probed with a four-object preflight budget.",
    ),
    MalformedHeaderCase(
        name="pdf-header-cyclic",
        modality="pdf",
        corruption="cyclic",
        payload=pdf_cyclic_page_tree(),
        boundary=PDF_BOUNDARY,
        expected_reason="pdf_page_tree_invalid",
        notes="The root page-tree node lists itself as its only child.",
    ),
    MalformedHeaderCase(
        name="pdf-header-inconsistent",
        modality="pdf",
        corruption="inconsistent",
        payload=pdf_inconsistent_page_count(),
        boundary=PDF_BOUNDARY,
        expected_reason="page_count_mismatch",
        rejected=False,
        notes="Declared /Count 3 disagrees with the single page-tree leaf.",
    ),
    MalformedHeaderCase(
        name="pdf-header-unsupported",
        modality="pdf",
        corruption="unsupported",
        payload=pdf_unsupported_encryption(),
        boundary=PDF_BOUNDARY,
        expected_reason="pdf_encrypted",
        notes="The trailer declares an encryption dictionary the reader refuses.",
    ),
    MalformedHeaderCase(
        name="dicom-header-truncated",
        modality="dicom",
        corruption="truncated",
        payload=dicom_truncated_magic(),
        boundary=DICOM_BOUNDARY,
        expected_reason="unknown",
        declared_media_type="application/dicom",
        declared_fields=(("frames", 1), ("width", 8), ("height", 8)),
        notes="131 bytes: the magic cannot complete inside the bounded prefix.",
    ),
    MalformedHeaderCase(
        name="dicom-header-oversized",
        modality="dicom",
        corruption="oversized",
        payload=dicom_oversized_preamble(),
        boundary=DICOM_BOUNDARY,
        expected_reason="unknown",
        declared_media_type="application/dicom",
        declared_fields=(("frames", 1), ("width", 8), ("height", 8)),
        notes="A 192-byte preamble pushes the DICM magic past byte 132.",
    ),
    MalformedHeaderCase(
        name="dicom-header-cyclic",
        modality="dicom",
        corruption="cyclic",
        payload=dicom_cyclic_preamble(),
        boundary=DICOM_BOUNDARY,
        expected_reason="unknown",
        declared_media_type="application/dicom",
        declared_fields=(("frames", 1), ("width", 8), ("height", 8)),
        notes="A repeated 128-byte preamble block leaves the magic past the bound.",
    ),
    MalformedHeaderCase(
        name="dicom-header-inconsistent",
        modality="dicom",
        corruption="inconsistent",
        payload=dicom_inconsistent_magic(),
        boundary=DICOM_BOUNDARY,
        expected_reason="mismatch",
        expected_detected="application/dicom",
        declared_media_type="image/bmp",
        declared_fields=(("width", 8), ("height", 8)),
        notes="Detected DICOM magic contradicts a manifest that declares a BMP.",
    ),
    MalformedHeaderCase(
        name="dicom-header-unsupported",
        modality="dicom",
        corruption="unsupported",
        payload=dicom_unsupported_magic(),
        boundary=DICOM_BOUNDARY,
        expected_reason="unknown",
        declared_media_type="application/dicom",
        declared_fields=(("frames", 1), ("width", 8), ("height", 8)),
        notes="A foreign four-byte magic at offset 128 is not a supported type.",
    ),
)

# Pinned digests: any change to a fixture's bytes must be a deliberate update of
# this table, and ``tests/unit/multimodal/test_malformed_headers.py`` fails if
# the two disagree.
PAYLOAD_SHA256: Final[dict[str, str]] = {
    "pdf-header-truncated": "2d47a05c37cefe96d0ca40933f463fec60543485766bdfe65c409888ab0fb458",
    "pdf-header-oversized": "fe47f44c3b1898b41daa81a497367327e51a10f16b65f45533e0add964689072",
    "pdf-header-cyclic": "2c8bd7d603b3b3824573b14c0c3e8c298cf17ce614467032095ee6cfd4ef1fb2",
    "pdf-header-inconsistent": "9a41cf50fb323714c0fec107ca02585be7faf5bee7d7d40996cfb769c95c3414",
    "pdf-header-unsupported": "31e535aebc9e70c0ddd4cc83eb0f03bf544d249511119cef7e14411b93647288",
    "dicom-header-truncated": "d88a0bcff195c53a5ede2f1ac882d86004856f066509753645592ee8137b45fa",
    "dicom-header-oversized": "7d031dddfc0eb962755ae291646056f0f53ffd66b5f8e8120200987899d89e06",
    "dicom-header-cyclic": "8390f0485df65cc420c58b8f5bef7d56b6d7f1a9d447969a8e464a8881cc1965",
    "dicom-header-inconsistent": "cb1428dc9b113fbc1824b995841c2f5c4b13a62da76f7060cb07035b72caf115",
    "dicom-header-unsupported": "94ee0fb8873ffb0122b56fb6c35d9d674bf57eee05580569825b8a040ae6887b",
}


def cases_for(modality: str) -> tuple[MalformedHeaderCase, ...]:
    """Return the fixtures for one modality in table order."""
    if modality not in MODALITIES:
        raise ValueError("fixture modality is unsupported")
    return tuple(case for case in MALFORMED_HEADER_CASES if case.modality == modality)


def case_by_name(name: str) -> MalformedHeaderCase:
    """Return one fixture by its stable name."""
    for case in MALFORMED_HEADER_CASES:
        if case.name == name:
            return case
    raise KeyError(name)


def corruption_matrix() -> dict[str, tuple[str, ...]]:
    """Return the covered corruption classes per modality."""
    return {
        modality: tuple(case.corruption for case in cases_for(modality))
        for modality in MODALITIES
    }
