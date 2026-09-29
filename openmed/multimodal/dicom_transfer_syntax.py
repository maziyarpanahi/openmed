"""Bounded, dependency-free DICOM transfer-syntax preflight.

The DICOM File Meta Information group declares how pixel data is encoded. A
pipeline can size and route work from those declared bytes alone, before a
pixel decoder or an imaging library is imported. This module parses only the
bounded file-meta header: the 128-byte preamble, the ``DICM`` magic, and the
group ``0002`` elements needed to read ``(0002,0010) TransferSyntaxUID``.

Nothing else is returned. Patient tags, other UIDs, pixel bytes, file paths,
and text metadata never appear in a report, and no codec is imported, loaded,
or exercised.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

__all__ = [
    "DEFAULT_MAX_FILE_META_BYTES",
    "DEFAULT_MAX_FILE_META_ELEMENTS",
    "DEFAULT_MAX_TRANSFER_SYNTAX_UID_BYTES",
    "TRANSFER_SYNTAX_CATALOG",
    "TRANSFER_SYNTAX_OUTCOMES",
    "TRANSFER_SYNTAX_REASON_CODES",
    "TRANSFER_SYNTAX_SCHEMA_VERSION",
    "TransferSyntaxCapability",
    "TransferSyntaxError",
    "TransferSyntaxOutcome",
    "TransferSyntaxReport",
    "describe_transfer_syntax",
    "read_dicom_transfer_syntax",
]

TRANSFER_SYNTAX_SCHEMA_VERSION: Final[str] = (
    "openmed.multimodal.dicom_transfer_syntax.v1"
)

DEFAULT_MAX_FILE_META_BYTES: Final[int] = 64 * 1024
DEFAULT_MAX_FILE_META_ELEMENTS: Final[int] = 64
DEFAULT_MAX_TRANSFER_SYNTAX_UID_BYTES: Final[int] = 128

_DICOM_PREAMBLE_BYTES: Final[int] = 128
_DICOM_MAGIC: Final[bytes] = b"DICM"
_FILE_META_GROUP: Final[int] = 0x0002
_GROUP_LENGTH_TAG: Final[tuple[int, int]] = (0x0002, 0x0000)
_TRANSFER_SYNTAX_TAG: Final[tuple[int, int]] = (0x0002, 0x0010)
_EXPLICIT_VR_SHORT: Final[frozenset[bytes]] = frozenset(
    {
        b"AE",
        b"AS",
        b"AT",
        b"CS",
        b"DA",
        b"DS",
        b"DT",
        b"FD",
        b"FL",
        b"IS",
        b"LO",
        b"LT",
        b"PN",
        b"SH",
        b"SL",
        b"SS",
        b"ST",
        b"TM",
        b"UI",
        b"UL",
        b"US",
    }
)
_EXPLICIT_VR_LONG: Final[frozenset[bytes]] = frozenset(
    {
        b"OB",
        b"OD",
        b"OF",
        b"OL",
        b"OV",
        b"OW",
        b"SQ",
        b"SV",
        b"UC",
        b"UN",
        b"UR",
        b"UT",
        b"UV",
    }
)
_UID_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9]+(?:\.[0-9]+)*$")
_UID_ROOT: Final[str] = "1.2.840.10008."


class TransferSyntaxOutcome(str, Enum):
    """Closed set of routing outcomes for a declared transfer syntax.

    Values:
        NATIVE: Uncompressed little-endian syntaxes the pipeline can read with
            built-in decoders.
        OPTIONAL_DECODER: Compressed syntaxes that need an installed codec
            before pixel data can be decoded.
        REVIEW: Syntaxes that are retired or need an extra parsing decision
            before decoding is attempted.
        UNSUPPORTED: Unusable, unknown, private, or malformed declarations.
    """

    NATIVE = "native"
    OPTIONAL_DECODER = "optional_decoder"
    REVIEW = "review"
    UNSUPPORTED = "unsupported"


TRANSFER_SYNTAX_OUTCOMES: Final[frozenset[str]] = frozenset(
    outcome.value for outcome in TransferSyntaxOutcome
)

TRANSFER_SYNTAX_REASON_CODES: Final[tuple[str, ...]] = (
    "file_meta_element_limit",
    "file_meta_group_length_mismatch",
    "file_meta_malformed",
    "file_meta_missing",
    "file_meta_truncated",
    "transfer_syntax_big_endian_retired",
    "transfer_syntax_deflated",
    "transfer_syntax_explicit_little_endian",
    "transfer_syntax_implicit_little_endian",
    "transfer_syntax_jpeg2000_lossless",
    "transfer_syntax_jpeg2000_lossy",
    "transfer_syntax_jpeg2000_multi",
    "transfer_syntax_jpeg2000_multi_lossless",
    "transfer_syntax_jpeg_baseline",
    "transfer_syntax_jpeg_extended",
    "transfer_syntax_jpeg_lossless",
    "transfer_syntax_jpeg_lossless_sv1",
    "transfer_syntax_jpeg_ls_lossless",
    "transfer_syntax_jpeg_ls_near_lossless",
    "transfer_syntax_malformed",
    "transfer_syntax_missing",
    "transfer_syntax_private",
    "transfer_syntax_rle_lossless",
    "transfer_syntax_unknown",
)


@dataclass(frozen=True, slots=True)
class TransferSyntaxCapability:
    """Catalog entry describing one declared transfer syntax.

    Attributes:
        uid: The transfer-syntax identifier this entry describes.
        name: Human-readable DICOM name of the syntax.
        outcome: Routing outcome reported when the syntax is declared.
        reason: Stable reason code paired with the outcome.
        decoder_required: Whether an optional pixel-data codec is needed.
        retired: Whether DICOM has retired the syntax.
    """

    uid: str
    name: str
    outcome: TransferSyntaxOutcome
    reason: str
    decoder_required: bool
    retired: bool = False

    def __post_init__(self) -> None:
        if not _valid_uid(self.uid):
            raise ValueError("uid must be a DICOM UID")
        if type(self.name) is not str or not self.name:
            raise ValueError("name must be a non-empty string")
        if not isinstance(self.outcome, TransferSyntaxOutcome):
            raise ValueError("outcome must be a TransferSyntaxOutcome")
        if self.reason not in TRANSFER_SYNTAX_REASON_CODES:
            raise ValueError("reason must be a known reason code")
        if type(self.decoder_required) is not bool or type(self.retired) is not bool:
            raise ValueError("decoder_required and retired must be booleans")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready mapping for this entry."""

        return {
            "decoder_required": self.decoder_required,
            "name": self.name,
            "outcome": self.outcome.value,
            "reason": self.reason,
            "retired": self.retired,
            "uid": self.uid,
        }


@dataclass(frozen=True, slots=True)
class TransferSyntaxReport:
    """Bounded file-meta preflight result without pixels, paths, or patient tags.

    Attributes:
        outcome: Routing outcome for the declared transfer syntax.
        reason: Primary stable reason code for the outcome.
        reason_codes: All stable reason codes, in insertion order.
        transfer_syntax_uid: Declared ``(0002,0010)`` value, if readable.
        transfer_syntax_name: Catalog name for the declared value, if known.
        decoder_required: Whether an optional codec is needed before decoding.
        file_meta_bytes: Bytes consumed by the parsed file-meta group.
        element_count: Number of group ``0002`` elements parsed.
        schema_version: Version of the report schema.
    """

    outcome: TransferSyntaxOutcome
    reason: str
    reason_codes: tuple[str, ...]
    transfer_syntax_uid: str | None
    transfer_syntax_name: str | None
    decoder_required: bool
    file_meta_bytes: int
    element_count: int
    schema_version: str = TRANSFER_SYNTAX_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, TransferSyntaxOutcome):
            raise ValueError("outcome must be a TransferSyntaxOutcome")
        if not self.reason_codes:
            raise ValueError("reason_codes must not be empty")
        for code in self.reason_codes:
            if code not in TRANSFER_SYNTAX_REASON_CODES:
                raise ValueError("reason_codes must be known reason codes")
        if self.reason != self.reason_codes[0]:
            raise ValueError("reason must be the first reason code")
        if self.transfer_syntax_uid is not None and not _valid_uid(
            self.transfer_syntax_uid
        ):
            raise ValueError("transfer_syntax_uid must be a DICOM UID")
        if type(self.decoder_required) is not bool:
            raise ValueError("decoder_required must be a boolean")
        for count in (self.file_meta_bytes, self.element_count):
            if type(count) is not int or count < 0:
                raise ValueError("counts must be non-negative integers")
        if self.schema_version != TRANSFER_SYNTAX_SCHEMA_VERSION:
            raise ValueError("schema_version is unsupported")

    @property
    def ok(self) -> bool:
        """Whether the declaration is readable without an extra decision."""

        return self.outcome is TransferSyntaxOutcome.NATIVE

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready mapping for this report.

        The mapping is limited to transfer-syntax metadata: no patient tags,
        no UIDs other than the declared transfer syntax, no pixel bytes, and
        no file paths.
        """

        return {
            "schema_version": self.schema_version,
            "outcome": self.outcome.value,
            "reason": self.reason,
            "reason_codes": list(self.reason_codes),
            "transfer_syntax_uid": self.transfer_syntax_uid,
            "transfer_syntax_name": self.transfer_syntax_name,
            "decoder_required": self.decoder_required,
            "file_meta_bytes": self.file_meta_bytes,
            "element_count": self.element_count,
        }

    def to_json(self) -> str:
        """Return compact JSON with sorted keys."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


class TransferSyntaxError(ValueError):
    """Value-free failure for invalid reader arguments.

    Content problems are reported through :class:`TransferSyntaxReport`; this
    error is raised only when a caller-supplied argument cannot be used.
    """

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)


def _valid_uid(value: object, *, max_bytes: int = 64) -> bool:
    if type(value) is not str or not value or len(value) > max_bytes:
        return False
    if _UID_RE.fullmatch(value) is None:
        return False
    for component in value.split("."):
        if len(component) > 1 and component.startswith("0"):
            return False
    return True


def _capability(
    uid: str,
    name: str,
    outcome: TransferSyntaxOutcome,
    reason: str,
    *,
    decoder_required: bool = False,
    retired: bool = False,
) -> TransferSyntaxCapability:
    return TransferSyntaxCapability(
        uid=uid,
        name=name,
        outcome=outcome,
        reason=reason,
        decoder_required=decoder_required,
        retired=retired,
    )


TRANSFER_SYNTAX_CATALOG: Final[dict[str, TransferSyntaxCapability]] = {
    "1.2.840.10008.1.2": _capability(
        "1.2.840.10008.1.2",
        "Implicit VR Little Endian",
        TransferSyntaxOutcome.NATIVE,
        "transfer_syntax_implicit_little_endian",
    ),
    "1.2.840.10008.1.2.1": _capability(
        "1.2.840.10008.1.2.1",
        "Explicit VR Little Endian",
        TransferSyntaxOutcome.NATIVE,
        "transfer_syntax_explicit_little_endian",
    ),
    "1.2.840.10008.1.2.1.99": _capability(
        "1.2.840.10008.1.2.1.99",
        "Deflated Explicit VR Little Endian",
        TransferSyntaxOutcome.REVIEW,
        "transfer_syntax_deflated",
    ),
    "1.2.840.10008.1.2.2": _capability(
        "1.2.840.10008.1.2.2",
        "Explicit VR Big Endian",
        TransferSyntaxOutcome.REVIEW,
        "transfer_syntax_big_endian_retired",
        retired=True,
    ),
    "1.2.840.10008.1.2.4.50": _capability(
        "1.2.840.10008.1.2.4.50",
        "JPEG Baseline (Process 1)",
        TransferSyntaxOutcome.OPTIONAL_DECODER,
        "transfer_syntax_jpeg_baseline",
        decoder_required=True,
    ),
    "1.2.840.10008.1.2.4.51": _capability(
        "1.2.840.10008.1.2.4.51",
        "JPEG Extended (Process 2 and 4)",
        TransferSyntaxOutcome.OPTIONAL_DECODER,
        "transfer_syntax_jpeg_extended",
        decoder_required=True,
    ),
    "1.2.840.10008.1.2.4.57": _capability(
        "1.2.840.10008.1.2.4.57",
        "JPEG Lossless, Non-Hierarchical (Process 14)",
        TransferSyntaxOutcome.OPTIONAL_DECODER,
        "transfer_syntax_jpeg_lossless",
        decoder_required=True,
    ),
    "1.2.840.10008.1.2.4.70": _capability(
        "1.2.840.10008.1.2.4.70",
        "JPEG Lossless, Non-Hierarchical, First-Order Prediction",
        TransferSyntaxOutcome.OPTIONAL_DECODER,
        "transfer_syntax_jpeg_lossless_sv1",
        decoder_required=True,
    ),
    "1.2.840.10008.1.2.4.80": _capability(
        "1.2.840.10008.1.2.4.80",
        "JPEG-LS Lossless",
        TransferSyntaxOutcome.OPTIONAL_DECODER,
        "transfer_syntax_jpeg_ls_lossless",
        decoder_required=True,
    ),
    "1.2.840.10008.1.2.4.81": _capability(
        "1.2.840.10008.1.2.4.81",
        "JPEG-LS Lossy (Near-Lossless)",
        TransferSyntaxOutcome.OPTIONAL_DECODER,
        "transfer_syntax_jpeg_ls_near_lossless",
        decoder_required=True,
    ),
    "1.2.840.10008.1.2.4.90": _capability(
        "1.2.840.10008.1.2.4.90",
        "JPEG 2000 Image Compression (Lossless Only)",
        TransferSyntaxOutcome.OPTIONAL_DECODER,
        "transfer_syntax_jpeg2000_lossless",
        decoder_required=True,
    ),
    "1.2.840.10008.1.2.4.91": _capability(
        "1.2.840.10008.1.2.4.91",
        "JPEG 2000 Image Compression",
        TransferSyntaxOutcome.OPTIONAL_DECODER,
        "transfer_syntax_jpeg2000_lossy",
        decoder_required=True,
    ),
    "1.2.840.10008.1.2.4.92": _capability(
        "1.2.840.10008.1.2.4.92",
        "JPEG 2000 Part 2 Multi-component (Lossless Only)",
        TransferSyntaxOutcome.OPTIONAL_DECODER,
        "transfer_syntax_jpeg2000_multi_lossless",
        decoder_required=True,
    ),
    "1.2.840.10008.1.2.4.93": _capability(
        "1.2.840.10008.1.2.4.93",
        "JPEG 2000 Part 2 Multi-component",
        TransferSyntaxOutcome.OPTIONAL_DECODER,
        "transfer_syntax_jpeg2000_multi",
        decoder_required=True,
    ),
    "1.2.840.10008.1.2.5": _capability(
        "1.2.840.10008.1.2.5",
        "RLE Lossless",
        TransferSyntaxOutcome.OPTIONAL_DECODER,
        "transfer_syntax_rle_lossless",
        decoder_required=True,
    ),
}


def describe_transfer_syntax(uid: str) -> TransferSyntaxCapability | None:
    """Look up a transfer-syntax UID in the closed capability catalog.

    Returns ``None`` for a syntactically valid UID the catalog does not
    describe, and raises :class:`TransferSyntaxError` for a non-string or
    malformed identifier.
    """

    if not _valid_uid(uid):
        raise TransferSyntaxError("uid_invalid")
    return TRANSFER_SYNTAX_CATALOG.get(uid)


def read_dicom_transfer_syntax(
    source: bytes | bytearray | memoryview,
    *,
    max_bytes: int = DEFAULT_MAX_FILE_META_BYTES,
    max_elements: int = DEFAULT_MAX_FILE_META_ELEMENTS,
    max_uid_bytes: int = DEFAULT_MAX_TRANSFER_SYNTAX_UID_BYTES,
) -> TransferSyntaxReport:
    """Read the declared DICOM transfer syntax from bounded file-meta bytes.

    Only the preamble, the magic, and the group ``0002`` elements up to the end
    of the file-meta group are inspected. Truncated, malformed, unknown, and
    private declarations are reported as outcomes rather than raised, so an
    orchestrator can route them without catching content errors. Invalid
    arguments raise :class:`TransferSyntaxError`.
    """

    for name, value in (
        ("max_bytes", max_bytes),
        ("max_elements", max_elements),
        ("max_uid_bytes", max_uid_bytes),
    ):
        if type(value) is not int or value <= 0:
            raise TransferSyntaxError(f"{name}_invalid")
    if not isinstance(source, (bytes, bytearray, memoryview)):
        raise TransferSyntaxError("source_invalid")

    raw = bytes(source)
    bounded = raw[:max_bytes]
    if len(bounded) < _DICOM_PREAMBLE_BYTES + len(_DICOM_MAGIC):
        return _unsupported("file_meta_missing", file_meta_bytes=len(bounded))
    if bounded[_DICOM_PREAMBLE_BYTES : _DICOM_PREAMBLE_BYTES + 4] != _DICOM_MAGIC:
        return _unsupported(
            "file_meta_missing", file_meta_bytes=_DICOM_PREAMBLE_BYTES + 4
        )

    offset = _DICOM_PREAMBLE_BYTES + 4
    parsed_bytes = offset
    element_count = 0
    declared_uid: str | None = None
    group_length: int | None = None
    group_length_end: int | None = None
    element_limit_hit = False
    group_terminated = False
    truncated = len(raw) > max_bytes

    while offset < len(bounded):
        if element_count >= max_elements:
            element_limit_hit = True
            break
        if offset + 8 > len(bounded):
            truncated = True
            break
        group = int.from_bytes(bounded[offset : offset + 2], "little")
        element = int.from_bytes(bounded[offset + 2 : offset + 4], "little")
        if group != _FILE_META_GROUP:
            group_terminated = True
            break
        vr = bounded[offset + 4 : offset + 6]
        if vr in _EXPLICIT_VR_SHORT:
            length = int.from_bytes(bounded[offset + 6 : offset + 8], "little")
            value_start = offset + 8
        elif vr in _EXPLICIT_VR_LONG:
            if offset + 12 > len(bounded):
                truncated = True
                break
            length = int.from_bytes(bounded[offset + 8 : offset + 12], "little")
            value_start = offset + 12
        else:
            return _unsupported(
                "file_meta_malformed",
                file_meta_bytes=parsed_bytes,
                element_count=element_count,
            )
        if length % 2:
            return _unsupported(
                "file_meta_malformed",
                file_meta_bytes=parsed_bytes,
                element_count=element_count,
            )
        value_end = value_start + length
        if value_end > len(bounded):
            truncated = True
            break
        tag = (group, element)
        value = bounded[value_start:value_end]
        if tag == _GROUP_LENGTH_TAG:
            if vr != b"UL" or length != 4:
                return _unsupported(
                    "file_meta_malformed",
                    file_meta_bytes=parsed_bytes,
                    element_count=element_count,
                )
            group_length = int.from_bytes(value, "little")
            group_length_end = value_end
        elif tag == _TRANSFER_SYNTAX_TAG:
            if vr != b"UI":
                return _unsupported(
                    "file_meta_malformed",
                    file_meta_bytes=parsed_bytes,
                    element_count=element_count,
                )
            if len(value) > max_uid_bytes:
                return _unsupported(
                    "transfer_syntax_malformed",
                    file_meta_bytes=parsed_bytes,
                    element_count=element_count,
                )
            text = value.rstrip(b"\x00 ").decode("ascii", errors="replace")
            if not _valid_uid(text):
                return _unsupported(
                    "transfer_syntax_malformed",
                    file_meta_bytes=parsed_bytes,
                    element_count=element_count,
                )
            declared_uid = text
        element_count += 1
        parsed_bytes = value_end
        offset = value_end

    if declared_uid is None:
        if element_limit_hit:
            reason = "file_meta_element_limit"
        elif group_terminated:
            reason = "transfer_syntax_missing"
        elif truncated:
            reason = "file_meta_truncated"
        else:
            reason = "transfer_syntax_missing"
        return _unsupported(
            reason, file_meta_bytes=parsed_bytes, element_count=element_count
        )

    if group_terminated and group_length is not None and group_length_end is not None:
        if offset != group_length_end + group_length:
            return _unsupported(
                "file_meta_group_length_mismatch",
                file_meta_bytes=parsed_bytes,
                element_count=element_count,
                uid=declared_uid,
            )

    capability = TRANSFER_SYNTAX_CATALOG.get(declared_uid)
    if capability is None:
        reason = (
            "transfer_syntax_private"
            if not declared_uid.startswith(_UID_ROOT)
            else "transfer_syntax_unknown"
        )
        return _unsupported(
            reason,
            file_meta_bytes=parsed_bytes,
            element_count=element_count,
            uid=declared_uid,
        )
    return TransferSyntaxReport(
        outcome=capability.outcome,
        reason=capability.reason,
        reason_codes=(capability.reason,),
        transfer_syntax_uid=capability.uid,
        transfer_syntax_name=capability.name,
        decoder_required=capability.decoder_required,
        file_meta_bytes=parsed_bytes,
        element_count=element_count,
    )


def _unsupported(
    reason: str,
    *,
    file_meta_bytes: int,
    element_count: int = 0,
    uid: str | None = None,
) -> TransferSyntaxReport:
    capability = TRANSFER_SYNTAX_CATALOG.get(uid) if uid is not None else None
    return TransferSyntaxReport(
        outcome=TransferSyntaxOutcome.UNSUPPORTED,
        reason=reason,
        reason_codes=(reason,),
        transfer_syntax_uid=uid,
        transfer_syntax_name=capability.name if capability is not None else None,
        decoder_required=False,
        file_meta_bytes=file_meta_bytes,
        element_count=element_count,
    )
