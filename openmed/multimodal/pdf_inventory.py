"""Dependency-free, counts-only inventory of PDF content outside page text.

This is a conservative bounded object inventory, not a PDF conformance or
redaction-completeness certificate. Superseded definitions are included.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, BinaryIO, Final

from .abstention import AbstentionReason, AbstentionRecord, AbstentionStage
from .pdf_geometry import (
    DEFAULT_MAX_PDF_BYTES,
    DEFAULT_MAX_PDF_DECOMPRESSED_BYTES,
    DEFAULT_MAX_PDF_OBJECTS,
    DEFAULT_MAX_PDF_PAGES,
    PDF_REASON_CODES,
    PdfGeometryError,
    PdfGeometryStatus,
    _Document,
    _Name,
    _read_pdf_document,
    _Ref,
)

__all__ = [
    "PDF_CONTENT_REASON_CODES",
    "PDF_ANNOTATION_SUBTYPES",
    "PdfContentProfile",
    "PdfContentInventory",
    "PdfInventoryReport",
    "read_pdf_inventory",
]

PDF_CONTENT_REASON_CODES: Final = (
    "pdf_inventory_invalid",
    "pdf_annotations",
    "pdf_form_values",
    "pdf_embedded_files",
    "pdf_xfa",
    "pdf_optional_content",
    "pdf_javascript",
    "pdf_open_action",
    "pdf_launch_action",
    "pdf_incremental_revisions",
)
PDF_ANNOTATION_SUBTYPES: Final = (
    "Text",
    "Link",
    "FreeText",
    "Line",
    "Square",
    "Circle",
    "Polygon",
    "PolyLine",
    "Highlight",
    "Underline",
    "Squiggly",
    "StrikeOut",
    "Stamp",
    "Caret",
    "Ink",
    "Popup",
    "FileAttachment",
    "Sound",
    "Movie",
    "Widget",
    "Screen",
    "PrinterMark",
    "TrapNet",
    "Watermark",
    "3D",
    "Redact",
    "Projection",
    "RichMedia",
    "other",
)


class PdfContentProfile(str, Enum):
    """Hidden content policy: review all categories or strictly reject risk.

    STRICT rejects attachments, JavaScript, Launch, and any OpenAction.
    REVIEW retains the inventory but requires review for these categories.
    Neither profile executes actions or extracts source payloads.
    """

    REVIEW = "review"
    STRICT = "strict"


@dataclass(frozen=True, slots=True)
class PdfContentInventory:
    """Counts of parsed definitions, including superseded revisions.

    Annotation subtype names are closed buckets; unknown names become ``other``.
    Fields count dictionaries in the AcroForm tree or with field keys. Values
    count explicit non-null V/DV declarations, without inspecting their content.
    Attachments count unique embedded streams or resolved target dictionaries.
    Revisions count top-level startxref/EOF pairs, at least one.
    No count establishes text extraction or redaction coverage.
    """

    annotation_subtypes: tuple[tuple[str, int], ...] = ()
    field_count: int = 0
    field_value_count: int = 0
    embedded_file_count: int = 0
    xfa_packet_count: int = 0
    optional_content_group_count: int = 0
    javascript_action_count: int = 0
    open_action_count: int = 0
    launch_action_count: int = 0
    revision_count: int = 1

    def __post_init__(self) -> None:
        for field in fields(self):
            if field.name == "annotation_subtypes":
                continue
            value = getattr(self, field.name)
            if type(value) is not int or value < (
                1 if field.name == "revision_count" else 0
            ):
                raise PdfGeometryError("pdf_inventory_count_invalid")
        if type(self.annotation_subtypes) is not tuple:
            raise PdfGeometryError("pdf_inventory_subtypes_invalid")
        seen = set()
        for item in self.annotation_subtypes:
            if type(item) is not tuple or len(item) != 2:
                raise PdfGeometryError("pdf_inventory_subtypes_invalid")
            code, count = item
            if (
                type(code) is not str
                or code not in PDF_ANNOTATION_SUBTYPES
                or code in seen
                or type(count) is not int
                or count <= 0
            ):
                raise PdfGeometryError("pdf_inventory_subtypes_invalid")
            seen.add(code)

    def to_dict(self) -> dict[str, Any]:
        """Return counts with controlled category keys only."""
        return {
            "annotation_subtypes": dict(self.annotation_subtypes),
            "field_count": self.field_count,
            "field_value_count": self.field_value_count,
            "embedded_file_count": self.embedded_file_count,
            "xfa_packet_count": self.xfa_packet_count,
            "optional_content_group_count": self.optional_content_group_count,
            "javascript_action_count": self.javascript_action_count,
            "open_action_count": self.open_action_count,
            "launch_action_count": self.launch_action_count,
            "revision_count": self.revision_count,
        }


@dataclass(frozen=True, slots=True)
class PdfInventoryReport:
    """Preflight verdict, counts and controlled codes; no source data.

    Inventory is absent on parser failure. Strict policy rejection retains the
    successfully read inventory. Review and rejection carry an abstention.
    """

    status: PdfGeometryStatus
    reason_codes: tuple[str, ...]
    inventory: PdfContentInventory | None
    abstention: AbstentionRecord | None

    def __post_init__(self) -> None:
        if not isinstance(self.status, PdfGeometryStatus):
            raise PdfGeometryError("pdf_inventory_status_invalid")
        if type(self.reason_codes) is not tuple or any(
            type(code) is not str
            or code not in (*PDF_REASON_CODES, *PDF_CONTENT_REASON_CODES)
            for code in self.reason_codes
        ):
            raise PdfGeometryError("pdf_inventory_codes_invalid")
        if self.inventory is not None and not isinstance(
            self.inventory, PdfContentInventory
        ):
            raise PdfGeometryError("pdf_inventory_counts_invalid")
        if self.abstention is not None and (
            not isinstance(self.abstention, AbstentionRecord)
            or self.abstention.stage is not AbstentionStage.PREFLIGHT
        ):
            raise PdfGeometryError("pdf_inventory_abstention_invalid")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, content-free mapping."""
        return {
            "schema_version": "openmed.multimodal.pdf_inventory.v1",
            "status": self.status.value,
            "reason_codes": list(self.reason_codes),
            "inventory": None if self.inventory is None else self.inventory.to_dict(),
            "abstention": None
            if self.abstention is None
            else self.abstention.to_dict(),
        }

    def to_json(self) -> str:
        """Return compact deterministic JSON."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def read_pdf_inventory(
    source: bytes | bytearray | memoryview | BinaryIO,
    *,
    profile: PdfContentProfile = PdfContentProfile.STRICT,
    max_bytes: int = DEFAULT_MAX_PDF_BYTES,
    max_pages: int = DEFAULT_MAX_PDF_PAGES,
    max_objects: int = DEFAULT_MAX_PDF_OBJECTS,
    max_decompressed_bytes: int = DEFAULT_MAX_PDF_DECOMPRESSED_BYTES,
) -> PdfInventoryReport:
    """Inventory hidden PDF content locally within the geometry reader's bounds.

    Args:
        source: In-memory bytes or caller-owned binary stream. Seekable streams
            are restored. Source content is not written or returned.
        profile: Strict rejection or review policy for attachments/actions.
        max_bytes: Input byte ceiling.
        max_pages: Page-tree leaf ceiling.
        max_objects: Total indirect definitions, including object-stream entries.
        max_decompressed_bytes: Total object-stream expansion ceiling.

    Returns:
        Counts and controlled codes. Readable means no inventoried categories,
        not a guarantee of PHI absence or complete redaction.

    Raises:
        PdfGeometryError: Invalid arguments or a controlled source failure.
    """
    if not isinstance(profile, PdfContentProfile):
        raise PdfGeometryError("pdf_profile_invalid")
    limits = dict(
        max_bytes=max_bytes,
        max_pages=max_pages,
        max_objects=max_objects,
        max_decompressed_bytes=max_decompressed_bytes,
    )
    for key, value in limits.items():
        if type(value) is not int or value <= 0:
            raise PdfGeometryError(f"{key}_invalid")
    try:
        geometry, document = _read_pdf_document(source, **limits)
    except PdfGeometryError:
        raise
    except Exception:
        raise PdfGeometryError("source_read_error") from None
    if geometry.status is PdfGeometryStatus.REJECTED:
        return _report(geometry.reason_codes, None, rejected=True)
    assert document is not None
    if document.object_stream_failed or document.syntax_failed:
        code = (
            "pdf_object_stream_unsupported"
            if document.object_stream_failed
            else "pdf_inventory_invalid"
        )
        return _report((code,), None, rejected=True)
    try:
        inventory, hidden = _inventory(document)
    except _InventoryFailure:
        return _report(("pdf_inventory_invalid",), None, rejected=True)
    reasons = tuple(
        code
        for code in (*PDF_REASON_CODES, *PDF_CONTENT_REASON_CODES)
        if code in {*geometry.reason_codes, *hidden}
    )
    strict_rejection = profile is PdfContentProfile.STRICT and bool(
        hidden
        & {
            "pdf_embedded_files",
            "pdf_javascript",
            "pdf_open_action",
            "pdf_launch_action",
        }
    )
    return _report(reasons, inventory, rejected=strict_rejection)


def _report(
    codes: tuple[str, ...], inventory: PdfContentInventory | None, *, rejected: bool
) -> PdfInventoryReport:
    status = (
        PdfGeometryStatus.REJECTED
        if rejected
        else PdfGeometryStatus.REVIEW
        if codes
        else PdfGeometryStatus.READABLE
    )
    reason = AbstentionReason.PHI_UNCERTAINTY
    if rejected:
        if any(code.endswith("_limit") for code in codes):
            reason = AbstentionReason.RESOURCE_LIMIT
        elif inventory is None:
            reason = AbstentionReason.MALFORMED_MEDIA
        else:
            reason = AbstentionReason.UNSUPPORTED_MEDIA
    abstention = (
        None if not codes else AbstentionRecord(AbstentionStage.PREFLIGHT, reason)
    )
    return PdfInventoryReport(status, codes, inventory, abstention)


class _InventoryFailure(Exception):
    pass


def _resolve(document: _Document, value: Any) -> Any:
    resolved = document.resolve(value)
    if isinstance(value, _Ref) and resolved is None:
        raise _InventoryFailure
    return resolved


def _tree(document: _Document, roots: Any, validated: set[int]) -> set[int]:
    """Validate AcroForm/name trees without following Parent backreferences."""
    roots = _resolve(document, roots)
    if not isinstance(roots, list):
        raise _InventoryFailure
    found: set[int] = set()
    active: set[int] = set()
    stack = [(item, 0, False) for item in roots]
    while stack:
        item, depth, exit_node = stack.pop()
        node = _resolve(document, item)
        if not isinstance(node, dict) or depth > 64:
            raise _InventoryFailure
        identity = id(node)
        if exit_node:
            active.remove(identity)
            continue
        if identity in active:
            raise _InventoryFailure
        if identity in found or identity in validated:
            continue
        found.add(identity)
        active.add(identity)
        stack.append((node, depth, True))
        if "Kids" in node:
            kids = _resolve(document, node["Kids"])
            if not isinstance(kids, list):
                raise _InventoryFailure
            stack.extend((kid, depth + 1, False) for kid in kids)
    validated.update(found)
    return found


def _inventory(document: _Document) -> tuple[PdfContentInventory, set[str]]:
    dictionaries: dict[int, dict[str, Any]] = {}
    stack = list(document.inventory_values)
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            if id(value) in dictionaries:
                continue
            dictionaries[id(value)] = value
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    field_trees: set[int] = set()
    name_trees: set[int] = set()
    annotations: set[int] = set()
    fields: set[int] = set()
    attachments: set[int] = set()
    reasons: set[str] = set()
    counts = dict(
        xfa_packet_count=0,
        optional_content_group_count=0,
        javascript_action_count=0,
        open_action_count=0,
        launch_action_count=0,
    )
    for identity, node in dictionaries.items():
        kind = _resolve(document, node.get("Type"))
        if kind == _Name("Annot") or ("Rect" in node and "Subtype" in node):
            annotations.add(identity)
        if "Annots" in node:
            items = _resolve(document, node["Annots"])
            if not isinstance(items, list):
                raise _InventoryFailure
            for item in items:
                annotation = _resolve(document, item)
                if not isinstance(annotation, dict):
                    raise _InventoryFailure
                annotations.add(id(annotation))
        if any(key in node for key in ("FT", "T", "V", "DV")) and "S" not in node:
            fields.add(identity)
        if "Fields" in node:
            fields.update(_tree(document, node["Fields"], field_trees))
        if kind == _Name("EmbeddedFile"):
            attachments.add(identity)
        if "EF" in node:
            ef = _resolve(document, node["EF"])
            if not isinstance(ef, dict) or not ef:
                raise _InventoryFailure
            for item in ef.values():
                target = _resolve(document, item)
                if not isinstance(target, dict):
                    raise _InventoryFailure
                attachments.add(id(target))
        if "EmbeddedFiles" in node:
            reasons.add("pdf_embedded_files")
            _tree(document, [node["EmbeddedFiles"]], name_trees)
        if "XFA" in node and node["XFA"] is not None:
            xfa = _resolve(document, node["XFA"])
            if isinstance(xfa, list):
                if len(xfa) % 2:
                    raise _InventoryFailure
                counts["xfa_packet_count"] += len(xfa) // 2
            else:
                counts["xfa_packet_count"] += 1
        if kind == _Name("OCG"):
            counts["optional_content_group_count"] += 1
        if "OCProperties" in node or "OC" in node:
            reasons.add("pdf_optional_content")
        action = _resolve(document, node.get("S"))
        if action == _Name("JavaScript") or "JS" in node:
            counts["javascript_action_count"] += 1
        if "JavaScript" in node:
            reasons.add("pdf_javascript")
            _tree(document, [node["JavaScript"]], name_trees)
        if "OpenAction" in node and node["OpenAction"] is not None:
            _resolve(document, node["OpenAction"])
            counts["open_action_count"] += 1
        if action == _Name("Launch"):
            counts["launch_action_count"] += 1
    subtypes: dict[str, int] = {}
    for identity in annotations:
        node = dictionaries.get(identity)
        if node is None:
            raise _InventoryFailure
        subtype = _resolve(document, node.get("Subtype"))
        code = (
            subtype
            if isinstance(subtype, _Name) and subtype in PDF_ANNOTATION_SUBTYPES[:-1]
            else "other"
        )
        subtypes[code] = subtypes.get(code, 0) + 1
    # Annotation /T is an author title, not an AcroForm field name. Widgets
    # may combine an annotation and field definition, so keep those fields.
    fields.difference_update(
        identity
        for identity in annotations
        if _resolve(document, dictionaries[identity].get("Subtype")) != _Name("Widget")
    )
    if subtypes.get("FileAttachment"):
        reasons.add("pdf_embedded_files")
    field_values = sum(
        any(
            key in dictionaries[identity]
            and _resolve(document, dictionaries[identity][key]) is not None
            for key in ("V", "DV")
        )
        for identity in fields
    )
    inventory = PdfContentInventory(
        annotation_subtypes=tuple(
            (code, subtypes[code])
            for code in PDF_ANNOTATION_SUBTYPES
            if code in subtypes
        ),
        field_count=len(fields),
        field_value_count=field_values,
        embedded_file_count=len(attachments),
        revision_count=document.revision_count,
        **counts,
    )
    for count, code in (
        (len(annotations), "pdf_annotations"),
        (field_values, "pdf_form_values"),
        (len(attachments), "pdf_embedded_files"),
        (counts["xfa_packet_count"], "pdf_xfa"),
        (counts["optional_content_group_count"], "pdf_optional_content"),
        (counts["javascript_action_count"], "pdf_javascript"),
        (counts["open_action_count"], "pdf_open_action"),
        (counts["launch_action_count"], "pdf_launch_action"),
        (inventory.revision_count > 1, "pdf_incremental_revisions"),
    ):
        if count:
            reasons.add(code)
    return inventory, reasons
