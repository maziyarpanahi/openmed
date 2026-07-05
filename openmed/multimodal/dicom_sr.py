"""DICOM Structured Report (SR) content-tree to structured-text extractor.

DICOM SR objects (for example TID 1500 measurement reports) carry findings and
measurements in a nested, coded ``ContentSequence`` tree rather than in a flat
narrative. This module walks that tree deterministically into concept-name /
value / unit rows and a linearized narrative so SR reports can be reviewed and
handed to :func:`openmed.deidentify` / :func:`openmed.analyze_text` downstream.

The walker mirrors :mod:`openmed.multimodal.dicom`: ``pydicom`` is imported
lazily so the multimodal package stays importable without the optional imaging
extra, and header de-identification runs before any report text is emitted so
PHI in SR headers never leaks into the flattened output.

.. note::

    ``DICOM_SR_ADVISORY`` — the flattened output is a faithful, mechanical
    transcription of the coded SR content tree. It is **not** a clinical
    interpretation, does not add or infer findings, and must not be used to
    auto-trigger any clinical decision. Review by a qualified professional is
    required before any clinical use.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Sequence, TypedDict

from .base import ExtractedDocument, SourceSpan, register_handler
from .dicom import (
    DicomHeaderDeidPolicy,
    _coerce_policy,
    deidentify_dicom_headers,
)
from .exceptions import MissingDependencyError, UnsupportedDocumentError

_DICOM_SR_INSTALL_HINT = 'Install with: pip install "openmed[multimodal]".'

#: Human-readable advisory attached to every extraction. The flattened text is a
#: faithful transcription of coded SR content, never a clinical interpretation.
DICOM_SR_ADVISORY = (
    "This structured text is a faithful, mechanical transcription of the DICOM "
    "SR coded content tree. It is not a clinical interpretation, adds no "
    "findings, and must not auto-trigger clinical decisions. Review by a "
    "qualified professional is required before any clinical use."
)

# DICOM SR value types we walk. CONTAINER nodes group children; leaf value types
# carry a concept name plus a typed value. Value types outside this set are still
# recorded structurally (with an empty rendered value) so the tree stays intact.
_CONTAINER = "CONTAINER"
_NUM = "NUM"
_CODE = "CODE"
_TEXT = "TEXT"


class SrContentItem(TypedDict):
    """One node in a walked DICOM SR content tree.

    Attributes:
        concept_name: Human-readable concept name (``ConceptNameCodeSequence``
            code meaning), or an empty string for the anonymous root.
        value_type: DICOM SR ``ValueType`` (``CONTAINER``, ``NUM``, ``CODE``,
            ``TEXT``, ``DATE``, ``PNAME``, ...).
        value: Rendered value for the node. For ``NUM`` this is the numeric
            measured value as a string; for ``CODE`` the coded concept meaning;
            for ``TEXT`` the literal text; empty for ``CONTAINER``.
        unit_code: Coded measurement unit (UCUM code meaning) for ``NUM`` nodes,
            otherwise ``None``.
        relationship: ``RelationshipType`` of the node to its parent (for
            example ``CONTAINS``, ``HAS PROPERTIES``), or ``None`` at the root.
        template_id: Mapped ``TemplateIdentifier`` (TID) when the node declares a
            ``ContentTemplateSequence``, otherwise ``None``.
        node_path: 1-based index path from the root (for example ``"1.2.1"``)
            that uniquely and stably identifies the node in the tree.
    """

    concept_name: str
    value_type: str
    value: str
    unit_code: str | None
    relationship: str | None
    template_id: str | None
    node_path: str


def extract_dicom_sr(
    path: str | Path,
    *,
    policy: Any | None = None,
    deidentify_headers: bool = True,
) -> ExtractedDocument:
    """Extract a DICOM SR content tree as structured text plus a typed item list.

    The SR ``ContentSequence`` is walked recursively into an ordered list of
    :class:`SrContentItem` rows (preserving nested ``CONTAINER`` structure and a
    stable ``node_path``) and linearized into a readable narrative. Unless
    ``deidentify_headers`` is ``False``, SR header PHI is de-identified in a
    temporary copy before any text is produced so identifiers never reach the
    flattened output.

    Args:
        path: Path to a DICOM SR (``.dcm``) file.
        policy: Optional header de-identification policy forwarded to
            :func:`openmed.multimodal.deidentify_dicom_headers` (a
            :class:`~openmed.multimodal.DicomHeaderDeidPolicy`, mapping, or
            attribute-bearing object).
        deidentify_headers: When ``True`` (default), scrub SR header PHI before
            reading the content tree. Set ``False`` only for already
            de-identified inputs.

    Returns:
        An :class:`~openmed.multimodal.ExtractedDocument` whose ``text`` is the
        linearized narrative (ready for :func:`openmed.deidentify` /
        :func:`openmed.analyze_text`), whose ``spans`` map each rendered line
        back to its node, and whose ``metadata`` carries the typed
        ``content_items`` list, the ``DICOM_SR_ADVISORY`` string, and an
        audit-safe header de-identification report.

    Raises:
        MissingDependencyError: If ``pydicom`` is not installed.
        UnsupportedDocumentError: If the file is not a readable DICOM SR object.
    """
    pydicom = _import_pydicom()
    source = Path(path)

    header_report: dict[str, Any] | None = None
    read_path = source
    tempdir: Any = None
    try:
        if deidentify_headers:
            tempdir, read_path, header_report = _deidentified_copy(source, policy)

        try:
            dataset = pydicom.dcmread(read_path, force=True)
        except Exception as exc:  # noqa: BLE001 - normalize to a clear error.
            raise UnsupportedDocumentError(
                f"Could not read DICOM SR object at {source}: {exc}"
            ) from exc

        _ensure_sr_dataset(dataset)
        items = walk_sr_content_tree(dataset)
    finally:
        if tempdir is not None:
            tempdir.cleanup()

    text, spans = _linearize(items)
    metadata: dict[str, Any] = {
        "format": "dicom_sr",
        "advisory": DICOM_SR_ADVISORY,
        "content_items": items,
        "node_count": len(items),
        "headers_deidentified": bool(deidentify_headers),
    }
    if header_report is not None:
        metadata["dicom_header_deid"] = header_report
    return ExtractedDocument(text=text, spans=spans, metadata=metadata)


def walk_sr_content_tree(dataset: Any) -> list[SrContentItem]:
    """Walk a DICOM SR dataset ``ContentSequence`` into ordered content items.

    The dataset itself is the root ``CONTAINER``; its ``ContentSequence`` holds
    child nodes, each of which may recurse through its own ``ContentSequence``.
    Nodes are emitted in document order with a 1-based ``node_path`` so nested
    structure is preserved and independently testable.

    Args:
        dataset: A pydicom ``Dataset`` for an SR object (or any object exposing
            ``ValueType`` / ``ContentSequence`` attributes).

    Returns:
        Ordered list of :class:`SrContentItem` rows, root first.
    """
    root_item = _node_to_item(dataset, relationship=None, node_path="1")
    items: list[SrContentItem] = [root_item]
    _walk_children(dataset, parent_path="1", out=items)
    return items


def _walk_children(node: Any, *, parent_path: str, out: list[SrContentItem]) -> None:
    for index, child in enumerate(_content_sequence(node), start=1):
        child_path = f"{parent_path}.{index}"
        relationship = _string_value(getattr(child, "RelationshipType", None)) or None
        out.append(
            _node_to_item(child, relationship=relationship, node_path=child_path)
        )
        _walk_children(child, parent_path=child_path, out=out)


def _node_to_item(
    node: Any,
    *,
    relationship: str | None,
    node_path: str,
) -> SrContentItem:
    value_type = _string_value(getattr(node, "ValueType", None)) or _CONTAINER
    value, unit_code = _render_value(node, value_type)
    return SrContentItem(
        concept_name=_concept_name(node),
        value_type=value_type,
        value=value,
        unit_code=unit_code,
        relationship=relationship,
        template_id=_template_id(node),
        node_path=node_path,
    )


def _render_value(node: Any, value_type: str) -> tuple[str, str | None]:
    """Return ``(rendered_value, unit_code)`` for a node by its SR value type."""
    if value_type == _CONTAINER:
        return "", None
    if value_type == _NUM:
        return _render_num(node)
    if value_type == _CODE:
        return _coded_concept_meaning(getattr(node, "ConceptCodeSequence", None)), None
    if value_type == _TEXT:
        return _string_value(getattr(node, "TextValue", None)), None
    if value_type == "DATE":
        return _string_value(getattr(node, "Date", None)), None
    if value_type == "TIME":
        return _string_value(getattr(node, "Time", None)), None
    if value_type == "DATETIME":
        return _string_value(getattr(node, "DateTime", None)), None
    if value_type == "PNAME":
        return _string_value(getattr(node, "PersonName", None)), None
    if value_type == "UIDREF":
        return _string_value(getattr(node, "UID", None)), None
    return "", None


def _render_num(node: Any) -> tuple[str, str | None]:
    measured = _first_sequence_item(getattr(node, "MeasuredValueSequence", None))
    if measured is None:
        return "", None
    value = _string_value(getattr(measured, "NumericValue", None))
    unit = _coded_concept_meaning(
        getattr(measured, "MeasurementUnitsCodeSequence", None)
    )
    return value, (unit or None)


def _concept_name(node: Any) -> str:
    return _coded_concept_meaning(getattr(node, "ConceptNameCodeSequence", None))


def _coded_concept_meaning(sequence: Any) -> str:
    item = _first_sequence_item(sequence)
    if item is None:
        return ""
    return _string_value(getattr(item, "CodeMeaning", None))


def _template_id(node: Any) -> str | None:
    item = _first_sequence_item(getattr(node, "ContentTemplateSequence", None))
    if item is None:
        return None
    template_id = _string_value(getattr(item, "TemplateIdentifier", None))
    return template_id or None


def _content_sequence(node: Any) -> Sequence[Any]:
    sequence = getattr(node, "ContentSequence", None)
    if sequence is None:
        return ()
    return tuple(sequence)


def _first_sequence_item(sequence: Any) -> Any | None:
    if sequence is None:
        return None
    try:
        iterator = iter(sequence)
    except TypeError:
        return None
    for item in iterator:
        return item
    return None


def _string_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _linearize(items: Sequence[SrContentItem]) -> tuple[str, tuple[SourceSpan, ...]]:
    """Render content items to indented narrative text plus per-line spans."""
    lines: list[str] = []
    spans: list[SourceSpan] = []
    cursor = 0
    for index, item in enumerate(items):
        line = _render_line(item)
        if index > 0:
            cursor += 1  # newline separator
        start = cursor
        cursor += len(line)
        spans.append(
            SourceSpan(
                start=start,
                end=cursor,
                metadata={
                    "format": "dicom_sr",
                    "node_path": item["node_path"],
                    "value_type": item["value_type"],
                },
            )
        )
        lines.append(line)
    return "\n".join(lines), tuple(spans)


def _render_line(item: SrContentItem) -> str:
    depth = item["node_path"].count(".")
    indent = "  " * depth
    concept = item["concept_name"]
    value_type = item["value_type"]
    value = item["value"]

    if value_type == _CONTAINER:
        label = concept or "(report root)"
        return f"{indent}{label}"

    head = f"{indent}{concept}: " if concept else indent
    if value_type == _NUM:
        unit = item["unit_code"]
        rendered = f"{value} {unit}".strip() if unit else value
        return f"{head}{rendered}".rstrip()
    return f"{head}{value}".rstrip()


def _deidentified_copy(
    source: Path,
    policy: Any | None,
) -> tuple[Any, Path, dict[str, Any]]:
    """De-identify SR headers into a temp copy; return (tempdir, path, report)."""
    import tempfile

    resolved = _coerce_policy(policy)
    tempdir = tempfile.TemporaryDirectory(prefix="openmed-dicom-sr-")
    output_path = Path(tempdir.name) / "deid_sr.dcm"
    header_policy = DicomHeaderDeidPolicy(
        output_path=output_path,
        date_shift_days=resolved.date_shift_days,
        patient_key=resolved.patient_key,
        date_shift_max_days=resolved.date_shift_max_days,
        date_shift_secret=resolved.date_shift_secret,
        uid_salt=resolved.uid_salt,
        keep_year=resolved.keep_year,
    )
    try:
        result = deidentify_dicom_headers(source, policy=header_policy)
    except Exception:
        tempdir.cleanup()
        raise
    return tempdir, output_path, result.to_audit_report()


def _ensure_sr_dataset(dataset: Any) -> None:
    modality = _string_value(getattr(dataset, "Modality", None))
    has_content = getattr(dataset, "ContentSequence", None) is not None
    value_type = _string_value(getattr(dataset, "ValueType", None))
    if modality == "SR" or has_content or value_type == _CONTAINER:
        return
    raise UnsupportedDocumentError(
        "DICOM object is not a Structured Report (missing SR modality and "
        "ContentSequence)."
    )


def _import_pydicom() -> Any:
    try:
        return importlib.import_module("pydicom")
    except ImportError as exc:  # pragma: no cover - exercised without extra.
        raise MissingDependencyError(
            dependency="pydicom", instruction=_DICOM_SR_INSTALL_HINT
        ) from exc


def _dicom_sr_detector(path: str | Path) -> bool:
    """Detect DICOM SR files so ``.dcm`` dispatch prefers SR extraction."""
    pydicom = _import_pydicom()
    try:
        dataset = pydicom.dcmread(
            path, stop_before_pixels=True, force=True, specific_tags=["Modality"]
        )
    except Exception:  # noqa: BLE001 - non-SR or unreadable, let others handle it.
        return False
    return _string_value(getattr(dataset, "Modality", None)) == "SR"


def _dicom_sr_handler(
    path: str | Path,
    *,
    policy: Any = None,
    models: Any = None,
    lang: str | None = None,
) -> ExtractedDocument:
    return extract_dicom_sr(path, policy=policy)


# Registered with a content detector so SR objects flatten to reviewable text
# while non-SR ``.dcm`` files continue to hit the pixel/header redaction handler.
register_handler(
    ".dcm",
    _dicom_sr_handler,
    detector=_dicom_sr_detector,
    requires_multimodal=False,
)


__all__ = [
    "DICOM_SR_ADVISORY",
    "SrContentItem",
    "extract_dicom_sr",
    "walk_sr_content_tree",
]
