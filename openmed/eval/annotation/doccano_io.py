"""Doccano JSONL adapter for source-text-free annotation interchange records.

The adapter owns Doccano files only. It never contacts a Doccano server, never
imports the Doccano client library, and never copies the Doccano ``text`` field
into an interchange envelope: imports keep offsets plus an HMAC surface hash,
and both directions require the SHA-256 digest of the de-identified text the
caller passes in. Exported text is the caller's de-identified text; annotation
content that Doccano cannot express is reported as a declared loss instead of
being dropped silently.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from openmed.clinical.journey_contracts import (
    canonical_digest,
    canonical_json,
    derived_opaque_id,
    sha256_digest,
)
from openmed.eval.annotation.interchange import (
    MAX_ANNOTATION_CELL_CHARS,
    MAX_ANNOTATION_INPUT_BYTES,
    MAX_ANNOTATION_ROWS,
    AnnotationEnvelope,
    AnnotationExport,
    AnnotationInterchangeError,
    AnnotationLoss,
    AnnotationLossKind,
    AnnotationLossReport,
    AnnotationRecord,
    AnnotationState,
    AnnotationType,
    CoordinateConvention,
    build_annotation_envelope,
)
from openmed.eval.annotation.toolkit import (
    AnnotationIssue,
    AnnotationValidationError,
    span_from_offsets,
)

DOCCANO_FORMAT_NAME: Final = "Doccano JSONL"
DOCCANO_SOURCE_FORMAT: Final = "doccano"
DOCCANO_ENTITY_FIELDS: Final = frozenset({"end_offset", "id", "label", "start_offset"})
DOCCANO_RELATION_FIELDS: Final = frozenset({"from_id", "id", "to_id", "type"})
DOCCANO_DOCUMENT_FIELDS: Final = frozenset({"entities", "label", "relations", "text"})

_CONTROLLED_PATTERN: Final = re.compile(r"[a-z][a-z0-9_.:/-]{0,127}")
_DIGEST_PATTERN: Final = re.compile(r"sha256:[0-9a-f]{64}")
_RELATION_SLUG_PATTERN: Final = re.compile(r"[^a-z0-9_.:/-]+")
_OPAQUE_ID_PATTERN: Final = re.compile(r"[a-z][a-z0-9_]{0,31}_[A-Za-z0-9_-]{8,128}")


@dataclass(frozen=True, slots=True)
class DoccanoImport:
    """One imported Doccano document and its declared losses."""

    envelope: AnnotationEnvelope
    report: AnnotationLossReport


@dataclass(frozen=True, slots=True)
class _DoccanoEntity:
    """One Doccano span before it becomes an interchange record."""

    identifier: int
    start: int
    end: int
    label: str


def export_doccano(
    envelope: AnnotationEnvelope,
    *,
    text: str,
    text_digest: str,
    document_id: str | None = None,
) -> AnnotationExport:
    """Export the entities and relations of one document as Doccano JSONL.

    The caller supplies the de-identified text together with its SHA-256 digest;
    the export refuses to run when the digest does not match the text, so a
    document without an audited de-identification step cannot be exported.

    Args:
        envelope: Source-text-free annotation envelope to export.
        text: De-identified document text that the offsets refer to.
        text_digest: SHA-256 digest of ``text`` from the de-identification step.
        document_id: Document to export; required when the envelope holds more
            than one document.

    Returns:
        A single-line JSONL export and its loss report. Record types, review
        states and embeddings that Doccano cannot express are reported as
        declared losses.

    Raises:
        AnnotationInterchangeError: If ``text_digest`` is not a digest, does not
            match ``text``, the records span several documents, or a relation
            points at an entity that cannot be exported.
    """

    if not isinstance(text, str) or not text:
        raise AnnotationInterchangeError(
            "doccano export requires the de-identified text"
        )
    _require_digest(text_digest, "text_digest")
    if sha256_digest(text) != text_digest:
        raise AnnotationInterchangeError(
            "doccano export text does not match the supplied de-identification digest"
        )
    records = envelope.records
    if document_id is None:
        if len({record.document_id for record in records}) > 1:
            raise AnnotationInterchangeError(
                "doccano export requires an explicit document_id for a multi-document "
                "envelope"
            )
        document_id = records[0].document_id if records else None
    selected = tuple(record for record in records if record.document_id == document_id)
    if records and not selected:
        raise AnnotationInterchangeError(
            "doccano export document_id is not present in the envelope"
        )

    losses: list[AnnotationLoss] = []
    entities: list[dict[str, Any]] = []
    entity_ids: dict[str, int] = {}
    for record in selected:
        if record.annotation_type is not AnnotationType.ENTITY:
            continue
        if record.state is not AnnotationState.SUCCESS:
            losses.append(
                _loss(
                    record.annotation_id,
                    AnnotationLossKind.REVIEW_REQUIRED,
                    "doccano_state_not_exported",
                )
            )
            continue
        start = record.start
        end = record.end
        if not isinstance(start, int) or not isinstance(end, int) or end > len(text):
            raise AnnotationInterchangeError(
                "doccano export offsets fall outside the exported text"
            )
        entity_ids[record.annotation_id] = len(entities)
        entities.append(
            {
                "end_offset": end,
                "id": len(entities),
                "label": record.result["label"],
                "start_offset": start,
            }
        )
        if record.embedding is not None:
            losses.append(
                _loss(
                    record.annotation_id,
                    AnnotationLossKind.EMBEDDING_OMITTED,
                    "doccano_embedding_not_exported",
                )
            )
    relations: list[dict[str, Any]] = []
    for record in selected:
        if record.annotation_type is not AnnotationType.RELATION:
            continue
        if record.state is not AnnotationState.SUCCESS:
            losses.append(
                _loss(
                    record.annotation_id,
                    AnnotationLossKind.REVIEW_REQUIRED,
                    "doccano_state_not_exported",
                )
            )
            continue
        source = entity_ids.get(record.result["source_annotation_id"])
        target = entity_ids.get(record.result["target_annotation_id"])
        if source is None or target is None:
            raise AnnotationInterchangeError(
                "doccano export relations must point at exported entities"
            )
        relations.append(
            {
                "from_id": source,
                "id": len(relations),
                "to_id": target,
                "type": record.result["relation"],
            }
        )
    exported_types = {AnnotationType.ENTITY, AnnotationType.RELATION}
    for record in selected:
        if record.annotation_type not in exported_types:
            losses.append(
                _loss(
                    record.annotation_id,
                    AnnotationLossKind.UNSUPPORTED_ANNOTATION,
                    "doccano_annotation_type_not_exported",
                )
            )

    line = (
        canonical_json({"entities": entities, "relations": relations, "text": text})
        + "\n"
    )
    return AnnotationExport(
        text=line,
        report=_report(
            "export_doccano",
            losses,
            input_digest=envelope.envelope_digest,
            output_digest=canonical_digest(line),
        ),
    )


def import_doccano(
    payload: str | bytes,
    *,
    text_digest: str,
    doc_id: str,
    namespace: str = "default",
    hash_secret: str | bytes,
) -> DoccanoImport:
    """Import one Doccano JSONL document as a source-text-free envelope.

    Both Doccano span dialects are accepted: the relation-extraction dialect
    (``entities`` with ``start_offset``/``end_offset``/``label``) and the
    sequence-labeling dialect (``label`` with ``[start, end, label]`` triples).
    Mixing them fails closed. The Doccano ``text`` field is used only to verify
    ``text_digest`` and to hash the annotated surfaces, so the imported envelope
    never carries source text. Doccano fields this adapter does not model, and
    relation types that are rewritten into controlled identifiers, are reported
    as declared losses.

    Args:
        payload: A single JSONL line as text or UTF-8 encoded bytes.
        text_digest: SHA-256 digest of the de-identified document text.
        doc_id: Opaque document identifier (``<kind>_<material>``) recorded on
            every imported annotation.
        namespace: Controlled namespace recorded on every imported annotation.
        hash_secret: HMAC secret used to hash the annotated surfaces.

    Returns:
        The imported envelope and the loss report for the import.

    Raises:
        AnnotationValidationError: If the payload is not a single JSON object,
            the text digest does not match, offsets fall outside the document,
            a label is unknown to the OpenMed taxonomy, or a relation endpoint
            cannot be resolved.
        AnnotationInterchangeError: If ``doc_id`` is not an opaque identifier,
            the payload carries duplicate keys or non-finite numbers, or a
            relation type cannot be represented as a controlled identifier.
    """

    if not doc_id:
        raise AnnotationValidationError(
            AnnotationIssue("doc_id must be non-empty"),
            format_name=DOCCANO_FORMAT_NAME,
        )
    if _OPAQUE_ID_PATTERN.fullmatch(doc_id) is None:
        raise AnnotationInterchangeError("doc_id must be an opaque identifier")
    _require_digest(text_digest, "text_digest")
    line = _single_line(payload)
    document = _strict_load(line)
    if not isinstance(document, Mapping):
        raise _invalid("line 1 must be a JSON object")

    losses: list[AnnotationLoss] = []
    document_annotation_id = derived_opaque_id("doccano_document", doc_id)
    for _field in sorted(set(document) - DOCCANO_DOCUMENT_FIELDS):
        losses.append(
            _loss(
                document_annotation_id,
                AnnotationLossKind.UNSUPPORTED_ANNOTATION,
                "doccano_unsupported_field",
            )
        )

    text = document.get("text")
    if not isinstance(text, str) or not text:
        raise _invalid("line 1 must carry the de-identified text")
    if sha256_digest(text) != text_digest:
        raise _invalid(
            "doccano text does not match the supplied de-identification digest"
        )

    has_entities = "entities" in document
    has_labels = "label" in document
    if has_entities and has_labels:
        raise _invalid("line 1 must not mix Doccano entities and label spans")
    if not has_entities and not has_labels:
        raise _invalid("line 1 must carry Doccano entities or label spans")

    issues: list[AnnotationIssue] = []
    if has_entities:
        entities = _parse_entities(
            document["entities"],
            text_length=len(text),
            doc_id=doc_id,
            losses=losses,
            issues=issues,
        )
    else:
        entities = _parse_label_spans(
            document["label"], text_length=len(text), issues=issues
        )

    records: list[AnnotationRecord] = []
    entity_ids: dict[int, str] = {}
    for entity in entities:
        try:
            span = span_from_offsets(
                doc_id=doc_id,
                text=text,
                start=entity.start,
                end=entity.end,
                label=entity.label,
                hash_secret=hash_secret,
                metadata={"annotation_format": DOCCANO_SOURCE_FORMAT},
            )
        except AnnotationValidationError as exc:
            issues.extend(exc.issues)
            continue
        annotation_id = derived_opaque_id(
            "doccano_span",
            doc_id,
            str(entity.identifier),
            str(entity.start),
            str(entity.end),
        )
        entity_ids[entity.identifier] = annotation_id
        records.append(
            AnnotationRecord(
                annotation_id=annotation_id,
                document_id=doc_id,
                namespace=namespace,
                annotation_type=AnnotationType.ENTITY,
                coordinate_convention=CoordinateConvention.UNICODE_CODEPOINT,
                start=entity.start,
                end=entity.end,
                state=AnnotationState.SUCCESS,
                result={
                    "label": span.canonical_label.lower(),
                    "surface_hash": sha256_digest(span.text_hash),
                },
                metadata={"source_format": DOCCANO_SOURCE_FORMAT},
            )
        )
    if issues:
        raise AnnotationValidationError(issues, format_name=DOCCANO_FORMAT_NAME)

    records.extend(
        _relation_records(
            document.get("relations", []),
            doc_id=doc_id,
            namespace=namespace,
            entity_ids=entity_ids,
            losses=losses,
            issues=issues,
        )
    )
    if issues:
        raise AnnotationValidationError(issues, format_name=DOCCANO_FORMAT_NAME)

    envelope = build_annotation_envelope(records)
    return DoccanoImport(
        envelope=envelope,
        report=_report(
            "import_doccano",
            losses,
            input_digest=sha256_digest(line),
            output_digest=envelope.envelope_digest,
        ),
    )


def _parse_entities(
    value: Any,
    *,
    text_length: int,
    doc_id: str,
    losses: list[AnnotationLoss],
    issues: list[AnnotationIssue],
) -> list[_DoccanoEntity]:
    if not isinstance(value, list):
        raise _invalid("line 1 entities must be a JSON array")
    if len(value) > MAX_ANNOTATION_ROWS:
        raise _invalid("line 1 exceeds the Doccano annotation limit")
    entities: list[_DoccanoEntity] = []
    seen: set[int] = set()
    for position, item in enumerate(value, start=1):
        if not isinstance(item, Mapping):
            issues.append(
                AnnotationIssue(f"entity {position} must be a JSON object", line=1)
            )
            continue
        missing = DOCCANO_ENTITY_FIELDS - {"id"} - set(item)
        if missing:
            issues.append(
                AnnotationIssue(
                    f"entity {position} is missing {', '.join(sorted(missing))}",
                    line=1,
                )
            )
            continue
        identifier = item.get("id", position - 1)
        if type(identifier) is not int or identifier < 0 or identifier in seen:
            issues.append(
                AnnotationIssue(f"entity {position} has an invalid id", line=1)
            )
            continue
        offsets = _offsets(
            item["start_offset"],
            item["end_offset"],
            text_length=text_length,
            position=position,
            kind="entity",
            issues=issues,
        )
        label = item["label"]
        if not isinstance(label, str) or not label.strip():
            issues.append(
                AnnotationIssue(f"entity {position} has an invalid label", line=1)
            )
            continue
        if offsets is None:
            continue
        start, end = offsets
        seen.add(identifier)
        entities.append(_DoccanoEntity(identifier, start, end, label))
        for _field in sorted(set(item) - DOCCANO_ENTITY_FIELDS):
            losses.append(
                _loss(
                    derived_opaque_id(
                        "doccano_span", doc_id, str(identifier), str(start), str(end)
                    ),
                    AnnotationLossKind.UNSUPPORTED_ANNOTATION,
                    "doccano_unsupported_field",
                )
            )
    return entities


def _parse_label_spans(
    value: Any, *, text_length: int, issues: list[AnnotationIssue]
) -> list[_DoccanoEntity]:
    if not isinstance(value, list):
        raise _invalid("line 1 label spans must be a JSON array")
    if len(value) > MAX_ANNOTATION_ROWS:
        raise _invalid("line 1 exceeds the Doccano annotation limit")
    entities: list[_DoccanoEntity] = []
    for position, item in enumerate(value, start=1):
        if isinstance(item, (str, bytes)) or not isinstance(item, (list, tuple)):
            issues.append(
                AnnotationIssue(
                    f"label span {position} must be a [start, end, label] triple",
                    line=1,
                )
            )
            continue
        if len(item) != 3:
            issues.append(
                AnnotationIssue(
                    f"label span {position} must be a [start, end, label] triple",
                    line=1,
                )
            )
            continue
        start, end, label = item
        offsets = _offsets(
            start,
            end,
            text_length=text_length,
            position=position,
            kind="label span",
            issues=issues,
        )
        if not isinstance(label, str) or not label.strip():
            issues.append(
                AnnotationIssue(f"label span {position} has an invalid label", line=1)
            )
            continue
        if offsets is None:
            continue
        entities.append(_DoccanoEntity(position - 1, offsets[0], offsets[1], label))
    return entities


def _relation_records(
    value: Any,
    *,
    doc_id: str,
    namespace: str,
    entity_ids: dict[int, str],
    losses: list[AnnotationLoss],
    issues: list[AnnotationIssue],
) -> list[AnnotationRecord]:
    if not isinstance(value, list):
        raise _invalid("line 1 relations must be a JSON array")
    if len(value) > MAX_ANNOTATION_ROWS:
        raise _invalid("line 1 exceeds the Doccano annotation limit")
    records: list[AnnotationRecord] = []
    for position, item in enumerate(value, start=1):
        if not isinstance(item, Mapping):
            issues.append(
                AnnotationIssue(f"relation {position} must be a JSON object", line=1)
            )
            continue
        missing = DOCCANO_RELATION_FIELDS - {"id"} - set(item)
        if missing:
            issues.append(
                AnnotationIssue(
                    f"relation {position} is missing {', '.join(sorted(missing))}",
                    line=1,
                )
            )
            continue
        source = (
            entity_ids.get(item["from_id"]) if type(item["from_id"]) is int else None
        )
        target = entity_ids.get(item["to_id"]) if type(item["to_id"]) is int else None
        if source is None or target is None:
            issues.append(
                AnnotationIssue(
                    f"relation {position} refers to an entity that was not imported",
                    line=1,
                )
            )
            continue
        raw_type = item["type"]
        if not isinstance(raw_type, str) or not raw_type.strip():
            issues.append(
                AnnotationIssue(f"relation {position} has an invalid type", line=1)
            )
            continue
        code = _relation_code(raw_type)
        annotation_id = derived_opaque_id(
            "doccano_relation", doc_id, str(position), code
        )
        if code != raw_type:
            losses.append(
                _loss(
                    annotation_id,
                    AnnotationLossKind.UNSUPPORTED_ANNOTATION,
                    "doccano_relation_type_normalized",
                )
            )
        for _field in sorted(set(item) - DOCCANO_RELATION_FIELDS):
            losses.append(
                _loss(
                    annotation_id,
                    AnnotationLossKind.UNSUPPORTED_ANNOTATION,
                    "doccano_unsupported_field",
                )
            )
        records.append(
            AnnotationRecord(
                annotation_id=annotation_id,
                document_id=doc_id,
                namespace=namespace,
                annotation_type=AnnotationType.RELATION,
                coordinate_convention=CoordinateConvention.NONE,
                start=None,
                end=None,
                state=AnnotationState.SUCCESS,
                result={
                    "relation": code,
                    "source_annotation_id": source,
                    "target_annotation_id": target,
                },
                metadata={"source_format": DOCCANO_SOURCE_FORMAT},
            )
        )
    return records


def _offsets(
    start: Any,
    end: Any,
    *,
    text_length: int,
    position: int,
    kind: str,
    issues: list[AnnotationIssue],
) -> tuple[int, int] | None:
    if type(start) is not int or type(end) is not int:
        issues.append(
            AnnotationIssue(f"{kind} {position} offsets must be integers", line=1)
        )
        return None
    if start < 0 or start >= end or end > text_length:
        issues.append(
            AnnotationIssue(
                f"{kind} {position} offsets fall outside the document", line=1
            )
        )
        return None
    return start, end


def _relation_code(raw_type: str) -> str:
    slug = _RELATION_SLUG_PATTERN.sub("-", raw_type.strip().casefold()).strip("-")
    if slug and _CONTROLLED_PATTERN.match(slug) is None:
        slug = f"relation:{slug}"
    if not slug or _CONTROLLED_PATTERN.fullmatch(slug) is None:
        raise AnnotationInterchangeError(
            "doccano relation type is not a controlled identifier"
        )
    return slug


def _single_line(payload: str | bytes) -> str:
    if isinstance(payload, (bytes, bytearray, memoryview)):
        if len(payload) > MAX_ANNOTATION_INPUT_BYTES:
            raise _invalid("payload exceeds the annotation input limit")
        try:
            raw = bytes(payload).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _invalid("payload must be UTF-8 encoded") from exc
    elif isinstance(payload, str):
        raw = payload
    else:
        raise _invalid("payload must be text or UTF-8 bytes")
    if len(raw.encode("utf-8")) > MAX_ANNOTATION_INPUT_BYTES:
        raise _invalid("payload exceeds the annotation input limit")
    lines = [item.strip() for item in raw.splitlines() if item.strip()]
    if len(lines) != 1:
        raise _invalid("import one Doccano document per call")
    if len(lines[0]) > MAX_ANNOTATION_CELL_CHARS:
        raise _invalid("line 1 exceeds the annotation cell limit")
    return lines[0]


def _strict_load(line: str) -> Any:
    try:
        return json.loads(
            line,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite,
        )
    except AnnotationInterchangeError:
        raise
    except (ValueError, RecursionError) as exc:
        raise AnnotationValidationError(
            AnnotationIssue("line 1 must be a valid JSON object", line=1),
            format_name=DOCCANO_FORMAT_NAME,
        ) from exc


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise AnnotationInterchangeError("doccano payload contains duplicate keys")
        seen[key] = value
    return seen


def _reject_non_finite(token: str) -> Any:
    raise AnnotationInterchangeError("doccano payload contains a non-finite number")


def _report(
    operation: str,
    losses: list[AnnotationLoss],
    *,
    input_digest: str,
    output_digest: str,
) -> AnnotationLossReport:
    entries = tuple(
        sorted(losses, key=lambda loss: (loss.annotation_id, loss.kind.value))
    )
    state = (
        AnnotationState.PARTIAL
        if any(entry.lossy for entry in entries)
        else AnnotationState.SUCCESS
    )
    return AnnotationLossReport(
        operation=operation,
        state=state,
        input_digest=input_digest,
        output_digest=output_digest,
        entries=entries,
    )


def _loss(annotation_id: str, kind: AnnotationLossKind, code: str) -> AnnotationLoss:
    return AnnotationLoss(annotation_id=annotation_id, kind=kind, lossy=True, code=code)


def _require_digest(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise AnnotationInterchangeError(f"{field_name} must be a SHA-256 digest")
    return value


def _invalid(message: str) -> AnnotationValidationError:
    return AnnotationValidationError(
        AnnotationIssue(message, line=1), format_name=DOCCANO_FORMAT_NAME
    )


__all__ = [
    "DOCCANO_DOCUMENT_FIELDS",
    "DOCCANO_ENTITY_FIELDS",
    "DOCCANO_FORMAT_NAME",
    "DOCCANO_RELATION_FIELDS",
    "DOCCANO_SOURCE_FORMAT",
    "DoccanoImport",
    "export_doccano",
    "import_doccano",
]
