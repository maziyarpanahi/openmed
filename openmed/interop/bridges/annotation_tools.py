"""Clean-room adapters for fact-correction and registry-label tool rows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Final

from openmed.eval.annotation.interchange import (
    MAX_ANNOTATION_ROWS,
    AnnotationEnvelope,
    AnnotationInterchangeError,
    AnnotationRecord,
    AnnotationState,
    AnnotationType,
    CoordinateConvention,
    build_annotation_envelope,
)

_CORRECTION_FIELDS: Final = frozenset(
    {
        "annotation_id",
        "document_id",
        "evidence_ids",
        "fact_id",
        "field",
        "reason_code",
        "replacement_code",
    }
)
_REGISTRY_FIELDS: Final = frozenset(
    {
        "annotation_id",
        "document_id",
        "evidence_ids",
        "label",
        "record_id",
        "registry_id",
    }
)


def import_fact_correction_rows(
    rows: Iterable[Mapping[str, Any]], *, namespace: str = "default"
) -> AnnotationEnvelope:
    """Import strict, source-text-free fact-correction rows."""

    return _import_rows(
        rows,
        namespace=namespace,
        annotation_type=AnnotationType.FACT_CORRECTION,
        fields=_CORRECTION_FIELDS,
        result_fields=frozenset(
            {"evidence_ids", "fact_id", "field", "reason_code", "replacement_code"}
        ),
    )


def import_registry_label_rows(
    rows: Iterable[Mapping[str, Any]], *, namespace: str = "default"
) -> AnnotationEnvelope:
    """Import strict, source-text-free registry-label rows."""

    return _import_rows(
        rows,
        namespace=namespace,
        annotation_type=AnnotationType.REGISTRY_LABEL,
        fields=_REGISTRY_FIELDS,
        result_fields=frozenset({"evidence_ids", "label", "record_id", "registry_id"}),
    )


def export_fact_correction_rows(
    envelope: AnnotationEnvelope,
) -> tuple[dict[str, Any], ...]:
    """Export fact corrections to neutral tool rows."""

    return _export_rows(envelope, AnnotationType.FACT_CORRECTION)


def export_registry_label_rows(
    envelope: AnnotationEnvelope,
) -> tuple[dict[str, Any], ...]:
    """Export registry labels to neutral tool rows."""

    return _export_rows(envelope, AnnotationType.REGISTRY_LABEL)


def _import_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    namespace: str,
    annotation_type: AnnotationType,
    fields: frozenset[str],
    result_fields: frozenset[str],
) -> AnnotationEnvelope:
    records: list[AnnotationRecord] = []
    for index, row in enumerate(rows, start=1):
        if index > MAX_ANNOTATION_ROWS:
            raise AnnotationInterchangeError("annotation row limit exceeded")
        if not isinstance(row, Mapping) or set(row) != fields:
            raise AnnotationInterchangeError("annotation tool row fields are invalid")
        result = {key: row[key] for key in result_fields}
        records.append(
            AnnotationRecord(
                annotation_id=row["annotation_id"],
                document_id=row["document_id"],
                namespace=namespace,
                annotation_type=annotation_type,
                coordinate_convention=CoordinateConvention.NONE,
                start=None,
                end=None,
                state=AnnotationState.SUCCESS,
                result=result,
                metadata={"source_format": "annotation_tool"},
            )
        )
    return build_annotation_envelope(records)


def _export_rows(
    envelope: AnnotationEnvelope, annotation_type: AnnotationType
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for record in envelope.records:
        if record.annotation_type is not annotation_type:
            continue
        rows.append(
            {
                "annotation_id": record.annotation_id,
                "document_id": record.document_id,
                **{
                    key: list(value) if isinstance(value, tuple) else value
                    for key, value in record.result.items()
                },
            }
        )
    return tuple(rows)


__all__ = [
    "export_fact_correction_rows",
    "export_registry_label_rows",
    "import_fact_correction_rows",
    "import_registry_label_rows",
]
