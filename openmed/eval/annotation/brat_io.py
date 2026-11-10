"""BRAT text-bound standoff readers and writers for ``OpenMedSpan``."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from openmed.core.schemas import OpenMedSpan
from openmed.eval.annotation.toolkit import (
    AnnotationIssue,
    AnnotationTask,
    AnnotationValidationError,
    span_from_offsets,
    validate_spans,
)


def parse_brat(
    text: str,
    annotations: str,
    *,
    doc_id: str,
    hash_secret: str | bytes,
) -> tuple[OpenMedSpan, ...]:
    """Parse BRAT text-bound annotations against exact source text.

    The supported BRAT subset is one continuous text-bound annotation per
    ``T`` record. Relations, attributes, events, and discontinuous spans are
    rejected with line-specific guidance because they cannot map losslessly to
    one ``OpenMedSpan``.

    Args:
        text: Exact contents of the paired ``.txt`` file.
        annotations: Contents of the paired ``.ann`` file.
        doc_id: Document identifier stored on imported spans.
        hash_secret: Key used to HMAC imported span surfaces.

    Returns:
        Imported spans in deterministic offset order.

    Raises:
        AnnotationValidationError: If any standoff record is malformed.
    """

    if not doc_id:
        raise AnnotationValidationError(
            AnnotationIssue("doc_id must be non-empty"),
            format_name="BRAT standoff",
        )
    issues: list[AnnotationIssue] = []
    spans: list[OpenMedSpan] = []
    annotation_ids: set[str] = set()

    for line_number, line in enumerate(annotations.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split("\t", 2)
        annotation_id = fields[0].strip() if fields else ""
        if len(fields) != 3:
            issues.append(
                AnnotationIssue(
                    "expected three tab-separated fields: ID, label offsets, text",
                    line=line_number,
                    annotation_id=annotation_id or None,
                )
            )
            continue
        if not annotation_id.startswith("T") or not annotation_id[1:].isdigit():
            issues.append(
                AnnotationIssue(
                    "unsupported record; only text-bound IDs such as T1 are allowed",
                    line=line_number,
                    annotation_id=annotation_id or None,
                )
            )
            continue
        if annotation_id in annotation_ids:
            issues.append(
                AnnotationIssue(
                    "duplicate annotation ID",
                    line=line_number,
                    annotation_id=annotation_id,
                )
            )
            continue
        annotation_ids.add(annotation_id)

        descriptor = fields[1].strip()
        if ";" in descriptor:
            issues.append(
                AnnotationIssue(
                    "discontinuous spans are not supported; split into continuous spans",
                    line=line_number,
                    annotation_id=annotation_id,
                )
            )
            continue
        parts = descriptor.split()
        if len(parts) != 3:
            issues.append(
                AnnotationIssue(
                    "expected 'LABEL START END' in the second field",
                    line=line_number,
                    annotation_id=annotation_id,
                )
            )
            continue
        label, raw_start, raw_end = parts
        try:
            start, end = int(raw_start), int(raw_end)
        except ValueError:
            issues.append(
                AnnotationIssue(
                    "start and end offsets must be integers",
                    line=line_number,
                    annotation_id=annotation_id,
                )
            )
            continue
        if start < 0 or end > len(text) or start >= end:
            issues.append(
                AnnotationIssue(
                    f"offsets {start}:{end} must satisfy "
                    f"0 <= start < end <= {len(text)}",
                    line=line_number,
                    annotation_id=annotation_id,
                )
            )
            continue
        expected_surface = _brat_surface(text[start:end])
        if fields[2] != expected_surface:
            issues.append(
                AnnotationIssue(
                    f"annotated text does not match source offsets {start}:{end}",
                    line=line_number,
                    annotation_id=annotation_id,
                )
            )
            continue
        try:
            spans.append(
                span_from_offsets(
                    doc_id=doc_id,
                    text=text,
                    start=start,
                    end=end,
                    label=label,
                    hash_secret=hash_secret,
                    entity_type=label,
                    metadata={
                        "annotation_format": "brat",
                        "brat_id": annotation_id,
                    },
                )
            )
        except AnnotationValidationError as exc:
            issues.extend(
                AnnotationIssue(
                    issue.message,
                    line=line_number,
                    annotation_id=annotation_id,
                )
                for issue in exc.issues
            )

    if issues:
        raise AnnotationValidationError(issues, format_name="BRAT standoff")
    try:
        return validate_spans(text, spans, doc_id=doc_id, hash_secret=hash_secret)
    except AnnotationValidationError as exc:
        raise AnnotationValidationError(
            exc.issues, format_name="BRAT standoff"
        ) from exc


def format_brat(text: str, spans: Iterable[OpenMedSpan]) -> str:
    """Serialize continuous spans as deterministic BRAT ``T`` records.

    Args:
        text: Exact source text whose offsets the spans reference.
        spans: Canonical spans from one document.

    Returns:
        UTF-8-ready contents for a BRAT ``.ann`` file.

    Raises:
        AnnotationValidationError: If a span is malformed or belongs to a
            different document.
    """

    materialized = tuple(spans)
    doc_id = materialized[0].doc_id if materialized else None
    ordered = validate_spans(text, materialized, doc_id=doc_id)
    lines = [
        f"T{index}\t{span.canonical_label} {span.start} {span.end}\t"
        f"{_brat_surface(text[span.start : span.end])}"
        for index, span in enumerate(ordered, start=1)
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def read_brat(
    text_path: str | Path,
    ann_path: str | Path | None = None,
    *,
    doc_id: str | None = None,
    hash_secret: str | bytes,
    synthetic: bool = False,
) -> AnnotationTask:
    """Read paired BRAT ``.txt`` and ``.ann`` files as an annotation task.

    Args:
        text_path: Path to the source ``.txt`` file.
        ann_path: Optional explicit ``.ann`` path; defaults beside ``text_path``.
        doc_id: Optional document ID; defaults to the text file stem.
        hash_secret: Key used to HMAC imported span surfaces.
        synthetic: Whether the caller attests that the source is synthetic.

    Returns:
        The validated source document and imported spans.
    """

    resolved_text_path = Path(text_path)
    resolved_ann_path = (
        Path(ann_path)
        if ann_path is not None
        else resolved_text_path.with_suffix(".ann")
    )
    text = _read_exact(resolved_text_path)
    annotations = _read_exact(resolved_ann_path)
    resolved_doc_id = doc_id or resolved_text_path.stem
    spans = parse_brat(
        text,
        annotations,
        doc_id=resolved_doc_id,
        hash_secret=hash_secret,
    )
    return AnnotationTask(
        doc_id=resolved_doc_id,
        text=text,
        spans=spans,
        synthetic=synthetic,
        metadata={"annotation_format": "brat", "synthetic": synthetic},
    )


def write_brat(
    text_path: str | Path,
    text: str,
    spans: Iterable[OpenMedSpan],
    *,
    ann_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write paired BRAT files and return their resolved paths.

    Args:
        text_path: Destination for the source ``.txt`` file.
        text: Exact source text.
        spans: Canonical spans from one document.
        ann_path: Optional explicit ``.ann`` destination.

    Returns:
        The resolved text and annotation paths, in that order.
    """

    resolved_text_path = Path(text_path)
    resolved_ann_path = (
        Path(ann_path)
        if ann_path is not None
        else resolved_text_path.with_suffix(".ann")
    )
    annotations = format_brat(text, spans)
    resolved_text_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_ann_path.parent.mkdir(parents=True, exist_ok=True)
    _write_exact(resolved_text_path, text)
    _write_exact(resolved_ann_path, annotations)
    return resolved_text_path, resolved_ann_path


def _brat_surface(text: str) -> str:
    return text.replace("\r", " ").replace("\n", " ")


def _read_exact(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def _write_exact(path: Path, contents: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(contents)


__all__ = ["format_brat", "parse_brat", "read_brat", "write_brat"]
