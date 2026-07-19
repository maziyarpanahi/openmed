"""Shared contracts for synthetic annotation tasks and annotation imports."""

from __future__ import annotations

import hmac
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from openmed.core.labels import OTHER, normalize_label
from openmed.core.schemas import OpenMedSpan, hmac_text_hash


@dataclass(frozen=True)
class AnnotationIssue:
    """One actionable problem found while reading annotation data."""

    message: str
    line: int | None = None
    annotation_id: str | None = None

    def render(self) -> str:
        """Return a concise location-prefixed description."""

        location: list[str] = []
        if self.line is not None:
            location.append(f"line {self.line}")
        if self.annotation_id is not None:
            location.append(f"annotation {self.annotation_id}")
        prefix = ", ".join(location)
        return f"{prefix}: {self.message}" if prefix else self.message


class AnnotationValidationError(ValueError):
    """Raised with one or more actionable annotation validation issues."""

    def __init__(
        self,
        issues: AnnotationIssue | Iterable[AnnotationIssue],
        *,
        format_name: str = "annotation",
    ) -> None:
        if isinstance(issues, AnnotationIssue):
            normalized = (issues,)
        else:
            normalized = tuple(issues)
        if not normalized:
            normalized = (AnnotationIssue("validation failed"),)
        self.issues = normalized
        self.format_name = format_name
        details = "\n".join(f"- {issue.render()}" for issue in normalized)
        super().__init__(f"Invalid {format_name} data:\n{details}")


@dataclass(frozen=True)
class AnnotationTask:
    """A source document and its pre-labels for human annotation review."""

    doc_id: str
    text: str
    spans: Sequence[OpenMedSpan] = field(default_factory=tuple)
    synthetic: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.doc_id:
            raise AnnotationValidationError(
                AnnotationIssue("doc_id must be non-empty"),
                format_name="annotation task",
            )
        if not self.text:
            raise AnnotationValidationError(
                AnnotationIssue("source text must be non-empty"),
                format_name="annotation task",
            )
        if not isinstance(self.metadata, Mapping):
            raise AnnotationValidationError(
                AnnotationIssue("metadata must be a mapping"),
                format_name="annotation task",
            )
        validated = validate_spans(self.text, self.spans, doc_id=self.doc_id)
        object.__setattr__(self, "spans", validated)
        object.__setattr__(self, "metadata", dict(self.metadata))


SpanProposal = OpenMedSpan | Mapping[str, Any]
Prelabeler = Callable[[str], Iterable[SpanProposal]]


def validate_spans(
    text: str,
    spans: Iterable[OpenMedSpan],
    *,
    doc_id: str | None = None,
    hash_secret: str | bytes | None = None,
    allow_overlap: bool = True,
) -> tuple[OpenMedSpan, ...]:
    """Validate span bounds, document identity, hashes, and overlap policy.

    Args:
        text: Exact source text whose character offsets the spans reference.
        spans: Canonical spans to validate.
        doc_id: Optional expected document identifier.
        hash_secret: Optional key used to verify each span's surface HMAC.
        allow_overlap: Whether overlapping and nested spans are permitted.

    Returns:
        The spans in deterministic offset and label order.

    Raises:
        AnnotationValidationError: If one or more spans are malformed.
    """

    if hash_secret is not None:
        _require_hash_secret(hash_secret)
    materialized = tuple(spans)
    type_issues = [
        AnnotationIssue(
            f"span {index} must be an OpenMedSpan, got {type(span).__name__}"
        )
        for index, span in enumerate(materialized, start=1)
        if not isinstance(span, OpenMedSpan)
    ]
    if type_issues:
        raise AnnotationValidationError(type_issues)
    ordered = tuple(
        sorted(
            materialized,
            key=lambda span: (span.start, span.end, span.canonical_label),
        )
    )
    issues: list[AnnotationIssue] = []
    seen: set[tuple[int, int, str]] = set()
    previous: OpenMedSpan | None = None

    for index, span in enumerate(ordered, start=1):
        location = f"span {index} ({span.start}:{span.end})"
        if doc_id is not None and span.doc_id != doc_id:
            issues.append(
                AnnotationIssue(f"{location} belongs to a different document")
            )
        if span.start < 0 or span.end > len(text) or span.start >= span.end:
            issues.append(
                AnnotationIssue(
                    f"{location} must satisfy 0 <= start < end <= {len(text)}"
                )
            )
            continue
        identity = (span.start, span.end, span.canonical_label)
        if identity in seen:
            issues.append(AnnotationIssue(f"duplicate {location} and label"))
        seen.add(identity)
        if not allow_overlap and previous is not None and span.start < previous.end:
            issues.append(
                AnnotationIssue(
                    f"{location} overlaps the preceding span "
                    f"({previous.start}:{previous.end}); this format supports "
                    "one label per token"
                )
            )
        if previous is None or span.end > previous.end:
            previous = span
        if hash_secret is not None:
            expected_hash = hmac_text_hash(text[span.start : span.end], hash_secret)
            if not hmac.compare_digest(span.text_hash, expected_hash):
                issues.append(
                    AnnotationIssue(
                        f"{location} text_hash does not match the source text"
                    )
                )

    if issues:
        raise AnnotationValidationError(issues)
    return ordered


def span_from_offsets(
    *,
    doc_id: str,
    text: str,
    start: int,
    end: int,
    label: str,
    hash_secret: str | bytes,
    entity_type: str | None = None,
    score: float | None = None,
    detector: str | None = "annotation_import",
    metadata: Mapping[str, Any] | None = None,
) -> OpenMedSpan:
    """Build a canonical span from validated source-text offsets.

    Unknown external labels are rejected instead of silently mapping to
    ``OTHER`` so annotation mistakes remain visible to reviewers.

    Args:
        doc_id: Document identifier stored on the span.
        text: Exact source text whose offsets are being imported.
        start: Inclusive character offset.
        end: Exclusive character offset.
        label: External or canonical entity label.
        hash_secret: Key used to HMAC the source surface.
        entity_type: Optional original entity type; defaults to ``label``.
        score: Optional pre-labeler confidence.
        detector: Optional source recorded on the span.
        metadata: Optional non-PHI provenance metadata.

    Returns:
        A canonical span whose hash matches the referenced source surface.

    Raises:
        AnnotationValidationError: If offsets or labels are invalid.
    """

    _require_hash_secret(hash_secret)
    if not doc_id:
        raise AnnotationValidationError(AnnotationIssue("doc_id must be non-empty"))
    if not isinstance(start, int) or not isinstance(end, int):
        raise AnnotationValidationError(
            AnnotationIssue("span start and end must be integers")
        )
    if start < 0 or end > len(text) or start >= end:
        raise AnnotationValidationError(
            AnnotationIssue(
                f"span {start}:{end} must satisfy 0 <= start < end <= {len(text)}"
            )
        )
    source_label = label.strip()
    if not source_label:
        raise AnnotationValidationError(AnnotationIssue("span label must be non-empty"))
    canonical_label = normalize_label(source_label)
    if canonical_label == OTHER and source_label.upper() != OTHER:
        raise AnnotationValidationError(
            AnnotationIssue(
                f"unknown label {source_label!r}; use an OpenMed canonical label"
            )
        )
    try:
        return OpenMedSpan(
            doc_id=doc_id,
            start=start,
            end=end,
            text_hash=hmac_text_hash(text[start:end], hash_secret),
            entity_type=entity_type or source_label,
            canonical_label=canonical_label,
            score=score,
            detector=detector,
            metadata=dict(metadata or {}),
        )
    except (TypeError, ValueError) as exc:
        raise AnnotationValidationError(AnnotationIssue(str(exc))) from exc


def generate_synthetic_annotation_task(
    doc_id: str,
    text: str,
    *,
    hash_secret: str | bytes,
    spans: Iterable[SpanProposal] | None = None,
    prelabeler: Prelabeler | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> AnnotationTask:
    """Generate a synthetic, pre-labeled task for human correction.

    Exactly one of ``spans`` and ``prelabeler`` may be supplied. Mapping-based
    proposals need ``start``, ``end``, and one of ``canonical_label``, ``label``,
    or ``entity_type``. Existing ``OpenMedSpan`` records are preserved.

    Args:
        doc_id: Stable synthetic document identifier.
        text: Synthetic source text to annotate.
        hash_secret: Key used to HMAC proposal surfaces.
        spans: Optional precomputed span proposals.
        prelabeler: Optional callback that returns proposals for ``text``.
        metadata: Optional task metadata.

    Returns:
        A synthetic annotation task marked as needing human review.
    """

    if spans is not None and prelabeler is not None:
        raise AnnotationValidationError(
            AnnotationIssue("provide spans or prelabeler, not both"),
            format_name="annotation proposal",
        )
    proposals = prelabeler(text) if prelabeler is not None else (spans or ())
    normalized: list[OpenMedSpan] = []
    issues: list[AnnotationIssue] = []

    for index, proposal in enumerate(proposals, start=1):
        if isinstance(proposal, OpenMedSpan):
            normalized.append(proposal)
            continue
        if not isinstance(proposal, Mapping):
            issues.append(
                AnnotationIssue(f"proposal {index} must be an OpenMedSpan or mapping")
            )
            continue
        try:
            start = int(proposal["start"])
            end = int(proposal["end"])
        except (KeyError, TypeError, ValueError):
            issues.append(
                AnnotationIssue(
                    f"proposal {index} requires integer start and end offsets"
                )
            )
            continue
        label = str(
            proposal.get("canonical_label")
            or proposal.get("label")
            or proposal.get("entity_type")
            or ""
        )
        try:
            normalized.append(
                span_from_offsets(
                    doc_id=doc_id,
                    text=text,
                    start=start,
                    end=end,
                    label=label,
                    hash_secret=hash_secret,
                    entity_type=str(proposal.get("entity_type") or label),
                    score=proposal.get("score"),
                    detector=str(proposal.get("detector") or "annotation_prelabeler"),
                    metadata=proposal.get("metadata") or {},
                )
            )
        except AnnotationValidationError as exc:
            issues.extend(
                AnnotationIssue(f"proposal {index}: {issue.message}")
                for issue in exc.issues
            )

    if issues:
        raise AnnotationValidationError(issues, format_name="annotation proposal")
    task_metadata = {
        **dict(metadata or {}),
        "prelabeled": bool(normalized),
        "review_status": "needs_human_review",
        "synthetic": True,
    }
    return AnnotationTask(
        doc_id=doc_id,
        text=text,
        spans=validate_spans(
            text,
            normalized,
            doc_id=doc_id,
            hash_secret=hash_secret,
        ),
        synthetic=True,
        metadata=task_metadata,
    )


generate_annotation_task = generate_synthetic_annotation_task


def _require_hash_secret(hash_secret: str | bytes) -> None:
    if not isinstance(hash_secret, (str, bytes)):
        raise AnnotationValidationError(
            AnnotationIssue("hash_secret must be a string or bytes")
        )
    if not hash_secret:
        raise AnnotationValidationError(
            AnnotationIssue("hash_secret must be non-empty")
        )


__all__ = [
    "AnnotationIssue",
    "AnnotationTask",
    "AnnotationValidationError",
    "Prelabeler",
    "SpanProposal",
    "generate_annotation_task",
    "generate_synthetic_annotation_task",
    "span_from_offsets",
    "validate_spans",
]
