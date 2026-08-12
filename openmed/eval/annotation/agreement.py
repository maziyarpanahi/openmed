"""Privacy-safe inter-annotator agreement for character-offset spans.

The scorer treats the union of annotation offsets as the item universe.  A
missing annotation is an explicit category, which means span-presence
disagreements contribute to kappa just like label disagreements.  Only
offsets and labels are retained in reports; source text and arbitrary input
metadata are deliberately ignored.
"""

from __future__ import annotations

import operator
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

MatchMode = Literal["exact", "overlap"]

_MISSING = object()
_SCHEMA_VERSION = "openmed.eval.annotation.agreement.v1"


@dataclass(frozen=True, slots=True)
class Annotation:
    """A minimal annotation record containing only an offset and label."""

    start: int
    end: int
    label: str

    def __post_init__(self) -> None:
        """Validate the offset contract and normalize surrounding whitespace."""

        if isinstance(self.start, bool) or not isinstance(self.start, int):
            raise TypeError("annotation start must be an integer")
        if isinstance(self.end, bool) or not isinstance(self.end, int):
            raise TypeError("annotation end must be an integer")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("annotation offsets must satisfy 0 <= start < end")
        if not isinstance(self.label, str):
            raise TypeError("annotation label must be a string")
        label = self.label.strip()
        if not label:
            raise ValueError("annotation label must be non-empty")
        object.__setattr__(self, "label", label)


# Descriptive aliases make the small record convenient for callers while
# keeping one canonical representation in the report implementation.
AnnotationSpan = Annotation
SpanAnnotation = Annotation


@dataclass(frozen=True, slots=True)
class _AlignedAnnotations:
    offsets: tuple[tuple[int, int], ...]
    rows: tuple[tuple[Any, ...], ...]
    labels: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class InterAnnotatorAgreement:
    """Agreement report for two or more annotators.

    ``cohen_kappa`` is populated for two annotators and ``fleiss_kappa`` for
    three or more.  ``observed_agreement`` and ``expected_agreement`` expose
    the proportions used by the selected kappa calculation.  Disagreement
    entries contain only ``offset`` and ``labels`` and are therefore safe to
    pass to a review or active-learning queue.
    """

    n_annotators: int
    n_items: int
    cohen_kappa: float | None
    fleiss_kappa: float | None
    observed_agreement: float
    expected_agreement: float
    mean_span_f1: float
    per_label: Mapping[str, float]
    per_relation: Mapping[str, float]
    disagreements: tuple[Mapping[str, Any], ...]
    match: MatchMode = "exact"

    @property
    def kappa(self) -> float:
        """Return the applicable Cohen or Fleiss kappa value."""

        if self.cohen_kappa is not None:
            return self.cohen_kappa
        if self.fleiss_kappa is not None:
            return self.fleiss_kappa
        return 1.0

    @property
    def agreement(self) -> float:
        """Return the overall reliability score used for queue triage."""

        return self.kappa

    @property
    def overall_agreement(self) -> float:
        """Compatibility name for the overall kappa score."""

        return self.kappa

    @property
    def label_agreement(self) -> Mapping[str, float]:
        """Return the per-label observed-agreement breakdown."""

        return self.per_label

    def to_active_learning_queue(self) -> tuple[Mapping[str, Any], ...]:
        """Return PHI-free disagreement candidates for the labeling queue.

        The queue accepts ``start``, ``end``, and ``label`` fields.  When
        several labels are present, the remaining labels are retained as
        ``matched_label`` so label-confusion ranking can use them without
        exposing source text.
        """

        candidates: list[Mapping[str, Any]] = []
        for item in self.disagreements:
            start, end = item["offset"]
            labels = tuple(item["labels"])
            if not labels:
                continue
            candidate: dict[str, Any] = {
                "end": end,
                "kind": "annotator_disagreement",
                "label": labels[0],
                "start": start,
                "uncertainty": 1.0,
            }
            if len(labels) > 1:
                candidate["matched_label"] = labels[1]
            candidates.append(candidate)
        return tuple(candidates)

    @property
    def queue_items(self) -> tuple[Mapping[str, Any], ...]:
        """Return the active-learning queue view of disagreement examples."""

        return self.to_active_learning_queue()

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible, raw-text-free report."""

        return {
            "schema_version": _SCHEMA_VERSION,
            "match": self.match,
            "n_annotators": self.n_annotators,
            "n_items": self.n_items,
            "cohen_kappa": self.cohen_kappa,
            "fleiss_kappa": self.fleiss_kappa,
            "kappa": self.kappa,
            "overall_agreement": self.overall_agreement,
            "observed_agreement": self.observed_agreement,
            "expected_agreement": self.expected_agreement,
            "mean_span_f1": self.mean_span_f1,
            "per_label": dict(self.per_label),
            "per_relation": dict(self.per_relation),
            "disagreements": [
                {
                    "offset": list(item["offset"]),
                    "labels": list(item["labels"]),
                }
                for item in self.disagreements
            ],
            "active_learning_queue": [
                dict(item) for item in self.to_active_learning_queue()
            ],
        }


# ``AgreementReport`` is the task-facing name; the longer name remains useful
# to callers that already use the evaluation metrics terminology.
AgreementReport = InterAnnotatorAgreement


def _coerce_offset(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"annotation {field} must be an integer")
    try:
        offset = operator.index(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"annotation {field} must be an integer") from exc
    if offset < 0:
        raise ValueError("annotation offsets must be non-negative")
    return int(offset)


def _coerce_annotation(value: Any) -> Annotation:
    """Coerce tuples, mappings, or span-like objects without retaining text."""

    if isinstance(value, Annotation):
        return value

    if isinstance(value, Mapping):
        start = value.get("start")
        end = value.get("end")
        label = value.get("label")
        if label is None:
            label = value.get("canonical_label")
        if label is None:
            label = value.get("entity_type")
    elif hasattr(value, "start") and hasattr(value, "end"):
        start = getattr(value, "start")
        end = getattr(value, "end")
        label = getattr(value, "label", None)
        if label is None:
            label = getattr(value, "canonical_label", None)
        if label is None:
            label = getattr(value, "entity_type", None)
    else:
        try:
            start, end, label = value
        except (TypeError, ValueError) as exc:
            raise TypeError("annotations must be (start, end, label) records") from exc

    if start is None or end is None or label is None:
        raise ValueError("annotations require start, end, and label")
    return Annotation(
        start=_coerce_offset(start, "start"),
        end=_coerce_offset(end, "end"),
        label=str(label),
    )


def _normalize_annotations(annotations: Iterable[Any]) -> tuple[Annotation, ...]:
    normalized = tuple(_coerce_annotation(annotation) for annotation in annotations)
    seen: set[tuple[int, int]] = set()
    for annotation in normalized:
        key = (annotation.start, annotation.end)
        if key in seen:
            raise ValueError(
                "an annotator cannot contain duplicate spans at the same offset"
            )
        seen.add(key)
    return normalized


def _validate_match(match: str) -> MatchMode:
    if match not in {"exact", "overlap"}:
        raise ValueError("match must be either 'exact' or 'overlap'")
    return match  # type: ignore[return-value]


def _overlaps(left: Annotation, right: Annotation) -> bool:
    return left.start < right.end and right.start < left.end


def _row_sort_key(row: Sequence[Any]) -> tuple[tuple[int, str], ...]:
    values: list[tuple[int, str]] = []
    for category in row:
        if category is _MISSING:
            values.append((0, ""))
        elif isinstance(category, tuple):
            values.append((2, "|".join(category)))
        else:
            values.append((1, str(category)))
    return tuple(values)


def _category(labels: Sequence[str]) -> Any:
    unique = tuple(sorted(set(labels)))
    if len(unique) == 1:
        return unique[0]
    return ("multiple", *unique)


def _align_exact(annotators: Sequence[Sequence[Annotation]]) -> _AlignedAnnotations:
    maps: list[dict[tuple[int, int], str]] = []
    for annotator in annotators:
        maps.append({(span.start, span.end): span.label for span in annotator})

    offsets = tuple(sorted({offset for mapping in maps for offset in mapping}))
    rows = tuple(
        tuple(mapping.get(offset, _MISSING) for mapping in maps) for offset in offsets
    )
    labels = tuple(
        tuple(sorted({category for category in row if category is not _MISSING}))
        for row in rows
    )
    return _AlignedAnnotations(offsets=offsets, rows=rows, labels=labels)


def _align_overlap(annotators: Sequence[Sequence[Annotation]]) -> _AlignedAnnotations:
    records = [
        (annotator_index, span)
        for annotator_index, annotations in enumerate(annotators)
        for span in annotations
    ]
    if not records:
        return _AlignedAnnotations(offsets=(), rows=(), labels=())

    parents = list(range(len(records)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left_index, (left_annotator, left_span) in enumerate(records):
        for right_index in range(left_index + 1, len(records)):
            right_annotator, right_span = records[right_index]
            if left_annotator != right_annotator and _overlaps(left_span, right_span):
                union(left_index, right_index)

    groups: dict[int, list[tuple[int, Annotation]]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[find(index)].append(record)

    aligned: list[tuple[tuple[int, int], tuple[Any, ...], tuple[str, ...]]] = []
    for group in groups.values():
        by_annotator: dict[int, list[Annotation]] = defaultdict(list)
        for annotator_index, span in group:
            by_annotator[annotator_index].append(span)

        row: list[Any] = []
        group_labels = tuple(sorted({span.label for _, span in group}))
        for annotator_index in range(len(annotators)):
            row.append(
                _category(
                    [span.label for span in by_annotator.get(annotator_index, ())]
                )
                if annotator_index in by_annotator
                else _MISSING
            )
        offset = (
            min(span.start for _, span in group),
            max(span.end for _, span in group),
        )
        aligned.append((offset, tuple(row), group_labels))

    aligned.sort(key=lambda item: (item[0], item[2], _row_sort_key(item[1])))
    return _AlignedAnnotations(
        offsets=tuple(item[0] for item in aligned),
        rows=tuple(item[1] for item in aligned),
        labels=tuple(item[2] for item in aligned),
    )


def _align(
    annotators: Sequence[Sequence[Annotation]],
    *,
    match: MatchMode,
) -> _AlignedAnnotations:
    return _align_exact(annotators) if match == "exact" else _align_overlap(annotators)


def _kappa(observed: float, expected: float) -> float:
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def _cohen_values(rows: Sequence[Sequence[Any]]) -> tuple[float, float, float]:
    if not rows:
        return 1.0, 1.0, 1.0

    total = len(rows)
    observed = sum(row[0] == row[1] for row in rows) / total
    first = defaultdict(int)
    second = defaultdict(int)
    for first_category, second_category in rows:
        first[first_category] += 1
        second[second_category] += 1
    expected = sum(
        first[category] * second[category] for category in set(first) | set(second)
    ) / (total * total)
    return observed, expected, _kappa(observed, expected)


def _fleiss_values(
    rows: Sequence[Sequence[Any]], n_annotators: int
) -> tuple[float, float, float]:
    if not rows:
        return 1.0, 1.0, 1.0

    n_items = len(rows)
    category_totals = defaultdict(int)
    item_agreements: list[float] = []
    for row in rows:
        counts = defaultdict(int)
        for category in row:
            counts[category] += 1
            category_totals[category] += 1
        item_agreements.append(
            (sum(count * count for count in counts.values()) - n_annotators)
            / (n_annotators * (n_annotators - 1))
        )

    observed = sum(item_agreements) / n_items
    expected = sum(
        (count / (n_items * n_annotators)) ** 2 for count in category_totals.values()
    )
    return observed, expected, _kappa(observed, expected)


def _materialize(
    annotators: Iterable[Iterable[Any]],
) -> tuple[tuple[Annotation, ...], ...]:
    materialized = tuple(_normalize_annotations(annotator) for annotator in annotators)
    if len(materialized) < 2:
        raise ValueError("agreement requires at least two annotators")
    return materialized


def cohen_kappa(
    annotator_a: Iterable[Any],
    annotator_b: Iterable[Any],
    *,
    match: MatchMode = "exact",
) -> float:
    """Return Cohen kappa for exactly two offset-based annotators."""

    _validate_match(match)
    annotators = (
        _normalize_annotations(annotator_a),
        _normalize_annotations(annotator_b),
    )
    return _cohen_values(_align(annotators, match=match).rows)[2]


def fleiss_kappa(
    annotators: Iterable[Iterable[Any]],
    *,
    match: MatchMode = "exact",
) -> float:
    """Return Fleiss kappa for at least three offset-based annotators."""

    _validate_match(match)
    materialized = tuple(_normalize_annotations(annotator) for annotator in annotators)
    if len(materialized) < 3:
        raise ValueError("fleiss_kappa requires at least three annotators")
    return _fleiss_values(
        _align(materialized, match=match).rows,
        len(materialized),
    )[2]


def _span_f1(
    left: Sequence[Annotation],
    right: Sequence[Annotation],
    *,
    match: MatchMode,
) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0

    unmatched = set(range(len(right)))
    true_positives = 0
    for left_span in sorted(left, key=lambda span: (span.start, span.end, span.label)):
        candidates = [
            index
            for index in unmatched
            if right[index].label == left_span.label
            and (
                (left_span.start, left_span.end)
                == (right[index].start, right[index].end)
                if match == "exact"
                else _overlaps(left_span, right[index])
            )
        ]
        if candidates:
            selected = max(
                candidates,
                key=lambda index: (
                    min(left_span.end, right[index].end)
                    - max(left_span.start, right[index].start),
                    -right[index].start,
                    -right[index].end,
                ),
            )
            unmatched.remove(selected)
            true_positives += 1
    return (2.0 * true_positives) / (len(left) + len(right))


def _mean_pairwise_span_f1(
    annotators: Sequence[Sequence[Annotation]],
    *,
    match: MatchMode,
) -> float:
    scores = [
        _span_f1(annotators[left], annotators[right], match=match)
        for left in range(len(annotators))
        for right in range(left + 1, len(annotators))
    ]
    return sum(scores) / len(scores) if scores else 1.0


def _per_label(
    aligned: _AlignedAnnotations,
) -> dict[str, float]:
    scores: dict[str, list[float]] = defaultdict(list)
    for row, labels in zip(aligned.rows, aligned.labels):
        if not labels:
            continue
        agreed = all(category == row[0] for category in row)
        for label in labels:
            scores[label].append(1.0 if agreed else 0.0)
    return {
        label: sum(values) / len(values) for label, values in sorted(scores.items())
    }


def _per_relation(
    relations: Sequence[Mapping[str, Iterable[Any]]],
    *,
    match: MatchMode,
) -> dict[str, float]:
    relation_types = sorted(
        {relation_type for mapping in relations for relation_type in mapping}
    )
    result: dict[str, float] = {}
    for relation_type in relation_types:
        relation_sets = tuple(
            _normalize_annotations(mapping.get(relation_type, ()))
            for mapping in relations
        )
        aligned = _align(relation_sets, match=match)
        result[relation_type] = _cohen_or_fleiss_observed(
            aligned.rows,
            len(relation_sets),
        )
    return result


def _cohen_or_fleiss_observed(
    rows: Sequence[Sequence[Any]], n_annotators: int
) -> float:
    if n_annotators == 2:
        return _cohen_values(rows)[0]
    return _fleiss_values(rows, n_annotators)[0]


def inter_annotator_agreement(
    annotators: Iterable[Iterable[Any]],
    *,
    match: MatchMode = "exact",
    relations: Sequence[Mapping[str, Iterable[Any]]] | None = None,
) -> InterAnnotatorAgreement:
    """Build an agreement report for two or more annotation sets.

    Args:
        annotators: One iterable of ``(start, end, label)`` records per rater.
            Existing span-like objects with ``start``, ``end``, and ``label``
            or ``canonical_label`` attributes are accepted too.
        match: ``"exact"`` aligns identical offsets; ``"overlap"`` aligns
            positive-overlap spans into deterministic offset components.
        relations: Optional relation-type mappings, one mapping per annotator.
            Relation values use the same span record contract and are reported
            as observed agreement by relation type.

    Returns:
        A deterministic, raw-text-free report suitable for JSON serialization
        or conversion to active-learning queue candidates.
    """

    _validate_match(match)
    materialized = _materialize(annotators)
    if relations is not None:
        if len(relations) != len(materialized):
            raise ValueError("relations must contain one mapping per annotator")
        if any(not isinstance(mapping, Mapping) for mapping in relations):
            raise TypeError("relations must contain mappings")

    aligned = _align(materialized, match=match)
    if len(materialized) == 2:
        observed, expected, cohen = _cohen_values(aligned.rows)
        fleiss = None
    else:
        observed, expected, fleiss = _fleiss_values(
            aligned.rows,
            len(materialized),
        )
        cohen = None

    disagreements = tuple(
        {
            "offset": offset,
            "labels": labels,
        }
        for offset, row, labels in zip(
            aligned.offsets,
            aligned.rows,
            aligned.labels,
        )
        if any(category != row[0] for category in row)
    )

    return InterAnnotatorAgreement(
        n_annotators=len(materialized),
        n_items=len(aligned.offsets),
        cohen_kappa=cohen,
        fleiss_kappa=fleiss,
        observed_agreement=observed,
        expected_agreement=expected,
        mean_span_f1=_mean_pairwise_span_f1(materialized, match=match),
        per_label=_per_label(aligned),
        per_relation=(
            _per_relation(relations, match=match) if relations is not None else {}
        ),
        disagreements=disagreements,
        match=match,
    )


def agreement_report(
    annotators: Iterable[Iterable[Any]],
    *,
    match: MatchMode = "exact",
    relations: Sequence[Mapping[str, Iterable[Any]]] | None = None,
) -> InterAnnotatorAgreement:
    """Alias for :func:`inter_annotator_agreement` with a report-oriented name."""

    return inter_annotator_agreement(
        annotators,
        match=match,
        relations=relations,
    )


def cohen_kappa_agreement(
    *annotators: Iterable[Any],
    match: MatchMode = "exact",
) -> float:
    """Compatibility wrapper for :func:`cohen_kappa`."""

    if len(annotators) != 2:
        raise ValueError("cohen_kappa_agreement requires exactly two annotators")
    return cohen_kappa(annotators[0], annotators[1], match=match)


def fleiss_kappa_agreement(
    annotators: Iterable[Iterable[Any]],
    *,
    match: MatchMode = "exact",
) -> float:
    """Compatibility wrapper for :func:`fleiss_kappa`."""

    return fleiss_kappa(annotators, match=match)


__all__ = [
    "AgreementReport",
    "Annotation",
    "AnnotationSpan",
    "InterAnnotatorAgreement",
    "SpanAnnotation",
    "agreement_report",
    "cohen_kappa",
    "cohen_kappa_agreement",
    "fleiss_kappa",
    "fleiss_kappa_agreement",
    "inter_annotator_agreement",
]
