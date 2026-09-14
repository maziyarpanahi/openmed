"""Deterministic, value-free review-yield metrics for clinical relations.

The evaluator accepts relation class and reviewer disposition metadata, then
projects the input to aggregate counts. Candidate text, endpoint values,
case identifiers, reviewer identities, scores, and other arbitrary fields are
never retained or serialized. The result is assistive evidence for a human
review workflow, not a clinical decision or a reviewer-ranking mechanism.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final

RELATION_REVIEW_YIELD_SCHEMA_VERSION: Final = "openmed.eval.relation_review_yield.v1"
REVIEW_YIELD_DISCLAIMER: Final = (
    "Relation review-yield metrics are aggregate assistive evidence for human "
    "review, not a clinical decision, compliance certification, or autonomous "
    "safety guarantee."
)

REVIEW_ACCEPTED: Final = "accepted"
REVIEW_CORRECTED: Final = "corrected"
REVIEW_REJECTED: Final = "rejected"
REVIEW_DUPLICATE: Final = "duplicate"
REVIEW_DEFERRED: Final = "deferred"
RELATION_REVIEW_DISPOSITIONS: tuple[str, ...] = (
    REVIEW_ACCEPTED,
    REVIEW_CORRECTED,
    REVIEW_REJECTED,
    REVIEW_DUPLICATE,
    REVIEW_DEFERRED,
)

_COUNT_FIELDS: tuple[str, ...] = RELATION_REVIEW_DISPOSITIONS
_REVIEWED_DISPOSITIONS = frozenset(
    {
        REVIEW_ACCEPTED,
        REVIEW_CORRECTED,
        REVIEW_REJECTED,
        REVIEW_DUPLICATE,
    }
)
_RELATION_CLASS_RE = re.compile(r"[A-Z0-9][A-Z0-9_]{0,63}\Z")
_MISSING = object()


class RelationReviewYieldError(ValueError):
    """Raised when a review-yield record is missing safe control metadata."""


class RelationReviewDisposition(str, Enum):
    """Controlled human-review outcome for one relation candidate."""

    ACCEPTED = REVIEW_ACCEPTED
    CORRECTED = REVIEW_CORRECTED
    REJECTED = REVIEW_REJECTED
    DUPLICATE = REVIEW_DUPLICATE
    DEFERRED = REVIEW_DEFERRED


@dataclass(frozen=True, slots=True)
class RelationReviewCandidate:
    """Safe projection of one candidate onto class and review disposition.

    Only the two fields below are retained. Callers may pass richer candidate
    records to :func:`normalize_relation_review_candidate`; source text,
    endpoint values, identifiers, reviewer metadata, and scores are discarded.
    """

    relation_class: str
    disposition: str

    def __post_init__(self) -> None:
        """Normalize the controlled fields without retaining arbitrary input."""

        object.__setattr__(
            self,
            "relation_class",
            _normalize_relation_class(self.relation_class),
        )
        object.__setattr__(
            self,
            "disposition",
            _normalize_disposition(self.disposition),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RelationReviewCandidate":
        """Build a safe candidate projection from a mapping.

        Unknown fields are intentionally ignored after the two control fields
        are selected, so rich relation records cannot enter report artifacts.
        """

        if not isinstance(payload, Mapping):
            raise RelationReviewYieldError("review candidate must be a mapping")
        return normalize_relation_review_candidate(payload)

    def to_dict(self) -> dict[str, str]:
        """Return the value-free candidate projection."""

        return {
            "disposition": self.disposition,
            "relation_class": self.relation_class,
        }

    def __getitem__(self, key: str) -> str:
        """Support mapping-style access to the safe projection."""

        return self.to_dict()[key]


@dataclass(frozen=True, slots=True)
class RelationReviewYieldMetrics:
    """Aggregate review outcomes and rates for one relation class.

    ``review_yield`` uses completed reviews as its denominator and counts an
    accepted or corrected candidate as useful. Deferred candidates are not
    completed reviews, so ``surface_yield`` is also provided to show useful
    outcomes across every surfaced candidate, including deferred work.
    """

    relation_class: str
    accepted: int = 0
    corrected: int = 0
    rejected: int = 0
    duplicate: int = 0
    deferred: int = 0

    def __post_init__(self) -> None:
        """Validate the aggregate counts and normalize the class label."""

        object.__setattr__(
            self,
            "relation_class",
            _normalize_relation_class(self.relation_class),
        )
        for field_name in _COUNT_FIELDS:
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise RelationReviewYieldError(
                    "review-yield counts must be non-negative integers"
                )

    @property
    def candidate_count(self) -> int:
        """Return all candidates surfaced for this relation class."""

        return sum(getattr(self, field_name) for field_name in _COUNT_FIELDS)

    @property
    def surfaced_count(self) -> int:
        """Alias for :attr:`candidate_count`."""

        return self.candidate_count

    @property
    def reviewed_count(self) -> int:
        """Return candidates with a completed review disposition."""

        return sum(getattr(self, field_name) for field_name in _REVIEWED_DISPOSITIONS)

    @property
    def useful_count(self) -> int:
        """Return accepted and corrected candidates."""

        return self.accepted + self.corrected

    @property
    def accepted_count(self) -> int:
        """Return the accepted candidate count."""

        return self.accepted

    @property
    def corrected_count(self) -> int:
        """Return the corrected candidate count."""

        return self.corrected

    @property
    def rejected_count(self) -> int:
        """Return the rejected candidate count."""

        return self.rejected

    @property
    def duplicate_count(self) -> int:
        """Return the duplicate candidate count."""

        return self.duplicate

    @property
    def deferred_count(self) -> int:
        """Return the deferred candidate count."""

        return self.deferred

    @property
    def review_yield(self) -> float:
        """Return useful completed reviews divided by completed reviews."""

        if self.reviewed_count == 0:
            return 0.0
        return self.useful_count / self.reviewed_count

    @property
    def surface_yield(self) -> float:
        """Return useful completed reviews divided by surfaced candidates."""

        if self.candidate_count == 0:
            return 0.0
        return self.useful_count / self.candidate_count

    @property
    def reviewed_rate(self) -> float:
        """Return the completed-review fraction among surfaced candidates."""

        if self.candidate_count == 0:
            return 0.0
        return self.reviewed_count / self.candidate_count

    @property
    def accepted_rate(self) -> float:
        """Return accepted candidates as a fraction of surfaced candidates."""

        return self._candidate_rate(self.accepted)

    @property
    def corrected_rate(self) -> float:
        """Return corrected candidates as a fraction of surfaced candidates."""

        return self._candidate_rate(self.corrected)

    @property
    def rejected_rate(self) -> float:
        """Return rejected candidates as a fraction of surfaced candidates."""

        return self._candidate_rate(self.rejected)

    @property
    def duplicate_rate(self) -> float:
        """Return duplicate candidates as a fraction of surfaced candidates."""

        return self._candidate_rate(self.duplicate)

    @property
    def deferred_rate(self) -> float:
        """Return deferred candidates as a fraction of surfaced candidates."""

        return self._candidate_rate(self.deferred)

    @property
    def disposition_counts(self) -> dict[str, int]:
        """Return all disposition counts in their canonical order."""

        return {
            disposition: int(getattr(self, disposition))
            for disposition in RELATION_REVIEW_DISPOSITIONS
        }

    def _candidate_rate(self, numerator: int) -> float:
        if self.candidate_count == 0:
            return 0.0
        return numerator / self.candidate_count

    def to_dict(self, *, include_relation_class: bool = True) -> dict[str, Any]:
        """Return deterministic counts and rates without candidate details."""

        payload: dict[str, Any] = {
            "accepted": self.accepted,
            "accepted_rate": self.accepted_rate,
            "corrected": self.corrected,
            "corrected_rate": self.corrected_rate,
            "deferred": self.deferred,
            "deferred_rate": self.deferred_rate,
            "duplicate": self.duplicate,
            "duplicate_rate": self.duplicate_rate,
            "rejected": self.rejected,
            "rejected_rate": self.rejected_rate,
            "candidate_count": self.candidate_count,
            "reviewed_count": self.reviewed_count,
            "reviewed_rate": self.reviewed_rate,
            "review_yield": self.review_yield,
            "surface_yield": self.surface_yield,
            "useful_count": self.useful_count,
        }
        if include_relation_class:
            payload["relation_class"] = self.relation_class
        return payload

    def __getitem__(self, key: str) -> Any:
        """Support mapping-style access to metric fields."""

        return self.to_dict()[key]


@dataclass(frozen=True, slots=True)
class RelationReviewYieldReport:
    """Deterministic aggregate report for guarded relation review outcomes."""

    overall: RelationReviewYieldMetrics
    by_relation_class: Mapping[str, RelationReviewYieldMetrics]
    schema_version: str = RELATION_REVIEW_YIELD_SCHEMA_VERSION
    disclaimer: str = REVIEW_YIELD_DISCLAIMER

    def __post_init__(self) -> None:
        """Normalize relation-class ordering and validate the report contract."""

        if self.schema_version != RELATION_REVIEW_YIELD_SCHEMA_VERSION:
            raise RelationReviewYieldError("unsupported review-yield schema version")
        if self.disclaimer != REVIEW_YIELD_DISCLAIMER:
            raise RelationReviewYieldError("review-yield disclaimer cannot be changed")
        ordered: dict[str, RelationReviewYieldMetrics] = {}
        for relation_class, metrics in self.by_relation_class.items():
            normalized_class = _normalize_relation_class(relation_class)
            if not isinstance(metrics, RelationReviewYieldMetrics):
                raise RelationReviewYieldError("relation-class metrics are invalid")
            if metrics.relation_class != normalized_class:
                raise RelationReviewYieldError(
                    "relation-class metric key does not match its relation class"
                )
            ordered[normalized_class] = metrics
        object.__setattr__(self, "by_relation_class", dict(sorted(ordered.items())))

    @property
    def relation_classes(self) -> tuple[str, ...]:
        """Return relation classes in deterministic order."""

        return tuple(self.by_relation_class)

    @property
    def candidate_count(self) -> int:
        """Return the total number of surfaced candidates."""

        return self.overall.candidate_count

    @property
    def total_candidate_count(self) -> int:
        """Alias for :attr:`candidate_count`."""

        return self.candidate_count

    @property
    def review_yield(self) -> float:
        """Return the overall completed-review yield."""

        return self.overall.review_yield

    @property
    def surface_yield(self) -> float:
        """Return the overall surfaced-candidate yield."""

        return self.overall.surface_yield

    def to_dict(self) -> dict[str, Any]:
        """Return a stable aggregate-only JSON-ready report payload."""

        return {
            "by_relation_class": {
                relation_class: metrics.to_dict()
                for relation_class, metrics in self.by_relation_class.items()
            },
            "candidate_count": self.candidate_count,
            "disclaimer": self.disclaimer,
            "overall": self.overall.to_dict(include_relation_class=False),
            "relation_class_count": len(self.by_relation_class),
            "schema_version": self.schema_version,
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the report to deterministic JSON."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write deterministic JSON to *path*."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path

    def to_markdown(self) -> str:
        """Render a deterministic, aggregate-only Markdown report."""

        lines = [
            "# Guarded Clinical Relation Review Yield",
            "",
            self.disclaimer,
            "",
            "| Relation class | Candidates | Reviewed | Accepted | Corrected | "
            "Rejected | Duplicate | Deferred | Review yield | Surface yield |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for relation_class, metrics in [("Overall", self.overall)] + list(
            self.by_relation_class.items()
        ):
            lines.append(
                f"| `{relation_class}` | {metrics.candidate_count} | "
                f"{metrics.reviewed_count} | {metrics.accepted} | "
                f"{metrics.corrected} | {metrics.rejected} | {metrics.duplicate} | "
                f"{metrics.deferred} | {metrics.review_yield:.6f} | "
                f"{metrics.surface_yield:.6f} |"
            )
        lines.extend(
            [
                "",
                "`Review yield` is `(accepted + corrected) / reviewed`, where "
                "deferred candidates are not completed reviews.",
                "",
                "`Surface yield` is `(accepted + corrected) / candidates` and "
                "includes deferred candidates in the surfaced denominator.",
                "",
            ]
        )
        return "\n".join(lines) + "\n"

    def write_markdown(self, path: str | Path) -> Path:
        """Write deterministic Markdown to *path*."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown(), encoding="utf-8")
        return output_path

    def __getitem__(self, key: str) -> Any:
        """Support mapping-style access to report fields."""

        return self.to_dict()[key]


def normalize_relation_review_candidate(
    candidate: Any,
) -> RelationReviewCandidate:
    """Project one candidate-like record to safe control metadata.

    Mappings and objects may contain richer relation data. Only a relation
    class and one of the supported disposition fields are read; all other
    fields are ignored and can never appear in the resulting report.
    """

    if isinstance(candidate, RelationReviewCandidate):
        return candidate

    relation_class = _read_field(
        candidate,
        (
            "relation_class",
            "relation_type",
            "predicate",
            "type",
            "class",
            "label",
        ),
    )
    if relation_class is _MISSING:
        relation = _read_field(candidate, ("relation",))
        if relation is not _MISSING:
            if isinstance(relation, str):
                relation_class = relation
            else:
                relation_class = _read_field(
                    relation,
                    (
                        "relation_class",
                        "relation_type",
                        "predicate",
                        "type",
                        "label",
                    ),
                )

    disposition = _read_field(
        candidate,
        (
            "disposition",
            "review_disposition",
            "review_status",
            "review_state",
            "review_outcome",
            "outcome",
            "status",
            "decision",
        ),
    )
    if disposition is _MISSING:
        review = _read_field(candidate, ("review",))
        if review is not _MISSING:
            disposition = _read_field(
                review,
                ("disposition", "status", "state", "decision"),
            )

    if relation_class is _MISSING:
        raise RelationReviewYieldError("review candidate relation class is required")
    if disposition is _MISSING:
        raise RelationReviewYieldError("review candidate disposition is required")
    return RelationReviewCandidate(
        relation_class=_normalize_relation_class(relation_class),
        disposition=_normalize_disposition(disposition),
    )


def compute_relation_review_yield(
    candidates: Iterable[Any],
    *,
    relation_classes: Iterable[Any] | None = None,
) -> RelationReviewYieldReport:
    """Compute deterministic review-yield counts by relation class.

    Args:
        candidates: Candidate-like records. Each record must expose a
            relation class and one of the five controlled review dispositions.
            Rich records are projected to those fields only.
        relation_classes: Optional additional controlled relation classes to
            include with zero counts. This is useful for stable dashboards.

    Returns:
        An aggregate-only report. ``review_yield`` is useful completed reviews
        divided by completed reviews; ``surface_yield`` uses all surfaced
        candidates, including deferred candidates, as its denominator.

    Raises:
        RelationReviewYieldError: If an input is missing safe control metadata
            or contains an unsupported class/disposition.
    """

    if isinstance(candidates, (str, bytes, Mapping)):
        raise RelationReviewYieldError("candidates must be an iterable of records")
    try:
        iterator = iter(candidates)
    except TypeError:
        raise RelationReviewYieldError(
            "candidates must be an iterable of records"
        ) from None

    counters: dict[str, Counter[str]] = {}
    for candidate in iterator:
        normalized = normalize_relation_review_candidate(candidate)
        counter = counters.setdefault(normalized.relation_class, Counter())
        counter[normalized.disposition] += 1

    if relation_classes is not None:
        if isinstance(relation_classes, (str, bytes, Mapping)):
            raise RelationReviewYieldError(
                "relation_classes must be an iterable of class labels"
            )
        try:
            declared_classes = iter(relation_classes)
        except TypeError:
            raise RelationReviewYieldError(
                "relation_classes must be an iterable of class labels"
            ) from None
        for relation_class in declared_classes:
            normalized_class = _normalize_relation_class(relation_class)
            counters.setdefault(normalized_class, Counter())

    by_relation_class = {
        relation_class: _metrics_from_counter(relation_class, counter)
        for relation_class, counter in sorted(counters.items())
    }
    overall_counter: Counter[str] = Counter()
    for counter in counters.values():
        overall_counter.update(counter)

    return RelationReviewYieldReport(
        overall=_metrics_from_counter("OVERALL", overall_counter),
        by_relation_class=by_relation_class,
    )


def build_relation_review_yield_report(
    candidates: Iterable[Any],
    *,
    relation_classes: Iterable[Any] | None = None,
) -> RelationReviewYieldReport:
    """Build a relation review-yield report using the canonical evaluator."""

    return compute_relation_review_yield(
        candidates,
        relation_classes=relation_classes,
    )


def _metrics_from_counter(
    relation_class: str,
    counter: Mapping[str, int],
) -> RelationReviewYieldMetrics:
    return RelationReviewYieldMetrics(
        relation_class=relation_class,
        accepted=int(counter.get(REVIEW_ACCEPTED, 0)),
        corrected=int(counter.get(REVIEW_CORRECTED, 0)),
        rejected=int(counter.get(REVIEW_REJECTED, 0)),
        duplicate=int(counter.get(REVIEW_DUPLICATE, 0)),
        deferred=int(counter.get(REVIEW_DEFERRED, 0)),
    )


def _read_field(candidate: Any, keys: tuple[str, ...]) -> Any:
    if isinstance(candidate, Mapping):
        for key in keys:
            if key in candidate:
                return candidate[key]
        return _MISSING
    if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
        if len(candidate) == 2:
            return candidate[0] if keys[0].startswith("relation") else candidate[1]
        return _MISSING
    for key in keys:
        try:
            value = getattr(candidate, key, _MISSING)
        except Exception:
            raise RelationReviewYieldError(
                "review candidate field is unreadable"
            ) from None
        if value is not _MISSING:
            return value
    return _MISSING


def _normalize_relation_class(value: Any) -> str:
    if not isinstance(value, str):
        raise RelationReviewYieldError("relation class must be a controlled string")
    normalized = re.sub(r"[-\s]+", "_", value.strip().upper())
    if not _RELATION_CLASS_RE.fullmatch(normalized):
        raise RelationReviewYieldError("relation class must be a safe identifier")
    return normalized


def _normalize_disposition(value: Any) -> str:
    if isinstance(value, RelationReviewDisposition):
        return value.value
    if not isinstance(value, str):
        raise RelationReviewYieldError("review disposition must be controlled")
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "accept": REVIEW_ACCEPTED,
        "correct": REVIEW_CORRECTED,
        "reject": REVIEW_REJECTED,
        "defer": REVIEW_DEFERRED,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in RELATION_REVIEW_DISPOSITIONS:
        raise RelationReviewYieldError("review disposition is unsupported")
    return normalized


__all__ = [
    "RELATION_REVIEW_DISPOSITIONS",
    "RELATION_REVIEW_YIELD_SCHEMA_VERSION",
    "REVIEW_ACCEPTED",
    "REVIEW_CORRECTED",
    "REVIEW_DEFERRED",
    "REVIEW_DUPLICATE",
    "REVIEW_REJECTED",
    "REVIEW_YIELD_DISCLAIMER",
    "RelationReviewCandidate",
    "RelationReviewDisposition",
    "RelationReviewYieldError",
    "RelationReviewYieldMetrics",
    "RelationReviewYieldReport",
    "build_relation_review_yield_report",
    "compute_relation_review_yield",
    "normalize_relation_review_candidate",
]
