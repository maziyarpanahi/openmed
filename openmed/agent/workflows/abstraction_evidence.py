"""Value-free evidence chains for clinical chart abstraction.

The contract records source digests and character offsets, never source text or
normalized clinical values. Finalization is deterministic and local. It fails
closed when required fields are absent, source evidence is missing or entirely
generated, or review has not approved every represented field.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

ABSTRACTION_EVIDENCE_SCHEMA: Final = "openmed.agent.workflows.abstraction_evidence.v1"

_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
_FIELD_ID_RE = re.compile(r"[a-z][a-z0-9_]{0,63}(?:\.[a-z][a-z0-9_]{0,63}){0,7}")
_MAX_OFFSET: Final = (1 << 63) - 1
_MAX_CHAINS: Final = 10_000
_MAX_SOURCES_PER_CHAIN: Final = 1_024
_ISSUE_CODES: Final = frozenset(
    {
        "generated_only_evidence",
        "missing_field_evidence",
        "missing_source_evidence",
        "review_not_approved",
    }
)


class SourceKind(str, Enum):
    """Closed vocabulary for the origin of a supporting span."""

    CLINICAL_RECORD = "clinical_record"
    GENERATED_TEXT = "generated_text"


class TransformationKind(str, Enum):
    """Closed vocabulary for the transformation that produced a fact."""

    RULE = "rule"
    MODEL = "model"


class ReviewerState(str, Enum):
    """Closed human-review state for an abstracted field."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class AbstractionEvidenceError(ValueError):
    """A value-free evidence validation or finalization error.

    Args:
        code: Stable machine-readable error code.
        field_name: Optional public contract field associated with the error.
        report: Optional metadata-only report for a failed finalization.
    """

    def __init__(
        self,
        code: str,
        field_name: str | None = None,
        *,
        report: AbstractionEvidenceReport | None = None,
    ) -> None:
        self.code = code
        self.field_name = field_name
        self.report = report
        message = code if field_name is None else f"{field_name}: {code}"
        super().__init__(message)


@dataclass(frozen=True, slots=True, order=True)
class SourceLocation:
    """A digest-addressed span without source content or an external ID.

    Args:
        source_digest: SHA-256 digest of the containing source artifact.
        start_offset: Inclusive character offset in the canonical source.
        end_offset: Exclusive character offset in the canonical source.
        kind: Whether the span came from a clinical record or generated text.
    """

    source_digest: str
    start_offset: int
    end_offset: int
    kind: SourceKind = SourceKind.CLINICAL_RECORD

    def __post_init__(self) -> None:
        _validate_digest(self.source_digest, "source_digest")
        _validate_offset(self.start_offset, "start_offset")
        _validate_offset(self.end_offset, "end_offset")
        if self.end_offset <= self.start_offset:
            raise AbstractionEvidenceError("invalid_span", "end_offset")
        if type(self.kind) is not SourceKind:
            raise AbstractionEvidenceError("invalid_source_kind", "kind")

    @property
    def location_digest(self) -> str:
        """Return a stable digest for the complete source location."""

        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, str | int]:
        """Return deterministic, value-free source metadata."""

        return {
            "end_offset": self.end_offset,
            "kind": self.kind.value,
            "source_digest": self.source_digest,
            "start_offset": self.start_offset,
        }


@dataclass(frozen=True, slots=True, init=False)
class AbstractionEvidenceChain:
    """Evidence and review metadata for one normalized abstracted field.

    ``normalized_fact_digest`` binds the chain to a fact held inside the
    trusted abstraction boundary without retaining that fact in audit data.
    ``transformation_digest`` similarly binds the exact rule or model artifact.
    """

    field_id: str
    source_locations: tuple[SourceLocation, ...] = field(repr=False)
    normalized_fact_digest: str
    transformation_kind: TransformationKind
    transformation_digest: str
    uncertainty: float
    reviewer_state: ReviewerState

    def __init__(
        self,
        field_id: str,
        source_locations: Iterable[SourceLocation],
        normalized_fact_digest: str,
        transformation_kind: TransformationKind,
        transformation_digest: str,
        uncertainty: float,
        reviewer_state: ReviewerState,
    ) -> None:
        object.__setattr__(self, "field_id", _validate_field_id(field_id))
        object.__setattr__(
            self,
            "source_locations",
            _normalize_sources(source_locations),
        )
        object.__setattr__(
            self,
            "normalized_fact_digest",
            _validate_digest(normalized_fact_digest, "normalized_fact_digest"),
        )
        if type(transformation_kind) is not TransformationKind:
            raise AbstractionEvidenceError(
                "invalid_transformation_kind", "transformation_kind"
            )
        object.__setattr__(self, "transformation_kind", transformation_kind)
        object.__setattr__(
            self,
            "transformation_digest",
            _validate_digest(transformation_digest, "transformation_digest"),
        )
        object.__setattr__(self, "uncertainty", _validate_uncertainty(uncertainty))
        if type(reviewer_state) is not ReviewerState:
            raise AbstractionEvidenceError("invalid_reviewer_state", "reviewer_state")
        object.__setattr__(self, "reviewer_state", reviewer_state)

    @property
    def has_clinical_source(self) -> bool:
        """Return whether at least one span comes from a clinical record."""

        return any(
            source.kind is SourceKind.CLINICAL_RECORD
            for source in self.source_locations
        )

    @property
    def chain_digest(self) -> str:
        """Return a stable digest over the complete evidence chain."""

        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic evidence metadata without clinical values."""

        return {
            "field_id": self.field_id,
            "normalized_fact_digest": self.normalized_fact_digest,
            "reviewer_state": self.reviewer_state.value,
            "source_locations": [
                location.to_dict() for location in self.source_locations
            ],
            "transformation_digest": self.transformation_digest,
            "transformation_kind": self.transformation_kind.value,
            "uncertainty": self.uncertainty,
        }


@dataclass(frozen=True, slots=True, order=True)
class AbstractionEvidenceIssue:
    """One closed reason that prevents evidence finalization."""

    code: str
    field_id: str

    def __post_init__(self) -> None:
        if type(self.code) is not str or self.code not in _ISSUE_CODES:
            raise AbstractionEvidenceError("invalid_issue_code", "code")
        _validate_field_id(self.field_id)

    def to_dict(self) -> dict[str, str]:
        """Return stable issue metadata."""

        return {"code": self.code, "field_id": self.field_id}


@dataclass(frozen=True, slots=True)
class AbstractionEvidenceReport:
    """Deterministic, value-free finalization report."""

    evidence_digest: str
    required_fields_digest: str
    report_digest: str
    issues: tuple[AbstractionEvidenceIssue, ...]
    chain_count: int
    required_field_count: int
    clinical_source_count: int
    approved_field_count: int
    schema: str = ABSTRACTION_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        _validate_digest(self.evidence_digest, "evidence_digest")
        _validate_digest(self.required_fields_digest, "required_fields_digest")
        _validate_digest(self.report_digest, "report_digest")
        if type(self.issues) is not tuple or any(
            type(issue) is not AbstractionEvidenceIssue for issue in self.issues
        ):
            raise AbstractionEvidenceError("invalid_issues", "issues")
        if self.issues != tuple(sorted(self.issues)):
            raise AbstractionEvidenceError("issues_not_sorted", "issues")
        for field_name in (
            "chain_count",
            "required_field_count",
            "clinical_source_count",
            "approved_field_count",
        ):
            _validate_count(getattr(self, field_name), field_name)
        if self.clinical_source_count > self.chain_count:
            raise AbstractionEvidenceError(
                "count_out_of_range", "clinical_source_count"
            )
        if self.approved_field_count > self.chain_count:
            raise AbstractionEvidenceError("count_out_of_range", "approved_field_count")
        if self.schema != ABSTRACTION_EVIDENCE_SCHEMA:
            raise AbstractionEvidenceError("invalid_schema", "schema")

    @property
    def is_finalizable(self) -> bool:
        """Return whether every finalization gate passed."""

        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic counts, digests, and closed issue codes."""

        return {
            "approved_field_count": self.approved_field_count,
            "chain_count": self.chain_count,
            "clinical_source_count": self.clinical_source_count,
            "evidence_digest": self.evidence_digest,
            "is_finalizable": self.is_finalizable,
            "issues": [issue.to_dict() for issue in self.issues],
            "report_digest": self.report_digest,
            "required_field_count": self.required_field_count,
            "required_fields_digest": self.required_fields_digest,
            "schema": self.schema,
        }

    def to_json(self) -> str:
        """Serialize the report with stable key ordering."""

        return _canonical_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class FinalizedAbstractionEvidence:
    """Content-free receipt for evidence that passed all finalization gates."""

    evidence_digest: str
    required_fields_digest: str
    report_digest: str
    chain_count: int
    schema: str = ABSTRACTION_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        _validate_digest(self.evidence_digest, "evidence_digest")
        _validate_digest(self.required_fields_digest, "required_fields_digest")
        _validate_digest(self.report_digest, "report_digest")
        _validate_count(self.chain_count, "chain_count")
        if self.schema != ABSTRACTION_EVIDENCE_SCHEMA:
            raise AbstractionEvidenceError("invalid_schema", "schema")

    def to_dict(self) -> dict[str, str | int]:
        """Return deterministic receipt metadata."""

        return {
            "chain_count": self.chain_count,
            "evidence_digest": self.evidence_digest,
            "report_digest": self.report_digest,
            "required_fields_digest": self.required_fields_digest,
            "schema": self.schema,
        }

    def to_json(self) -> str:
        """Serialize the receipt with stable key ordering."""

        return _canonical_json(self.to_dict())


@dataclass(frozen=True, slots=True, init=False)
class ChartAbstractionEvidence:
    """A unique, deterministically ordered set of field evidence chains."""

    chains: tuple[AbstractionEvidenceChain, ...] = field(repr=False)

    def __init__(self, chains: Iterable[AbstractionEvidenceChain]) -> None:
        normalized = _normalize_chains(chains)
        object.__setattr__(self, "chains", normalized)

    @property
    def evidence_digest(self) -> str:
        """Return a stable digest over all ordered field chains."""

        return _digest([chain.chain_digest for chain in self.chains])

    def evaluate(self, required_fields: Iterable[str]) -> AbstractionEvidenceReport:
        """Evaluate source and review coverage for required field identifiers."""

        required = _normalize_required_fields(required_fields)
        chain_by_field = {chain.field_id: chain for chain in self.chains}
        issues: list[AbstractionEvidenceIssue] = []

        for field_id in required:
            if field_id not in chain_by_field:
                issues.append(
                    AbstractionEvidenceIssue("missing_field_evidence", field_id)
                )

        for chain in self.chains:
            if not chain.source_locations:
                issues.append(
                    AbstractionEvidenceIssue("missing_source_evidence", chain.field_id)
                )
            elif not chain.has_clinical_source:
                issues.append(
                    AbstractionEvidenceIssue("generated_only_evidence", chain.field_id)
                )
            if chain.reviewer_state is not ReviewerState.APPROVED:
                issues.append(
                    AbstractionEvidenceIssue("review_not_approved", chain.field_id)
                )

        normalized_issues = tuple(sorted(issues))
        required_fields_digest = _digest(list(required))
        report_fields = {
            "approved_field_count": sum(
                chain.reviewer_state is ReviewerState.APPROVED for chain in self.chains
            ),
            "chain_count": len(self.chains),
            "clinical_source_count": sum(
                chain.has_clinical_source for chain in self.chains
            ),
            "evidence_digest": self.evidence_digest,
            "issues": [issue.to_dict() for issue in normalized_issues],
            "required_field_count": len(required),
            "required_fields_digest": required_fields_digest,
            "schema": ABSTRACTION_EVIDENCE_SCHEMA,
        }
        return AbstractionEvidenceReport(
            evidence_digest=self.evidence_digest,
            required_fields_digest=required_fields_digest,
            report_digest=_digest(report_fields),
            issues=normalized_issues,
            chain_count=len(self.chains),
            required_field_count=len(required),
            clinical_source_count=report_fields["clinical_source_count"],
            approved_field_count=report_fields["approved_field_count"],
        )

    def finalize(self, required_fields: Iterable[str]) -> FinalizedAbstractionEvidence:
        """Return a receipt or fail closed with a metadata-only report."""

        report = self.evaluate(required_fields)
        if not report.is_finalizable:
            raise AbstractionEvidenceError(
                "evidence_not_finalizable",
                report=report,
            )
        return FinalizedAbstractionEvidence(
            evidence_digest=report.evidence_digest,
            required_fields_digest=report.required_fields_digest,
            report_digest=report.report_digest,
            chain_count=report.chain_count,
        )


def _normalize_sources(
    sources: Iterable[SourceLocation],
) -> tuple[SourceLocation, ...]:
    values = _bounded_tuple(
        sources,
        field_name="source_locations",
        maximum=_MAX_SOURCES_PER_CHAIN,
    )
    if any(type(source) is not SourceLocation for source in values):
        raise AbstractionEvidenceError("invalid_source", "source_locations")
    normalized = tuple(sorted(values))
    if len(set(normalized)) != len(normalized):
        raise AbstractionEvidenceError("duplicate_source", "source_locations")
    return normalized


def _normalize_chains(
    chains: Iterable[AbstractionEvidenceChain],
) -> tuple[AbstractionEvidenceChain, ...]:
    values = _bounded_tuple(chains, field_name="chains", maximum=_MAX_CHAINS)
    if any(type(chain) is not AbstractionEvidenceChain for chain in values):
        raise AbstractionEvidenceError("invalid_chain", "chains")
    normalized = tuple(sorted(values, key=lambda chain: chain.field_id))
    field_ids = tuple(chain.field_id for chain in normalized)
    if len(set(field_ids)) != len(field_ids):
        raise AbstractionEvidenceError("duplicate_field", "chains")
    return normalized


def _normalize_required_fields(fields: Iterable[str]) -> tuple[str, ...]:
    values = _bounded_tuple(
        fields,
        field_name="required_fields",
        maximum=_MAX_CHAINS,
    )
    if not values:
        raise AbstractionEvidenceError("empty_collection", "required_fields")
    normalized = tuple(sorted(_validate_field_id(value) for value in values))
    if len(set(normalized)) != len(normalized):
        raise AbstractionEvidenceError("duplicate_field", "required_fields")
    return normalized


def _bounded_tuple(value: Any, *, field_name: str, maximum: int) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise AbstractionEvidenceError("invalid_collection", field_name)
    try:
        iterator = iter(value)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise AbstractionEvidenceError("invalid_collection", field_name) from None

    items: list[Any] = []
    try:
        for item in iterator:
            if len(items) >= maximum:
                raise AbstractionEvidenceError("too_many_items", field_name)
            items.append(item)
    except (KeyboardInterrupt, SystemExit, AbstractionEvidenceError):
        raise
    except Exception:
        raise AbstractionEvidenceError("invalid_collection", field_name) from None
    return tuple(items)


def _validate_field_id(value: Any) -> str:
    if type(value) is not str or _FIELD_ID_RE.fullmatch(value) is None:
        raise AbstractionEvidenceError("invalid_field_id", "field_id")
    return value


def _validate_digest(value: Any, field_name: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise AbstractionEvidenceError("invalid_digest", field_name)
    return value


def _validate_offset(value: Any, field_name: str) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_OFFSET:
        raise AbstractionEvidenceError("invalid_offset", field_name)
    return value


def _validate_count(value: Any, field_name: str) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_CHAINS:
        raise AbstractionEvidenceError("invalid_count", field_name)
    return value


def _validate_uncertainty(value: Any) -> float:
    if type(value) not in (int, float):
        raise AbstractionEvidenceError("invalid_uncertainty", "uncertainty")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise AbstractionEvidenceError("invalid_uncertainty", "uncertainty")
    return normalized


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value: Any) -> str:
    payload = _canonical_json(value).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


__all__ = [
    "ABSTRACTION_EVIDENCE_SCHEMA",
    "AbstractionEvidenceChain",
    "AbstractionEvidenceError",
    "AbstractionEvidenceIssue",
    "AbstractionEvidenceReport",
    "ChartAbstractionEvidence",
    "FinalizedAbstractionEvidence",
    "ReviewerState",
    "SourceKind",
    "SourceLocation",
    "TransformationKind",
]
