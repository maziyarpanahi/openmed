"""PHI-safe evidence contract for social-determinant findings.

The contract is intentionally an envelope around an upstream SDOH extractor.
It records the type of evidence, the assertion boundary, the document section,
the source offsets, and the review state. It never stores source excerpts,
normalized values, document identifiers, or other raw input values. Callers
remain responsible for extracting and reviewing the underlying source text.

All helpers are local and deterministic. The real Social History Annotated
Corpus (SHAC) is DUA-gated and eval-only; this module contains no corpus data
and does not load a network resource.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

SDOH_EVIDENCE_SCHEMA_VERSION = 1

SDOH_EVIDENCE_ADVISORY = (
    "SDOH evidence is an assistive, clinician-reviewable annotation. It is not "
    "a diagnosis, a clinical decision, or an automated eligibility decision."
)

EvidenceType: TypeAlias = Literal[
    "self_report",
    "proxy_report",
    "clinician_observation",
    "structured_record",
    "inferred",
    "unknown",
    "refused",
]
AssertionStatus: TypeAlias = Literal[
    "present",
    "absent",
    "uncertain",
    "unknown",
    "refused",
]
SourceSection: TypeAlias = Literal[
    "social_history",
    "history",
    "assessment",
    "plan",
    "other",
    "unknown",
    "refused",
]
ReviewStatus: TypeAlias = Literal[
    "unreviewed",
    "needs_review",
    "reviewed",
    "rejected",
    "unknown",
    "refused",
]

# The SDOH-prefixed aliases make the public contract self-describing while the
# shorter aliases above keep annotations readable for downstream callers.
SDOHAssertion: TypeAlias = AssertionStatus
SDOHSourceSection: TypeAlias = SourceSection
SDOHReviewStatus: TypeAlias = ReviewStatus

SpanOffset: TypeAlias = tuple[int, int]

SELF_REPORT: EvidenceType = "self_report"
PROXY_REPORT: EvidenceType = "proxy_report"
CLINICIAN_OBSERVATION: EvidenceType = "clinician_observation"
STRUCTURED_RECORD: EvidenceType = "structured_record"
INFERRED: EvidenceType = "inferred"
EVIDENCE_UNKNOWN: EvidenceType = "unknown"
EVIDENCE_REFUSED: EvidenceType = "refused"

PRESENT: AssertionStatus = "present"
ABSENT: AssertionStatus = "absent"
UNCERTAIN: AssertionStatus = "uncertain"
ASSERTION_UNKNOWN: AssertionStatus = "unknown"
ASSERTION_REFUSED: AssertionStatus = "refused"

SOCIAL_HISTORY: SourceSection = "social_history"
HISTORY: SourceSection = "history"
ASSESSMENT: SourceSection = "assessment"
PLAN: SourceSection = "plan"
OTHER_SECTION: SourceSection = "other"
SECTION_UNKNOWN: SourceSection = "unknown"
SECTION_REFUSED: SourceSection = "refused"

UNREVIEWED: ReviewStatus = "unreviewed"
NEEDS_REVIEW: ReviewStatus = "needs_review"
REVIEWED: ReviewStatus = "reviewed"
REJECTED: ReviewStatus = "rejected"
REVIEW_UNKNOWN: ReviewStatus = "unknown"
REVIEW_REFUSED: ReviewStatus = "refused"

EVIDENCE_TYPES: tuple[EvidenceType, ...] = (
    SELF_REPORT,
    PROXY_REPORT,
    CLINICIAN_OBSERVATION,
    STRUCTURED_RECORD,
    INFERRED,
    EVIDENCE_UNKNOWN,
    EVIDENCE_REFUSED,
)
ASSERTION_STATUSES: tuple[AssertionStatus, ...] = (
    PRESENT,
    ABSENT,
    UNCERTAIN,
    ASSERTION_UNKNOWN,
    ASSERTION_REFUSED,
)
SOURCE_SECTIONS: tuple[SourceSection, ...] = (
    SOCIAL_HISTORY,
    HISTORY,
    ASSESSMENT,
    PLAN,
    OTHER_SECTION,
    SECTION_UNKNOWN,
    SECTION_REFUSED,
)
REVIEW_STATUSES: tuple[ReviewStatus, ...] = (
    UNREVIEWED,
    NEEDS_REVIEW,
    REVIEWED,
    REJECTED,
    REVIEW_UNKNOWN,
    REVIEW_REFUSED,
)

SDOH_DETERMINANTS: tuple[str, ...] = (
    "alcohol_use",
    "caregiving",
    "disability",
    "education",
    "employment",
    "employment_status",
    "financial_strain",
    "food_insecurity",
    "healthcare_access",
    "housing",
    "housing_insecurity",
    "insurance",
    "interpersonal_safety",
    "occupation",
    "social_support",
    "substance_use",
    "tobacco",
    "transportation",
    "utilities",
    "other",
    "unknown",
    "refused",
)

_CHOICE_ALIASES: dict[str, dict[str, str]] = {
    "evidence_type": {
        "reported": SELF_REPORT,
        "self_reported": SELF_REPORT,
        "proxy": PROXY_REPORT,
        "observed": CLINICIAN_OBSERVATION,
        "observation": CLINICIAN_OBSERVATION,
        "structured": STRUCTURED_RECORD,
    },
    "assertion": {
        "affirmed": PRESENT,
        "confirmed": PRESENT,
        "negated": ABSENT,
        "refuted": ABSENT,
        "unconfirmed": UNCERTAIN,
    },
    "review_status": {
        "pending": UNREVIEWED,
        "needs_human_review": NEEDS_REVIEW,
        "approved": REVIEWED,
    },
    "source_section": {
        "social history": SOCIAL_HISTORY,
        "social-history": SOCIAL_HISTORY,
        "past medical history": HISTORY,
        "past_medical_history": HISTORY,
    },
}

_SAFE_LABEL_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")


@dataclass(frozen=True)
class SDOHSourceSpan:
    """A non-empty half-open source span represented only by offsets.

    No excerpt or source text is accepted. The offsets are meaningful only in
    the caller's local source document and are safe to place in an audit
    record.
    """

    start: int
    end: int

    def __post_init__(self) -> None:
        start, end = _span_offset((self.start, self.end), "source span")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    def to_tuple(self) -> SpanOffset:
        """Return the validated half-open offset pair."""

        return self.start, self.end

    def to_dict(self) -> dict[str, int]:
        """Return offsets without any source excerpt."""

        return {"start": self.start, "end": self.end}

    @classmethod
    def from_obj(cls, value: object) -> "SDOHSourceSpan":
        """Build a source span from offsets or an offset-only mapping."""

        return cls(*_span_offset(value, "source span"))


@dataclass(frozen=True)
class SDOHEvidence:
    """One PHI-safe SDOH evidence record.

    Args:
        evidence_type: How the upstream extractor obtained the signal.
        assertion: Explicit assertion boundary. ``unknown`` and ``refused``
            are first-class values and are never treated as affirmative.
        source_section: Canonical document section containing the signal.
        source_span: Required half-open character offsets into caller-owned
            source text. Only offsets are retained.
        review_status: Human-review state. New records default to
            ``needs_review`` so they cannot be mistaken for approved output.
        determinant: Optional controlled determinant label. It is normalized to
            a short safe token; raw source values are not accepted or stored.

    The record is an annotation contract, not a diagnosis, eligibility result,
    or autonomous decision.
    """

    evidence_type: EvidenceType
    assertion: AssertionStatus
    source_section: SourceSection
    source_span: SpanOffset | SDOHSourceSpan
    review_status: ReviewStatus = NEEDS_REVIEW
    determinant: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence_type",
            _choice(self.evidence_type, "evidence_type", EVIDENCE_TYPES),
        )
        object.__setattr__(
            self,
            "assertion",
            _choice(self.assertion, "assertion", ASSERTION_STATUSES),
        )
        object.__setattr__(
            self,
            "source_section",
            _source_section(self.source_section),
        )
        object.__setattr__(self, "source_span", _span_offset(self.source_span))
        object.__setattr__(
            self,
            "review_status",
            _choice(self.review_status, "review_status", REVIEW_STATUSES),
        )
        object.__setattr__(self, "determinant", _determinant(self.determinant))

    @property
    def status(self) -> AssertionStatus:
        """Return the assertion status under the common ``status`` name."""

        return self.assertion

    @property
    def span(self) -> SpanOffset:
        """Return the offset pair under the legacy ``span`` name."""

        return self.source_span

    @property
    def evidence(self) -> "SDOHEvidence":
        """Return this record as the emitted finding envelope."""

        return self

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic, raw-text-free mapping."""

        payload: dict[str, object] = {
            "evidence_type": self.evidence_type,
            "assertion": self.assertion,
            "source_section": self.source_section,
            "source_span": list(self.source_span),
            "review_status": self.review_status,
        }
        if self.determinant is not None:
            payload["determinant"] = self.determinant
        return payload

    def to_json(self) -> str:
        """Return canonical JSON containing no source text."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SDOHEvidence":
        """Rebuild a record while ignoring untrusted raw-text fields.

        ``text``, ``value``, ``surface``, ``excerpt``, and unknown extension
        keys are deliberately not copied into the result. A source span and an
        explicit assertion are mandatory; use ``unknown`` or ``refused`` when
        the upstream extractor cannot establish a stronger state.
        """

        if not isinstance(payload, Mapping):
            raise TypeError("SDOH evidence payload must be a mapping")

        evidence_type = _first(payload, "evidence_type", "type", "kind")
        assertion = _first(payload, "assertion", "assertion_status", "status")
        source_section = _first(
            payload,
            "source_section",
            "section",
            "section_name",
        )
        source_span = _first(
            payload,
            "source_span",
            "span",
            "source_offsets",
            "offset",
        )
        review_status = _first(payload, "review_status", "review")
        if review_status is _MISSING:
            review_status = NEEDS_REVIEW

        if evidence_type is _MISSING:
            raise ValueError("SDOH evidence requires an evidence type")
        if assertion is _MISSING:
            raise ValueError(
                "SDOH evidence requires an explicit assertion; use unknown or refused"
            )
        if source_section is _MISSING:
            raise ValueError("SDOH evidence requires a source section")
        if source_span is _MISSING:
            raise ValueError("SDOH evidence requires a source span")

        determinant = _first(payload, "determinant", "category", "domain")
        if determinant is _MISSING:
            determinant = None

        return cls(
            evidence_type=evidence_type,
            assertion=_nested_value(assertion, "status"),
            source_section=_nested_value(
                source_section,
                "name",
                "label",
                "section",
            ),
            source_span=source_span,
            review_status=_nested_value(review_status, "status"),
            determinant=determinant,
        )

    @classmethod
    def from_json(cls, payload: str) -> "SDOHEvidence":
        """Parse canonical JSON into a validated evidence record."""

        if not isinstance(payload, str):
            raise TypeError("SDOH evidence JSON must be a string")
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            raise ValueError("invalid SDOH evidence JSON") from None
        return cls.from_dict(decoded)


# ``SDOHFinding`` is the name used by extractors and downstream consumers. It
# intentionally aliases the same safe envelope rather than adding a second
# class that could accidentally reintroduce a raw value field.
SDOHFinding = SDOHEvidence


@dataclass(frozen=True)
class SDOHEvidenceReport:
    """Deterministic, PHI-safe report of validated SDOH evidence."""

    findings: tuple[SDOHEvidence, ...]
    disclaimer: str = SDOH_EVIDENCE_ADVISORY
    schema_version: int = SDOH_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        normalized = tuple(validate_sdoh_evidence(item) for item in self.findings)
        object.__setattr__(self, "findings", tuple(sorted(normalized, key=_sort_key)))
        if self.disclaimer != SDOH_EVIDENCE_ADVISORY:
            raise ValueError("SDOH evidence reports require the canonical advisory")
        if self.schema_version != SDOH_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("unsupported SDOH evidence report schema")

    def to_dict(self) -> dict[str, object]:
        """Return a sorted report containing offsets and controlled labels only."""

        return {
            "schema_version": self.schema_version,
            "disclaimer": self.disclaimer,
            "findings": [finding.to_dict() for finding in self.findings],
        }

    def to_json(self) -> str:
        """Return canonical JSON for the safe report."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def build_sdoh_evidence(
    evidence_type: EvidenceType,
    assertion: AssertionStatus,
    source_section: SourceSection,
    source_span: SpanOffset | SDOHSourceSpan,
    review_status: ReviewStatus = NEEDS_REVIEW,
    *,
    determinant: str | None = None,
) -> SDOHEvidence:
    """Build one validated, raw-text-free SDOH evidence record."""

    return SDOHEvidence(
        evidence_type=evidence_type,
        assertion=assertion,
        source_section=source_section,
        source_span=source_span,
        review_status=review_status,
        determinant=determinant,
    )


build_sdoh_finding = build_sdoh_evidence


def validate_sdoh_evidence(candidate: Any) -> SDOHEvidence:
    """Validate an evidence record or a serialized mapping.

    The returned object is the only supported report/input representation. Raw
    source values supplied in a mapping are not retained.
    """

    if isinstance(candidate, SDOHEvidence):
        return SDOHEvidence.from_dict(candidate.to_dict())
    if isinstance(candidate, Mapping):
        return SDOHEvidence.from_dict(candidate)
    raise TypeError("SDOH evidence must be a record or mapping")


validate_sdoh_finding = validate_sdoh_evidence


def validate_sdoh_findings(
    candidates: Iterable[Any],
) -> tuple[SDOHEvidence, ...]:
    """Validate an iterable of findings without retaining raw input fields."""

    if isinstance(candidates, (str, bytes)):
        raise TypeError("SDOH findings must be an iterable of records")
    return tuple(validate_sdoh_evidence(candidate) for candidate in candidates)


def build_sdoh_evidence_report(candidates: Iterable[Any]) -> SDOHEvidenceReport:
    """Build a deterministic safe report from records or serialized mappings."""

    return SDOHEvidenceReport(findings=validate_sdoh_findings(candidates))


def evidence_from_sdoh_finding(
    finding: Any,
    *,
    evidence_type: EvidenceType = EVIDENCE_UNKNOWN,
    assertion: AssertionStatus | None = None,
    source_section: SourceSection = SOCIAL_HISTORY,
    review_status: ReviewStatus = NEEDS_REVIEW,
) -> SDOHEvidence:
    """Wrap an upstream SDOH finding without copying its raw value.

    This adapter accepts the existing ``openmed.clinical.sdoh.SDOHFinding``
    shape (``category`` and ``span``) and keeps only the safe category label and
    offsets. A missing upstream status becomes explicit ``unknown``.
    """

    category = _field(finding, "category")
    source_span = _field(finding, "span", "source_span")
    if source_span is _MISSING:
        raise ValueError("upstream SDOH finding requires a source span")
    upstream_status = _field(finding, "status")
    resolved_assertion = assertion
    if resolved_assertion is None:
        resolved_assertion = _status_to_assertion(upstream_status)
    return build_sdoh_evidence(
        evidence_type=evidence_type,
        assertion=resolved_assertion,
        source_section=source_section,
        source_span=source_span,
        review_status=review_status,
        determinant=None if category is _MISSING else category,
    )


def _choice(value: object, field_name: str, allowed: Sequence[str]) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip().casefold().replace("-", "_").replace(" ", "_")
    normalized = _CHOICE_ALIASES.get(field_name, {}).get(normalized, normalized)
    if normalized not in allowed:
        raise ValueError(f"{field_name} has an unsupported value")
    return normalized


def _source_section(value: object) -> SourceSection:
    if not isinstance(value, str):
        raise TypeError("source_section must be a string")
    normalized = value.strip().casefold()
    normalized = _CHOICE_ALIASES["source_section"].get(normalized, normalized)
    normalized = normalized.replace("-", "_").replace(" ", "_")
    if normalized in SOURCE_SECTIONS:
        return normalized  # type: ignore[return-value]
    # Custom source labels are reduced to an opaque, safe bucket. This prevents
    # a report from becoming a transport for arbitrary source text.
    if _SAFE_LABEL_RE.fullmatch(normalized):
        return OTHER_SECTION
    raise ValueError("source_section must be a safe section label")


def _determinant(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("determinant must be a string when provided")
    normalized = re.sub(r"\s+", "_", value.strip().casefold())
    if not normalized:
        return None
    if not _SAFE_LABEL_RE.fullmatch(normalized):
        raise ValueError("determinant must be a safe controlled label")
    if normalized not in SDOH_DETERMINANTS:
        raise ValueError("determinant must be a known controlled label")
    return normalized


def _span_offset(value: object, field_name: str = "source span") -> SpanOffset:
    if isinstance(value, SDOHSourceSpan):
        return value.to_tuple()
    if isinstance(value, Mapping):
        value = (value.get("start"), value.get("end"))
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise TypeError(f"{field_name} must be a two-item offset sequence")
    start, end = value
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
    ):
        raise TypeError(f"{field_name} offsets must be integers")
    if start < 0 or end <= start:
        raise ValueError(f"{field_name} must satisfy 0 <= start < end")
    return start, end


_MISSING = object()


def _first(mapping: Mapping[str, Any], *keys: str) -> object:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return _MISSING


def _nested_value(value: object, *keys: str) -> object:
    if isinstance(value, Mapping):
        nested = _first(value, *keys)
        if nested is not _MISSING:
            return nested
    return value


def _field(value: object, *keys: str) -> object:
    if isinstance(value, Mapping):
        return _first(value, *keys)
    for key in keys:
        candidate = getattr(value, key, _MISSING)
        if candidate is not _MISSING:
            return candidate
    return _MISSING


def _status_to_assertion(value: object) -> AssertionStatus:
    if value is _MISSING or value is None:
        return ASSERTION_UNKNOWN
    normalized = str(value).strip().casefold()
    if normalized in {"current", "present", "affirmed", "active"}:
        return PRESENT
    if normalized in {"never", "absent", "negated", "refuted"}:
        return ABSENT
    if normalized in {"unknown", "unspecified", ""}:
        return ASSERTION_UNKNOWN
    return UNCERTAIN


def _sort_key(item: SDOHEvidence) -> tuple[object, ...]:
    return (
        item.source_span,
        item.determinant or "",
        item.evidence_type,
        item.assertion,
        item.source_section,
        item.review_status,
    )


__all__ = [
    "ABSENT",
    "ASSESSMENT",
    "ASSERTION_REFUSED",
    "ASSERTION_STATUSES",
    "ASSERTION_UNKNOWN",
    "AssertionStatus",
    "CLINICIAN_OBSERVATION",
    "EVIDENCE_REFUSED",
    "EvidenceType",
    "EVIDENCE_TYPES",
    "EVIDENCE_UNKNOWN",
    "HISTORY",
    "INFERRED",
    "NEEDS_REVIEW",
    "OTHER_SECTION",
    "PLAN",
    "PRESENT",
    "PROXY_REPORT",
    "REJECTED",
    "REVIEWED",
    "REVIEW_REFUSED",
    "REVIEW_STATUSES",
    "ReviewStatus",
    "REVIEW_UNKNOWN",
    "SDOHAssertion",
    "SDOH_EVIDENCE_ADVISORY",
    "SDOH_DETERMINANTS",
    "SDOH_EVIDENCE_SCHEMA_VERSION",
    "SDOHEvidence",
    "SDOHEvidenceReport",
    "SDOHFinding",
    "SDOHReviewStatus",
    "SDOHSourceSection",
    "SDOHSourceSpan",
    "SELF_REPORT",
    "SECTION_REFUSED",
    "SECTION_UNKNOWN",
    "SOCIAL_HISTORY",
    "SOURCE_SECTIONS",
    "SourceSection",
    "SpanOffset",
    "STRUCTURED_RECORD",
    "UNCERTAIN",
    "UNREVIEWED",
    "build_sdoh_evidence",
    "build_sdoh_evidence_report",
    "build_sdoh_finding",
    "evidence_from_sdoh_finding",
    "validate_sdoh_evidence",
    "validate_sdoh_finding",
    "validate_sdoh_findings",
]
