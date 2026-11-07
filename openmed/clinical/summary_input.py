"""Fail-closed, PHI-safe input contract for clinical summary generation.

The post-de-identification summary stage must consume evidence that is typed,
traceable to a source, bound to the active de-identification policy, and
explicitly reviewed.  This module validates that envelope without inspecting
or retaining note text.  Rejections are reported as deterministic category
counts, so a caller can safely expose the result in logs or audit reports.

The contract is local-only.  It does not load a model, contact a service, or
resolve a source reference.  A source reference is an opaque caller-owned
token; the source itself remains outside this module.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

SUMMARY_INPUT_SCHEMA_VERSION = 1
SUMMARY_INPUT_ADVISORY = (
    "Summary input accepts only typed, policy-bound, reviewed evidence; "
    "it is not a clinical decision or compliance certification."
)


class SummaryEvidenceType(str, Enum):
    """Evidence types that may enter the summary stage.

    These names describe structured evidence envelopes, not free-text fields.
    Deployments can restrict this set through :class:`SummaryInputContract`.
    """

    AGGREGATE_COUNT = "aggregate_count"
    ASSERTION = "assertion"
    CLINICAL_ASSERTION = "clinical_assertion"
    CODED_CONCEPT = "coded_concept"
    EVENT = "event"
    FINDING = "finding"
    LAB_RESULT = "lab_result"
    MEASUREMENT = "measurement"
    MEDICATION = "medication"
    PROBLEM = "problem"
    PROCEDURE = "procedure"
    STRUCTURED_CODE = "structured_code"
    STRUCTURED_EVENT = "structured_event"
    STRUCTURED_FACT = "structured_fact"
    STRUCTURED_MEASUREMENT = "structured_measurement"
    SUMMARY_CARD = "summary_card"


# Short alias for callers that use the generic evidence terminology.
EvidenceType = SummaryEvidenceType
SUMMARY_EVIDENCE_TYPES = frozenset(item.value for item in SummaryEvidenceType)
DEFAULT_ALLOWED_EVIDENCE_TYPES = SUMMARY_EVIDENCE_TYPES

SUMMARY_REVIEW_STATUSES = frozenset({"approved", "reviewed", "verified"})


class SummaryInputRejectionCategory(str, Enum):
    """Stable, PHI-free categories used by validation reports."""

    INVALID_CONTAINER = "invalid_container"
    MISSING_EVIDENCE_TYPE = "missing_evidence_type"
    UNKNOWN_EVIDENCE_TYPE = "unknown_evidence_type"
    MISSING_SOURCE_REFERENCE = "missing_source_reference"
    INVALID_SOURCE_REFERENCE = "invalid_source_reference"
    MISSING_POLICY_FINGERPRINT = "missing_policy_fingerprint"
    INVALID_POLICY_FINGERPRINT = "invalid_policy_fingerprint"
    POLICY_FINGERPRINT_MISMATCH = "policy_fingerprint_mismatch"
    MISSING_REVIEW_STATUS = "missing_review_status"
    UNVERIFIED_REVIEW_STATUS = "unverified_review_status"
    RAW_FIELD = "raw_field"
    INVALID_SAFE_FIELD = "invalid_safe_field"


REJECTION_CATEGORIES = tuple(item.value for item in SummaryInputRejectionCategory)

_POLICY_FINGERPRINT_RE = re.compile(r"^sha256:[0-9a-f]{64}$", re.ASCII)
_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$", re.ASCII)
_FIELD_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$", re.ASCII)
_RAW_FIELD_NAMES = frozenset(
    {
        "content",
        "description",
        "display",
        "document_text",
        "excerpt",
        "identifier",
        "metadata",
        "name",
        "note",
        "patient_id",
        "payload",
        "raw",
        "raw_text",
        "source_text",
        "surface",
        "text",
        "value",
    }
)
_SAFE_FIELD_NAMES = frozenset(
    {
        "assertion",
        "category",
        "code",
        "coding_system",
        "confidence_bucket",
        "count",
        "direction",
        "event_type",
        "experiencer",
        "source_kind",
        "status",
        "temporality",
        "trend",
        "unit",
        "value_kind",
    }
)
_SAFE_FIELD_VALUE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+%=-]{0,127}$", re.ASCII)
_EVIDENCE_KEY_ALIASES = {
    "attributes": "fields",
    "evidence_type": "evidence_type",
    "fields": "fields",
    "policy": "policy_fingerprint",
    "policy_fingerprint": "policy_fingerprint",
    "review": "review_status",
    "review_status": "review_status",
    "source_ref": "source_ref",
    "source_reference": "source_ref",
    "summary_fields": "fields",
    "type": "evidence_type",
}
_ENVELOPE_KEYS = frozenset({"evidence", "items"})


def _enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def _safe_source_token(value: Any) -> str:
    if not isinstance(value, str) or not _SAFE_TOKEN_RE.fullmatch(value):
        raise ValueError("source reference must be an opaque token")
    if re.search(r"\d{6,}", value, re.ASCII):
        raise ValueError("source reference must not contain a long identifier")
    return value


def _normalize_policy_fingerprint(value: Any) -> str:
    value = _enum_value(value)
    if not isinstance(value, str) or not _POLICY_FINGERPRINT_RE.fullmatch(value):
        raise ValueError("policy fingerprint must be a canonical SHA-256 value")
    return value


def _normalize_review_status(value: Any) -> str:
    value = _enum_value(value)
    if not isinstance(value, str):
        raise ValueError("review status is not approved")
    normalized = value.strip().casefold().replace("-", "_").replace(" ", "_")
    if normalized not in SUMMARY_REVIEW_STATUSES:
        raise ValueError("review status is not approved")
    return "approved"


def _normalize_evidence_type(value: Any) -> str:
    value = _enum_value(value)
    if not isinstance(value, str):
        raise ValueError("evidence type is not allowed")
    normalized = value.strip().casefold().replace("-", "_").replace(" ", "_")
    if normalized not in SUMMARY_EVIDENCE_TYPES:
        raise ValueError("evidence type is not allowed")
    return normalized


def _normalize_allowed_evidence_types(
    values: Iterable[str] | None,
) -> frozenset[str]:
    if values is None:
        return DEFAULT_ALLOWED_EVIDENCE_TYPES
    if isinstance(values, (str, bytes)):
        raise TypeError("allowed evidence types must be an iterable of types")
    normalized = frozenset(_normalize_evidence_type(value) for value in values)
    if not normalized:
        raise ValueError("at least one allowed evidence type is required")
    return normalized


@dataclass(frozen=True)
class SummarySourceReference:
    """An opaque, PHI-free reference to evidence provenance.

    ``source_id`` is deliberately a token rather than source text.  Callers
    may use a local token such as ``source:synthetic-note-001`` or a SHA-256
    reference.  Optional offsets identify a reviewed span without copying it.
    """

    source_id: str
    start: int | None = None
    end: int | None = None
    kind: str = "document"

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _safe_source_token(self.source_id))
        kind = _safe_source_token(self.kind)
        object.__setattr__(self, "kind", kind)
        if (self.start is None) != (self.end is None):
            raise ValueError("source reference offsets must be supplied together")
        if self.start is not None and (
            isinstance(self.start, bool)
            or not isinstance(self.start, int)
            or isinstance(self.end, bool)
            or not isinstance(self.end, int)
            or self.start < 0
            or self.end <= self.start
        ):
            raise ValueError("source reference offsets are invalid")

    @classmethod
    def from_obj(cls, value: Any) -> "SummarySourceReference":
        """Build a source reference from a token or a safe reference mapping."""

        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(source_id=value)
        if not isinstance(value, Mapping):
            raise ValueError("source reference must be a token or mapping")

        aliases = {
            "document_id": "source_id",
            "id": "source_id",
            "kind": "kind",
            "ref": "source_id",
            "source_id": "source_id",
            "start": "start",
            "end": "end",
        }
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or key.casefold() not in aliases:
                raise ValueError("source reference contains an unsupported field")
            canonical = aliases[key.casefold()]
            if canonical in normalized:
                raise ValueError("source reference contains duplicate fields")
            normalized[canonical] = item
        if "source_id" not in normalized:
            raise ValueError("source reference is missing its opaque identifier")
        return cls(**normalized)

    def to_dict(self) -> dict[str, Any]:
        """Return the safe, deterministic source-reference representation."""

        payload: dict[str, Any] = {"source_id": self.source_id, "kind": self.kind}
        if self.start is not None:
            payload["start"] = self.start
            payload["end"] = self.end
        return payload


def _normalize_safe_fields(fields: Any) -> dict[str, bool | float | int | str]:
    if fields is None:
        return {}
    if not isinstance(fields, Mapping):
        raise ValueError("summary fields must be a mapping")

    normalized: dict[str, bool | float | int | str] = {}
    for raw_name, value in fields.items():
        if not isinstance(raw_name, str):
            raise ValueError("summary fields contain an unapproved field")
        name = raw_name.strip().casefold().replace("-", "_").replace(" ", "_")
        if not _FIELD_NAME_RE.fullmatch(name) or name not in _SAFE_FIELD_NAMES:
            raise ValueError("summary fields contain an unapproved field")
        if isinstance(value, bool):
            safe_value: bool | float | int | str = value
        elif isinstance(value, int):
            if value < 0:
                raise ValueError("summary field counts must be non-negative")
            safe_value = value
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("summary field numbers must be finite")
            safe_value = value
        elif isinstance(value, str) and _SAFE_FIELD_VALUE_RE.fullmatch(value):
            safe_value = value
        else:
            raise ValueError("summary field value is not an approved scalar")
        if name == "count" and (
            isinstance(safe_value, bool) or not isinstance(safe_value, int)
        ):
            raise ValueError("summary field count must be a non-negative integer")
        normalized[name] = safe_value
    return dict(sorted(normalized.items()))


def _safe_fields_rejection_category(fields: Any) -> str:
    """Classify a malformed field mapping without exposing its contents."""

    if not isinstance(fields, Mapping):
        return SummaryInputRejectionCategory.INVALID_SAFE_FIELD.value
    for raw_name in fields:
        if not isinstance(raw_name, str):
            return SummaryInputRejectionCategory.RAW_FIELD.value
        name = raw_name.strip().casefold().replace("-", "_").replace(" ", "_")
        if not _FIELD_NAME_RE.fullmatch(name) or name not in _SAFE_FIELD_NAMES:
            return SummaryInputRejectionCategory.RAW_FIELD.value
    return SummaryInputRejectionCategory.INVALID_SAFE_FIELD.value


@dataclass(frozen=True)
class SummaryEvidence:
    """One typed, policy-bound, reviewed evidence record.

    Only small, scalar summary fields are accepted.  Free text, raw values,
    identifiers, excerpts, and arbitrary payload mappings are intentionally not
    part of this type.
    """

    evidence_type: str | SummaryEvidenceType
    source_ref: SummarySourceReference | str | Mapping[str, Any]
    policy_fingerprint: str
    review_status: str
    fields: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "evidence_type", _normalize_evidence_type(self.evidence_type)
        )
        object.__setattr__(
            self, "source_ref", SummarySourceReference.from_obj(self.source_ref)
        )
        object.__setattr__(
            self,
            "policy_fingerprint",
            _normalize_policy_fingerprint(self.policy_fingerprint),
        )
        object.__setattr__(
            self, "review_status", _normalize_review_status(self.review_status)
        )
        object.__setattr__(self, "fields", _normalize_safe_fields(self.fields))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SummaryEvidence":
        """Rebuild a validated evidence record from its safe wire form."""

        if not isinstance(value, Mapping):
            raise TypeError("summary evidence must be a mapping")
        return cls(
            evidence_type=value.get("evidence_type"),
            source_ref=value.get("source_ref"),
            policy_fingerprint=value.get("policy_fingerprint"),
            review_status=value.get("review_status"),
            fields=value.get("fields", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, PHI-free evidence envelope."""

        return {
            "evidence_type": self.evidence_type,
            "fields": dict(self.fields),
            "policy_fingerprint": self.policy_fingerprint,
            "review_status": self.review_status,
            "source_ref": self.source_ref.to_dict(),
        }


@dataclass(frozen=True)
class SummaryInputValidationResult:
    """Counts-only result of validating a summary input collection."""

    accepted: tuple[SummaryEvidence, ...] = ()
    rejection_counts: Mapping[str, int] = field(default_factory=dict)
    total_count: int = 0
    schema_version: int = SUMMARY_INPUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        accepted = tuple(self.accepted)
        if any(not isinstance(item, SummaryEvidence) for item in accepted):
            raise TypeError("accepted summary input must contain SummaryEvidence")
        if (
            isinstance(self.total_count, bool)
            or not isinstance(self.total_count, int)
            or self.total_count < len(accepted)
        ):
            raise ValueError("summary input total count is invalid")
        counts: dict[str, int] = {}
        for category, count in self.rejection_counts.items():
            category = _enum_value(category)
            if category not in REJECTION_CATEGORIES:
                raise ValueError("summary input rejection category is not supported")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("summary input rejection count is invalid")
            if count:
                counts[str(category)] = count
        object.__setattr__(self, "accepted", accepted)
        object.__setattr__(self, "rejection_counts", dict(sorted(counts.items())))

    @property
    def valid(self) -> bool:
        """Return whether every supplied item passed the contract."""

        return not self.rejection_counts

    @property
    def is_valid(self) -> bool:
        """Alias for :attr:`valid` used by validation-oriented callers."""

        return self.valid

    @property
    def accepted_count(self) -> int:
        """Return the number of evidence items admitted to the summarizer."""

        return len(self.accepted)

    @property
    def rejected_count(self) -> int:
        """Return the number of evidence items not admitted."""

        return self.total_count - self.accepted_count

    @property
    def evidence(self) -> tuple[SummaryEvidence, ...]:
        """Return the validated evidence tuple for a summarizer."""

        return self.accepted

    @property
    def nonzero_rejection_counts(self) -> dict[str, int]:
        """Return only non-zero rejection categories in stable order."""

        return dict(self.rejection_counts)

    def to_dict(self) -> dict[str, Any]:
        """Return a counts-only, deterministic validation report."""

        return {
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "rejection_counts": {
                category: self.rejection_counts.get(category, 0)
                for category in REJECTION_CATEGORIES
            },
            "schema_version": self.schema_version,
            "valid": self.valid,
        }

    def to_json(self) -> str:
        """Serialize the counts-only report with canonical JSON settings."""

        return json.dumps(self.to_dict(), allow_nan=False, separators=(",", ":"))

    def require_valid(self) -> "SummaryInputValidationResult":
        """Raise a PHI-free error unless all evidence passed validation."""

        if not self.valid:
            raise SummaryInputValidationError(self)
        return self


class SummaryInputValidationError(ValueError):
    """Raised when a caller requires a fully valid summary input."""

    def __init__(self, result: SummaryInputValidationResult) -> None:
        self.result = result
        summary = ", ".join(
            f"{category}={count}" for category, count in result.rejection_counts.items()
        )
        super().__init__(f"summary input rejected: {summary}")


# Backward-friendly name for callers that use the shorter error terminology.
SummaryInputError = SummaryInputValidationError


def _resolve_expected_policy_fingerprint(
    policy_fingerprint: str | None,
    expected_policy_fingerprint: str | None,
) -> str | None:
    if policy_fingerprint is not None and expected_policy_fingerprint is not None:
        first = _normalize_policy_fingerprint(policy_fingerprint)
        second = _normalize_policy_fingerprint(expected_policy_fingerprint)
        if first != second:
            raise ValueError("policy fingerprint arguments disagree")
        return first
    value = (
        policy_fingerprint
        if policy_fingerprint is not None
        else expected_policy_fingerprint
    )
    return None if value is None else _normalize_policy_fingerprint(value)


def _mapping_to_evidence(
    value: Mapping[str, Any],
    *,
    allowed_evidence_types: frozenset[str],
    expected_policy_fingerprint: str | None,
) -> tuple[SummaryEvidence | None, tuple[str, ...]]:
    failures: set[str] = set()
    normalized: dict[str, Any] = {}
    for raw_key, item in value.items():
        if not isinstance(raw_key, str):
            failures.add(SummaryInputRejectionCategory.RAW_FIELD.value)
            continue
        key = raw_key.strip().casefold().replace("-", "_").replace(" ", "_")
        canonical = _EVIDENCE_KEY_ALIASES.get(key)
        if canonical is None or canonical in normalized:
            failures.add(SummaryInputRejectionCategory.RAW_FIELD.value)
            continue
        normalized[canonical] = item

    required = {
        "evidence_type": SummaryInputRejectionCategory.MISSING_EVIDENCE_TYPE.value,
        "source_ref": SummaryInputRejectionCategory.MISSING_SOURCE_REFERENCE.value,
        "policy_fingerprint": SummaryInputRejectionCategory.MISSING_POLICY_FINGERPRINT.value,
        "review_status": SummaryInputRejectionCategory.MISSING_REVIEW_STATUS.value,
    }
    for key, category in required.items():
        if key not in normalized:
            failures.add(category)

    evidence_type: str | None = None
    if "evidence_type" in normalized:
        try:
            evidence_type = _normalize_evidence_type(normalized["evidence_type"])
        except (TypeError, ValueError):
            failures.add(SummaryInputRejectionCategory.UNKNOWN_EVIDENCE_TYPE.value)
        else:
            if evidence_type not in allowed_evidence_types:
                failures.add(SummaryInputRejectionCategory.UNKNOWN_EVIDENCE_TYPE.value)

    source_ref: SummarySourceReference | None = None
    if "source_ref" in normalized:
        try:
            source_ref = SummarySourceReference.from_obj(normalized["source_ref"])
        except (TypeError, ValueError):
            failures.add(SummaryInputRejectionCategory.INVALID_SOURCE_REFERENCE.value)

    fingerprint: str | None = None
    if "policy_fingerprint" in normalized:
        try:
            fingerprint = _normalize_policy_fingerprint(
                normalized["policy_fingerprint"]
            )
        except (TypeError, ValueError):
            failures.add(SummaryInputRejectionCategory.INVALID_POLICY_FINGERPRINT.value)
        else:
            if (
                expected_policy_fingerprint is not None
                and fingerprint != expected_policy_fingerprint
            ):
                failures.add(
                    SummaryInputRejectionCategory.POLICY_FINGERPRINT_MISMATCH.value
                )

    review_status: str | None = None
    if "review_status" in normalized:
        try:
            review_status = _normalize_review_status(normalized["review_status"])
        except (TypeError, ValueError):
            failures.add(SummaryInputRejectionCategory.UNVERIFIED_REVIEW_STATUS.value)

    fields: dict[str, bool | float | int | str] = {}
    if "fields" in normalized:
        try:
            fields = _normalize_safe_fields(normalized["fields"])
        except (TypeError, ValueError):
            failures.add(_safe_fields_rejection_category(normalized["fields"]))

    if failures:
        return None, _ordered_categories(failures)
    if (
        evidence_type is None
        or source_ref is None
        or fingerprint is None
        or review_status is None
    ):
        return None, (SummaryInputRejectionCategory.INVALID_CONTAINER.value,)
    return (
        SummaryEvidence(
            evidence_type=evidence_type,
            source_ref=source_ref,
            policy_fingerprint=fingerprint,
            review_status=review_status,
            fields=fields,
        ),
        (),
    )


def _instance_failures(
    value: SummaryEvidence,
    *,
    allowed_evidence_types: frozenset[str],
    expected_policy_fingerprint: str | None,
) -> tuple[str, ...]:
    failures: set[str] = set()
    if value.evidence_type not in allowed_evidence_types:
        failures.add(SummaryInputRejectionCategory.UNKNOWN_EVIDENCE_TYPE.value)
    if not isinstance(value.source_ref, SummarySourceReference):
        failures.add(SummaryInputRejectionCategory.INVALID_SOURCE_REFERENCE.value)
    try:
        fingerprint = _normalize_policy_fingerprint(value.policy_fingerprint)
    except (TypeError, ValueError):
        failures.add(SummaryInputRejectionCategory.INVALID_POLICY_FINGERPRINT.value)
    else:
        if (
            expected_policy_fingerprint is not None
            and fingerprint != expected_policy_fingerprint
        ):
            failures.add(
                SummaryInputRejectionCategory.POLICY_FINGERPRINT_MISMATCH.value
            )
    try:
        _normalize_review_status(value.review_status)
    except (TypeError, ValueError):
        failures.add(SummaryInputRejectionCategory.UNVERIFIED_REVIEW_STATUS.value)
    try:
        _normalize_safe_fields(value.fields)
    except (TypeError, ValueError):
        failures.add(_safe_fields_rejection_category(value.fields))
    return _ordered_categories(failures)


def _ordered_categories(categories: Iterable[str]) -> tuple[str, ...]:
    category_set = set(categories)
    return tuple(
        category for category in REJECTION_CATEGORIES if category in category_set
    )


def _input_items(value: Any) -> tuple[tuple[Any, ...] | None, str | None]:
    if isinstance(value, SummaryEvidence):
        return (value,), None
    if isinstance(value, Mapping):
        keys = {key.casefold() for key in value if isinstance(key, str)}
        envelope = keys & _ENVELOPE_KEYS
        if envelope and not (keys & set(_EVIDENCE_KEY_ALIASES)):
            if len(envelope) != 1 or len(keys) != 1:
                return None, SummaryInputRejectionCategory.INVALID_CONTAINER.value
            item_key = next(iter(envelope))
            nested = value[item_key]
            if isinstance(nested, (str, bytes, Mapping)) or nested is None:
                return None, SummaryInputRejectionCategory.INVALID_CONTAINER.value
            try:
                return tuple(nested), None
            except TypeError:
                return None, SummaryInputRejectionCategory.INVALID_CONTAINER.value
        return (value,), None
    if value is None or isinstance(value, (str, bytes)):
        return None, SummaryInputRejectionCategory.INVALID_CONTAINER.value
    try:
        return tuple(value), None
    except TypeError:
        return None, SummaryInputRejectionCategory.INVALID_CONTAINER.value


def _evidence_sort_key(value: SummaryEvidence) -> tuple[Any, ...]:
    reference = value.source_ref
    return (
        reference.source_id,
        -1 if reference.start is None else reference.start,
        -1 if reference.end is None else reference.end,
        value.evidence_type,
        value.policy_fingerprint,
        value.review_status,
        json.dumps(value.fields, sort_keys=True, separators=(",", ":")),
    )


def validate_summary_input(
    evidence: Any,
    *,
    policy_fingerprint: str | None = None,
    expected_policy_fingerprint: str | None = None,
    allowed_evidence_types: Iterable[str] | None = None,
) -> SummaryInputValidationResult:
    """Validate evidence before passing it to a summary generator.

    Args:
        evidence: One :class:`SummaryEvidence`, a safe wire mapping, or an
            iterable of either.  A mapping with one ``evidence`` or ``items``
            key is accepted as a batch envelope.
        policy_fingerprint: Fingerprint that every item must match.  The
            ``expected_policy_fingerprint`` spelling is an equivalent alias.
        expected_policy_fingerprint: Alias for ``policy_fingerprint``.
        allowed_evidence_types: Optional restriction of the built-in evidence
            type allowlist.

    Returns:
        A deterministic result.  Rejection details are counts only; no input
        value, source content, or exception text is copied into the result.

    The validator never performs a network call and fails closed for malformed
    containers, untyped records, raw fields, unbound policy fingerprints, and
    unapproved review statuses.
    """

    expected = _resolve_expected_policy_fingerprint(
        policy_fingerprint, expected_policy_fingerprint
    )
    allowed = _normalize_allowed_evidence_types(allowed_evidence_types)
    items, container_failure = _input_items(evidence)
    if container_failure is not None or items is None:
        return SummaryInputValidationResult(
            rejection_counts={
                container_failure
                or SummaryInputRejectionCategory.INVALID_CONTAINER.value: 1
            },
            total_count=1,
        )

    accepted: list[SummaryEvidence] = []
    rejection_counts: Counter[str] = Counter()
    for item in items:
        if isinstance(item, SummaryEvidence):
            normalized, failures = (
                item,
                _instance_failures(
                    item,
                    allowed_evidence_types=allowed,
                    expected_policy_fingerprint=expected,
                ),
            )
        elif isinstance(item, Mapping):
            normalized, failures = _mapping_to_evidence(
                item,
                allowed_evidence_types=allowed,
                expected_policy_fingerprint=expected,
            )
        else:
            normalized, failures = (
                None,
                (SummaryInputRejectionCategory.INVALID_CONTAINER.value,),
            )
        if failures:
            rejection_counts.update(failures)
        elif normalized is not None:
            accepted.append(normalized)

    accepted.sort(key=_evidence_sort_key)
    return SummaryInputValidationResult(
        accepted=tuple(accepted),
        rejection_counts=rejection_counts,
        total_count=len(items),
    )


def guard_summary_input(
    evidence: Any,
    *,
    policy_fingerprint: str | None = None,
    expected_policy_fingerprint: str | None = None,
    allowed_evidence_types: Iterable[str] | None = None,
) -> tuple[SummaryEvidence, ...]:
    """Return only admitted evidence, raising if any item is rejected."""

    result = validate_summary_input(
        evidence,
        policy_fingerprint=policy_fingerprint,
        expected_policy_fingerprint=expected_policy_fingerprint,
        allowed_evidence_types=allowed_evidence_types,
    )
    result.require_valid()
    return result.accepted


def build_summary_input(
    evidence: Any,
    *,
    policy_fingerprint: str | None = None,
    expected_policy_fingerprint: str | None = None,
    allowed_evidence_types: Iterable[str] | None = None,
) -> tuple[SummaryEvidence, ...]:
    """Build the guarded tuple consumed by a summary generator."""

    return guard_summary_input(
        evidence,
        policy_fingerprint=policy_fingerprint,
        expected_policy_fingerprint=expected_policy_fingerprint,
        allowed_evidence_types=allowed_evidence_types,
    )


validate_summary_inputs = validate_summary_input


@dataclass(frozen=True)
class SummaryInputContract:
    """Reusable validator configuration for one summary pipeline."""

    policy_fingerprint: str | None = None
    allowed_evidence_types: Iterable[str] | None = None

    def __post_init__(self) -> None:
        fingerprint = _resolve_expected_policy_fingerprint(
            self.policy_fingerprint, None
        )
        allowed = _normalize_allowed_evidence_types(self.allowed_evidence_types)
        object.__setattr__(self, "policy_fingerprint", fingerprint)
        object.__setattr__(self, "allowed_evidence_types", allowed)

    @property
    def expected_policy_fingerprint(self) -> str | None:
        """Return the configured policy fingerprint, if one was pinned."""

        return self.policy_fingerprint

    def validate(self, evidence: Any) -> SummaryInputValidationResult:
        """Validate evidence against this contract configuration."""

        return validate_summary_input(
            evidence,
            policy_fingerprint=self.policy_fingerprint,
            allowed_evidence_types=self.allowed_evidence_types,
        )

    def require_valid(self, evidence: Any) -> tuple[SummaryEvidence, ...]:
        """Return admitted evidence or raise a counts-only validation error."""

        return self.validate(evidence).require_valid().accepted


__all__ = [
    "DEFAULT_ALLOWED_EVIDENCE_TYPES",
    "EvidenceType",
    "REJECTION_CATEGORIES",
    "SUMMARY_EVIDENCE_TYPES",
    "SUMMARY_INPUT_ADVISORY",
    "SUMMARY_INPUT_SCHEMA_VERSION",
    "SUMMARY_REVIEW_STATUSES",
    "SummaryEvidence",
    "SummaryEvidenceType",
    "SummaryInputContract",
    "SummaryInputError",
    "SummaryInputRejectionCategory",
    "SummaryInputValidationError",
    "SummaryInputValidationResult",
    "SummarySourceReference",
    "build_summary_input",
    "guard_summary_input",
    "validate_summary_input",
    "validate_summary_inputs",
]
