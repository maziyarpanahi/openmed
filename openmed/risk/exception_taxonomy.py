"""PHI-safe taxonomy validation for intentional local exceptions.

The validator is deliberately closed-world. Exception records may contain only
versioned, allow-listed categories, reason codes, scopes, evidence digests,
expiry timestamps, and owner-free approval metadata. It never stores or echoes
an arbitrary payload, free-form reason, identifier, path, or exception text.

Validation is local and deterministic when callers provide an explicit
``as_of`` time. No network, telemetry client, filesystem, or clock access is
needed by the default path.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Final

EXCEPTION_TAXONOMY_SCHEMA_VERSION: Final = 1
EXCEPTION_TAXONOMY_VERSION: Final = "1.0"
EXCEPTION_TAXONOMY_REPORT_TYPE: Final = "no_phi_exception_taxonomy"

# Short aliases make the version contract easy to discover for callers that
# use the generic taxonomy terminology.
TAXONOMY_SCHEMA_VERSION: Final = EXCEPTION_TAXONOMY_SCHEMA_VERSION
TAXONOMY_VERSION: Final = EXCEPTION_TAXONOMY_VERSION

EXCEPTION_SURFACES: Final = ("telemetry", "audit")
EXCEPTION_CATEGORIES: Final = (
    "local_suppression",
    "local_allowance",
    "synthetic_fixture",
    "operational_fallback",
)
EVIDENCE_KINDS: Final = (
    "test",
    "review",
    "policy",
    "fixture",
    "incident",
)
APPROVAL_STATUSES: Final = ("approved", "reviewed")
APPROVAL_ROLES: Final = (
    "privacy_reviewer",
    "release_reviewer",
    "maintainer",
    "test_reviewer",
)

_DIGEST_RE: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_UTC: Final = timezone.utc
_MAX_EVIDENCE_REFERENCES: Final = 8
_MAX_RECORD_KEYS: Final = 8
_MAX_DATETIME_LENGTH: Final = 40

_RECORD_FIELDS: Final = frozenset(
    {
        "schema_version",
        "taxonomy_version",
        "category",
        "reason_code",
        "scope",
        "evidence",
        "expires_at",
        "approval",
    }
)
_EVIDENCE_FIELDS: Final = frozenset({"kind", "digest"})
_APPROVAL_FIELDS: Final = frozenset(
    {"status", "role", "approval_digest", "approved_at"}
)
_TAXONOMY_FIELDS: Final = frozenset(
    {"schema_version", "taxonomy_version", "report_type", "categories"}
)

_REASON_CODES: Final = frozenset(
    {
        "false_positive_reviewed",
        "policy_exclusion",
        "compatibility_boundary",
        "synthetic_only",
        "bounded_degradation",
    }
)


class ExceptionTaxonomyError(ValueError):
    """Raised when a typed exception record cannot be constructed safely."""


_FINDING_MESSAGES: Final = {
    "RECORD_NOT_MAPPING": "exception record must be a mapping",
    "UNSUPPORTED_SURFACE": "validation surface is not supported",
    "UNSUPPORTED_FIELD": "record contains a field outside the closed schema",
    "MISSING_FIELD": "record is missing a required field",
    "INVALID_TYPE": "record field has an invalid type",
    "UNSUPPORTED_SCHEMA_VERSION": "record schema version is not supported",
    "UNSUPPORTED_TAXONOMY_VERSION": "record taxonomy version is not supported",
    "UNSUPPORTED_CATEGORY": "exception category is not in the taxonomy",
    "UNSUPPORTED_REASON_CODE": "exception reason code is not in the taxonomy",
    "UNSUPPORTED_SCOPE": "exception scope is not supported",
    "INVALID_EVIDENCE": "evidence reference is invalid",
    "DUPLICATE_EVIDENCE": "evidence kinds must be unique",
    "TOO_MANY_EVIDENCE": "record contains too many evidence references",
    "MISSING_REQUIRED_EVIDENCE": "required evidence is missing",
    "INVALID_DIGEST": "evidence and approval values must be SHA-256 digests",
    "INVALID_TIMESTAMP": "timestamp must be an explicit UTC value",
    "EXPIRY_BEFORE_APPROVAL": "exception expiry must be after approval",
    "EXPIRY_TOO_LONG": "exception expiry exceeds the category bound",
    "EXPIRED": "exception has expired at the validation time",
    "APPROVAL_NOT_EFFECTIVE": "approval is not effective at the validation time",
    "INVALID_APPROVAL": "approval metadata is invalid",
    "UNSUPPORTED_APPROVAL_STATUS": "approval status is not supported",
    "UNSUPPORTED_APPROVAL_ROLE": "approval role is not supported",
    "INVALID_TAXONOMY": "taxonomy definition does not match the supported version",
}


@dataclass(frozen=True)
class TaxonomyFinding:
    """A stable, non-sensitive validation finding."""

    code: str
    path: str

    def __post_init__(self) -> None:
        if self.code not in _FINDING_MESSAGES:
            raise ValueError("unknown taxonomy finding code")
        if self.path not in {
            "$",
            "$.schema_version",
            "$.taxonomy_version",
            "$.category",
            "$.reason_code",
            "$.scope",
            "$.evidence",
            "$.evidence[*]",
            "$.evidence[*].kind",
            "$.evidence[*].digest",
            "$.expires_at",
            "$.approval",
            "$.approval.status",
            "$.approval.role",
            "$.approval.approval_digest",
            "$.approval.approved_at",
            "$.surface",
        }:
            raise ValueError("taxonomy finding path must be structural")

    @property
    def message(self) -> str:
        """Return a fixed message that never includes input data."""

        return _FINDING_MESSAGES[self.code]

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-safe finding without arbitrary input values."""

        return {"code": self.code, "message": self.message, "path": self.path}


@dataclass(frozen=True)
class EvidenceReference:
    """A typed, content-free reference to review evidence."""

    kind: str
    digest: str

    def __post_init__(self) -> None:
        if type(self.kind) is not str or self.kind not in EVIDENCE_KINDS:
            raise ExceptionTaxonomyError("unsupported evidence kind")
        _require_digest(self.digest)

    def to_dict(self) -> dict[str, str]:
        """Return the allow-listed evidence representation."""

        return {"kind": self.kind, "digest": self.digest}


@dataclass(frozen=True)
class ApprovalMetadata:
    """Owner-free approval metadata for a local exception.

    The schema intentionally records a role and digest instead of a person,
    email address, ticket body, or free-form approver identity.
    """

    status: str
    role: str
    approval_digest: str
    approved_at: datetime

    def __post_init__(self) -> None:
        if type(self.status) is not str or self.status not in APPROVAL_STATUSES:
            raise ExceptionTaxonomyError("unsupported approval status")
        if type(self.role) is not str or self.role not in APPROVAL_ROLES:
            raise ExceptionTaxonomyError("unsupported approval role")
        _require_digest(self.approval_digest)
        normalized = _require_utc_datetime(self.approved_at)
        object.__setattr__(self, "approved_at", normalized)

    def to_dict(self) -> dict[str, Any]:
        """Return approval metadata without an owner or free-form payload."""

        return {
            "status": self.status,
            "role": self.role,
            "approval_digest": self.approval_digest,
            "approved_at": _format_utc_datetime(self.approved_at),
        }


@dataclass(frozen=True)
class TaxonomyRule:
    """The fixed requirements for one exception category."""

    category: str
    reason_codes: tuple[str, ...]
    required_evidence: tuple[str, ...]
    max_expiry_days: int
    allowed_scopes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned rule in deterministic order."""

        return {
            "category": self.category,
            "reason_codes": list(self.reason_codes),
            "required_evidence": list(self.required_evidence),
            "max_expiry_days": self.max_expiry_days,
            "allowed_scopes": list(self.allowed_scopes),
        }


_RULE_DEFINITIONS: Final = {
    "local_suppression": TaxonomyRule(
        category="local_suppression",
        reason_codes=("false_positive_reviewed", "policy_exclusion"),
        required_evidence=("test", "review"),
        max_expiry_days=90,
        allowed_scopes=EXCEPTION_SURFACES,
    ),
    "local_allowance": TaxonomyRule(
        category="local_allowance",
        reason_codes=("false_positive_reviewed", "compatibility_boundary"),
        required_evidence=("test", "review"),
        max_expiry_days=90,
        allowed_scopes=EXCEPTION_SURFACES,
    ),
    "synthetic_fixture": TaxonomyRule(
        category="synthetic_fixture",
        reason_codes=("synthetic_only",),
        required_evidence=("fixture", "test"),
        max_expiry_days=30,
        allowed_scopes=EXCEPTION_SURFACES,
    ),
    "operational_fallback": TaxonomyRule(
        category="operational_fallback",
        reason_codes=("bounded_degradation",),
        required_evidence=("incident", "test"),
        max_expiry_days=7,
        allowed_scopes=EXCEPTION_SURFACES,
    ),
}


def _default_rules() -> tuple[TaxonomyRule, ...]:
    return tuple(_RULE_DEFINITIONS[category] for category in EXCEPTION_CATEGORIES)


@dataclass(frozen=True)
class ExceptionRecord:
    """A canonical, PHI-free exception record.

    Use :func:`validate_exception_record` before emitting a record to telemetry
    or an audit log. The constructor validates field types and allow-lists; the
    taxonomy validator additionally checks category-specific requirements and
    expiry at a caller-supplied time.
    """

    category: str
    reason_code: str
    scope: str
    evidence: tuple[EvidenceReference, ...]
    expires_at: datetime
    approval: ApprovalMetadata
    taxonomy_version: str = EXCEPTION_TAXONOMY_VERSION
    schema_version: int = EXCEPTION_TAXONOMY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int:
            raise ExceptionTaxonomyError("invalid schema version")
        if self.schema_version != EXCEPTION_TAXONOMY_SCHEMA_VERSION:
            raise ExceptionTaxonomyError("unsupported schema version")
        if (
            type(self.taxonomy_version) is not str
            or self.taxonomy_version != EXCEPTION_TAXONOMY_VERSION
        ):
            raise ExceptionTaxonomyError("unsupported taxonomy version")
        if type(self.category) is not str or self.category not in EXCEPTION_CATEGORIES:
            raise ExceptionTaxonomyError("unsupported exception category")
        rule = _RULE_DEFINITIONS[self.category]
        if (
            type(self.reason_code) is not str
            or self.reason_code not in rule.reason_codes
        ):
            raise ExceptionTaxonomyError("unsupported exception reason code")
        if type(self.scope) is not str or self.scope not in rule.allowed_scopes:
            raise ExceptionTaxonomyError("unsupported exception scope")
        if not isinstance(self.evidence, tuple):
            raise ExceptionTaxonomyError("evidence must be a tuple")
        if len(self.evidence) > _MAX_EVIDENCE_REFERENCES:
            raise ExceptionTaxonomyError("too many evidence references")
        evidence_kinds: set[str] = set()
        for reference in self.evidence:
            if not isinstance(reference, EvidenceReference):
                raise ExceptionTaxonomyError("invalid evidence reference")
            if reference.kind in evidence_kinds:
                raise ExceptionTaxonomyError("duplicate evidence kind")
            evidence_kinds.add(reference.kind)
        normalized_expiry = _require_utc_datetime(self.expires_at)
        if not isinstance(self.approval, ApprovalMetadata):
            raise ExceptionTaxonomyError("invalid approval metadata")
        if normalized_expiry <= self.approval.approved_at:
            raise ExceptionTaxonomyError("expiry must be after approval")
        object.__setattr__(self, "expires_at", normalized_expiry)
        object.__setattr__(self, "evidence", tuple(self.evidence))

    def to_dict(self) -> dict[str, Any]:
        """Return the exact canonical record schema."""

        return {
            "schema_version": self.schema_version,
            "taxonomy_version": self.taxonomy_version,
            "category": self.category,
            "reason_code": self.reason_code,
            "scope": self.scope,
            "evidence": [item.to_dict() for item in self.evidence],
            "expires_at": _format_utc_datetime(self.expires_at),
            "approval": self.approval.to_dict(),
        }

    def to_json(self) -> str:
        """Serialize the record deterministically without arbitrary data."""

        return _canonical_json(self.to_dict())

    @property
    def digest(self) -> str:
        """Return a stable digest suitable for a safe audit reference."""

        return _sha256_digest(self.to_json().encode("utf-8"))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ExceptionRecord":
        """Construct a record only when the complete taxonomy is satisfied."""

        result = validate_exception_record(payload, surface="telemetry")
        if not result.valid or result.record is None:
            raise ExceptionTaxonomyError("invalid exception record")
        return result.record


@dataclass(frozen=True)
class ExceptionTaxonomy:
    """The immutable, versioned no-PHI exception taxonomy."""

    schema_version: int = EXCEPTION_TAXONOMY_SCHEMA_VERSION
    taxonomy_version: str = EXCEPTION_TAXONOMY_VERSION
    report_type: str = EXCEPTION_TAXONOMY_REPORT_TYPE
    rules: tuple[TaxonomyRule, ...] = _default_rules()

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != EXCEPTION_TAXONOMY_SCHEMA_VERSION
        ):
            raise ExceptionTaxonomyError("unsupported taxonomy schema version")
        if (
            type(self.taxonomy_version) is not str
            or self.taxonomy_version != EXCEPTION_TAXONOMY_VERSION
        ):
            raise ExceptionTaxonomyError("unsupported taxonomy version")
        if (
            type(self.report_type) is not str
            or self.report_type != EXCEPTION_TAXONOMY_REPORT_TYPE
        ):
            raise ExceptionTaxonomyError("unsupported taxonomy report type")
        if not isinstance(self.rules, tuple):
            raise ExceptionTaxonomyError("taxonomy rules must be a tuple")
        if tuple(self.rules) != _default_rules():
            raise ExceptionTaxonomyError("taxonomy rules are not supported")

    def rule_for(self, category: str) -> TaxonomyRule | None:
        """Return the fixed rule for a category, or ``None``."""

        return _RULE_DEFINITIONS.get(category)

    def to_dict(self) -> dict[str, Any]:
        """Return the complete taxonomy definition without free-form data."""

        return {
            "schema_version": self.schema_version,
            "taxonomy_version": self.taxonomy_version,
            "report_type": self.report_type,
            "categories": [rule.to_dict() for rule in self.rules],
        }

    def to_json(self) -> str:
        """Serialize the taxonomy deterministically."""

        return _canonical_json(self.to_dict())

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ExceptionTaxonomy":
        """Load only the exact bundled taxonomy schema."""

        if not isinstance(payload, Mapping):
            raise ExceptionTaxonomyError("invalid taxonomy")
        if set(payload) != _TAXONOMY_FIELDS:
            raise ExceptionTaxonomyError("invalid taxonomy")
        try:
            candidate = {
                "schema_version": payload["schema_version"],
                "taxonomy_version": payload["taxonomy_version"],
                "report_type": payload["report_type"],
                "categories": payload["categories"],
            }
            if _canonical_json(candidate) != _canonical_json(
                DEFAULT_EXCEPTION_TAXONOMY.to_dict()
            ):
                raise ExceptionTaxonomyError("invalid taxonomy")
        except (TypeError, ValueError, KeyError):
            raise ExceptionTaxonomyError("invalid taxonomy") from None
        return cls()


@dataclass(frozen=True)
class TaxonomyValidationResult:
    """A deterministic validation result containing no input payload."""

    surface: str
    valid: bool
    errors: tuple[TaxonomyFinding, ...]
    taxonomy_version: str = EXCEPTION_TAXONOMY_VERSION
    record_digest: str | None = None
    record: ExceptionRecord | None = None

    def __post_init__(self) -> None:
        if self.surface not in EXCEPTION_SURFACES:
            raise ValueError("unsupported validation surface")
        if not isinstance(self.errors, tuple):
            raise TypeError("errors must be a tuple")
        if self.valid != (not self.errors):
            raise ValueError("validation result validity does not match findings")
        if self.record is not None and not isinstance(self.record, ExceptionRecord):
            raise TypeError("record must be an ExceptionRecord")

    @property
    def error_codes(self) -> tuple[str, ...]:
        """Return stable finding codes in validation order."""

        return tuple(finding.code for finding in self.errors)

    def __bool__(self) -> bool:
        return self.valid

    def to_dict(self) -> dict[str, Any]:
        """Return a safe report with no normalized input record."""

        return {
            "taxonomy_version": self.taxonomy_version,
            "surface": self.surface,
            "valid": self.valid,
            "record_digest": self.record_digest,
            "error_count": len(self.errors),
            "errors": [finding.to_dict() for finding in self.errors],
        }

    def to_json(self) -> str:
        """Serialize the safe validation report deterministically."""

        return _canonical_json(self.to_dict())


DEFAULT_EXCEPTION_TAXONOMY: Final = ExceptionTaxonomy()


def validate_exception_record(
    record: Mapping[str, Any] | ExceptionRecord,
    *,
    surface: str = "telemetry",
    as_of: datetime | date | str | None = None,
    taxonomy: ExceptionTaxonomy = DEFAULT_EXCEPTION_TAXONOMY,
) -> TaxonomyValidationResult:
    """Validate one canonical exception record for telemetry or audit use.

    ``as_of`` is optional so structural validation remains deterministic and
    clock-free. When supplied, it must be an explicit UTC timestamp and expiry
    is evaluated at that timestamp. Findings contain only fixed codes, paths,
    and messages; arbitrary record values are never returned.
    """

    if surface not in EXCEPTION_SURFACES:
        return _result(
            surface="telemetry",
            errors=("UNSUPPORTED_SURFACE", "$.surface"),
        )
    if not isinstance(taxonomy, ExceptionTaxonomy):
        return _result(surface=surface, errors=("INVALID_TAXONOMY", "$"))

    if isinstance(record, ExceptionRecord):
        candidate = record
        findings: list[TaxonomyFinding] = []
    elif isinstance(record, Mapping):
        candidate, findings = _parse_record_mapping(record)
    else:
        return _result(
            surface=surface,
            errors=("RECORD_NOT_MAPPING", "$"),
        )

    if candidate is not None and not findings:
        _validate_candidate(candidate, surface=surface, as_of=as_of, findings=findings)

    if findings:
        return _result(surface=surface, errors=tuple(findings))
    assert candidate is not None
    return TaxonomyValidationResult(
        surface=surface,
        valid=True,
        errors=(),
        taxonomy_version=taxonomy.taxonomy_version,
        record_digest=candidate.digest,
        record=candidate,
    )


def validate_telemetry_record(
    record: Mapping[str, Any] | ExceptionRecord,
    *,
    as_of: datetime | date | str | None = None,
    taxonomy: ExceptionTaxonomy = DEFAULT_EXCEPTION_TAXONOMY,
) -> TaxonomyValidationResult:
    """Validate a PHI-free exception record emitted as telemetry."""

    return validate_exception_record(
        record,
        surface="telemetry",
        as_of=as_of,
        taxonomy=taxonomy,
    )


def validate_audit_record(
    record: Mapping[str, Any] | ExceptionRecord,
    *,
    as_of: datetime | date | str | None = None,
    taxonomy: ExceptionTaxonomy = DEFAULT_EXCEPTION_TAXONOMY,
) -> TaxonomyValidationResult:
    """Validate a PHI-free exception record emitted to an audit log."""

    return validate_exception_record(
        record,
        surface="audit",
        as_of=as_of,
        taxonomy=taxonomy,
    )


# Concise aliases for callers that already use the surface as the verb.
validate_telemetry = validate_telemetry_record
validate_audit = validate_audit_record
validate_record = validate_exception_record


def _parse_record_mapping(
    payload: Mapping[str, Any],
) -> tuple[ExceptionRecord | None, list[TaxonomyFinding]]:
    findings: list[TaxonomyFinding] = []
    if len(payload) > _MAX_RECORD_KEYS or set(payload) - _RECORD_FIELDS:
        findings.append(TaxonomyFinding("UNSUPPORTED_FIELD", "$"))
    for field in sorted(_RECORD_FIELDS - set(payload)):
        findings.append(TaxonomyFinding("MISSING_FIELD", _field_path(field)))

    schema_version = payload.get("schema_version")
    if type(schema_version) is not int:
        findings.append(TaxonomyFinding("INVALID_TYPE", "$.schema_version"))
    elif schema_version != EXCEPTION_TAXONOMY_SCHEMA_VERSION:
        findings.append(
            TaxonomyFinding("UNSUPPORTED_SCHEMA_VERSION", "$.schema_version")
        )

    taxonomy_version = payload.get("taxonomy_version")
    if type(taxonomy_version) is not str:
        findings.append(TaxonomyFinding("INVALID_TYPE", "$.taxonomy_version"))
    elif taxonomy_version != EXCEPTION_TAXONOMY_VERSION:
        findings.append(
            TaxonomyFinding("UNSUPPORTED_TAXONOMY_VERSION", "$.taxonomy_version")
        )

    category = payload.get("category")
    if type(category) is not str:
        findings.append(TaxonomyFinding("INVALID_TYPE", "$.category"))
    elif category not in EXCEPTION_CATEGORIES:
        findings.append(TaxonomyFinding("UNSUPPORTED_CATEGORY", "$.category"))

    reason_code = payload.get("reason_code")
    if type(reason_code) is not str:
        findings.append(TaxonomyFinding("INVALID_TYPE", "$.reason_code"))
    elif reason_code not in _REASON_CODES:
        findings.append(TaxonomyFinding("UNSUPPORTED_REASON_CODE", "$.reason_code"))

    scope = payload.get("scope")
    if type(scope) is not str:
        findings.append(TaxonomyFinding("INVALID_TYPE", "$.scope"))
    elif scope not in EXCEPTION_SURFACES:
        findings.append(TaxonomyFinding("UNSUPPORTED_SCOPE", "$.scope"))

    evidence = _parse_evidence(payload.get("evidence"), findings)
    expires_at = _parse_timestamp(
        payload.get("expires_at"),
        "$.expires_at",
        findings,
    )
    approval = _parse_approval(payload.get("approval"), findings)

    if findings:
        return None, findings
    assert isinstance(category, str)
    assert isinstance(reason_code, str)
    assert isinstance(scope, str)
    assert evidence is not None
    assert expires_at is not None
    assert approval is not None
    try:
        return (
            ExceptionRecord(
                category=category,
                reason_code=reason_code,
                scope=scope,
                evidence=evidence,
                expires_at=expires_at,
                approval=approval,
                taxonomy_version=taxonomy_version,
                schema_version=schema_version,
            ),
            findings,
        )
    except ExceptionTaxonomyError as error:
        # The constructor uses only fixed messages. Map the structural case to
        # a stable finding instead of returning a user-provided value.
        return None, [_finding_for_constructor_error(str(error))]


def _parse_evidence(
    value: Any,
    findings: list[TaxonomyFinding],
) -> tuple[EvidenceReference, ...] | None:
    if not isinstance(value, (list, tuple)):
        findings.append(TaxonomyFinding("INVALID_TYPE", "$.evidence"))
        return None
    if len(value) > _MAX_EVIDENCE_REFERENCES:
        findings.append(TaxonomyFinding("TOO_MANY_EVIDENCE", "$.evidence"))
    parsed: list[EvidenceReference] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping) or set(item) != _EVIDENCE_FIELDS:
            findings.append(TaxonomyFinding("INVALID_EVIDENCE", "$.evidence[*]"))
            continue
        kind = item.get("kind")
        digest = item.get("digest")
        if type(kind) is not str or kind not in EVIDENCE_KINDS:
            findings.append(TaxonomyFinding("INVALID_EVIDENCE", "$.evidence[*].kind"))
            continue
        if kind in seen:
            findings.append(TaxonomyFinding("DUPLICATE_EVIDENCE", "$.evidence[*]"))
            continue
        if type(digest) is not str or not _DIGEST_RE.fullmatch(digest):
            findings.append(TaxonomyFinding("INVALID_DIGEST", "$.evidence[*].digest"))
            continue
        seen.add(kind)
        parsed.append(EvidenceReference(kind=kind, digest=digest))
    return tuple(parsed)


def _parse_approval(
    value: Any,
    findings: list[TaxonomyFinding],
) -> ApprovalMetadata | None:
    if not isinstance(value, Mapping) or set(value) != _APPROVAL_FIELDS:
        findings.append(TaxonomyFinding("INVALID_APPROVAL", "$.approval"))
        return None
    status = value.get("status")
    role = value.get("role")
    digest = value.get("approval_digest")
    approved_at = _parse_timestamp(
        value.get("approved_at"),
        "$.approval.approved_at",
        findings,
    )
    if type(status) is not str:
        findings.append(TaxonomyFinding("INVALID_APPROVAL", "$.approval.status"))
    elif status not in APPROVAL_STATUSES:
        findings.append(
            TaxonomyFinding("UNSUPPORTED_APPROVAL_STATUS", "$.approval.status")
        )
    if type(role) is not str:
        findings.append(TaxonomyFinding("INVALID_APPROVAL", "$.approval.role"))
    elif role not in APPROVAL_ROLES:
        findings.append(TaxonomyFinding("UNSUPPORTED_APPROVAL_ROLE", "$.approval.role"))
    if type(digest) is not str or not _DIGEST_RE.fullmatch(digest):
        findings.append(TaxonomyFinding("INVALID_DIGEST", "$.approval.approval_digest"))
    if findings and any(finding.path.startswith("$.approval") for finding in findings):
        return None
    assert isinstance(status, str)
    assert isinstance(role, str)
    assert isinstance(digest, str)
    assert approved_at is not None
    try:
        return ApprovalMetadata(
            status=status,
            role=role,
            approval_digest=digest,
            approved_at=approved_at,
        )
    except ExceptionTaxonomyError:
        findings.append(TaxonomyFinding("INVALID_APPROVAL", "$.approval"))
        return None


def _parse_timestamp(
    value: Any,
    path: str,
    findings: list[TaxonomyFinding],
) -> datetime | None:
    try:
        return _require_utc_datetime(value)
    except (TypeError, ValueError):
        findings.append(TaxonomyFinding("INVALID_TIMESTAMP", path))
        return None


def _validate_candidate(
    candidate: ExceptionRecord,
    *,
    surface: str,
    as_of: datetime | date | str | None,
    findings: list[TaxonomyFinding],
) -> None:
    rule = _RULE_DEFINITIONS.get(candidate.category)
    if rule is None:
        findings.append(TaxonomyFinding("UNSUPPORTED_CATEGORY", "$.category"))
        return
    if candidate.reason_code not in rule.reason_codes:
        findings.append(TaxonomyFinding("UNSUPPORTED_REASON_CODE", "$.reason_code"))
    if candidate.scope != surface or candidate.scope not in rule.allowed_scopes:
        findings.append(TaxonomyFinding("UNSUPPORTED_SCOPE", "$.scope"))
    present_evidence = {reference.kind for reference in candidate.evidence}
    if not set(rule.required_evidence).issubset(present_evidence):
        findings.append(TaxonomyFinding("MISSING_REQUIRED_EVIDENCE", "$.evidence"))
    if candidate.expires_at > candidate.approval.approved_at + timedelta(
        days=rule.max_expiry_days
    ):
        findings.append(TaxonomyFinding("EXPIRY_TOO_LONG", "$.expires_at"))
    if as_of is None:
        return
    try:
        comparison_time = _require_utc_datetime(as_of)
    except (TypeError, ValueError):
        findings.append(TaxonomyFinding("INVALID_TIMESTAMP", "$.expires_at"))
        return
    if candidate.approval.approved_at > comparison_time:
        findings.append(
            TaxonomyFinding("APPROVAL_NOT_EFFECTIVE", "$.approval.approved_at")
        )
    if candidate.expires_at <= comparison_time:
        findings.append(TaxonomyFinding("EXPIRED", "$.expires_at"))


def _result(
    *,
    surface: str,
    errors: tuple[TaxonomyFinding, ...] | tuple[str, str],
) -> TaxonomyValidationResult:
    if surface not in EXCEPTION_SURFACES:
        surface = "telemetry"
    if errors and isinstance(errors[0], TaxonomyFinding):
        findings = errors
    else:
        code, path = errors
        findings = (TaxonomyFinding(code, path),)
    return TaxonomyValidationResult(
        surface=surface,
        valid=False,
        errors=findings,
    )


def _finding_for_constructor_error(message: str) -> TaxonomyFinding:
    if "reason code" in message:
        return TaxonomyFinding("UNSUPPORTED_REASON_CODE", "$.reason_code")
    if "scope" in message:
        return TaxonomyFinding("UNSUPPORTED_SCOPE", "$.scope")
    if "expiry" in message:
        return TaxonomyFinding("EXPIRY_BEFORE_APPROVAL", "$.expires_at")
    if "evidence" in message:
        return TaxonomyFinding("INVALID_EVIDENCE", "$.evidence")
    return TaxonomyFinding("INVALID_TYPE", "$")


def _field_path(field: str) -> str:
    if field == "approval":
        return "$.approval"
    return f"$.{field}"


def _require_digest(value: Any) -> None:
    if type(value) is not str or not _DIGEST_RE.fullmatch(value):
        raise ExceptionTaxonomyError("invalid digest")


def _require_utc_datetime(value: Any) -> datetime:
    if isinstance(value, date) and not isinstance(value, datetime):
        value = datetime.combine(value, time.min, tzinfo=_UTC)
    elif isinstance(value, str):
        if len(value) > _MAX_DATETIME_LENGTH:
            raise ValueError("timestamp is too long")
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            raise ValueError("invalid timestamp") from None
    if not isinstance(value, datetime):
        raise TypeError("timestamp must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    normalized = value.astimezone(_UTC)
    if normalized.year < 1 or normalized.year > 9999:
        raise ValueError("timestamp is outside the supported range")
    return normalized


def _format_utc_datetime(value: datetime) -> str:
    normalized = _require_utc_datetime(value)
    timespec = "microseconds" if normalized.microsecond else "seconds"
    return normalized.isoformat(timespec=timespec).replace("+00:00", "Z")


def _sha256_digest(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "APPROVAL_ROLES",
    "APPROVAL_STATUSES",
    "DEFAULT_EXCEPTION_TAXONOMY",
    "EVIDENCE_KINDS",
    "EXCEPTION_CATEGORIES",
    "EXCEPTION_SURFACES",
    "EXCEPTION_TAXONOMY_REPORT_TYPE",
    "EXCEPTION_TAXONOMY_SCHEMA_VERSION",
    "EXCEPTION_TAXONOMY_VERSION",
    "ApprovalMetadata",
    "ExceptionRecord",
    "ExceptionTaxonomy",
    "ExceptionTaxonomyError",
    "EvidenceReference",
    "TaxonomyFinding",
    "TaxonomyRule",
    "TaxonomyValidationResult",
    "TAXONOMY_SCHEMA_VERSION",
    "TAXONOMY_VERSION",
    "validate_audit",
    "validate_audit_record",
    "validate_exception_record",
    "validate_record",
    "validate_telemetry",
    "validate_telemetry_record",
]
