"""Deterministic, raw-text-free privacy regression corpus manifests.

The manifest is a release-facing contract for synthetic privacy evaluations. A
caller may provide a local fixture to :func:`make_privacy_case`, but the
fixture is reduced to a content hash and length before it is retained in a
case or written to disk. This keeps the registry useful for coverage checks
without turning reports, exceptions, or committed fixtures into a place where
source text can leak.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PRIVACY_CORPUS_SCHEMA_VERSION = "openmed.eval.privacy_corpus.v1"
PRIVACY_CORPUS_MANIFEST_ID = "openmed-privacy-regression-synthetic-v1"

PRIVACY_SEVERITIES = ("info", "low", "medium", "high", "critical")
PRIVACY_SEVERITY_RANK = {
    severity: rank for rank, severity in enumerate(PRIVACY_SEVERITIES)
}

_HASH_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]*")
_MANIFEST_FIELDS = {
    "cases",
    "manifest_hash",
    "manifest_id",
    "policy_profiles",
    "required_categories",
    "schema_version",
    "synthetic_only",
}
_CASE_FIELDS = {
    "case_id",
    "category",
    "expected_findings",
    "fixture_hash",
    "fixture_length",
    "policy_profile_id",
    "severity",
    "synthetic_only",
    "tags",
}
_PROFILE_FIELDS = {
    "critical_leakage_expected",
    "expected_critical_leakage",
    "profile_id",
    "required_categories",
    "required_severities",
}
_FINDING_FIELDS = {
    "critical_leakage",
    "expected_count",
    "finding_id",
    "severity",
}


@dataclass(frozen=True)
class PrivacyFindingExpectation:
    """Expected aggregate outcome for one finding category in a case."""

    finding_id: str
    severity: str
    expected_count: int
    critical_leakage: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the raw-text-free JSON representation."""

        return {
            "critical_leakage": self.critical_leakage,
            "expected_count": self.expected_count,
            "finding_id": self.finding_id,
            "severity": self.severity,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PrivacyFindingExpectation":
        """Create an expectation from a JSON-compatible mapping."""

        _reject_unknown_fields(value, _FINDING_FIELDS, "finding")
        finding_id = _required_token(value.get("finding_id"), "finding_id")
        severity = _required_severity(value.get("severity"))
        expected_count = _non_negative_int(
            value.get("expected_count"), "expected_count"
        )
        critical_leakage = value.get("critical_leakage", False)
        if not isinstance(critical_leakage, bool):
            raise ValueError("critical_leakage must be boolean")
        return cls(
            finding_id=finding_id,
            severity=severity,
            expected_count=expected_count,
            critical_leakage=critical_leakage,
        )


ExpectedFinding = PrivacyFindingExpectation


@dataclass(frozen=True)
class PrivacyPolicyProfile:
    """Coverage and critical-leakage expectations for one privacy policy."""

    profile_id: str
    required_categories: tuple[str, ...]
    required_severities: tuple[str, ...]
    expected_critical_leakage: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return the raw-text-free JSON representation."""

        return {
            "expected_critical_leakage": self.expected_critical_leakage,
            "profile_id": self.profile_id,
            "required_categories": list(self.required_categories),
            "required_severities": list(self.required_severities),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PrivacyPolicyProfile":
        """Create a policy profile from a JSON-compatible mapping."""

        _reject_unknown_fields(value, _PROFILE_FIELDS, "policy profile")
        profile_id = _required_token(value.get("profile_id"), "profile_id")
        required_categories = _required_tokens(
            value.get("required_categories"), "required_categories"
        )
        required_severities = _required_severities(
            value.get("required_severities"), "required_severities"
        )
        expected_critical_leakage = value.get(
            "expected_critical_leakage",
            value.get("critical_leakage_expected", 0),
        )
        return cls(
            profile_id=profile_id,
            required_categories=required_categories,
            required_severities=required_severities,
            expected_critical_leakage=_non_negative_int(
                expected_critical_leakage,
                "expected_critical_leakage",
            ),
        )


PolicyProfile = PrivacyPolicyProfile


@dataclass(frozen=True)
class PrivacyCase:
    """One synthetic privacy case after its fixture has been reduced to a hash."""

    case_id: str
    category: str
    policy_profile_id: str
    fixture_hash: str
    fixture_length: int
    severity: str
    expected_findings: tuple[PrivacyFindingExpectation, ...]
    tags: tuple[str, ...] = ()
    synthetic_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a representation that cannot contain fixture source text."""

        return {
            "case_id": self.case_id,
            "category": self.category,
            "expected_findings": [
                finding.to_dict() for finding in self.expected_findings
            ],
            "fixture_hash": self.fixture_hash,
            "fixture_length": self.fixture_length,
            "policy_profile_id": self.policy_profile_id,
            "severity": self.severity,
            "synthetic_only": self.synthetic_only,
            "tags": list(self.tags),
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        allow_fixture: bool = False,
    ) -> "PrivacyCase":
        """Create a case, hashing an optional local fixture before retention.

        ``allow_fixture`` is used only by manifest builders. Manifest loading
        keeps it disabled so a persisted case containing a raw ``text`` or
        ``fixture`` field is rejected instead of being silently accepted.
        """

        allowed_fields = _CASE_FIELDS | (
            {"fixture", "text"} if allow_fixture else set()
        )
        _reject_unknown_fields(value, allowed_fields, "case")

        case_id = _required_token(value.get("case_id"), "case_id")
        category = _required_token(value.get("category"), "category")
        policy_profile_id = _required_token(
            value.get("policy_profile_id"), "policy_profile_id"
        )
        severity = _required_severity(value.get("severity"))
        synthetic_only = value.get("synthetic_only", True)
        if synthetic_only is not True:
            raise ValueError("privacy cases must declare synthetic_only=true")

        expected_values = value.get("expected_findings")
        if not isinstance(expected_values, Sequence) or isinstance(
            expected_values, (str, bytes)
        ):
            raise ValueError("expected_findings must be a list")
        expected_findings = tuple(
            sorted(
                (_coerce_finding(value) for value in expected_values),
                key=lambda finding: finding.finding_id,
            )
        )
        if not expected_findings:
            raise ValueError("expected_findings must not be empty")
        if len({finding.finding_id for finding in expected_findings}) != len(
            expected_findings
        ):
            raise ValueError("expected finding identifiers must be unique")

        raw_fixture = _raw_fixture_from_mapping(value) if allow_fixture else None
        fixture_hash = value.get("fixture_hash")
        if fixture_hash is None and raw_fixture is None:
            raise ValueError("case requires fixture_hash or a local fixture")
        if fixture_hash is None:
            fixture_hash = compute_privacy_fixture_hash(raw_fixture)
        fixture_hash = _required_hash(fixture_hash, "fixture_hash")
        if raw_fixture is not None:
            computed_hash = compute_privacy_fixture_hash(raw_fixture)
            if computed_hash != fixture_hash:
                raise ValueError("fixture_hash does not match the local fixture")

        fixture_length = value.get("fixture_length")
        if fixture_length is None:
            if raw_fixture is None:
                raise ValueError("fixture_length is required with fixture_hash")
            fixture_length = _fixture_length(raw_fixture)
        fixture_length = _non_negative_int(fixture_length, "fixture_length")

        tags = _optional_tokens(value.get("tags", ()), "tags")
        return cls(
            case_id=case_id,
            category=category,
            policy_profile_id=policy_profile_id,
            fixture_hash=fixture_hash,
            fixture_length=fixture_length,
            severity=severity,
            expected_findings=expected_findings,
            tags=tags,
            synthetic_only=True,
        )


PrivacyCorpusCase = PrivacyCase


def make_privacy_case(
    case_id: str,
    fixture: Any,
    *,
    category: str,
    policy_profile_id: str,
    severity: str,
    expected_findings: Iterable[PrivacyFindingExpectation | Mapping[str, Any]],
    fixture_length: int | None = None,
    tags: Iterable[str] = (),
) -> PrivacyCase:
    """Hash a local fixture and return a source-free privacy case.

    The fixture is used only during this call. It is never copied into the
    returned case, its JSON representation, or validation errors.
    """

    payload: dict[str, Any] = {
        "case_id": case_id,
        "category": category,
        "policy_profile_id": policy_profile_id,
        "severity": severity,
        "expected_findings": list(expected_findings),
        "fixture": fixture,
        "tags": list(tags),
    }
    if fixture_length is not None:
        payload["fixture_length"] = fixture_length
    return PrivacyCase.from_mapping(payload, allow_fixture=True)


@dataclass(frozen=True)
class PrivacyCoverageReport:
    """Aggregate manifest coverage with no fixture source values."""

    required_categories: tuple[str, ...]
    covered_categories: tuple[str, ...]
    missing_categories: tuple[str, ...]
    required_severities: tuple[str, ...]
    covered_severities: tuple[str, ...]
    missing_severities: tuple[str, ...]
    covered_profiles: tuple[str, ...]
    missing_profiles: tuple[str, ...]
    profile_category_gaps: tuple[str, ...]
    profile_severity_gaps: tuple[str, ...]
    expected_critical_leakage: int
    profile_critical_leakage_gaps: tuple[str, ...]
    valid: bool

    @property
    def complete(self) -> bool:
        """Return whether every declared coverage requirement is satisfied."""

        return self.valid

    def to_dict(self) -> dict[str, Any]:
        """Return a report safe for logs and persisted evaluation artifacts."""

        return {
            "covered_categories": list(self.covered_categories),
            "covered_profiles": list(self.covered_profiles),
            "covered_severities": list(self.covered_severities),
            "expected_critical_leakage": self.expected_critical_leakage,
            "missing_categories": list(self.missing_categories),
            "missing_profiles": list(self.missing_profiles),
            "missing_severities": list(self.missing_severities),
            "profile_category_gaps": list(self.profile_category_gaps),
            "profile_critical_leakage_gaps": list(self.profile_critical_leakage_gaps),
            "profile_severity_gaps": list(self.profile_severity_gaps),
            "required_categories": list(self.required_categories),
            "required_severities": list(self.required_severities),
            "valid": self.valid,
        }


@dataclass(frozen=True)
class PrivacyCorpusManifest:
    """Versioned synthetic privacy corpus contract."""

    manifest_id: str
    cases: tuple[PrivacyCase, ...]
    policy_profiles: tuple[PrivacyPolicyProfile, ...]
    required_categories: tuple[str, ...]
    manifest_hash: str
    schema_version: str = PRIVACY_CORPUS_SCHEMA_VERSION
    synthetic_only: bool = True

    def case(self, case_id: str) -> PrivacyCase:
        """Return a case by identifier."""

        for case in self.cases:
            if case.case_id == case_id:
                return case
        raise KeyError("unknown privacy corpus case")

    def profile(self, profile_id: str) -> PrivacyPolicyProfile:
        """Return a policy profile by identifier."""

        for profile in self.policy_profiles:
            if profile.profile_id == profile_id:
                return profile
        raise KeyError("unknown privacy policy profile")

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic raw-text-free manifest payload."""

        return {
            "cases": [case.to_dict() for case in self.cases],
            "manifest_hash": self.manifest_hash,
            "manifest_id": self.manifest_id,
            "policy_profiles": [profile.to_dict() for profile in self.policy_profiles],
            "required_categories": list(self.required_categories),
            "schema_version": self.schema_version,
            "synthetic_only": self.synthetic_only,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PrivacyCorpusManifest":
        """Load and validate a persisted raw-text-free manifest."""

        _reject_unknown_fields(value, _MANIFEST_FIELDS, "manifest")
        manifest_id = _required_token(value.get("manifest_id"), "manifest_id")
        schema_version = value.get("schema_version")
        if schema_version != PRIVACY_CORPUS_SCHEMA_VERSION:
            raise ValueError("unsupported privacy corpus schema")
        synthetic_only = value.get("synthetic_only")
        if synthetic_only is not True:
            raise ValueError(
                "privacy corpus manifests must declare synthetic_only=true"
            )

        raw_profiles = value.get("policy_profiles")
        if not isinstance(raw_profiles, Sequence) or isinstance(
            raw_profiles, (str, bytes)
        ):
            raise ValueError("policy_profiles must be a list")
        profiles = tuple(
            PrivacyPolicyProfile.from_mapping(profile) for profile in raw_profiles
        )

        raw_cases = value.get("cases")
        if not isinstance(raw_cases, Sequence) or isinstance(raw_cases, (str, bytes)):
            raise ValueError("cases must be a list")
        cases = tuple(PrivacyCase.from_mapping(case) for case in raw_cases)
        required_categories = _required_tokens(
            value.get("required_categories"), "required_categories"
        )
        manifest_hash = _required_hash(value.get("manifest_hash"), "manifest_hash")
        manifest = cls(
            manifest_id=manifest_id,
            cases=tuple(sorted(cases, key=lambda case: case.case_id)),
            policy_profiles=tuple(
                sorted(profiles, key=lambda profile: profile.profile_id)
            ),
            required_categories=required_categories,
            manifest_hash=manifest_hash,
            schema_version=schema_version,
            synthetic_only=True,
        )
        validate_privacy_corpus_manifest(manifest)
        return manifest


def build_privacy_corpus_manifest(
    cases: Iterable[PrivacyCase | Mapping[str, Any]] | None = None,
    policy_profiles: Iterable[PrivacyPolicyProfile | Mapping[str, Any]] | None = None,
    *,
    manifest_id: str = PRIVACY_CORPUS_MANIFEST_ID,
    required_categories: Iterable[str] | None = None,
) -> PrivacyCorpusManifest:
    """Build a deterministic manifest from source-free cases or local fixtures.

    Mapping cases may contain a temporary ``fixture`` or ``text`` value. That
    value is hashed during construction and is absent from the result. Cases
    passed as :class:`PrivacyCase` instances are already source-free.
    """

    if cases is None:
        cases = DEFAULT_PRIVACY_CASES
    if policy_profiles is None:
        policy_profiles = DEFAULT_PRIVACY_POLICY_PROFILES

    normalized_cases = tuple(_coerce_case(case, allow_fixture=True) for case in cases)
    normalized_profiles = tuple(_coerce_profile(profile) for profile in policy_profiles)
    normalized_cases = tuple(sorted(normalized_cases, key=lambda case: case.case_id))
    normalized_profiles = tuple(
        sorted(normalized_profiles, key=lambda profile: profile.profile_id)
    )
    if required_categories is None:
        categories = {
            category
            for profile in normalized_profiles
            for category in profile.required_categories
        }
    else:
        categories = set(_required_tokens(required_categories, "required_categories"))
    manifest_id = _required_token(manifest_id, "manifest_id")
    required_category_tuple = tuple(sorted(categories))
    payload = {
        "cases": [case.to_dict() for case in normalized_cases],
        "manifest_id": manifest_id,
        "policy_profiles": [profile.to_dict() for profile in normalized_profiles],
        "required_categories": list(required_category_tuple),
        "schema_version": PRIVACY_CORPUS_SCHEMA_VERSION,
        "synthetic_only": True,
    }
    manifest = PrivacyCorpusManifest(
        manifest_id=manifest_id,
        cases=normalized_cases,
        policy_profiles=normalized_profiles,
        required_categories=required_category_tuple,
        manifest_hash=_hash_json(payload),
    )
    validate_privacy_corpus_manifest(manifest)
    return manifest


def compute_privacy_fixture_hash(fixture: Any) -> str:
    """Return a deterministic content hash without retaining fixture content."""

    try:
        if isinstance(fixture, str):
            return _hash_bytes(fixture.encode("utf-8"))
        if isinstance(fixture, bytes):
            return _hash_bytes(fixture)
        normalized = _canonical_fixture_value(fixture)
        return _hash_json(normalized)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("fixture must contain finite JSON-compatible values") from None


compute_fixture_hash = compute_privacy_fixture_hash


def compute_privacy_corpus_manifest_hash(
    manifest: PrivacyCorpusManifest | Mapping[str, Any],
) -> str:
    """Return the content hash of a manifest without its stored hash field."""

    if isinstance(manifest, PrivacyCorpusManifest):
        payload = manifest.to_dict()
    elif isinstance(manifest, Mapping):
        payload = dict(manifest)
    else:
        raise TypeError("manifest must be a PrivacyCorpusManifest or mapping")
    payload.pop("manifest_hash", None)
    return _hash_json(payload)


compute_manifest_hash = compute_privacy_corpus_manifest_hash


def privacy_corpus_coverage(
    manifest: PrivacyCorpusManifest | Mapping[str, Any],
) -> PrivacyCoverageReport:
    """Compute coverage from a manifest without checking its stored hash."""

    normalized = _coerce_manifest(manifest)
    _validate_manifest_structure(normalized)
    return _coverage_report(normalized)


def validate_privacy_corpus_manifest(
    manifest: PrivacyCorpusManifest | Mapping[str, Any],
) -> PrivacyCoverageReport:
    """Validate safety, integrity, deterministic hashing, and coverage.

    The returned report contains only identifiers, categories, severities, and
    counts. Invalid input raises a generic, raw-text-free ``ValueError``.
    """

    normalized = _coerce_manifest(manifest)
    _validate_manifest_structure(normalized)
    expected_hash = compute_privacy_corpus_manifest_hash(normalized)
    if normalized.manifest_hash != expected_hash:
        raise ValueError("privacy corpus manifest_hash does not match contents")
    report = _coverage_report(normalized)
    if not report.valid:
        raise ValueError("privacy corpus manifest coverage is incomplete")
    return report


def write_privacy_corpus_manifest(
    path: str | Path,
    manifest: PrivacyCorpusManifest | None = None,
) -> Path:
    """Write a validated deterministic manifest and return its local path."""

    output_path = Path(path)
    payload = (manifest or default_privacy_corpus_manifest()).to_dict()
    validate_privacy_corpus_manifest(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def default_privacy_corpus_manifest() -> PrivacyCorpusManifest:
    """Return the committed, metadata-only synthetic privacy manifest."""

    return build_privacy_corpus_manifest(
        DEFAULT_PRIVACY_CASES,
        DEFAULT_PRIVACY_POLICY_PROFILES,
        manifest_id=PRIVACY_CORPUS_MANIFEST_ID,
    )


def load_privacy_corpus_manifest(
    path: str | Path | None = None,
) -> PrivacyCorpusManifest:
    """Load a local manifest, or return the built-in metadata-only manifest."""

    if path is None:
        return default_privacy_corpus_manifest()
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("privacy corpus manifest is unavailable or invalid") from None
    if not isinstance(payload, Mapping):
        raise ValueError("privacy corpus manifest must be a JSON object")
    return PrivacyCorpusManifest.from_mapping(payload)


def _coerce_case(
    value: PrivacyCase | Mapping[str, Any],
    *,
    allow_fixture: bool,
) -> PrivacyCase:
    if isinstance(value, PrivacyCase):
        return PrivacyCase.from_mapping(value.to_dict())
    if isinstance(value, Mapping):
        return PrivacyCase.from_mapping(value, allow_fixture=allow_fixture)
    raise TypeError("cases must contain PrivacyCase instances or mappings")


def _coerce_profile(
    value: PrivacyPolicyProfile | Mapping[str, Any],
) -> PrivacyPolicyProfile:
    if isinstance(value, PrivacyPolicyProfile):
        return PrivacyPolicyProfile.from_mapping(value.to_dict())
    if isinstance(value, Mapping):
        return PrivacyPolicyProfile.from_mapping(value)
    raise TypeError("policy_profiles must contain profile instances or mappings")


def _coerce_finding(
    value: PrivacyFindingExpectation | Mapping[str, Any],
) -> PrivacyFindingExpectation:
    if isinstance(value, PrivacyFindingExpectation):
        return value
    if isinstance(value, Mapping):
        return PrivacyFindingExpectation.from_mapping(value)
    raise TypeError("expected_findings must contain expectation mappings")


def _coerce_manifest(
    value: PrivacyCorpusManifest | Mapping[str, Any],
) -> PrivacyCorpusManifest:
    if isinstance(value, PrivacyCorpusManifest):
        return value
    if isinstance(value, Mapping):
        return PrivacyCorpusManifest.from_mapping(value)
    raise TypeError("manifest must be a PrivacyCorpusManifest or mapping")


def _validate_manifest_structure(manifest: PrivacyCorpusManifest) -> None:
    if manifest.schema_version != PRIVACY_CORPUS_SCHEMA_VERSION:
        raise ValueError("unsupported privacy corpus schema")
    _required_token(manifest.manifest_id, "manifest_id")
    if manifest.synthetic_only is not True:
        raise ValueError("privacy corpus manifests must declare synthetic_only=true")
    if not manifest.cases:
        raise ValueError("privacy corpus manifest requires cases")
    if not manifest.policy_profiles:
        raise ValueError("privacy corpus manifest requires policy_profiles")
    _required_tokens(manifest.required_categories, "required_categories")

    case_ids = [case.case_id for case in manifest.cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("privacy case identifiers must be unique")
    profile_ids = [profile.profile_id for profile in manifest.policy_profiles]
    if len(set(profile_ids)) != len(profile_ids):
        raise ValueError("policy profile identifiers must be unique")
    profile_id_set = set(profile_ids)
    for case in manifest.cases:
        if case.policy_profile_id not in profile_id_set:
            raise ValueError("privacy case references an unknown policy profile")
        _validate_case(case)
    for profile in manifest.policy_profiles:
        _validate_profile(profile)


def _validate_case(case: PrivacyCase) -> None:
    _required_token(case.case_id, "case_id")
    _required_token(case.category, "category")
    _required_token(case.policy_profile_id, "policy_profile_id")
    _required_hash(case.fixture_hash, "fixture_hash")
    _non_negative_int(case.fixture_length, "fixture_length")
    _required_severity(case.severity)
    if case.synthetic_only is not True:
        raise ValueError("privacy cases must declare synthetic_only=true")
    if not case.expected_findings:
        raise ValueError("expected_findings must not be empty")
    finding_ids = [finding.finding_id for finding in case.expected_findings]
    if len(set(finding_ids)) != len(finding_ids):
        raise ValueError("expected finding identifiers must be unique")
    for finding in case.expected_findings:
        PrivacyFindingExpectation.from_mapping(finding.to_dict())
    _optional_tokens(case.tags, "tags")


def _validate_profile(profile: PrivacyPolicyProfile) -> None:
    _required_token(profile.profile_id, "profile_id")
    _required_tokens(profile.required_categories, "required_categories")
    _required_severities(profile.required_severities, "required_severities")
    _non_negative_int(
        profile.expected_critical_leakage,
        "expected_critical_leakage",
    )


def _coverage_report(manifest: PrivacyCorpusManifest) -> PrivacyCoverageReport:
    covered_categories = tuple(sorted({case.category for case in manifest.cases}))
    required_categories = tuple(sorted(manifest.required_categories))
    missing_categories = tuple(
        category
        for category in required_categories
        if category not in covered_categories
    )
    required_severities = tuple(
        sorted(
            {
                severity
                for profile in manifest.policy_profiles
                for severity in profile.required_severities
            },
            key=lambda severity: PRIVACY_SEVERITY_RANK[severity],
        )
    )
    covered_severities = tuple(
        sorted(
            {case.severity for case in manifest.cases},
            key=lambda severity: PRIVACY_SEVERITY_RANK[severity],
        )
    )
    missing_severities = tuple(
        severity
        for severity in required_severities
        if severity not in covered_severities
    )
    covered_profiles = tuple(
        sorted({case.policy_profile_id for case in manifest.cases})
    )
    profile_ids = tuple(
        sorted(profile.profile_id for profile in manifest.policy_profiles)
    )
    missing_profiles = tuple(
        profile_id for profile_id in profile_ids if profile_id not in covered_profiles
    )

    profile_category_gaps: list[str] = []
    profile_severity_gaps: list[str] = []
    profile_critical_leakage_gaps: list[str] = []
    expected_critical_leakage = 0
    cases_by_profile = {
        profile_id: tuple(
            case for case in manifest.cases if case.policy_profile_id == profile_id
        )
        for profile_id in profile_ids
    }
    for profile in manifest.policy_profiles:
        profile_cases = cases_by_profile[profile.profile_id]
        profile_categories = {case.category for case in profile_cases}
        profile_severities = {case.severity for case in profile_cases}
        missing_profile_categories = sorted(
            set(profile.required_categories) - profile_categories
        )
        missing_profile_severities = sorted(
            set(profile.required_severities) - profile_severities,
            key=lambda severity: PRIVACY_SEVERITY_RANK[severity],
        )
        if missing_profile_categories:
            profile_category_gaps.append(profile.profile_id)
        if missing_profile_severities:
            profile_severity_gaps.append(profile.profile_id)

        profile_leakage = sum(
            finding.expected_count
            for case in profile_cases
            for finding in case.expected_findings
            if finding.critical_leakage
        )
        expected_critical_leakage += profile_leakage
        if profile_leakage != profile.expected_critical_leakage:
            profile_critical_leakage_gaps.append(profile.profile_id)

    valid = not (
        missing_categories
        or missing_severities
        or missing_profiles
        or profile_category_gaps
        or profile_severity_gaps
        or profile_critical_leakage_gaps
    )
    return PrivacyCoverageReport(
        required_categories=required_categories,
        covered_categories=covered_categories,
        missing_categories=missing_categories,
        required_severities=required_severities,
        covered_severities=covered_severities,
        missing_severities=missing_severities,
        covered_profiles=covered_profiles,
        missing_profiles=missing_profiles,
        profile_category_gaps=tuple(sorted(profile_category_gaps)),
        profile_severity_gaps=tuple(sorted(profile_severity_gaps)),
        expected_critical_leakage=expected_critical_leakage,
        profile_critical_leakage_gaps=tuple(sorted(profile_critical_leakage_gaps)),
        valid=valid,
    )


def _raw_fixture_from_mapping(value: Mapping[str, Any]) -> Any | None:
    if "fixture" in value:
        return value["fixture"]
    if "text" in value:
        return value["text"]
    return None


def _fixture_length(fixture: Any) -> int:
    if isinstance(fixture, str | bytes):
        return len(fixture)
    if isinstance(fixture, Mapping):
        text = fixture.get("text")
        if isinstance(text, str | bytes):
            return len(text)
    try:
        return len(_canonical_json(_canonical_fixture_value(fixture)).encode("utf-8"))
    except (TypeError, ValueError, OverflowError):
        raise ValueError("fixture must contain finite JSON-compatible values") from None


def _canonical_fixture_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("fixture mapping keys must be strings")
            normalized[key] = _canonical_fixture_value(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonical_fixture_value(item) for item in value]
    if isinstance(value, bytes):
        return {
            "byte_length": len(value),
            "byte_sha256": hashlib.sha256(value).hexdigest(),
            "type": "bytes",
        }
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("fixture contains a non-finite number")
        return value
    raise TypeError("fixture value is not JSON-compatible")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _hash_json(value: Any) -> str:
    return (
        f"sha256:{hashlib.sha256(_canonical_json(value).encode('utf-8')).hexdigest()}"
    )


def _hash_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _required_token(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _TOKEN_PATTERN.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase safe identifier")
    return value


def _required_tokens(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a list of identifiers")
    tokens = tuple(sorted({_required_token(item, field) for item in value}))
    if not tokens:
        raise ValueError(f"{field} must not be empty")
    return tokens


def _optional_tokens(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a list of identifiers")
    return tuple(sorted({_required_token(item, field) for item in value}))


def _required_severities(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a list of severities")
    severities = tuple(
        sorted(
            {_required_severity(item) for item in value},
            key=lambda severity: PRIVACY_SEVERITY_RANK[severity],
        )
    )
    if not severities:
        raise ValueError(f"{field} must not be empty")
    return severities


def _required_severity(value: Any) -> str:
    if value not in PRIVACY_SEVERITIES:
        raise ValueError("severity is not supported")
    return value


def _required_hash(value: Any, field: str) -> str:
    if not isinstance(value, str) or _HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a sha256 hash")
    return value


def _non_negative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _reject_unknown_fields(
    value: Mapping[str, Any],
    allowed: set[str],
    field: str,
) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{field} contains unsupported fields")


def _default_case_fixture(case_id: str, seed: str) -> dict[str, str]:
    return {
        "case_id": case_id,
        "generator": "openmed.synthetic.privacy.v1",
        "seed": seed,
    }


def _default_case(
    case_id: str,
    category: str,
    profile_id: str,
    severity: str,
    seed: str,
    findings: tuple[PrivacyFindingExpectation, ...],
    *,
    fixture_length: int,
    tags: tuple[str, ...] = (),
) -> PrivacyCase:
    return make_privacy_case(
        case_id,
        _default_case_fixture(case_id, seed),
        category=category,
        policy_profile_id=profile_id,
        severity=severity,
        expected_findings=findings,
        fixture_length=fixture_length,
        tags=tags,
    )


DEFAULT_PRIVACY_POLICY_PROFILES = (
    PrivacyPolicyProfile(
        profile_id="strict_redaction",
        required_categories=("direct_identifier", "quasi_identifier"),
        required_severities=("high", "critical"),
        expected_critical_leakage=0,
    ),
    PrivacyPolicyProfile(
        profile_id="context_safe_harbor",
        required_categories=("clinical_context", "negative_context"),
        required_severities=("low", "medium"),
        expected_critical_leakage=0,
    ),
)

_NO_CRITICAL_LEAKAGE = PrivacyFindingExpectation(
    finding_id="critical_leakage",
    severity="critical",
    expected_count=0,
    critical_leakage=True,
)

DEFAULT_PRIVACY_CASES = (
    _default_case(
        "direct_identifier_boundary",
        "direct_identifier",
        "strict_redaction",
        "critical",
        "seed-a",
        (
            PrivacyFindingExpectation(
                finding_id="direct_identifier",
                severity="critical",
                expected_count=1,
            ),
            _NO_CRITICAL_LEAKAGE,
        ),
        fixture_length=64,
        tags=("boundary", "critical_leakage_zero"),
    ),
    _default_case(
        "quasi_identifier_context",
        "quasi_identifier",
        "strict_redaction",
        "high",
        "seed-b",
        (
            PrivacyFindingExpectation(
                finding_id="quasi_identifier",
                severity="high",
                expected_count=1,
            ),
            PrivacyFindingExpectation(
                finding_id="critical_leakage",
                severity="high",
                expected_count=0,
                critical_leakage=True,
            ),
        ),
        fixture_length=72,
        tags=("cross_field", "critical_leakage_zero"),
    ),
    _default_case(
        "clinical_context_negative",
        "clinical_context",
        "context_safe_harbor",
        "medium",
        "seed-c",
        (
            PrivacyFindingExpectation(
                finding_id="clinical_context",
                severity="medium",
                expected_count=0,
            ),
            _NO_CRITICAL_LEAKAGE,
        ),
        fixture_length=48,
        tags=("safe_harbor", "critical_leakage_zero"),
    ),
    _default_case(
        "negative_context_safe",
        "negative_context",
        "context_safe_harbor",
        "low",
        "seed-d",
        (
            PrivacyFindingExpectation(
                finding_id="negative_context",
                severity="low",
                expected_count=1,
            ),
            PrivacyFindingExpectation(
                finding_id="critical_leakage",
                severity="low",
                expected_count=0,
                critical_leakage=True,
            ),
        ),
        fixture_length=40,
        tags=("negative", "critical_leakage_zero"),
    ),
)
