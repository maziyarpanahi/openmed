"""Deterministic completeness checks for uncertainty disclosures.

The audit accepts claim metadata and checks the structural parts of an
uncertainty disclosure that a guarded output needs before it is shown to a
reviewer: at least one uncertainty category and reason code, evidence and
provenance references, an allowed review state, and bounded display hints.

This module deliberately does not interpret a claim or decide whether an
uncertainty statement is clinically correct. Reports contain only aggregate
counts, fixed issue codes, and SHA-256 claim keys. Input identifiers,
categories, reason codes, references, display values, and other metadata are
never copied into a report or an exception. The audit is local-only and has no
network or filesystem side effects.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

UNCERTAINTY_DISCLOSURE_SCHEMA_VERSION = 1
UNCERTAINTY_DISCLOSURE_ADVISORY = (
    "Uncertainty disclosure completeness is an assistive structural audit, not "
    "a clinical interpretation, compliance certification, or clinical decision."
)

DEFAULT_REQUIRED_DISPLAY_HINTS = ("max_chars", "max_items")
DISPLAY_HINT_LIMITS = MappingProxyType(
    {
        "max_chars": (1, 4096),
        "max_items": (1, 100),
        "max_lines": (1, 100),
    }
)

UNCERTAINTY_REVIEW_STATES = frozenset(
    {
        "approved",
        "complete",
        "in_review",
        "pending_review",
        "needs_review",
        "not_reviewed",
        "pending",
        "requires_review",
        "rejected",
        "reviewed",
        "unreviewed",
    }
)

MISSING_UNCERTAINTY_CATEGORIES = "missing_uncertainty_categories"
INVALID_UNCERTAINTY_CATEGORIES = "invalid_uncertainty_categories"
MISSING_REQUIRED_CATEGORY = "missing_required_category"
DUPLICATE_UNCERTAINTY_CATEGORIES = "duplicate_uncertainty_categories"
MISSING_REASON_CODES = "missing_reason_codes"
INVALID_REASON_CODES = "invalid_reason_codes"
DUPLICATE_REASON_CODES = "duplicate_reason_codes"
MISSING_EVIDENCE_REFERENCES = "missing_evidence_references"
INVALID_EVIDENCE_REFERENCES = "invalid_evidence_references"
DUPLICATE_EVIDENCE_REFERENCES = "duplicate_evidence_references"
MISSING_REVIEW_STATE = "missing_review_state"
INVALID_REVIEW_STATE = "invalid_review_state"
MISSING_DISPLAY_HINTS = "missing_display_hints"
INVALID_DISPLAY_HINTS = "invalid_display_hints"

UNCERTAINTY_DISCLOSURE_ISSUE_CODES = (
    MISSING_UNCERTAINTY_CATEGORIES,
    INVALID_UNCERTAINTY_CATEGORIES,
    MISSING_REQUIRED_CATEGORY,
    DUPLICATE_UNCERTAINTY_CATEGORIES,
    MISSING_REASON_CODES,
    INVALID_REASON_CODES,
    DUPLICATE_REASON_CODES,
    MISSING_EVIDENCE_REFERENCES,
    INVALID_EVIDENCE_REFERENCES,
    DUPLICATE_EVIDENCE_REFERENCES,
    MISSING_REVIEW_STATE,
    INVALID_REVIEW_STATE,
    MISSING_DISPLAY_HINTS,
    INVALID_DISPLAY_HINTS,
)

_ISSUE_CODE_SET = frozenset(UNCERTAINTY_DISCLOSURE_ISSUE_CODES)
_CLAIM_ID_KEYS = ("claim_id", "id", "key")
_DISCLOSURE_CONTAINER_KEYS = (
    "uncertainty_disclosure",
    "uncertainty",
    "disclosure",
)
_CATEGORY_KEYS = (
    "uncertainty_categories",
    "uncertainty_category",
    "categories",
    "category",
)
_REASON_CODE_KEYS = (
    "reason_codes",
    "uncertainty_reason_codes",
    "uncertainty_reason_code",
    "reason_code",
)
_EVIDENCE_REFERENCE_KEYS = (
    "evidence_references",
    "evidence_reference",
    "evidence_refs",
    "evidence_ref",
    "provenance_references",
    "provenance_reference",
    "provenance_refs",
    "provenance_ref",
    "evidence",
)
_REVIEW_STATE_KEYS = ("review_state", "review_status", "review")
_DISPLAY_HINT_KEYS = ("display_hints", "display_hint", "display_bounds", "display")
_REFERENCE_ID_KEYS = (
    "reference_id",
    "ref_id",
    "ref",
    "evidence_id",
    "provenance_id",
    "id",
    "key",
    "hash",
)
_DISPLAY_HINT_ALIASES = {
    "max_length": "max_chars",
    "max_display_chars": "max_chars",
    "max_count": "max_items",
    "max_display_items": "max_items",
    "max_line_count": "max_lines",
}
_MISSING = object()


@dataclass(frozen=True)
class UncertaintyDisclosureFinding:
    """Privacy-safe structural findings for one claim.

    ``claim_key`` is a SHA-256 digest, not the input claim identifier. Issue
    codes are fixed taxonomy values and carry no claim content.
    """

    claim_key: str
    issue_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.claim_key.startswith("sha256:") or len(self.claim_key) != 71:
            raise ValueError("claim_key must be an opaque SHA-256 identifier")
        if any(code not in _ISSUE_CODE_SET for code in self.issue_codes):
            raise ValueError("finding contains an unknown issue code")
        normalized = tuple(
            code
            for code in UNCERTAINTY_DISCLOSURE_ISSUE_CODES
            if code in self.issue_codes
        )
        object.__setattr__(self, "issue_codes", normalized)

    def to_dict(self) -> dict[str, Any]:
        """Return the finding without claim metadata or source values."""

        return {
            "claim_key": self.claim_key,
            "issue_codes": list(self.issue_codes),
        }


@dataclass(frozen=True)
class UncertaintyDisclosureReport:
    """Deterministic, count-oriented uncertainty disclosure audit report."""

    checked_claims: int
    compliant_claims: int
    findings: tuple[UncertaintyDisclosureFinding, ...]
    issue_counts: Mapping[str, int]
    schema_version: int = UNCERTAINTY_DISCLOSURE_SCHEMA_VERSION
    advisory: str = UNCERTAINTY_DISCLOSURE_ADVISORY

    def __post_init__(self) -> None:
        if self.checked_claims < 0 or self.compliant_claims < 0:
            raise ValueError("claim counts must be non-negative")
        if self.compliant_claims > self.checked_claims:
            raise ValueError("compliant claim count cannot exceed checked claim count")
        findings = tuple(sorted(self.findings, key=_finding_sort_key))
        counts = {
            code: int(self.issue_counts.get(code, 0))
            for code in UNCERTAINTY_DISCLOSURE_ISSUE_CODES
        }
        if any(count < 0 for count in counts.values()):
            raise ValueError("issue counts must be non-negative")
        object.__setattr__(self, "findings", findings)
        object.__setattr__(self, "issue_counts", MappingProxyType(counts))
        object.__setattr__(self, "advisory", UNCERTAINTY_DISCLOSURE_ADVISORY)

    @property
    def non_compliant_claims(self) -> int:
        """Return the number of claims with at least one finding."""

        return self.checked_claims - self.compliant_claims

    @property
    def incomplete_claims(self) -> int:
        """Alias for :attr:`non_compliant_claims`."""

        return self.non_compliant_claims

    @property
    def is_complete(self) -> bool:
        """Return whether every checked claim passed the structural audit."""

        return self.non_compliant_claims == 0

    @property
    def issues(self) -> tuple[UncertaintyDisclosureFinding, ...]:
        """Return privacy-safe findings as an ergonomic alias."""

        return self.findings

    @property
    def summary(self) -> dict[str, int | bool]:
        """Return aggregate counts without claim metadata."""

        return {
            "checked_claims": self.checked_claims,
            "compliant_claims": self.compliant_claims,
            "non_compliant_claims": self.non_compliant_claims,
            "complete": self.is_complete,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible report containing no raw claim values."""

        return {
            "schema_version": self.schema_version,
            "summary": self.summary,
            "issue_counts": dict(self.issue_counts),
            "findings": [finding.to_dict() for finding in self.findings],
            "advisory": self.advisory,
        }


def audit_uncertainty_disclosures(
    claims: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    *,
    required_categories: Iterable[str] | str | None = None,
    required_uncertainty_categories: Iterable[str] | str | None = None,
    min_evidence_references: int = 1,
    required_display_hints: Iterable[str] | str = DEFAULT_REQUIRED_DISPLAY_HINTS,
) -> UncertaintyDisclosureReport:
    """Audit claim metadata for complete uncertainty disclosures.

    Args:
        claims: One claim mapping or an iterable of claim mappings. The claim
            identifier may be supplied as ``claim_id``, ``id``, or ``key``.
            Disclosure fields may be top-level or nested under
            ``uncertainty_disclosure``/``uncertainty``.
        required_categories: Optional category tokens that every claim must
            declare. With no configured tokens, a non-empty category field is
            still required. Tokens are compared internally and never emitted.
        required_uncertainty_categories: Backwards-compatible, more explicit
            alias for ``required_categories``.
        min_evidence_references: Minimum number of evidence/provenance
            references required per claim.
        required_display_hints: Bounded display fields required per claim.
            The defaults require ``max_chars`` in ``[1, 4096]`` and
            ``max_items`` in ``[1, 100]``. ``max_lines`` is also supported.

    Returns:
        A deterministic report with opaque claim keys, fixed issue codes, and
        aggregate counts only.

    Raises:
        TypeError: If claims or configuration values have the wrong shape.
        ValueError: If numeric configuration is outside its supported range.
    """

    if required_categories is not None and required_uncertainty_categories is not None:
        raise ValueError("provide only one required category configuration")
    configured_categories = (
        required_uncertainty_categories
        if required_uncertainty_categories is not None
        else required_categories
    )
    required_category_tokens = _required_tokens(configured_categories)
    required_hint_keys = _required_hint_keys(required_display_hints)
    if (
        isinstance(min_evidence_references, bool)
        or not isinstance(min_evidence_references, int)
        or min_evidence_references < 1
    ):
        raise ValueError("min_evidence_references must be a positive integer")

    claim_items = _claim_items(claims)
    findings: list[UncertaintyDisclosureFinding] = []
    issue_counts = {code: 0 for code in UNCERTAINTY_DISCLOSURE_ISSUE_CODES}
    compliant_claims = 0

    for claim in claim_items:
        claim_key = _opaque_claim_key(claim)
        issue_codes = _audit_claim(
            claim,
            required_category_tokens=required_category_tokens,
            min_evidence_references=min_evidence_references,
            required_hint_keys=required_hint_keys,
        )
        if not issue_codes:
            compliant_claims += 1
            continue
        finding = UncertaintyDisclosureFinding(claim_key, tuple(issue_codes))
        findings.append(finding)
        for code in finding.issue_codes:
            issue_counts[code] += 1

    return UncertaintyDisclosureReport(
        checked_claims=len(claim_items),
        compliant_claims=compliant_claims,
        findings=tuple(findings),
        issue_counts=issue_counts,
    )


def audit_uncertainty_disclosure(
    claims: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    **kwargs: Any,
) -> UncertaintyDisclosureReport:
    """Singular-name alias for :func:`audit_uncertainty_disclosures`."""

    return audit_uncertainty_disclosures(claims, **kwargs)


def _audit_claim(
    claim: Mapping[str, Any],
    *,
    required_category_tokens: frozenset[str],
    min_evidence_references: int,
    required_hint_keys: tuple[str, ...],
) -> list[str]:
    sources = _field_sources(claim)
    issues: list[str] = []

    categories_present, raw_categories = _first_field(sources, _CATEGORY_KEYS)
    categories, category_valid = _category_tokens(raw_categories)
    if not categories_present:
        issues.append(MISSING_UNCERTAINTY_CATEGORIES)
    elif not category_valid:
        issues.append(INVALID_UNCERTAINTY_CATEGORIES)
    elif not categories:
        issues.append(MISSING_UNCERTAINTY_CATEGORIES)
    else:
        category_keys = frozenset(categories)
        if len(category_keys) != len(categories):
            issues.append(DUPLICATE_UNCERTAINTY_CATEGORIES)
        if required_category_tokens and not required_category_tokens.issubset(
            category_keys
        ):
            issues.append(MISSING_REQUIRED_CATEGORY)

    reason_present, raw_reasons = _first_field(sources, _REASON_CODE_KEYS)
    reasons, reason_valid = _reason_tokens(raw_reasons)
    if not reason_present or not reasons:
        issues.append(MISSING_REASON_CODES)
    elif not reason_valid:
        issues.append(INVALID_REASON_CODES)
    elif len(set(reasons)) != len(reasons):
        issues.append(DUPLICATE_REASON_CODES)

    evidence_present, raw_evidence = _first_field(sources, _EVIDENCE_REFERENCE_KEYS)
    evidence, evidence_valid = _reference_tokens(raw_evidence)
    if not evidence_present or not evidence:
        issues.append(MISSING_EVIDENCE_REFERENCES)
    elif not evidence_valid:
        issues.append(INVALID_EVIDENCE_REFERENCES)
    elif len(evidence) < min_evidence_references:
        issues.append(MISSING_EVIDENCE_REFERENCES)
    elif len(set(evidence)) != len(evidence):
        issues.append(DUPLICATE_EVIDENCE_REFERENCES)

    review_present, raw_review = _first_field(sources, _REVIEW_STATE_KEYS)
    if not review_present:
        issues.append(MISSING_REVIEW_STATE)
    elif not _valid_review_state(raw_review):
        issues.append(INVALID_REVIEW_STATE)

    display_present, raw_display = _first_field(sources, _DISPLAY_HINT_KEYS)
    display_issues = _display_hint_issues(
        raw_display if display_present else _MISSING,
        required_hint_keys,
    )
    issues.extend(display_issues)

    return [code for code in UNCERTAINTY_DISCLOSURE_ISSUE_CODES if code in issues]


def _claim_items(
    claims: Mapping[str, Any] | Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    if isinstance(claims, Mapping):
        return [_coerce_claim(claims)]
    if isinstance(claims, (str, bytes)):
        raise TypeError("claims must be a mapping or iterable of mappings")
    try:
        items = list(claims)
    except TypeError as exc:
        raise TypeError("claims must be a mapping or iterable of mappings") from exc
    return [_coerce_claim(item) for item in items]


def _coerce_claim(claim: Any) -> Mapping[str, Any]:
    if isinstance(claim, Mapping):
        return claim
    to_dict = getattr(claim, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, Mapping):
            return converted
    raise TypeError("each claim must be a mapping")


def _field_sources(claim: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    sources: list[Mapping[str, Any]] = []
    metadata = claim.get("metadata")
    candidates: list[Mapping[str, Any]] = [claim]
    if isinstance(metadata, Mapping):
        candidates.append(metadata)
    for candidate in candidates:
        for key in _DISCLOSURE_CONTAINER_KEYS:
            nested = candidate.get(key)
            if isinstance(nested, Mapping):
                sources.append(nested)
    sources.extend(candidates)
    return tuple(sources)


def _first_field(
    sources: Iterable[Mapping[str, Any]], keys: Iterable[str]
) -> tuple[bool, Any]:
    key_set = tuple(keys)
    for source in sources:
        for key in key_set:
            if key in source:
                return True, source[key]
    return False, None


def _required_tokens(value: Iterable[str] | str | None) -> frozenset[str]:
    if value is None:
        return frozenset()
    tokens, valid = _text_tokens(value, mapping_mode="keys")
    if not valid or not tokens:
        raise TypeError("required categories must contain non-empty strings")
    return frozenset(tokens)


def _required_hint_keys(value: Iterable[str] | str) -> tuple[str, ...]:
    if isinstance(value, str):
        raw_keys = (value,)
    else:
        if isinstance(value, (bytes, Mapping)):
            raise TypeError("required display hints must contain hint names")
        try:
            raw_keys = tuple(value)
        except TypeError as exc:
            raise TypeError("required display hints must contain hint names") from exc
    keys: list[str] = []
    for raw_key in raw_keys:
        if not isinstance(raw_key, str) or not raw_key.strip():
            raise TypeError("required display hints must contain hint names")
        key = _DISPLAY_HINT_ALIASES.get(raw_key.strip().casefold(), raw_key.strip())
        if key not in DISPLAY_HINT_LIMITS:
            raise ValueError("required display hint is not bounded")
        if key not in keys:
            keys.append(key)
    return tuple(keys)


def _category_tokens(value: Any) -> tuple[list[str], bool]:
    return _text_tokens(value, mapping_mode="keys")


def _reason_tokens(value: Any) -> tuple[list[str], bool]:
    if isinstance(value, Mapping):
        flattened: list[Any] = []
        for nested in value.values():
            if isinstance(nested, str):
                flattened.append(nested)
            elif isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
                flattened.extend(nested)
            else:
                return [], False
        return _text_tokens(flattened, mapping_mode="values")
    return _text_tokens(value, mapping_mode="values")


def _reference_tokens(value: Any) -> tuple[list[str], bool]:
    if isinstance(value, Mapping):
        reference = _first_present(value, _REFERENCE_ID_KEYS)
        if reference is not _MISSING:
            return _text_tokens(reference, mapping_mode="values")
        if value and all(isinstance(key, str) for key in value):
            return _text_tokens(tuple(value), mapping_mode="keys")
        return [], False
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        tokens: list[str] = []
        for item in value:
            if isinstance(item, Mapping):
                reference = _first_present(item, _REFERENCE_ID_KEYS)
                if reference is _MISSING:
                    return [], False
                item_tokens, valid = _text_tokens(reference, mapping_mode="values")
            else:
                item_tokens, valid = _text_tokens(item, mapping_mode="values")
            if not valid:
                return [], False
            tokens.extend(item_tokens)
        return tokens, True
    return _text_tokens(value, mapping_mode="values")


def _text_tokens(value: Any, *, mapping_mode: str) -> tuple[list[str], bool]:
    if isinstance(value, str):
        values: Iterable[Any] = (value,)
    elif isinstance(value, Mapping):
        if not value:
            return [], True
        values = value.keys() if mapping_mode == "keys" else value.values()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = value
    else:
        return [], False

    tokens: list[str] = []
    for item in values:
        if not isinstance(item, str) or not item.strip():
            return [], False
        tokens.append(item.strip().casefold())
    return tokens, True


def _valid_review_state(value: Any) -> bool:
    if isinstance(value, Mapping):
        value = _first_present(value, ("state", "status"))
        if value is _MISSING:
            return False
    if not isinstance(value, str) or not value.strip():
        return False
    normalized = value.strip().casefold().replace("-", "_").replace(" ", "_")
    return normalized in UNCERTAINTY_REVIEW_STATES


def _display_hint_issues(value: Any, required_keys: tuple[str, ...]) -> list[str]:
    if value is _MISSING:
        return [MISSING_DISPLAY_HINTS]
    if not isinstance(value, Mapping):
        return [INVALID_DISPLAY_HINTS]

    normalized: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str):
            continue
        key = _DISPLAY_HINT_ALIASES.get(raw_key.strip().casefold(), raw_key.strip())
        normalized[key] = raw_value

    issues: list[str] = []
    missing = any(key not in normalized for key in required_keys)
    invalid = any(
        key in normalized and not _bounded_hint_value(key, normalized[key])
        for key in required_keys
    )
    if missing:
        issues.append(MISSING_DISPLAY_HINTS)
    if invalid:
        issues.append(INVALID_DISPLAY_HINTS)
    return issues


def _bounded_hint_value(key: str, value: Any) -> bool:
    if key not in DISPLAY_HINT_LIMITS:
        return False
    if isinstance(value, bool) or not isinstance(value, int):
        return False
    lower, upper = DISPLAY_HINT_LIMITS[key]
    return lower <= value <= upper


def _first_present(mapping: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return _MISSING


def _opaque_claim_key(claim: Mapping[str, Any]) -> str:
    claim_id = _first_present(claim, _CLAIM_ID_KEYS)
    if claim_id is _MISSING or claim_id is None or claim_id == "":
        # Hash the canonical metadata shape when no identifier is supplied so
        # the report remains stable if the caller presents claims in a
        # different order. The ordinal is intentionally not emitted or hashed.
        seed = _canonical_json(claim)
    else:
        seed = _canonical_json(claim_id)
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
        )
    except (TypeError, ValueError, RecursionError):
        return f"{type(value).__module__}.{type(value).__qualname__}"


def _json_default(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _finding_sort_key(
    finding: UncertaintyDisclosureFinding,
) -> tuple[str, tuple[str, ...]]:
    return finding.claim_key, finding.issue_codes


__all__ = [
    "DEFAULT_REQUIRED_DISPLAY_HINTS",
    "DISPLAY_HINT_LIMITS",
    "DUPLICATE_EVIDENCE_REFERENCES",
    "DUPLICATE_REASON_CODES",
    "DUPLICATE_UNCERTAINTY_CATEGORIES",
    "INVALID_DISPLAY_HINTS",
    "INVALID_EVIDENCE_REFERENCES",
    "INVALID_REASON_CODES",
    "INVALID_REVIEW_STATE",
    "INVALID_UNCERTAINTY_CATEGORIES",
    "MISSING_DISPLAY_HINTS",
    "MISSING_EVIDENCE_REFERENCES",
    "MISSING_REASON_CODES",
    "MISSING_REQUIRED_CATEGORY",
    "MISSING_REVIEW_STATE",
    "MISSING_UNCERTAINTY_CATEGORIES",
    "UNCERTAINTY_DISCLOSURE_ADVISORY",
    "UNCERTAINTY_DISCLOSURE_ISSUE_CODES",
    "UNCERTAINTY_DISCLOSURE_SCHEMA_VERSION",
    "UNCERTAINTY_REVIEW_STATES",
    "UncertaintyDisclosureFinding",
    "UncertaintyDisclosureReport",
    "audit_uncertainty_disclosure",
    "audit_uncertainty_disclosures",
]
