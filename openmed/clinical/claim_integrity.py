"""Deterministic, privacy-safe integrity checks for clinical claim packets.

The packet boundary accepts structured claim, citation, review, and policy
records.  Values participate in a local canonical SHA-256 digest, but the
public canonical form and integrity report contain only counts, reason codes,
and digests.  They never render record identifiers or clinical text.

This module is a review and tamper-detection aid.  It performs no inference,
filesystem discovery, telemetry, or network access and does not certify a
clinical conclusion or a compliance program.
"""

from __future__ import annotations

import dataclasses
import json
import math
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any, Final

from openmed.core.audit import stable_hash

__all__ = [
    "CLAIM_PACKET_DIGEST_ALGORITHM",
    "CLAIM_PACKET_INTEGRITY_ADVISORY",
    "CLAIM_PACKET_INTEGRITY_KIND",
    "CLAIM_PACKET_SCHEMA_VERSION",
    "CLAIM_INTEGRITY_SCHEMA_VERSION",
    "CLAIM_INTEGRITY_KIND",
    "CLAIM_INTEGRITY_ADVISORY",
    "CLAIM_PACKET_INTEGRITY_SCHEMA_VERSION",
    "DUPLICATE_RECORD_REASON",
    "DUPLICATE_REFERENCE_REASON",
    "DIGEST_MISMATCH_REASON",
    "INVALID_PACKET_REASON",
    "INVALID_REFERENCE_REASON",
    "MISSING_RECORD_REASON",
    "MISSING_REFERENCE_REASON",
    "MUTATED_RECORD_REASON",
    "MUTATED_REFERENCE_REASON",
    "REORDERED_REFERENCE_REASON",
    "UNEXPECTED_RECORD_REASON",
    "UNEXPECTED_REFERENCE_REASON",
    "REASON_DUPLICATE_RECORD",
    "REASON_DUPLICATE_REFERENCE",
    "REASON_DIGEST_MISMATCH",
    "REASON_INVALID_PACKET",
    "REASON_INVALID_REFERENCE",
    "REASON_MISSING_RECORD",
    "REASON_MISSING_REFERENCE",
    "REASON_MUTATED_RECORD",
    "REASON_MUTATED_REFERENCE",
    "REASON_REORDERED_REFERENCE",
    "REASON_UNEXPECTED_RECORD",
    "REASON_UNEXPECTED_REFERENCE",
    "CLAIM_PACKET_INTEGRITY_REASONS",
    "ClaimIntegrityError",
    "ClaimPacketIntegrityError",
    "ClaimPacketIntegrityReport",
    "build_claim_packet_digest",
    "build_claim_packet_integrity_report",
    "calculate_claim_packet_digest",
    "canonicalize_claim_packet",
    "check_claim_packet_integrity",
    "claim_packet_digest",
    "compute_claim_packet_digest",
    "compute_claim_packet_integrity_digest",
    "compute_claim_packet_integrity",
    "is_claim_packet_integrity_valid",
    "validate_claim_packet_integrity",
    "verify_claim_packet_digest",
    "verify_claim_packet_integrity",
    "ClaimPacketIntegrity",
]

CLAIM_PACKET_SCHEMA_VERSION: Final = 1
CLAIM_INTEGRITY_SCHEMA_VERSION: Final = CLAIM_PACKET_SCHEMA_VERSION
CLAIM_PACKET_INTEGRITY_SCHEMA_VERSION: Final = CLAIM_PACKET_SCHEMA_VERSION
CLAIM_PACKET_INTEGRITY_KIND: Final = "openmed.clinical_claim_packet_integrity"
CLAIM_INTEGRITY_KIND: Final = CLAIM_PACKET_INTEGRITY_KIND
CLAIM_PACKET_DIGEST_ALGORITHM: Final = "sha256"
CLAIM_PACKET_INTEGRITY_ADVISORY: Final = (
    "Claim-packet integrity is deterministic assistive review metadata, not a "
    "clinical decision, compliance certification, or substitute for qualified "
    "clinical judgment."
)
CLAIM_INTEGRITY_ADVISORY: Final = CLAIM_PACKET_INTEGRITY_ADVISORY

MISSING_REFERENCE_REASON: Final = "missing_reference"
DUPLICATE_REFERENCE_REASON: Final = "duplicate_reference"
REORDERED_REFERENCE_REASON: Final = "reordered_reference"
MUTATED_REFERENCE_REASON: Final = "mutated_reference"
MISSING_RECORD_REASON: Final = "missing_record"
DUPLICATE_RECORD_REASON: Final = "duplicate_record"
MUTATED_RECORD_REASON: Final = "mutated_record"
UNEXPECTED_RECORD_REASON: Final = "unexpected_record"
UNEXPECTED_REFERENCE_REASON: Final = "unexpected_reference"
DIGEST_MISMATCH_REASON: Final = "digest_mismatch"
INVALID_REFERENCE_REASON: Final = "invalid_reference"
INVALID_PACKET_REASON: Final = "invalid_packet"
_INVALID_EXPECTED_DIGEST_REASON: Final = "invalid_expected_digest"
_EXPECTED_PACKET_MISMATCH_REASON: Final = "expected_packet_mismatch"

REASON_MISSING_REFERENCE: Final = MISSING_REFERENCE_REASON
REASON_DUPLICATE_REFERENCE: Final = DUPLICATE_REFERENCE_REASON
REASON_REORDERED_REFERENCE: Final = REORDERED_REFERENCE_REASON
REASON_MUTATED_REFERENCE: Final = MUTATED_REFERENCE_REASON
REASON_MISSING_RECORD: Final = MISSING_RECORD_REASON
REASON_DUPLICATE_RECORD: Final = DUPLICATE_RECORD_REASON
REASON_MUTATED_RECORD: Final = MUTATED_RECORD_REASON
REASON_UNEXPECTED_RECORD: Final = UNEXPECTED_RECORD_REASON
REASON_UNEXPECTED_REFERENCE: Final = UNEXPECTED_REFERENCE_REASON
REASON_DIGEST_MISMATCH: Final = DIGEST_MISMATCH_REASON
REASON_INVALID_REFERENCE: Final = INVALID_REFERENCE_REASON
REASON_INVALID_PACKET: Final = INVALID_PACKET_REASON

CLAIM_PACKET_INTEGRITY_REASONS: Final = (
    INVALID_PACKET_REASON,
    INVALID_REFERENCE_REASON,
    MISSING_REFERENCE_REASON,
    DUPLICATE_REFERENCE_REASON,
    REORDERED_REFERENCE_REASON,
    MUTATED_REFERENCE_REASON,
    MISSING_RECORD_REASON,
    DUPLICATE_RECORD_REASON,
    MUTATED_RECORD_REASON,
    UNEXPECTED_RECORD_REASON,
    UNEXPECTED_REFERENCE_REASON,
    DIGEST_MISMATCH_REASON,
    _INVALID_EXPECTED_DIGEST_REASON,
    _EXPECTED_PACKET_MISMATCH_REASON,
)

_DIGEST_LENGTH = 64
_DIGEST_PREFIX = "sha256:"
_SECTION_NAMES: Final = ("claims", "citations", "reviews", "policy")
_SECTION_ALIASES: Final = {
    "claims": ("claims", "claim_records", "claim"),
    "citations": ("citations", "citation_records", "citation"),
    "reviews": ("reviews", "review_records", "review"),
    "policy": ("policy", "policies", "policy_record"),
}
_ID_FIELDS: Final = {
    "claims": ("claim_id", "record_id", "id", "key"),
    "citations": (
        "citation_id",
        "reference_id",
        "evidence_id",
        "record_id",
        "id",
        "key",
    ),
    "reviews": ("review_id", "record_id", "id", "key"),
    "policy": ("policy_id", "record_id", "id", "key"),
}
_REFERENCE_FIELDS: Final = {
    "claims": frozenset(
        {
            "citation",
            "citations",
            "citation_id",
            "citation_ids",
            "citation_ref",
            "citation_refs",
            "evidence",
            "evidence_id",
            "evidence_ids",
            "evidence_ref",
            "evidence_refs",
            "policy",
            "policy_id",
            "policy_ids",
            "reference",
            "references",
            "reference_id",
            "reference_ids",
            "source",
            "source_id",
            "source_ids",
            "source_ref",
            "source_refs",
        }
    ),
    "citations": frozenset(
        {
            "claim",
            "claim_id",
            "claim_ids",
            "claim_ref",
            "claim_refs",
        }
    ),
    "reviews": frozenset(
        {
            "claim",
            "claim_id",
            "claim_ids",
            "claim_ref",
            "claim_refs",
            "citation",
            "citations",
            "citation_id",
            "citation_ids",
            "citation_ref",
            "citation_refs",
            "evidence",
            "evidence_id",
            "evidence_ids",
            "evidence_ref",
            "evidence_refs",
            "policy",
            "policy_id",
            "policy_ids",
            "reference",
            "references",
            "reference_id",
            "reference_ids",
        }
    ),
    "policy": frozenset(
        {
            "claim",
            "claim_id",
            "claim_ids",
            "claim_ref",
            "claim_refs",
            "review",
            "review_id",
            "review_ids",
            "review_ref",
            "review_refs",
        }
    ),
}
_RECORD_HINT_FIELDS: Final = frozenset(
    field
    for fields in (*_ID_FIELDS.values(), *_REFERENCE_FIELDS.values())
    for field in fields
) | frozenset(
    {
        "claim",
        "claim_text",
        "citation_text",
        "content",
        "kind",
        "policy_text",
        "review_text",
        "rules",
        "status",
        "summary",
        "text",
        "value",
        "version",
    }
)
_REASON_ORDER: Final = {
    reason: index for index, reason in enumerate(CLAIM_PACKET_INTEGRITY_REASONS)
}


class ClaimIntegrityError(ValueError):
    """Raised for malformed claim-packet input using only safe reason codes."""

    def __init__(self, code: str = INVALID_PACKET_REASON) -> None:
        safe_code = (
            code if code in CLAIM_PACKET_INTEGRITY_REASONS else INVALID_PACKET_REASON
        )
        self.code = safe_code
        super().__init__(f"claim packet rejected: {safe_code}")


ClaimPacketIntegrityError = ClaimIntegrityError


@dataclass(frozen=True)
class _Reference:
    """Internal reference identity; raw record values never leave this module."""

    field: str
    target: str
    identifier: str
    position: int

    @property
    def signature(self) -> tuple[str, str]:
        return self.target, self.identifier

    @property
    def digest(self) -> str:
        return stable_hash(
            {
                "kind": "openmed.claim-packet-reference",
                "target": self.target,
                "identifier": self.identifier,
            }
        )


@dataclass(frozen=True)
class _Record:
    section: str
    identifier: str
    canonical: dict[str, Any]
    references: tuple[_Reference, ...]
    digest: str
    content_digest: str


@dataclass(frozen=True)
class _Packet:
    records: dict[str, tuple[_Record, ...]]
    digest: str
    reference_count: int
    missing_reference_count: int
    duplicate_reference_count: int
    duplicate_record_count: int
    invalid_reference_count: int


@dataclass(frozen=True)
class ClaimPacketIntegrityReport:
    """Aggregate-only result of a claim-packet integrity check.

    The report intentionally exposes no record keys, identifiers, reference
    values, or clinical text.  Use ``digest`` and the fixed reason codes to
    correlate a report with a controlled review surface.
    """

    schema_version: int
    digest: str
    expected_digest: str | None
    digest_matches: bool
    passed: bool
    issues: tuple[str, ...]
    claim_count: int
    citation_count: int
    review_count: int
    policy_count: int
    reference_count: int
    missing_reference_count: int
    duplicate_reference_count: int
    invalid_reference_count: int
    missing_record_count: int = 0
    duplicate_record_count: int = 0
    mutated_record_count: int = 0
    mutated_reference_count: int = 0
    reordered_reference_count: int = 0
    unexpected_record_count: int = 0
    unexpected_reference_count: int = 0

    @property
    def integrity_digest(self) -> str:
        """Return the computed packet digest."""

        return self.digest

    @property
    def failure_reasons(self) -> tuple[str, ...]:
        """Return the stable reason-code alias used by other clinical reports."""

        return self.issues

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic report containing no raw input values."""

        return {
            "artifact": CLAIM_PACKET_INTEGRITY_KIND,
            "schema_version": self.schema_version,
            "digest_algorithm": CLAIM_PACKET_DIGEST_ALGORITHM,
            "digest": self.digest,
            "expected_digest": self.expected_digest,
            "digest_matches": self.digest_matches,
            "passed": self.passed,
            "issues": list(self.issues),
            "failure_reasons": list(self.failure_reasons),
            "claim_count": self.claim_count,
            "citation_count": self.citation_count,
            "review_count": self.review_count,
            "policy_count": self.policy_count,
            "reference_count": self.reference_count,
            "missing_reference_count": self.missing_reference_count,
            "duplicate_reference_count": self.duplicate_reference_count,
            "invalid_reference_count": self.invalid_reference_count,
            "missing_record_count": self.missing_record_count,
            "duplicate_record_count": self.duplicate_record_count,
            "mutated_record_count": self.mutated_record_count,
            "mutated_reference_count": self.mutated_reference_count,
            "reordered_reference_count": self.reordered_reference_count,
            "unexpected_record_count": self.unexpected_record_count,
            "unexpected_reference_count": self.unexpected_reference_count,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize the aggregate report deterministically."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            indent=indent,
        )

    def to_markdown(self) -> str:
        """Render aggregate review metadata without record values."""

        reasons = ", ".join(self.issues) or "none"
        lines = [
            "# Claim Packet Integrity",
            "",
            "Aggregate integrity metadata only; claim, citation, review, and "
            "policy values are never rendered.",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Schema version | {self.schema_version} |",
            f"| Claims | {self.claim_count} |",
            f"| Citations | {self.citation_count} |",
            f"| Reviews | {self.review_count} |",
            f"| Policies | {self.policy_count} |",
            f"| References | {self.reference_count} |",
            f"| Missing references | {self.missing_reference_count} |",
            f"| Duplicate references | {self.duplicate_reference_count} |",
            f"| Mutated references | {self.mutated_reference_count} |",
            f"| Reordered references | {self.reordered_reference_count} |",
            f"| Digest matches | {self.digest_matches} |",
            f"| Verdict | {'pass' if self.passed else 'fail'} |",
            f"| Failure reasons | `{reasons}` |",
        ]
        return "\n".join(lines) + "\n"

    def __getitem__(self, key: str) -> Any:
        """Provide mapping-style access to the safe serialized report."""

        return self.to_dict()[key]

    def __bool__(self) -> bool:
        return self.passed


ClaimPacketIntegrity = ClaimPacketIntegrityReport


def _error(code: str = INVALID_PACKET_REASON) -> ClaimIntegrityError:
    return ClaimIntegrityError(code)


def _normalise_key(value: Any) -> str:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if type(value) is int:
        return str(value)
    if type(value) is bool:
        return "true" if value else "false"
    raise _error(INVALID_PACKET_REASON)


def _canonical_value(value: Any) -> Any:
    """Build a JSON-safe canonical value for internal hashing only."""

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        try:
            items = value.items()
            for raw_key, raw_value in items:
                key = _normalise_key(raw_key)
                if key in result:
                    raise _error(INVALID_PACKET_REASON)
                result[key] = _canonical_value(raw_value)
        except ClaimIntegrityError:
            raise
        except Exception:
            raise _error(INVALID_PACKET_REASON) from None
        return {key: result[key] for key in sorted(result)}

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        try:
            return _canonical_value(dataclasses.asdict(value))
        except ClaimIntegrityError:
            raise
        except Exception:
            raise _error(INVALID_PACKET_REASON) from None

    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]

    if isinstance(value, (set, frozenset)):
        values = [_canonical_value(item) for item in value]
        return sorted(
            values,
            key=lambda item: json.dumps(
                item,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

    if value is None or type(value) is bool or type(value) is int:
        return value

    if type(value) is float:
        if not math.isfinite(value):
            raise _error(INVALID_PACKET_REASON)
        return value

    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)

    if isinstance(value, bytes):
        return {"__type__": "bytes", "value": value.hex()}

    if isinstance(value, (datetime, date, time)):
        return {"__type__": type(value).__name__, "value": value.isoformat()}

    if isinstance(value, Decimal):
        if not value.is_finite():
            raise _error(INVALID_PACKET_REASON)
        return {"__type__": "decimal", "value": str(value)}

    try:
        to_dict = getattr(value, "to_dict", None)
    except Exception:
        raise _error(INVALID_PACKET_REASON) from None
    if callable(to_dict):
        try:
            converted = to_dict()
        except Exception:
            raise _error(INVALID_PACKET_REASON) from None
        if isinstance(converted, Mapping):
            return _canonical_value(converted)

    raise _error(INVALID_PACKET_REASON)


def _as_mapping(value: Any) -> Mapping[Any, Any] | None:
    if isinstance(value, Mapping):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        try:
            converted = dataclasses.asdict(value)
        except Exception:
            raise _error(INVALID_PACKET_REASON) from None
        return converted if isinstance(converted, Mapping) else None
    try:
        to_dict = getattr(value, "to_dict", None)
    except Exception:
        raise _error(INVALID_PACKET_REASON) from None
    if callable(to_dict):
        try:
            converted = to_dict()
        except Exception:
            raise _error(INVALID_PACKET_REASON) from None
        if isinstance(converted, Mapping):
            return converted
    try:
        attributes = vars(value)
    except (TypeError, ValueError):
        return None
    return attributes if isinstance(attributes, Mapping) else None


def _field_name(value: Any) -> str:
    return _normalise_key(value).casefold()


def _mapping_value(mapping: Mapping[Any, Any], field: str) -> Any:
    for key, value in mapping.items():
        if _field_name(key) == field:
            return value
    return None


def _has_field(mapping: Mapping[Any, Any], field: str) -> bool:
    return any(_field_name(key) == field for key in mapping)


def _identifier(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if type(value) is int:
        return str(value)
    if isinstance(value, str):
        normalized = unicodedata.normalize("NFC", value.strip())
        return normalized or None
    return None


def _identifier_from_mapping(value: Any, target: str) -> str | None:
    mapping = _as_mapping(value)
    if mapping is None:
        return _identifier(value)
    fields = (
        *_ID_FIELDS.get(target, ()),
        "reference_id",
        "evidence_id",
        "source_id",
        "id",
        "key",
    )
    for field in fields:
        candidate = _mapping_value(mapping, field)
        identifier = _identifier(candidate)
        if identifier is not None:
            return identifier
    return None


def _reference_target(section: str, field: str) -> str | None:
    if field.startswith("policy") or field == "policy":
        return None if section == "policy" else "policy"
    if field.startswith("claim") or field == "claim":
        return None if section == "claims" else "claims"
    if field.startswith("review") or field == "review":
        return None if section == "reviews" else "reviews"
    if field.startswith(("citation", "evidence", "source")):
        if section in {"claims", "reviews"}:
            return "citations"
    if field.startswith("reference"):
        if section == "claims":
            return "citations"
        if section == "reviews":
            return "claims"
    return None


def _reference_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        if _identifier_from_mapping(value, "citations") is not None:
            return [value]
        try:
            items = list(value.items())
        except Exception:
            raise _error(INVALID_REFERENCE_REASON) from None
        if items and all(_as_mapping(item) is not None for _, item in items):
            return sorted(
                (key for key, _ in items),
                key=lambda key: _identifier(key) or _normalise_key(key),
            )
        return [value]
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    try:
        return list(value)
    except (TypeError, ValueError):
        return [value]


def _looks_like_record(value: Mapping[Any, Any]) -> bool:
    try:
        fields = {_field_name(key) for key in value}
    except Exception:
        raise _error(INVALID_PACKET_REASON) from None
    return bool(fields & _RECORD_HINT_FIELDS)


def _coerce_records(value: Any, section: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        if not value:
            return []
        if section == "policy" or _looks_like_record(value):
            return [value]
        try:
            items = list(value.items())
        except Exception:
            raise _error(INVALID_PACKET_REASON) from None
        if items:
            records: list[Any] = []
            for key, item in items:
                item_mapping = _as_mapping(item)
                if item_mapping is None:
                    records.append({"id": key, "value": item})
                    continue
                record = dict(item_mapping)
                if not any(_has_field(record, field) for field in _ID_FIELDS[section]):
                    record[_ID_FIELDS[section][0]] = key
                records.append(record)
            return records
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    try:
        return list(value)
    except (TypeError, ValueError):
        return [value]


def _section_value(packet: Mapping[Any, Any], section: str) -> Any:
    for alias in _SECTION_ALIASES[section]:
        for key, value in packet.items():
            if _field_name(key) == alias:
                return value
    return None


def _resolve_packet_input(
    packet_or_claims: Any,
    citations: Any,
    reviews: Any,
    policy: Any,
    claims: Any,
) -> Mapping[str, Any]:
    if claims is not None:
        if packet_or_claims is not None:
            raise _error(INVALID_PACKET_REASON)
        return {
            "claims": claims,
            "citations": citations,
            "reviews": reviews,
            "policy": policy,
        }

    supplied_sections = any(item is not None for item in (citations, reviews, policy))
    if supplied_sections:
        return {
            "claims": packet_or_claims,
            "citations": citations,
            "reviews": reviews,
            "policy": policy,
        }

    if packet_or_claims is None:
        return {section: None for section in _SECTION_NAMES}

    packet_mapping = _as_mapping(packet_or_claims)
    if packet_mapping is not None and any(
        _section_value(packet_mapping, section) is not None
        for section in _SECTION_NAMES
    ):
        return packet_mapping
    return {"claims": packet_or_claims}


def _record_identifier(
    mapping: Mapping[Any, Any],
    section: str,
    canonical: dict[str, Any],
) -> str:
    for field in _ID_FIELDS[section]:
        candidate = _identifier(_mapping_value(mapping, field))
        if candidate is not None:
            return candidate
    return "anonymous:" + stable_hash(
        {"section": section, "record": canonical}
    ).removeprefix(_DIGEST_PREFIX)


def _reference_list(
    mapping: Mapping[Any, Any],
    section: str,
) -> tuple[tuple[_Reference, ...], int]:
    references: list[_Reference] = []
    invalid_count = 0
    position = 0
    try:
        fields = sorted(
            ((_field_name(key), value) for key, value in mapping.items()),
            key=lambda item: item[0],
        )
    except Exception:
        raise _error(INVALID_PACKET_REASON) from None

    for field, value in fields:
        if field not in _REFERENCE_FIELDS[section]:
            continue
        target = _reference_target(section, field)
        if target is None:
            continue
        try:
            items = _reference_items(value)
        except ClaimIntegrityError:
            invalid_count += 1
            continue
        for item in items:
            identifier = _identifier_from_mapping(item, target)
            if identifier is None:
                invalid_count += 1
                continue
            references.append(
                _Reference(
                    field=field,
                    target=target,
                    identifier=identifier,
                    position=position,
                )
            )
            position += 1
    return tuple(references), invalid_count


def _record_without_references(
    canonical: Mapping[str, Any],
    section: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field, value in canonical.items():
        if _reference_target(section, field) is None:
            result[field] = value
    return result


def _build_record(raw_record: Any, section: str) -> tuple[_Record, int]:
    mapping = _as_mapping(raw_record)
    if mapping is None:
        if isinstance(raw_record, (str, int)) and not isinstance(raw_record, bool):
            mapping = {"id": raw_record}
        else:
            raise _error(INVALID_PACKET_REASON)
    canonical = _canonical_value(mapping)
    if not isinstance(canonical, dict):
        raise _error(INVALID_PACKET_REASON)
    identifier = _record_identifier(mapping, section, canonical)
    references, invalid_count = _reference_list(mapping, section)
    record_digest = stable_hash(
        {
            "kind": "openmed.claim-packet-record",
            "schema_version": CLAIM_PACKET_SCHEMA_VERSION,
            "section": section,
            "identifier": identifier,
            "record": canonical,
            "references": [
                {
                    "field": reference.field,
                    "target": reference.target,
                    "identifier": reference.identifier,
                }
                for reference in references
            ],
        }
    )
    content_digest = stable_hash(
        {
            "kind": "openmed.claim-packet-record-content",
            "schema_version": CLAIM_PACKET_SCHEMA_VERSION,
            "section": section,
            "identifier": identifier,
            "record": _record_without_references(canonical, section),
        }
    )
    return (
        _Record(
            section=section,
            identifier=identifier,
            canonical=canonical,
            references=references,
            digest=record_digest,
            content_digest=content_digest,
        ),
        invalid_count,
    )


def _canonical_packet_payload(
    records: Mapping[str, tuple[_Record, ...]],
) -> dict[str, Any]:
    return {
        "kind": CLAIM_PACKET_INTEGRITY_KIND,
        "schema_version": CLAIM_PACKET_SCHEMA_VERSION,
        "sections": {
            section: [
                {
                    "identifier": record.identifier,
                    "record": record.canonical,
                    "references": [
                        {
                            "field": reference.field,
                            "target": reference.target,
                            "identifier": reference.identifier,
                        }
                        for reference in record.references
                    ],
                }
                for record in records[section]
            ]
            for section in _SECTION_NAMES
        },
    }


def _build_packet(packet_input: Mapping[str, Any]) -> _Packet:
    records: dict[str, tuple[_Record, ...]] = {}
    invalid_reference_count = 0
    duplicate_record_count = 0

    for section in _SECTION_NAMES:
        raw_records = _coerce_records(_section_value(packet_input, section), section)
        built: list[_Record] = []
        for raw_record in raw_records:
            record, invalid_count = _build_record(raw_record, section)
            built.append(record)
            invalid_reference_count += invalid_count
        built.sort(key=lambda record: (record.identifier, record.digest))
        counts = Counter(record.identifier for record in built)
        duplicate_record_count += sum(max(count - 1, 0) for count in counts.values())
        records[section] = tuple(built)

    available = {
        section: {record.identifier for record in section_records}
        for section, section_records in records.items()
    }
    reference_count = 0
    missing_reference_count = 0
    duplicate_reference_count = 0
    for section_records in records.values():
        for record in section_records:
            reference_count += len(record.references)
            signatures = [reference.signature for reference in record.references]
            reference_counts = Counter(signatures)
            duplicate_reference_count += sum(
                max(count - 1, 0) for count in reference_counts.values()
            )
            missing_reference_count += sum(
                1
                for reference in record.references
                if reference.identifier not in available[reference.target]
            )

    digest = stable_hash(_canonical_packet_payload(records))
    return _Packet(
        records=records,
        digest=digest,
        reference_count=reference_count,
        missing_reference_count=missing_reference_count,
        duplicate_reference_count=duplicate_reference_count,
        duplicate_record_count=duplicate_record_count,
        invalid_reference_count=invalid_reference_count,
    )


def _normalise_packet(
    packet_or_claims: Any = None,
    citations: Any = None,
    reviews: Any = None,
    policy: Any = None,
    *,
    claims: Any = None,
) -> _Packet:
    try:
        packet_input = _resolve_packet_input(
            packet_or_claims,
            citations,
            reviews,
            policy,
            claims,
        )
        return _build_packet(packet_input)
    except ClaimIntegrityError:
        raise
    except Exception:
        raise _error(INVALID_PACKET_REASON) from None


def _safe_digest(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip().casefold()
    if (
        candidate.startswith(_DIGEST_PREFIX)
        and len(candidate) == len(_DIGEST_PREFIX) + _DIGEST_LENGTH
        and all(
            character in "0123456789abcdef"
            for character in candidate[len(_DIGEST_PREFIX) :]
        )
    ):
        return candidate
    return None


def _public_canonical(packet: _Packet) -> dict[str, Any]:
    return {
        "artifact": CLAIM_PACKET_INTEGRITY_KIND,
        "schema_version": CLAIM_PACKET_SCHEMA_VERSION,
        "digest_algorithm": CLAIM_PACKET_DIGEST_ALGORITHM,
        "digest": packet.digest,
        "sections": {
            section: [
                {
                    "record_digest": record.digest,
                    "reference_count": len(record.references),
                    "reference_digests": [
                        reference.digest for reference in record.references
                    ],
                }
                for record in packet.records[section]
            ]
            for section in _SECTION_NAMES
        },
        "counts": {
            "claims": len(packet.records["claims"]),
            "citations": len(packet.records["citations"]),
            "reviews": len(packet.records["reviews"]),
            "policy": len(packet.records["policy"]),
            "references": packet.reference_count,
            "missing_references": packet.missing_reference_count,
            "duplicate_references": packet.duplicate_reference_count,
            "duplicate_records": packet.duplicate_record_count,
            "invalid_references": packet.invalid_reference_count,
        },
    }


def canonicalize_claim_packet(
    packet_or_claims: Any = None,
    citations: Any = None,
    reviews: Any = None,
    policy: Any = None,
    *,
    claims: Any = None,
) -> dict[str, Any]:
    """Return a versioned canonical representation containing digests only.

    Mapping-key order and record order are normalized.  Reference-list order
    is preserved because it is part of the integrity contract.  Raw values
    affect the internal digest but are never copied into the returned mapping.
    """

    return _public_canonical(
        _normalise_packet(
            packet_or_claims,
            citations,
            reviews,
            policy,
            claims=claims,
        )
    )


def compute_claim_packet_digest(
    packet_or_claims: Any = None,
    citations: Any = None,
    reviews: Any = None,
    policy: Any = None,
    *,
    claims: Any = None,
) -> str:
    """Return the deterministic versioned SHA-256 claim-packet digest."""

    return _normalise_packet(
        packet_or_claims,
        citations,
        reviews,
        policy,
        claims=claims,
    ).digest


def _record_index(packet: _Packet) -> dict[tuple[str, str], tuple[_Record, ...]]:
    return {
        (section, record.identifier): tuple(
            item for item in section_records if item.identifier == record.identifier
        )
        for section, section_records in packet.records.items()
        for record in section_records
    }


def _compare_packets(
    expected: _Packet,
    candidate: _Packet,
) -> dict[str, int]:
    counts = {
        "missing_record_count": 0,
        "mutated_record_count": 0,
        "mutated_reference_count": 0,
        "reordered_reference_count": 0,
        "unexpected_record_count": 0,
        "unexpected_reference_count": 0,
        "missing_reference_count": 0,
    }
    expected_index = _record_index(expected)
    candidate_index = _record_index(candidate)
    keys = sorted(set(expected_index) | set(candidate_index))
    for key in keys:
        expected_records = expected_index.get(key, ())
        candidate_records = candidate_index.get(key, ())
        if len(candidate_records) < len(expected_records):
            counts["missing_record_count"] += len(expected_records) - len(
                candidate_records
            )
        if len(candidate_records) > len(expected_records):
            counts["unexpected_record_count"] += len(candidate_records) - len(
                expected_records
            )
        for expected_record, candidate_record in zip(
            expected_records, candidate_records
        ):
            if expected_record.content_digest != candidate_record.content_digest:
                counts["mutated_record_count"] += 1
            expected_references = [
                reference.signature for reference in expected_record.references
            ]
            candidate_references = [
                reference.signature for reference in candidate_record.references
            ]
            if expected_references == candidate_references:
                continue
            if Counter(expected_references) == Counter(candidate_references):
                counts["reordered_reference_count"] += 1
                continue
            expected_counter = Counter(expected_references)
            candidate_counter = Counter(candidate_references)
            counts["missing_reference_count"] += sum(
                (expected_counter - candidate_counter).values()
            )
            counts["unexpected_reference_count"] += sum(
                (candidate_counter - expected_counter).values()
            )
            for expected_reference, candidate_reference in zip(
                expected_references,
                candidate_references,
            ):
                if (
                    expected_reference != candidate_reference
                    and expected_reference[0] == candidate_reference[0]
                ):
                    counts["mutated_reference_count"] += 1
    return counts


def _report(
    packet: _Packet,
    *,
    expected_digest: Any = None,
    expected_packet: _Packet | None = None,
) -> ClaimPacketIntegrityReport:
    issues: set[str] = set()
    safe_expected_digest = _safe_digest(expected_digest)
    if expected_digest is not None and safe_expected_digest is None:
        issues.add(_INVALID_EXPECTED_DIGEST_REASON)

    comparison = {
        "missing_record_count": 0,
        "mutated_record_count": 0,
        "mutated_reference_count": 0,
        "reordered_reference_count": 0,
        "unexpected_record_count": 0,
        "unexpected_reference_count": 0,
        "missing_reference_count": 0,
    }
    if expected_packet is not None:
        if safe_expected_digest is None:
            safe_expected_digest = expected_packet.digest
        elif safe_expected_digest != expected_packet.digest:
            issues.add(_EXPECTED_PACKET_MISMATCH_REASON)
        comparison = _compare_packets(expected_packet, packet)

    missing_reference_count = (
        packet.missing_reference_count + comparison["missing_reference_count"]
    )
    duplicate_reference_count = packet.duplicate_reference_count
    if packet.invalid_reference_count:
        issues.add(INVALID_REFERENCE_REASON)
    if missing_reference_count:
        issues.add(MISSING_REFERENCE_REASON)
    if duplicate_reference_count:
        issues.add(DUPLICATE_REFERENCE_REASON)
    if packet.duplicate_record_count:
        issues.add(DUPLICATE_RECORD_REASON)
    if comparison["missing_record_count"]:
        issues.add(MISSING_RECORD_REASON)
    if comparison["unexpected_record_count"]:
        issues.add(UNEXPECTED_RECORD_REASON)
    if comparison["unexpected_reference_count"]:
        issues.add(UNEXPECTED_REFERENCE_REASON)
    if comparison["mutated_record_count"]:
        issues.add(MUTATED_RECORD_REASON)
    if comparison["mutated_reference_count"]:
        issues.add(MUTATED_REFERENCE_REASON)
    if comparison["reordered_reference_count"]:
        issues.add(REORDERED_REFERENCE_REASON)

    digest_matches = (
        safe_expected_digest is None or packet.digest == safe_expected_digest
    )
    if safe_expected_digest is not None and not digest_matches:
        issues.add(DIGEST_MISMATCH_REASON)

    ordered_issues = tuple(
        sorted(issues, key=lambda reason: _REASON_ORDER.get(reason, 999))
    )
    return ClaimPacketIntegrityReport(
        schema_version=CLAIM_PACKET_SCHEMA_VERSION,
        digest=packet.digest,
        expected_digest=safe_expected_digest,
        digest_matches=digest_matches and _INVALID_EXPECTED_DIGEST_REASON not in issues,
        passed=not ordered_issues,
        issues=ordered_issues,
        claim_count=len(packet.records["claims"]),
        citation_count=len(packet.records["citations"]),
        review_count=len(packet.records["reviews"]),
        policy_count=len(packet.records["policy"]),
        reference_count=packet.reference_count,
        missing_reference_count=missing_reference_count,
        duplicate_reference_count=duplicate_reference_count,
        invalid_reference_count=packet.invalid_reference_count,
        missing_record_count=comparison["missing_record_count"],
        duplicate_record_count=packet.duplicate_record_count,
        mutated_record_count=comparison["mutated_record_count"],
        mutated_reference_count=comparison["mutated_reference_count"],
        reordered_reference_count=comparison["reordered_reference_count"],
        unexpected_record_count=comparison["unexpected_record_count"],
        unexpected_reference_count=comparison["unexpected_reference_count"],
    )


def check_claim_packet_integrity(
    packet_or_claims: Any = None,
    citations: Any = None,
    reviews: Any = None,
    policy: Any = None,
    *,
    claims: Any = None,
    expected_digest: Any = None,
    expected_packet: Any = None,
) -> ClaimPacketIntegrityReport:
    """Check a packet and optionally compare it with a baseline packet/digest.

    Positional inputs may be supplied as ``claims, citations, reviews,
    policy``.  Alternatively pass one mapping containing the four section
    names.  ``expected_packet`` enables safe classification of missing,
    reordered, and mutated references; an expected digest alone verifies the
    final binding without retaining any expected raw values.
    """

    if (
        expected_digest is None
        and reviews is None
        and policy is None
        and isinstance(citations, str)
        and _safe_digest(citations) is not None
    ):
        expected_digest = citations
        citations = None

    packet = _normalise_packet(
        packet_or_claims,
        citations,
        reviews,
        policy,
        claims=claims,
    )
    baseline = None
    if expected_packet is not None:
        baseline = _normalise_packet(expected_packet)
    return _report(
        packet,
        expected_digest=expected_digest,
        expected_packet=baseline,
    )


def verify_claim_packet_integrity(
    packet_or_claims: Any = None,
    expected_digest: Any = None,
    *,
    expected_packet: Any = None,
    citations: Any = None,
    reviews: Any = None,
    policy: Any = None,
    claims: Any = None,
) -> ClaimPacketIntegrityReport:
    """Return a safe report verifying a packet against an expected digest."""

    return check_claim_packet_integrity(
        packet_or_claims,
        citations,
        reviews,
        policy,
        claims=claims,
        expected_digest=expected_digest,
        expected_packet=expected_packet,
    )


def verify_claim_packet_digest(
    packet_or_claims: Any = None,
    expected_digest: Any = None,
    *,
    expected_packet: Any = None,
    citations: Any = None,
    reviews: Any = None,
    policy: Any = None,
    claims: Any = None,
) -> bool:
    """Return whether the packet passes its digest and structural checks."""

    return verify_claim_packet_integrity(
        packet_or_claims,
        expected_digest,
        expected_packet=expected_packet,
        citations=citations,
        reviews=reviews,
        policy=policy,
        claims=claims,
    ).passed


def is_claim_packet_integrity_valid(
    packet_or_claims: Any = None,
    expected_digest: Any = None,
    **kwargs: Any,
) -> bool:
    """Boolean alias for :func:`verify_claim_packet_digest`."""

    return verify_claim_packet_digest(
        packet_or_claims,
        expected_digest,
        **kwargs,
    )


def build_claim_packet_integrity_report(
    packet_or_claims: Any = None,
    citations: Any = None,
    reviews: Any = None,
    policy: Any = None,
    **kwargs: Any,
) -> ClaimPacketIntegrityReport:
    """Build an aggregate-only claim-packet integrity report."""

    return check_claim_packet_integrity(
        packet_or_claims,
        citations,
        reviews,
        policy,
        **kwargs,
    )


def validate_claim_packet_integrity(
    packet_or_claims: Any = None,
    citations: Any = None,
    reviews: Any = None,
    policy: Any = None,
    **kwargs: Any,
) -> ClaimPacketIntegrityReport:
    """Alias for :func:`check_claim_packet_integrity`."""

    return check_claim_packet_integrity(
        packet_or_claims,
        citations,
        reviews,
        policy,
        **kwargs,
    )


def compute_claim_packet_integrity(
    packet_or_claims: Any = None,
    citations: Any = None,
    reviews: Any = None,
    policy: Any = None,
    **kwargs: Any,
) -> ClaimPacketIntegrityReport:
    """Alias for the aggregate integrity check."""

    return check_claim_packet_integrity(
        packet_or_claims,
        citations,
        reviews,
        policy,
        **kwargs,
    )


build_claim_packet_digest = compute_claim_packet_digest
calculate_claim_packet_digest = compute_claim_packet_digest
claim_packet_digest = compute_claim_packet_digest
compute_claim_packet_integrity_digest = compute_claim_packet_digest
