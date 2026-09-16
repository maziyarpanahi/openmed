"""Deterministic, value-free coverage reports for guarded clinical claims.

The evidence coverage matrix is an operational review aid.  It records only
opaque claim identifiers, categorical evidence classes, review state, and
source fingerprints.  Claim text, evidence text, offsets, and clinical
interpretation are deliberately outside the contract and are ignored when
building a matrix.  All processing is in memory and local-first.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

EVIDENCE_COVERAGE_SCHEMA_VERSION = "openmed.clinical.evidence_coverage.v1"
EVIDENCE_COVERAGE_NOTE = (
    "Evidence coverage is a value-free review aid, not clinical interpretation."
)

COVERAGE_STATUSES = ("present", "missing", "conflicting", "unreviewed")
REVIEW_STATES = ("reviewed", "missing", "conflicting", "unreviewed")

CoverageStatus = Literal["present", "missing", "conflicting", "unreviewed"]
ReviewState = Literal["reviewed", "missing", "conflicting", "unreviewed"]

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+@-]{0,127}$")
_CLAIM_ID_KEYS = ("claim_id", "opaque_claim_id", "id")
_EVIDENCE_CLASS_KEYS = (
    "evidence_class",
    "evidence_type",
    "class",
    "category",
    "type",
)
_EVIDENCE_CONTAINER_KEYS = (
    "required_evidence",
    "required_evidence_classes",
    "requirements",
    "evidence",
)
_STATUS_KEYS = ("status", "coverage_status")
_REVIEW_STATE_KEYS = ("review_state", "state")
_FINGERPRINT_KEYS = ("source_fingerprints", "source_fingerprint", "source_hashes")
_STATUS_MAP_KEYS = ("statuses", "coverage_statuses", "status_by_class")
_REVIEW_STATE_MAP_KEYS = ("review_states", "review_state_by_class")
_FINGERPRINT_MAP_KEYS = (
    "source_fingerprints_by_class",
    "source_hashes_by_class",
)

_STATUS_ALIASES = {
    "present": "present",
    "available": "present",
    "complete": "present",
    "reviewed": "present",
    "supported": "present",
    "missing": "missing",
    "absent": "missing",
    "none": "missing",
    "conflicting": "conflicting",
    "conflict": "conflicting",
    "unreviewed": "unreviewed",
    "pending": "unreviewed",
    "unknown": "unreviewed",
}
_REVIEW_ALIASES = {
    "reviewed": "reviewed",
    "present": "reviewed",
    "available": "reviewed",
    "complete": "reviewed",
    "supported": "reviewed",
    "missing": "missing",
    "absent": "missing",
    "none": "missing",
    "conflicting": "conflicting",
    "conflict": "conflicting",
    "unreviewed": "unreviewed",
    "pending": "unreviewed",
    "unknown": "unreviewed",
}


class EvidenceCoverageError(ValueError):
    """Raised when a value-free evidence coverage input is invalid."""


def fingerprint_source(source: object) -> str:
    """Return a stable SHA-256 fingerprint without retaining source content.

    Strings and bytes are hashed directly.  JSON-compatible structured values
    are canonicalized before hashing.  The returned digest is the only source
    representation accepted in a serialized coverage report.
    """

    if isinstance(source, bytes):
        encoded = source
    elif isinstance(source, str):
        encoded = source.encode("utf-8")
    else:
        try:
            encoded = _canonical_json(source).encode("utf-8")
        except (TypeError, ValueError, OverflowError) as exc:
            raise EvidenceCoverageError("source cannot be fingerprinted") from exc
    return _sha256(encoded)


@dataclass(frozen=True, slots=True)
class EvidenceCoverageRecord:
    """One required evidence class for one opaque claim.

    ``source_fingerprints`` contains only SHA-256 digests.  A present record
    must have at least one fingerprint; missing records have none.  Conflicting
    and unreviewed records may carry zero or more fingerprints because the
    review state does not assert that a source has been accepted.
    """

    claim_id: str
    evidence_class: str
    status: CoverageStatus
    review_state: ReviewState | None = None
    source_fingerprints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        claim_id = _identifier(self.claim_id, "claim_id")
        evidence_class = _identifier(self.evidence_class, "evidence_class")
        status = _status(self.status)
        fingerprints = _fingerprints(self.source_fingerprints)
        review_state = (
            _review_state(self.review_state)
            if self.review_state is not None
            else _review_for_status(status)
        )

        if review_state != _review_for_status(status):
            raise EvidenceCoverageError("status and review_state are inconsistent")
        if status == "present" and not fingerprints:
            raise EvidenceCoverageError("present evidence requires a fingerprint")
        if status == "missing" and fingerprints:
            raise EvidenceCoverageError("missing evidence cannot have a fingerprint")

        object.__setattr__(self, "claim_id", claim_id)
        object.__setattr__(self, "evidence_class", evidence_class)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "review_state", review_state)
        object.__setattr__(self, "source_fingerprints", fingerprints)

    @property
    def source_fingerprint(self) -> str | None:
        """Return the first source fingerprint, if one is available."""

        return self.source_fingerprints[0] if self.source_fingerprints else None

    def to_dict(self, *, include_claim_id: bool = True) -> dict[str, Any]:
        """Return the value-free JSON representation of this record."""

        payload: dict[str, Any] = {
            "evidence_class": self.evidence_class,
            "status": self.status,
            "review_state": self.review_state,
            "source_fingerprints": list(self.source_fingerprints),
        }
        if include_claim_id:
            payload = {"claim_id": self.claim_id, **payload}
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "EvidenceCoverageRecord":
        """Build a record from a value-free mapping."""

        if not isinstance(payload, Mapping):
            raise EvidenceCoverageError("evidence record must be a mapping")
        claim_id = _first_field(payload, _CLAIM_ID_KEYS)
        evidence_class = _first_field(payload, _EVIDENCE_CLASS_KEYS)
        if claim_id is None or evidence_class is None:
            raise EvidenceCoverageError("evidence record requires an identifier")
        return _record_from_requirement(
            claim_id,
            evidence_class,
            payload,
        )


@dataclass(frozen=True, slots=True)
class EvidenceCoverageMatrix:
    """Sorted, hashable value-free evidence coverage rows."""

    records: tuple[EvidenceCoverageRecord, ...] = ()

    def __post_init__(self) -> None:
        normalized: list[EvidenceCoverageRecord] = []
        seen: set[tuple[str, str]] = set()
        for record in self.records:
            if not isinstance(record, EvidenceCoverageRecord):
                raise EvidenceCoverageError("matrix records must be typed records")
            key = (record.claim_id, record.evidence_class)
            if key in seen:
                raise EvidenceCoverageError(
                    "matrix contains duplicate evidence classes"
                )
            seen.add(key)
            normalized.append(record)
        normalized.sort(key=lambda item: (item.claim_id, item.evidence_class))
        object.__setattr__(self, "records", tuple(normalized))

    @property
    def claim_ids(self) -> tuple[str, ...]:
        """Return sorted opaque claim identifiers represented by the matrix."""

        return tuple(dict.fromkeys(record.claim_id for record in self.records))

    @property
    def claim_count(self) -> int:
        """Return the number of claims represented by the matrix."""

        return len(self.claim_ids)

    @property
    def required_evidence_count(self) -> int:
        """Return the number of required evidence-class cells."""

        return len(self.records)

    @property
    def status_counts(self) -> dict[str, int]:
        """Return fixed-order counts for all coverage statuses."""

        counts = dict.fromkeys(COVERAGE_STATUSES, 0)
        for record in self.records:
            counts[record.status] += 1
        return counts

    @property
    def evidence_class_counts(self) -> dict[str, dict[str, int]]:
        """Return fixed-order status counts for each evidence class."""

        classes = sorted({record.evidence_class for record in self.records})
        counts = {
            evidence_class: dict.fromkeys(COVERAGE_STATUSES, 0)
            for evidence_class in classes
        }
        for record in self.records:
            counts[record.evidence_class][record.status] += 1
        return counts

    @property
    def source_fingerprint_count(self) -> int:
        """Return the number of distinct source fingerprints in the matrix."""

        return len(
            {
                fingerprint
                for record in self.records
                for fingerprint in record.source_fingerprints
            }
        )

    @property
    def source_fingerprint_hash(self) -> str:
        """Return a digest of the sorted source fingerprints only."""

        fingerprints = sorted(
            {
                fingerprint
                for record in self.records
                for fingerprint in record.source_fingerprints
            }
        )
        return _digest(fingerprints)

    def claim_records(self, claim_id: str) -> tuple[EvidenceCoverageRecord, ...]:
        """Return the evidence cells for one opaque claim identifier."""

        normalized_id = _identifier(claim_id, "claim_id")
        return tuple(
            record for record in self.records if record.claim_id == normalized_id
        )

    def claim_hash(self, claim_id: str) -> str:
        """Return the deterministic digest for one claim's safe evidence rows."""

        records = self.claim_records(claim_id)
        if not records:
            raise EvidenceCoverageError("claim identifier is not present")
        return _digest(
            {
                "claim_id": records[0].claim_id,
                "requirements": [
                    record.to_dict(include_claim_id=False) for record in records
                ],
            }
        )

    @property
    def matrix_hash(self) -> str:
        """Return the deterministic digest for the complete safe matrix."""

        return _digest(
            {
                "schema_version": EVIDENCE_COVERAGE_SCHEMA_VERSION,
                "records": [record.to_dict() for record in self.records],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible coverage evidence."""

        claims: list[dict[str, Any]] = []
        for claim_id in self.claim_ids:
            records = self.claim_records(claim_id)
            claims.append(
                {
                    "claim_id": claim_id,
                    "requirements": [
                        record.to_dict(include_claim_id=False) for record in records
                    ],
                    "claim_hash": self.claim_hash(claim_id),
                }
            )
        return {
            "schema_version": EVIDENCE_COVERAGE_SCHEMA_VERSION,
            "claim_count": self.claim_count,
            "required_evidence_count": self.required_evidence_count,
            "status_counts": self.status_counts,
            "evidence_class_counts": self.evidence_class_counts,
            "source_fingerprint_count": self.source_fingerprint_count,
            "source_fingerprint_hash": self.source_fingerprint_hash,
            "claims": claims,
            "matrix_hash": self.matrix_hash,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize the matrix with stable key ordering and no raw values."""

        kwargs: dict[str, Any] = {
            "ensure_ascii": True,
            "allow_nan": False,
            "sort_keys": True,
        }
        if indent is None:
            kwargs["separators"] = (",", ":")
        else:
            kwargs["indent"] = indent
        return json.dumps(self.to_dict(), **kwargs)

    def to_markdown(self) -> str:
        """Render a stable value-free Markdown summary."""

        lines = [
            "# Clinical Evidence Coverage Matrix",
            "",
            EVIDENCE_COVERAGE_NOTE,
            "",
            "| Status | Count |",
            "|---|---:|",
        ]
        lines.extend(
            f"| `{status}` | {self.status_counts[status]} |"
            for status in COVERAGE_STATUSES
        )
        lines.extend(
            [
                "",
                f"Claims: {self.claim_count}",
                f"Required evidence cells: {self.required_evidence_count}",
                f"Source fingerprint count: {self.source_fingerprint_count}",
                f"Source fingerprint hash: `{self.source_fingerprint_hash}`",
                f"Matrix hash: `{self.matrix_hash}`",
                "",
                "| Claim ID | Evidence class | Status | Review state | Source count |",
                "|---|---|---|---|---:|",
            ]
        )
        lines.extend(
            "| `{}` | `{}` | `{}` | `{}` | {} |".format(
                record.claim_id,
                record.evidence_class,
                record.status,
                record.review_state,
                len(record.source_fingerprints),
            )
            for record in self.records
        )
        return "\n".join(lines) + "\n"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "EvidenceCoverageMatrix":
        """Reconstruct a matrix from :meth:`to_dict` output."""

        if not isinstance(payload, Mapping):
            raise EvidenceCoverageError("matrix payload must be a mapping")
        if payload.get("schema_version") != EVIDENCE_COVERAGE_SCHEMA_VERSION:
            raise EvidenceCoverageError("unsupported evidence coverage schema")
        claims = payload.get("claims")
        if not isinstance(claims, (Mapping, Sequence)) or isinstance(
            claims, (str, bytes)
        ):
            raise EvidenceCoverageError("matrix claims must be a collection")

        records: list[EvidenceCoverageRecord] = []
        supplied_claim_hashes: list[tuple[str, object]] = []
        claim_items = (
            claims.items()
            if isinstance(claims, Mapping)
            else ((None, item) for item in claims)
        )
        for hinted_claim_id, claim in claim_items:
            if not isinstance(claim, Mapping):
                raise EvidenceCoverageError("matrix claim must be a mapping")
            claim_id = _first_field(claim, _CLAIM_ID_KEYS) or hinted_claim_id
            requirements = claim.get("requirements", claim.get("required_evidence"))
            if claim_id is None or requirements is None:
                raise EvidenceCoverageError("matrix claim is incomplete")
            supplied_claim_hash = claim.get("claim_hash")
            if supplied_claim_hash is not None:
                supplied_claim_hashes.append(
                    (_identifier(claim_id, "claim_id"), supplied_claim_hash)
                )
            for requirement in _iter_requirement_items(requirements):
                records.append(_record_from_requirement(claim_id, None, requirement))

        matrix = cls(tuple(records))
        for claim_id, supplied_claim_hash in supplied_claim_hashes:
            if supplied_claim_hash != matrix.claim_hash(claim_id):
                raise EvidenceCoverageError("claim hash does not match its contents")
        supplied_hash = payload.get("matrix_hash")
        if supplied_hash is not None and supplied_hash != matrix.matrix_hash:
            raise EvidenceCoverageError("matrix hash does not match its contents")
        return matrix


def build_evidence_coverage_matrix(claims: object) -> EvidenceCoverageMatrix:
    """Build a deterministic value-free matrix from local claim metadata.

    Accepted inputs are a sequence of claim mappings/objects or a mapping from
    opaque claim IDs to requirement metadata.  A claim may use
    ``required_evidence``, ``required_evidence_classes``, ``requirements``, or
    ``evidence``.  Each requirement may provide ``evidence_class``, ``status``
    or ``review_state``, and ``source_fingerprint(s)``.  A ``source`` field is
    accepted only as an input convenience and is immediately fingerprinted;
    it is never retained or serialized.

    The builder reads no claim or evidence text.  Unknown metadata fields are
    ignored, which makes it safe to pass records that also contain upstream
    model output without copying that output into the report.
    """

    merged: dict[tuple[str, str], EvidenceCoverageRecord] = {}
    for hinted_claim_id, claim in _iter_claims(claims):
        claim_id = _claim_id(claim, hinted_claim_id)
        for requirement in _iter_claim_requirements(claim):
            record = _record_from_requirement(claim_id, None, requirement)
            key = (record.claim_id, record.evidence_class)
            previous = merged.get(key)
            merged[key] = (
                record if previous is None else _merge_records(previous, record)
            )
    return EvidenceCoverageMatrix(tuple(merged.values()))


def render_evidence_coverage_matrix(
    matrix_or_claims: EvidenceCoverageMatrix | object,
    *,
    format: Literal["dict", "json", "markdown"] = "dict",
    indent: int | None = None,
) -> dict[str, Any] | str:
    """Render a matrix or claim input as a dict, JSON string, or Markdown."""

    matrix = (
        matrix_or_claims
        if isinstance(matrix_or_claims, EvidenceCoverageMatrix)
        else build_evidence_coverage_matrix(matrix_or_claims)
    )
    if format == "dict":
        return matrix.to_dict()
    if format == "json":
        return matrix.to_json(indent=indent)
    if format == "markdown":
        return matrix.to_markdown()
    raise EvidenceCoverageError("unsupported matrix render format")


def _iter_claims(claims: object) -> Iterable[tuple[object | None, object]]:
    if claims is None:
        return
    if isinstance(claims, Mapping):
        if _first_field(claims, _CLAIM_ID_KEYS) is not None:
            yield None, claims
            return
        for claim_id, claim in claims.items():
            yield claim_id, claim
        return
    if isinstance(claims, (str, bytes)):
        raise EvidenceCoverageError("claims must be a collection")
    if isinstance(claims, Iterable):
        for claim in claims:
            yield None, claim
        return
    raise EvidenceCoverageError("claims must be a collection")


def _claim_id(claim: object, hinted_claim_id: object | None) -> str:
    raw_claim_id = hinted_claim_id
    if raw_claim_id is None:
        raw_claim_id = _first_field(claim, _CLAIM_ID_KEYS)
    if raw_claim_id is None:
        raise EvidenceCoverageError("claim requires an opaque claim_id")
    return _identifier(raw_claim_id, "claim_id")


def _iter_claim_requirements(claim: object) -> Iterable[object]:
    if isinstance(claim, EvidenceCoverageRecord):
        yield claim
        return

    if isinstance(claim, (str, bytes)):
        raise EvidenceCoverageError("claim has no required evidence classes")
    if isinstance(claim, Iterable) and not isinstance(claim, Mapping):
        yield from _iter_requirement_items(claim)
        return

    direct_container = _first_field(claim, ("required_evidence", "requirements"))
    if direct_container is not None:
        yield from _iter_requirement_items(direct_container)
        return

    required_classes = _field_value(claim, "required_evidence_classes")
    if required_classes is not None:
        explicit_evidence = _field_value(claim, "evidence")
        explicit_items = (
            list(_iter_requirement_items(explicit_evidence))
            if explicit_evidence is not None
            else []
        )
        explicit_classes = {
            _identifier(_requirement_class(item), "evidence_class")
            for item in explicit_items
            if _requirement_class(item) is not None
        }
        yield from explicit_items
        for required_item in _iter_requirement_items(required_classes):
            evidence_class = _requirement_class(required_item)
            if evidence_class is None:
                raise EvidenceCoverageError("required evidence needs an evidence_class")
            normalized_class = _identifier(evidence_class, "evidence_class")
            if normalized_class not in explicit_classes:
                yield _claim_level_requirement(claim, normalized_class, required_item)
        return

    evidence = _field_value(claim, "evidence")
    if evidence is not None:
        yield from _iter_requirement_items(evidence)
        return

    if _first_field(claim, _EVIDENCE_CLASS_KEYS) is not None:
        yield claim
        return

    if isinstance(claim, Mapping):
        claim_fields = {
            *_CLAIM_ID_KEYS,
            "source_fingerprints",
            "source_fingerprints_by_class",
            "source_hashes_by_class",
            "review_states",
            "review_state_by_class",
            "statuses",
            "coverage_statuses",
            "status_by_class",
        }
        for evidence_class, requirement in claim.items():
            if evidence_class in claim_fields:
                continue
            if isinstance(requirement, Mapping):
                item = dict(requirement)
                item.setdefault("evidence_class", evidence_class)
            elif requirement is None:
                item = {"evidence_class": evidence_class}
            else:
                item = {"evidence_class": evidence_class, "status": requirement}
            yield item
        return

    raise EvidenceCoverageError("claim has no required evidence classes")


def _claim_level_requirement(
    claim: object,
    evidence_class: str,
    requirement: object,
) -> dict[str, Any]:
    if isinstance(requirement, Mapping):
        item: dict[str, Any] = dict(requirement)
    else:
        item = {"evidence_class": evidence_class}
    item["evidence_class"] = evidence_class

    status = _mapped_claim_value(claim, _STATUS_MAP_KEYS, evidence_class)
    review_state = _mapped_claim_value(
        claim,
        _REVIEW_STATE_MAP_KEYS,
        evidence_class,
    )
    fingerprints = _mapped_claim_value(
        claim,
        _FINGERPRINT_MAP_KEYS,
        evidence_class,
    )
    if status is not None:
        item.setdefault("status", status)
    if review_state is not None:
        item.setdefault("review_state", review_state)
    if fingerprints is not None:
        item.setdefault("source_fingerprints", fingerprints)
    return item


def _requirement_class(requirement: object) -> object | None:
    evidence_class = _first_field(requirement, _EVIDENCE_CLASS_KEYS)
    if evidence_class is not None:
        return evidence_class
    if isinstance(requirement, bytes):
        try:
            return requirement.decode()
        except UnicodeDecodeError as exc:
            raise EvidenceCoverageError("evidence_class must be an identifier") from exc
    if isinstance(requirement, str):
        return requirement
    return None


def _mapped_claim_value(
    claim: object,
    field_names: Sequence[str],
    evidence_class: str,
) -> object | None:
    for field_name in field_names:
        value = _field_value(claim, field_name)
        if isinstance(value, Mapping) and evidence_class in value:
            return value[evidence_class]
    source_fingerprints = _field_value(claim, "source_fingerprints")
    if (
        field_names == _FINGERPRINT_MAP_KEYS
        and source_fingerprints is not None
        and not isinstance(source_fingerprints, Mapping)
    ):
        return source_fingerprints
    return None


def _iter_requirement_items(container: object) -> Iterable[object]:
    if isinstance(container, Mapping):
        if _first_field(container, _EVIDENCE_CLASS_KEYS) is not None:
            yield container
            return
        for evidence_class, requirement in container.items():
            if isinstance(requirement, Mapping):
                item = dict(requirement)
                item.setdefault("evidence_class", evidence_class)
            elif requirement is None:
                item = {"evidence_class": evidence_class}
            else:
                item = {"evidence_class": evidence_class, "status": requirement}
            yield item
        return
    if isinstance(container, (str, bytes)):
        yield {"evidence_class": container}
        return
    if isinstance(container, Iterable):
        yield from container
        return
    raise EvidenceCoverageError("required evidence must be a collection")


def _record_from_requirement(
    claim_id: object,
    hinted_evidence_class: object | None,
    requirement: object,
) -> EvidenceCoverageRecord:
    if isinstance(requirement, EvidenceCoverageRecord):
        if _identifier(claim_id, "claim_id") != requirement.claim_id:
            return EvidenceCoverageRecord(
                claim_id=_identifier(claim_id, "claim_id"),
                evidence_class=requirement.evidence_class,
                status=requirement.status,
                review_state=requirement.review_state,
                source_fingerprints=requirement.source_fingerprints,
            )
        return requirement

    raw_evidence_class = hinted_evidence_class
    if raw_evidence_class is None:
        raw_evidence_class = _first_field(requirement, _EVIDENCE_CLASS_KEYS)
    if raw_evidence_class is None and isinstance(requirement, (str, bytes)):
        raw_evidence_class = _requirement_class(requirement)
    if raw_evidence_class is None:
        raise EvidenceCoverageError("required evidence needs an evidence_class")

    raw_status = _first_field(requirement, _STATUS_KEYS)
    raw_review_state = _first_field(requirement, _REVIEW_STATE_KEYS)
    fingerprints = _requirement_fingerprints(requirement)
    if raw_status is None:
        if raw_review_state is None:
            status: CoverageStatus = "present" if fingerprints else "missing"
        else:
            review_state = _review_state(raw_review_state)
            if review_state == "reviewed" and not fingerprints:
                raise EvidenceCoverageError("reviewed evidence requires a fingerprint")
            status = _status_for_review(review_state, bool(fingerprints))
    else:
        status = _status(raw_status)

    review_state = (
        _review_state(raw_review_state)
        if raw_review_state is not None
        else _review_for_status(status)
    )
    return EvidenceCoverageRecord(
        claim_id=_identifier(claim_id, "claim_id"),
        evidence_class=_identifier(raw_evidence_class, "evidence_class"),
        status=status,
        review_state=review_state,
        source_fingerprints=fingerprints,
    )


def _requirement_fingerprints(requirement: object) -> tuple[str, ...]:
    value = _first_field(requirement, _FINGERPRINT_KEYS)
    source = _field_value(requirement, "source")
    fingerprints = list(_fingerprints(value)) if value is not None else []
    if source is not None:
        fingerprints.append(fingerprint_source(source))
    return tuple(sorted(set(fingerprints)))


def _merge_records(
    first: EvidenceCoverageRecord,
    second: EvidenceCoverageRecord,
) -> EvidenceCoverageRecord:
    if first.status == second.status:
        status = first.status
    else:
        status = "conflicting"
    fingerprints = tuple(
        sorted(set(first.source_fingerprints).union(second.source_fingerprints))
    )
    return EvidenceCoverageRecord(
        claim_id=first.claim_id,
        evidence_class=first.evidence_class,
        status=status,
        source_fingerprints=fingerprints,
    )


def _first_field(source: object, field_names: Sequence[str]) -> object | None:
    for field_name in field_names:
        value = _field_value(source, field_name)
        if value is not None:
            return value
    return None


def _field_value(source: object, field_name: str) -> object | None:
    if isinstance(source, Mapping):
        return source.get(field_name)
    return getattr(source, field_name, None)


def _identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise EvidenceCoverageError(f"{field_name} must be an opaque identifier")
    normalized = value.strip()
    if not _IDENTIFIER_RE.fullmatch(normalized):
        raise EvidenceCoverageError(f"{field_name} must be an opaque identifier")
    return normalized


def _status(value: object) -> CoverageStatus:
    if not isinstance(value, str):
        raise EvidenceCoverageError("coverage status is invalid")
    normalized = _STATUS_ALIASES.get(value.strip().casefold())
    if normalized not in COVERAGE_STATUSES:
        raise EvidenceCoverageError("coverage status is invalid")
    return normalized  # type: ignore[return-value]


def _review_state(value: object) -> ReviewState:
    if not isinstance(value, str):
        raise EvidenceCoverageError("review state is invalid")
    normalized = _REVIEW_ALIASES.get(value.strip().casefold())
    if normalized not in REVIEW_STATES:
        raise EvidenceCoverageError("review state is invalid")
    return normalized  # type: ignore[return-value]


def _status_for_review(
    review_state: ReviewState, has_fingerprints: bool
) -> CoverageStatus:
    if review_state == "reviewed":
        return "present" if has_fingerprints else "missing"
    if review_state == "missing":
        return "missing"
    if review_state == "conflicting":
        return "conflicting"
    return "unreviewed"


def _review_for_status(status: CoverageStatus) -> ReviewState:
    if status == "present":
        return "reviewed"
    return status  # type: ignore[return-value]


def _fingerprints(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        values: Iterable[object] = (value,)
    elif isinstance(value, Mapping):
        values = (value,)
    elif isinstance(value, Iterable):
        values = value
    else:
        values = (value,)

    normalized: set[str] = set()
    for item in values:
        if item is None:
            continue
        if isinstance(item, str) and _DIGEST_RE.fullmatch(item.strip().casefold()):
            normalized.add(item.strip().casefold())
        else:
            normalized.add(fingerprint_source(item))
    return tuple(sorted(normalized))


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _digest(value: object) -> str:
    try:
        return _sha256(_canonical_json(value).encode("utf-8"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise EvidenceCoverageError("value-free hash material is invalid") from exc


# Descriptive aliases keep the public surface easy to discover for callers
# that refer to a matrix cell or a source fingerprint rather than a record.
EvidenceCoverageCell = EvidenceCoverageRecord
EvidenceCoverageEntry = EvidenceCoverageRecord
EvidenceCoverage = EvidenceCoverageRecord
build_evidence_coverage = build_evidence_coverage_matrix
render_evidence_coverage = render_evidence_coverage_matrix


__all__ = [
    "COVERAGE_STATUSES",
    "CoverageStatus",
    "EVIDENCE_COVERAGE_NOTE",
    "EVIDENCE_COVERAGE_SCHEMA_VERSION",
    "EvidenceCoverage",
    "EvidenceCoverageCell",
    "EvidenceCoverageEntry",
    "EvidenceCoverageError",
    "EvidenceCoverageMatrix",
    "EvidenceCoverageRecord",
    "REVIEW_STATES",
    "ReviewState",
    "build_evidence_coverage",
    "build_evidence_coverage_matrix",
    "fingerprint_source",
    "render_evidence_coverage",
    "render_evidence_coverage_matrix",
]
