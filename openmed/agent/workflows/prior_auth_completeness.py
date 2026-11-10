"""Deterministic, value-free prior-authorization completeness scoring.

The scorer consumes only developer-authored requirement identifiers, schema
versions, digests, citation presence, and contradiction flags. It never reads
clinical content, makes a coverage decision, or generates a clinical claim.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

PRIOR_AUTH_COMPLETENESS_SCHEMA: Final = (
    "openmed.agent.workflows.prior_auth_completeness.v1"
)

_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
_IDENTIFIER_RE = re.compile(r"[a-z][a-z0-9_]{0,63}(?:\.[a-z][a-z0-9_]{0,63}){0,7}")
_MAX_ITEMS: Final = 10_000
_MAX_CITATIONS_PER_ITEM: Final = 1_024
_MAX_SCHEMA_VERSION: Final = (1 << 31) - 1


class MissingEvidenceCode(str, Enum):
    """Closed reasons that a schema requirement is incomplete."""

    REQUIRED_EVIDENCE_MISSING = "required_evidence_missing"
    REQUIRED_CITATION_MISSING = "required_citation_missing"


class ReviewerActionCode(str, Enum):
    """Closed actions that the completeness report can request."""

    PROVIDE_REQUIRED_EVIDENCE = "provide_required_evidence"
    ADD_REQUIRED_CITATION = "add_required_citation"
    REVIEW_CONTRADICTION = "review_contradiction"
    REVIEW_UNSUPPORTED_STATEMENT = "review_unsupported_statement"


class PriorAuthCompletenessError(ValueError):
    """A PHI-safe validation error.

    Args:
        code: Stable machine-readable error code.
        field_name: Optional public contract field associated with the error.
    """

    def __init__(self, code: str, field_name: str | None = None) -> None:
        self.code = code
        self.field_name = field_name
        message = code if field_name is None else f"{field_name}: {code}"
        super().__init__(message)


@dataclass(frozen=True, slots=True, order=True)
class PriorAuthRequirement:
    """One requirement in a versioned completeness schema.

    ``requirement_id`` must be a developer-authored identifier and must never
    contain a patient, member, provider, or clinical value.
    """

    requirement_id: str
    minimum_evidence_items: int = 1
    citation_required: bool = True

    def __post_init__(self) -> None:
        _validate_identifier(self.requirement_id, "requirement_id")
        if (
            type(self.minimum_evidence_items) is not int
            or not 1 <= self.minimum_evidence_items <= _MAX_ITEMS
        ):
            raise PriorAuthCompletenessError(
                "invalid_minimum_evidence_items", "minimum_evidence_items"
            )
        if type(self.citation_required) is not bool:
            raise PriorAuthCompletenessError(
                "invalid_citation_required", "citation_required"
            )

    def to_dict(self) -> dict[str, str | int | bool]:
        """Return deterministic, content-free requirement metadata."""

        return {
            "citation_required": self.citation_required,
            "minimum_evidence_items": self.minimum_evidence_items,
            "requirement_id": self.requirement_id,
        }


@dataclass(frozen=True, slots=True, init=False)
class PriorAuthRequirementSchema:
    """A named, versioned set of prior-authorization requirements."""

    schema_id: str
    version: int
    requirements: tuple[PriorAuthRequirement, ...]

    def __init__(
        self,
        schema_id: str,
        version: int,
        requirements: Iterable[PriorAuthRequirement],
    ) -> None:
        object.__setattr__(
            self, "schema_id", _validate_identifier(schema_id, "schema_id")
        )
        object.__setattr__(self, "version", _validate_version(version))
        values = _bounded_tuple(
            requirements,
            field_name="requirements",
            maximum=_MAX_ITEMS,
        )
        if not values:
            raise PriorAuthCompletenessError("empty_collection", "requirements")
        if any(type(value) is not PriorAuthRequirement for value in values):
            raise PriorAuthCompletenessError("invalid_requirement", "requirements")
        normalized = tuple(sorted(values, key=lambda value: value.requirement_id))
        identifiers = tuple(value.requirement_id for value in normalized)
        if len(set(identifiers)) != len(identifiers):
            raise PriorAuthCompletenessError("duplicate_requirement", "requirements")
        object.__setattr__(self, "requirements", normalized)

    @property
    def schema_digest(self) -> str:
        """Return a stable digest binding the schema name, version, and rules."""

        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic version and requirement metadata."""

        return {
            "requirements": [value.to_dict() for value in self.requirements],
            "schema_id": self.schema_id,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True, init=False)
class PacketEvidence:
    """Content-free evidence metadata for one packet statement.

    The evidence and citation digests bind content retained inside the caller's
    trusted boundary. ``contradiction_flag`` is an input from a separate local
    contradiction check; this module does not infer a clinical contradiction.
    """

    requirement_id: str
    evidence_digest: str
    citation_digests: tuple[str, ...] = field(repr=False)
    contradiction_flag: bool

    def __init__(
        self,
        requirement_id: str,
        evidence_digest: str,
        citation_digests: Iterable[str] = (),
        contradiction_flag: bool = False,
    ) -> None:
        object.__setattr__(
            self,
            "requirement_id",
            _validate_identifier(requirement_id, "requirement_id"),
        )
        object.__setattr__(
            self,
            "evidence_digest",
            _validate_digest(evidence_digest, "evidence_digest"),
        )
        citations = _bounded_tuple(
            citation_digests,
            field_name="citation_digests",
            maximum=_MAX_CITATIONS_PER_ITEM,
        )
        normalized = tuple(
            sorted(_validate_digest(value, "citation_digests") for value in citations)
        )
        if len(set(normalized)) != len(normalized):
            raise PriorAuthCompletenessError("duplicate_citation", "citation_digests")
        object.__setattr__(self, "citation_digests", normalized)
        if type(contradiction_flag) is not bool:
            raise PriorAuthCompletenessError(
                "invalid_contradiction_flag", "contradiction_flag"
            )
        object.__setattr__(self, "contradiction_flag", contradiction_flag)

    def to_dict(self) -> dict[str, Any]:
        """Return evidence metadata without statement or citation content."""

        return {
            "citation_digests": list(self.citation_digests),
            "contradiction_flag": self.contradiction_flag,
            "evidence_digest": self.evidence_digest,
            "requirement_id": self.requirement_id,
        }


@dataclass(frozen=True, slots=True, init=False)
class PriorAuthorizationPacket:
    """Digest-addressed packet metadata supplied to the local scorer."""

    packet_digest: str
    requirement_schema_id: str
    requirement_schema_version: int
    evidence: tuple[PacketEvidence, ...] = field(repr=False)

    def __init__(
        self,
        packet_digest: str,
        requirement_schema_id: str,
        requirement_schema_version: int,
        evidence: Iterable[PacketEvidence],
    ) -> None:
        object.__setattr__(
            self,
            "packet_digest",
            _validate_digest(packet_digest, "packet_digest"),
        )
        object.__setattr__(
            self,
            "requirement_schema_id",
            _validate_identifier(requirement_schema_id, "requirement_schema_id"),
        )
        object.__setattr__(
            self,
            "requirement_schema_version",
            _validate_version(
                requirement_schema_version,
                field_name="requirement_schema_version",
            ),
        )
        values = _bounded_tuple(evidence, field_name="evidence", maximum=_MAX_ITEMS)
        if any(type(value) is not PacketEvidence for value in values):
            raise PriorAuthCompletenessError("invalid_evidence", "evidence")
        normalized = tuple(
            sorted(
                values, key=lambda value: (value.requirement_id, value.evidence_digest)
            )
        )
        evidence_digests = tuple(value.evidence_digest for value in normalized)
        if len(set(evidence_digests)) != len(evidence_digests):
            raise PriorAuthCompletenessError("duplicate_evidence", "evidence")
        object.__setattr__(self, "evidence", normalized)

    @property
    def evidence_metadata_digest(self) -> str:
        """Return a digest of the ordered, content-free evidence metadata."""

        return _digest([value.to_dict() for value in self.evidence])


@dataclass(frozen=True, slots=True, order=True)
class MissingEvidenceFinding:
    """A missing-evidence code associated with a schema requirement."""

    code: MissingEvidenceCode
    requirement_id: str

    def __post_init__(self) -> None:
        if type(self.code) is not MissingEvidenceCode:
            raise PriorAuthCompletenessError("invalid_missing_evidence_code", "code")
        _validate_identifier(self.requirement_id, "requirement_id")

    def to_dict(self) -> dict[str, str]:
        """Return stable finding metadata."""

        return {"code": self.code.value, "requirement_id": self.requirement_id}


@dataclass(frozen=True, slots=True, order=True)
class ReviewerAction:
    """A bounded reviewer action associated with a requirement or statement."""

    code: ReviewerActionCode
    requirement_id: str

    def __post_init__(self) -> None:
        if type(self.code) is not ReviewerActionCode:
            raise PriorAuthCompletenessError("invalid_reviewer_action_code", "code")
        _validate_identifier(self.requirement_id, "requirement_id")

    def to_dict(self) -> dict[str, str]:
        """Return stable reviewer-action metadata."""

        return {"code": self.code.value, "requirement_id": self.requirement_id}


@dataclass(frozen=True, slots=True)
class PriorAuthCompletenessReport:
    """Deterministic scoring output containing no raw packet values."""

    packet_digest: str
    requirement_schema_digest: str
    evidence_metadata_digest: str
    report_digest: str
    missing_evidence: tuple[MissingEvidenceFinding, ...]
    reviewer_actions: tuple[ReviewerAction, ...]
    complete_requirement_count: int
    required_requirement_count: int
    completeness_score: float
    schema: str = PRIOR_AUTH_COMPLETENESS_SCHEMA

    def __post_init__(self) -> None:
        for field_name in (
            "packet_digest",
            "requirement_schema_digest",
            "evidence_metadata_digest",
            "report_digest",
        ):
            _validate_digest(getattr(self, field_name), field_name)
        if type(self.missing_evidence) is not tuple or any(
            type(value) is not MissingEvidenceFinding for value in self.missing_evidence
        ):
            raise PriorAuthCompletenessError(
                "invalid_missing_evidence", "missing_evidence"
            )
        if self.missing_evidence != tuple(sorted(self.missing_evidence)):
            raise PriorAuthCompletenessError(
                "missing_evidence_not_sorted", "missing_evidence"
            )
        if type(self.reviewer_actions) is not tuple or any(
            type(value) is not ReviewerAction for value in self.reviewer_actions
        ):
            raise PriorAuthCompletenessError(
                "invalid_reviewer_actions", "reviewer_actions"
            )
        if self.reviewer_actions != tuple(sorted(self.reviewer_actions)):
            raise PriorAuthCompletenessError(
                "reviewer_actions_not_sorted", "reviewer_actions"
            )
        _validate_count(self.complete_requirement_count, "complete_requirement_count")
        _validate_count(self.required_requirement_count, "required_requirement_count")
        if self.required_requirement_count == 0:
            raise PriorAuthCompletenessError(
                "empty_required_requirements", "required_requirement_count"
            )
        if self.complete_requirement_count > self.required_requirement_count:
            raise PriorAuthCompletenessError(
                "count_out_of_range", "complete_requirement_count"
            )
        if type(self.completeness_score) is not float or not (
            0.0 <= self.completeness_score <= 1.0
        ):
            raise PriorAuthCompletenessError(
                "invalid_completeness_score", "completeness_score"
            )
        expected_score = _score(
            self.complete_requirement_count,
            self.required_requirement_count,
        )
        if self.completeness_score != expected_score:
            raise PriorAuthCompletenessError(
                "inconsistent_completeness_score", "completeness_score"
            )
        if self.schema != PRIOR_AUTH_COMPLETENESS_SCHEMA:
            raise PriorAuthCompletenessError("invalid_schema", "schema")

    @property
    def has_missing_evidence(self) -> bool:
        """Return whether the report contains any completeness gap."""

        return bool(self.missing_evidence)

    @property
    def requires_reviewer_action(self) -> bool:
        """Return whether a bounded reviewer action was requested."""

        return bool(self.reviewer_actions)

    def to_dict(self) -> dict[str, Any]:
        """Return only identifiers, counts, scores, digests, and closed codes."""

        return {
            "complete_requirement_count": self.complete_requirement_count,
            "completeness_score": self.completeness_score,
            "evidence_metadata_digest": self.evidence_metadata_digest,
            "has_missing_evidence": self.has_missing_evidence,
            "missing_evidence": [value.to_dict() for value in self.missing_evidence],
            "packet_digest": self.packet_digest,
            "report_digest": self.report_digest,
            "required_requirement_count": self.required_requirement_count,
            "requirement_schema_digest": self.requirement_schema_digest,
            "requires_reviewer_action": self.requires_reviewer_action,
            "reviewer_actions": [value.to_dict() for value in self.reviewer_actions],
            "schema": self.schema,
        }

    def to_json(self) -> str:
        """Serialize the report with stable key ordering."""

        return _canonical_json(self.to_dict())


def score_prior_authorization_packet(
    packet: PriorAuthorizationPacket,
    requirement_schema: PriorAuthRequirementSchema,
) -> PriorAuthCompletenessReport:
    """Score packet evidence completeness against an exact schema version.

    The score is the fraction of requirements with enough evidence items and,
    when required, citations on every submitted item. Contradictions and items
    outside the requirement schema request review but do not alter that purely
    structural completeness score. This function never decides coverage.

    Args:
        packet: Digest-addressed packet evidence metadata.
        requirement_schema: The exact local requirement schema to apply.

    Returns:
        A deterministic, value-free completeness report.

    Raises:
        PriorAuthCompletenessError: If input types or schema versions mismatch.
    """

    if type(packet) is not PriorAuthorizationPacket:
        raise PriorAuthCompletenessError("invalid_packet", "packet")
    if type(requirement_schema) is not PriorAuthRequirementSchema:
        raise PriorAuthCompletenessError(
            "invalid_requirement_schema", "requirement_schema"
        )
    if packet.requirement_schema_id != requirement_schema.schema_id:
        raise PriorAuthCompletenessError("schema_id_mismatch", "requirement_schema_id")
    if packet.requirement_schema_version != requirement_schema.version:
        raise PriorAuthCompletenessError(
            "schema_version_mismatch", "requirement_schema_version"
        )

    evidence_by_requirement: dict[str, list[PacketEvidence]] = {}
    for evidence in packet.evidence:
        evidence_by_requirement.setdefault(evidence.requirement_id, []).append(evidence)

    requirement_ids = {
        requirement.requirement_id for requirement in requirement_schema.requirements
    }
    missing: list[MissingEvidenceFinding] = []
    actions: set[ReviewerAction] = set()
    complete_count = 0

    for requirement in requirement_schema.requirements:
        items = evidence_by_requirement.get(requirement.requirement_id, [])
        enough_evidence = len(items) >= requirement.minimum_evidence_items
        citations_complete = not requirement.citation_required or (
            enough_evidence and all(item.citation_digests for item in items)
        )

        if not enough_evidence:
            missing.append(
                MissingEvidenceFinding(
                    MissingEvidenceCode.REQUIRED_EVIDENCE_MISSING,
                    requirement.requirement_id,
                )
            )
            actions.add(
                ReviewerAction(
                    ReviewerActionCode.PROVIDE_REQUIRED_EVIDENCE,
                    requirement.requirement_id,
                )
            )
        if enough_evidence and not citations_complete:
            missing.append(
                MissingEvidenceFinding(
                    MissingEvidenceCode.REQUIRED_CITATION_MISSING,
                    requirement.requirement_id,
                )
            )
            actions.add(
                ReviewerAction(
                    ReviewerActionCode.ADD_REQUIRED_CITATION,
                    requirement.requirement_id,
                )
            )
        if enough_evidence and citations_complete:
            complete_count += 1

        if any(item.contradiction_flag for item in items):
            actions.add(
                ReviewerAction(
                    ReviewerActionCode.REVIEW_CONTRADICTION,
                    requirement.requirement_id,
                )
            )

    for requirement_id, items in evidence_by_requirement.items():
        if requirement_id not in requirement_ids:
            actions.add(
                ReviewerAction(
                    ReviewerActionCode.REVIEW_UNSUPPORTED_STATEMENT,
                    requirement_id,
                )
            )
            if any(item.contradiction_flag for item in items):
                actions.add(
                    ReviewerAction(
                        ReviewerActionCode.REVIEW_CONTRADICTION,
                        requirement_id,
                    )
                )

    normalized_missing = tuple(sorted(missing))
    normalized_actions = tuple(sorted(actions))
    required_count = len(requirement_schema.requirements)
    completeness_score = _score(complete_count, required_count)
    report_fields = {
        "complete_requirement_count": complete_count,
        "completeness_score": completeness_score,
        "evidence_metadata_digest": packet.evidence_metadata_digest,
        "missing_evidence": [value.to_dict() for value in normalized_missing],
        "packet_digest": packet.packet_digest,
        "required_requirement_count": required_count,
        "requirement_schema_digest": requirement_schema.schema_digest,
        "reviewer_actions": [value.to_dict() for value in normalized_actions],
        "schema": PRIOR_AUTH_COMPLETENESS_SCHEMA,
    }
    return PriorAuthCompletenessReport(
        packet_digest=packet.packet_digest,
        requirement_schema_digest=requirement_schema.schema_digest,
        evidence_metadata_digest=packet.evidence_metadata_digest,
        report_digest=_digest(report_fields),
        missing_evidence=normalized_missing,
        reviewer_actions=normalized_actions,
        complete_requirement_count=complete_count,
        required_requirement_count=required_count,
        completeness_score=completeness_score,
    )


def _validate_identifier(value: Any, field_name: str) -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise PriorAuthCompletenessError("invalid_identifier", field_name)
    return value


def _validate_digest(value: Any, field_name: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise PriorAuthCompletenessError("invalid_digest", field_name)
    return value


def _validate_version(value: Any, field_name: str = "version") -> int:
    if type(value) is not int or not 1 <= value <= _MAX_SCHEMA_VERSION:
        raise PriorAuthCompletenessError("invalid_version", field_name)
    return value


def _validate_count(value: Any, field_name: str) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_ITEMS:
        raise PriorAuthCompletenessError("invalid_count", field_name)
    return value


def _bounded_tuple(value: Any, *, field_name: str, maximum: int) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise PriorAuthCompletenessError("invalid_collection", field_name)
    try:
        iterator = iter(value)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise PriorAuthCompletenessError("invalid_collection", field_name) from None

    items: list[Any] = []
    try:
        for item in iterator:
            if len(items) >= maximum:
                raise PriorAuthCompletenessError("too_many_items", field_name)
            items.append(item)
    except (KeyboardInterrupt, SystemExit, PriorAuthCompletenessError):
        raise
    except Exception:
        raise PriorAuthCompletenessError("invalid_collection", field_name) from None
    return tuple(items)


def _score(complete: int, required: int) -> float:
    return round(complete / required, 6)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value: Any) -> str:
    payload = _canonical_json(value).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


__all__ = [
    "PRIOR_AUTH_COMPLETENESS_SCHEMA",
    "MissingEvidenceCode",
    "MissingEvidenceFinding",
    "PacketEvidence",
    "PriorAuthCompletenessError",
    "PriorAuthCompletenessReport",
    "PriorAuthRequirement",
    "PriorAuthRequirementSchema",
    "PriorAuthorizationPacket",
    "ReviewerAction",
    "ReviewerActionCode",
    "score_prior_authorization_packet",
]
