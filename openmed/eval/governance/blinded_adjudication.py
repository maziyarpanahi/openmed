"""Deterministic, identity-blinded packets for clinician adjudication.

Packet rendering is local-only. Public packets contain reviewer-facing evidence,
candidate text, a narrow rubric, and digest-bound conflict screening metadata.
Candidate identities and submission-manifest digests remain in a separately
stored, HMAC-sealed mapping for authorized post-adjudication audit.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

ADJUDICATION_PACKET_SCHEMA_VERSION = "openmed.eval.blinded_adjudication.packet.v1"
ADJUDICATION_MAPPING_SCHEMA_VERSION = "openmed.eval.blinded_adjudication.mapping.v1"
ADJUDICATION_BALANCE_SCHEMA_VERSION = "openmed.eval.blinded_adjudication.balance.v1"
ADJUDICATION_AUDIT_SCHEMA_VERSION = "openmed.eval.blinded_adjudication.audit.v1"

CONFLICT_SCREENING_CLEARED = "cleared"
CONFLICT_SCREENING_RECUSED = "recused"

AUDIT_VALID = "valid"
AUDIT_INVALID_PACKET_SET = "invalid_packet_set"
AUDIT_COMMITMENT_MISMATCH = "commitment_mismatch"
AUDIT_MAPPING_MISMATCH = "mapping_mismatch"

_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")
_PUBLIC_REF_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_ALIAS_RE = re.compile(r"candidate-[a-z]")
_CONFLICT_STATUSES = frozenset({CONFLICT_SCREENING_CLEARED, CONFLICT_SCREENING_RECUSED})
_MAX_CANDIDATES = 26
_MAX_CASES = 100_000
_MAX_CRITERIA = 8
_MAX_TEXT_LENGTH = 1_000_000


class BlindedAdjudicationError(ValueError):
    """Raised when a blinded packet set cannot be rendered safely."""


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _is_sha256(value: Any) -> bool:
    return type(value) is str and _SHA256_RE.fullmatch(value) is not None


def _is_public_ref(value: Any) -> bool:
    return type(value) is str and _PUBLIC_REF_RE.fullmatch(value) is not None


def _require_public_ref(value: Any, field: str) -> str:
    if not _is_public_ref(value):
        raise BlindedAdjudicationError(f"{field}: invalid_reference")
    return value


def _require_digest(value: Any, field: str) -> str:
    if not _is_sha256(value):
        raise BlindedAdjudicationError(f"{field}: invalid_digest")
    return value


def _require_text(value: Any, field: str, maximum: int = _MAX_TEXT_LENGTH) -> str:
    if type(value) is not str or not value.strip() or len(value) > maximum:
        raise BlindedAdjudicationError(f"{field}: invalid_text")
    return value


def _require_key(randomization_key: Any) -> bytes:
    if type(randomization_key) is not bytes or len(randomization_key) < 32:
        raise BlindedAdjudicationError("randomization_key: invalid_key")
    return randomization_key


def _digest_text(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _keyed_digest(key: bytes, domain: bytes, value: str) -> bytes:
    return hmac.new(
        key, domain + b"\0" + value.encode("utf-8"), hashlib.sha256
    ).digest()


def _alias(slot: int) -> str:
    return f"candidate-{chr(ord('a') + slot)}"


@dataclass(frozen=True, slots=True)
class RubricCriterion:
    """One bounded reviewer question with an integer response scale."""

    criterion_ref: str
    question: str
    minimum_score: int
    maximum_score: int

    def __post_init__(self) -> None:
        _require_public_ref(self.criterion_ref, "criterion_ref")
        _require_text(self.question, "question", maximum=500)
        if (
            type(self.minimum_score) is not int
            or type(self.maximum_score) is not int
            or self.minimum_score < 0
            or self.maximum_score > 10
            or self.minimum_score >= self.maximum_score
            or self.maximum_score - self.minimum_score > 4
        ):
            raise BlindedAdjudicationError("score_range: invalid_range")

    def to_dict(self) -> dict[str, Any]:
        """Return the reviewer-facing criterion document."""
        return {
            "criterion_ref": self.criterion_ref,
            "maximum_score": self.maximum_score,
            "minimum_score": self.minimum_score,
            "question": self.question,
        }


@dataclass(frozen=True, slots=True)
class SourceEvidence:
    """Reviewer-visible evidence shared equally by all candidate outputs."""

    evidence_ref: str
    content: str

    def __post_init__(self) -> None:
        _require_public_ref(self.evidence_ref, "evidence_ref")
        _require_text(self.content, "content")

    def to_dict(self) -> dict[str, str]:
        """Return the reviewer-facing evidence document."""
        return {"content": self.content, "evidence_ref": self.evidence_ref}


@dataclass(frozen=True, slots=True)
class ComparisonCase:
    """Private input containing one case and identity-keyed candidate outputs."""

    case_ref: str
    source_evidence: Sequence[SourceEvidence]
    candidate_outputs: Mapping[str, str]

    def __post_init__(self) -> None:
        _require_public_ref(self.case_ref, "case_ref")
        evidence = tuple(self.source_evidence)
        if not evidence or not all(
            isinstance(item, SourceEvidence) for item in evidence
        ):
            raise BlindedAdjudicationError("source_evidence: invalid_sequence")
        evidence_refs = tuple(item.evidence_ref for item in evidence)
        if len(set(evidence_refs)) != len(evidence_refs):
            raise BlindedAdjudicationError("source_evidence: duplicate_reference")
        if not isinstance(self.candidate_outputs, Mapping):
            raise BlindedAdjudicationError("candidate_outputs: invalid_mapping")
        outputs = dict(self.candidate_outputs)
        if not 2 <= len(outputs) <= _MAX_CANDIDATES:
            raise BlindedAdjudicationError("candidate_outputs: invalid_count")
        for candidate_identity, content in outputs.items():
            _require_public_ref(candidate_identity, "candidate_identity")
            _require_text(content, "candidate_output")
        object.__setattr__(self, "source_evidence", evidence)
        object.__setattr__(self, "candidate_outputs", MappingProxyType(outputs))


@dataclass(frozen=True, slots=True)
class ConflictOfInterestMetadata:
    """Digest-bound reviewer conflict screening metadata.

    The declaration itself stays with the evaluator. Only an opaque reviewer
    reference, its digest, and a closed screening status enter public packets.
    """

    reviewer_ref: str
    declaration_digest: str
    status: str = CONFLICT_SCREENING_CLEARED

    def __post_init__(self) -> None:
        _require_public_ref(self.reviewer_ref, "reviewer_ref")
        _require_digest(self.declaration_digest, "declaration_digest")
        if self.status not in _CONFLICT_STATUSES:
            raise BlindedAdjudicationError("conflict_status: invalid_status")

    def to_dict(self) -> dict[str, str]:
        """Return public conflict-screening metadata without declaration text."""
        return {
            "declaration_digest": self.declaration_digest,
            "reviewer_ref": self.reviewer_ref,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class BlindedCandidate:
    """A reviewer-visible candidate identified only by a positional alias."""

    alias: str
    content: str

    def __post_init__(self) -> None:
        if type(self.alias) is not str or _ALIAS_RE.fullmatch(self.alias) is None:
            raise BlindedAdjudicationError("candidate_alias: invalid_alias")
        _require_text(self.content, "candidate_content")

    def to_dict(self) -> dict[str, str]:
        """Return the blinded candidate document."""
        return {"alias": self.alias, "content": self.content}


@dataclass(frozen=True, slots=True)
class BlindedAdjudicationPacket:
    """A single identity-blinded, reviewer-facing comparison packet."""

    packet_set_ref: str
    case_ref: str
    holdout_commitment_digest: str
    mapping_commitment: str
    source_evidence: tuple[SourceEvidence, ...]
    candidates: tuple[BlindedCandidate, ...]
    rubric: tuple[RubricCriterion, ...]
    conflict_of_interest: ConflictOfInterestMetadata
    schema_version: str = ADJUDICATION_PACKET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ADJUDICATION_PACKET_SCHEMA_VERSION:
            raise BlindedAdjudicationError("schema_version: unsupported_version")
        _require_public_ref(self.packet_set_ref, "packet_set_ref")
        _require_public_ref(self.case_ref, "case_ref")
        _require_digest(self.holdout_commitment_digest, "holdout_commitment_digest")
        _require_digest(self.mapping_commitment, "mapping_commitment")
        if not self.source_evidence or not all(
            isinstance(item, SourceEvidence) for item in self.source_evidence
        ):
            raise BlindedAdjudicationError("source_evidence: invalid_sequence")
        if not 2 <= len(self.candidates) <= _MAX_CANDIDATES or not all(
            isinstance(item, BlindedCandidate) for item in self.candidates
        ):
            raise BlindedAdjudicationError("candidates: invalid_sequence")
        if tuple(item.alias for item in self.candidates) != tuple(
            _alias(index) for index in range(len(self.candidates))
        ):
            raise BlindedAdjudicationError("candidates: invalid_alias_order")
        if not 1 <= len(self.rubric) <= _MAX_CRITERIA or not all(
            isinstance(item, RubricCriterion) for item in self.rubric
        ):
            raise BlindedAdjudicationError("rubric: invalid_sequence")
        if not isinstance(self.conflict_of_interest, ConflictOfInterestMetadata):
            raise BlindedAdjudicationError("conflict_of_interest: invalid_metadata")

    def to_dict(self) -> dict[str, Any]:
        """Return the public packet without hidden candidate labels."""
        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "case_ref": self.case_ref,
            "conflict_of_interest": self.conflict_of_interest.to_dict(),
            "holdout_commitment_digest": self.holdout_commitment_digest,
            "mapping_commitment": self.mapping_commitment,
            "packet_set_ref": self.packet_set_ref,
            "rubric": [criterion.to_dict() for criterion in self.rubric],
            "schema_version": self.schema_version,
            "source_evidence": [
                evidence.to_dict() for evidence in self.source_evidence
            ],
        }

    def to_json(self) -> str:
        """Return stable compact JSON for the reviewer packet."""
        return _canonical_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class IdentityAssignment:
    """Private binding from a public alias to a submitted candidate."""

    alias: str
    candidate_identity: str
    submission_manifest_digest: str
    output_digest: str

    def __post_init__(self) -> None:
        if type(self.alias) is not str or _ALIAS_RE.fullmatch(self.alias) is None:
            raise BlindedAdjudicationError("assignment_alias: invalid_alias")
        _require_public_ref(self.candidate_identity, "candidate_identity")
        _require_digest(self.submission_manifest_digest, "submission_manifest_digest")
        _require_digest(self.output_digest, "output_digest")

    def to_private_dict(self) -> dict[str, str]:
        """Return the access-controlled identity binding for audit storage."""
        return {
            "alias": self.alias,
            "candidate_identity": self.candidate_identity,
            "output_digest": self.output_digest,
            "submission_manifest_digest": self.submission_manifest_digest,
        }


@dataclass(frozen=True, slots=True)
class IdentityMappingEntry:
    """Private identity assignments for one blinded packet."""

    case_ref: str
    assignments: tuple[IdentityAssignment, ...]

    def __post_init__(self) -> None:
        _require_public_ref(self.case_ref, "case_ref")
        assignments = tuple(self.assignments)
        if not 2 <= len(assignments) <= _MAX_CANDIDATES or not all(
            isinstance(assignment, IdentityAssignment) for assignment in assignments
        ):
            raise BlindedAdjudicationError("assignments: invalid_count")
        aliases = tuple(assignment.alias for assignment in assignments)
        if aliases != tuple(_alias(index) for index in range(len(assignments))):
            raise BlindedAdjudicationError("assignments: invalid_alias_order")
        identities = tuple(assignment.candidate_identity for assignment in assignments)
        if len(set(identities)) != len(identities):
            raise BlindedAdjudicationError("assignments: duplicate_identity")
        object.__setattr__(self, "assignments", assignments)

    def to_private_dict(self) -> dict[str, Any]:
        """Return this access-controlled mapping entry."""
        return {
            "assignments": [item.to_private_dict() for item in self.assignments],
            "case_ref": self.case_ref,
        }


def _mapping_payload(
    packet_set_ref: str,
    holdout_commitment_digest: str,
    entries: Sequence[IdentityMappingEntry],
) -> dict[str, Any]:
    return {
        "entries": [entry.to_private_dict() for entry in entries],
        "holdout_commitment_digest": holdout_commitment_digest,
        "packet_set_ref": packet_set_ref,
        "schema_version": ADJUDICATION_MAPPING_SCHEMA_VERSION,
    }


def _mapping_commitment(payload: Mapping[str, Any], key: bytes) -> str:
    digest = hmac.new(
        key,
        b"openmed.eval.blinded_adjudication.mapping.v1\0"
        + _canonical_json(payload).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"sha256:{digest}"


@dataclass(frozen=True, slots=True)
class SealedIdentityMapping:
    """Access-controlled identity mapping sealed by an evaluator-held key.

    ``to_private_dict`` intentionally names the disclosure boundary. Never
    attach that result to reviewer packets or ordinary benchmark reports.
    """

    packet_set_ref: str
    holdout_commitment_digest: str
    entries: tuple[IdentityMappingEntry, ...]
    mapping_commitment: str
    schema_version: str = ADJUDICATION_MAPPING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ADJUDICATION_MAPPING_SCHEMA_VERSION:
            raise BlindedAdjudicationError("schema_version: unsupported_version")
        _require_public_ref(self.packet_set_ref, "packet_set_ref")
        _require_digest(self.holdout_commitment_digest, "holdout_commitment_digest")
        _require_digest(self.mapping_commitment, "mapping_commitment")
        entries = tuple(self.entries)
        if (
            not entries
            or len(entries) > _MAX_CASES
            or not all(isinstance(entry, IdentityMappingEntry) for entry in entries)
        ):
            raise BlindedAdjudicationError("mapping_entries: invalid_count")
        case_refs = tuple(entry.case_ref for entry in entries)
        if len(set(case_refs)) != len(case_refs):
            raise BlindedAdjudicationError("mapping_entries: duplicate_case")
        object.__setattr__(self, "entries", entries)

    def to_private_dict(self) -> dict[str, Any]:
        """Return the identity-bearing document for protected audit storage."""
        document = _mapping_payload(
            self.packet_set_ref,
            self.holdout_commitment_digest,
            self.entries,
        )
        document["mapping_commitment"] = self.mapping_commitment
        return document

    def to_private_json(self) -> str:
        """Return canonical identity-bearing JSON for protected audit storage."""
        return _canonical_json(self.to_private_dict())


@dataclass(frozen=True, slots=True)
class PacketBalanceReport:
    """Identity-free summary of candidate position balance."""

    balanced: bool
    packet_count: int
    candidate_count: int
    maximum_slot_imbalance: int
    schema_version: str = ADJUDICATION_BALANCE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return aggregate balance evidence without hidden identities."""
        return {
            "balanced": self.balanced,
            "candidate_count": self.candidate_count,
            "maximum_slot_imbalance": self.maximum_slot_imbalance,
            "packet_count": self.packet_count,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class MappingAuditResult:
    """Identity-free result of a post-adjudication mapping integrity check."""

    valid: bool
    reason_codes: tuple[str, ...]
    packet_count: int
    schema_version: str = ADJUDICATION_AUDIT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a safe audit report containing only closed reason codes."""
        return {
            "packet_count": self.packet_count,
            "reason_codes": list(self.reason_codes),
            "schema_version": self.schema_version,
            "valid": self.valid,
        }


def validate_packet_balance(mapping: SealedIdentityMapping) -> PacketBalanceReport:
    """Validate identity rotation balance without disclosing identity labels.

    Balance means that, for every candidate, the counts across presentation
    slots differ by no more than one. The report exposes only aggregate counts.
    """
    if not isinstance(mapping, SealedIdentityMapping):
        raise BlindedAdjudicationError("mapping: invalid_mapping")
    candidate_count = len(mapping.entries[0].assignments)
    identity_set = {
        assignment.candidate_identity for assignment in mapping.entries[0].assignments
    }
    counts = {identity: [0] * candidate_count for identity in sorted(identity_set)}
    valid_shape = True
    for entry in mapping.entries:
        if (
            len(entry.assignments) != candidate_count
            or {assignment.candidate_identity for assignment in entry.assignments}
            != identity_set
        ):
            valid_shape = False
            continue
        for slot, assignment in enumerate(entry.assignments):
            counts[assignment.candidate_identity][slot] += 1

    maximum_imbalance = max(
        (max(slot_counts) - min(slot_counts) for slot_counts in counts.values()),
        default=0,
    )
    return PacketBalanceReport(
        balanced=valid_shape and maximum_imbalance <= 1,
        packet_count=len(mapping.entries),
        candidate_count=candidate_count,
        maximum_slot_imbalance=maximum_imbalance,
    )


def render_blinded_adjudication_packets(
    *,
    packet_set_ref: str,
    cases: Sequence[ComparisonCase],
    rubric: Sequence[RubricCriterion],
    conflict_of_interest: ConflictOfInterestMetadata,
    holdout_commitment_digest: str,
    submission_manifest_digests: Mapping[str, str],
    randomization_key: bytes,
) -> tuple[tuple[BlindedAdjudicationPacket, ...], SealedIdentityMapping]:
    """Render a deterministic and position-balanced blinded packet set.

    The evaluator-held key controls a deterministic case permutation and a
    balanced rotation of candidate positions. The same validated inputs and
    key produce byte-identical public packets and private mapping.

    Args:
        packet_set_ref: Public opaque identifier for this adjudication batch.
        cases: Private identity-keyed candidate outputs and shared evidence.
        rubric: Between one and eight narrow scoring criteria.
        conflict_of_interest: Digest-only reviewer screening metadata. Recused
            reviewers fail closed and receive no packets.
        holdout_commitment_digest: Published commitment for the governed cases.
        submission_manifest_digests: Sealed workflow manifest digest per hidden
            candidate identity.
        randomization_key: Evaluator-held key of at least 32 bytes.

    Returns:
        Public packets and a separate identity-bearing sealed mapping.
    """
    packet_set_ref = _require_public_ref(packet_set_ref, "packet_set_ref")
    holdout_commitment_digest = _require_digest(
        holdout_commitment_digest, "holdout_commitment_digest"
    )
    key = _require_key(randomization_key)
    if not isinstance(conflict_of_interest, ConflictOfInterestMetadata):
        raise BlindedAdjudicationError("conflict_of_interest: invalid_metadata")
    if conflict_of_interest.status != CONFLICT_SCREENING_CLEARED:
        raise BlindedAdjudicationError("conflict_of_interest: reviewer_recused")

    normalized_cases = tuple(cases)
    if (
        not normalized_cases
        or len(normalized_cases) > _MAX_CASES
        or not all(isinstance(case, ComparisonCase) for case in normalized_cases)
    ):
        raise BlindedAdjudicationError("cases: invalid_sequence")
    case_refs = tuple(case.case_ref for case in normalized_cases)
    if len(set(case_refs)) != len(case_refs):
        raise BlindedAdjudicationError("cases: duplicate_reference")

    normalized_rubric = tuple(rubric)
    if not 1 <= len(normalized_rubric) <= _MAX_CRITERIA or not all(
        isinstance(criterion, RubricCriterion) for criterion in normalized_rubric
    ):
        raise BlindedAdjudicationError("rubric: invalid_sequence")
    criterion_refs = tuple(item.criterion_ref for item in normalized_rubric)
    if len(set(criterion_refs)) != len(criterion_refs):
        raise BlindedAdjudicationError("rubric: duplicate_reference")

    first_identity_set = set(normalized_cases[0].candidate_outputs)
    if any(
        set(case.candidate_outputs) != first_identity_set for case in normalized_cases
    ):
        raise BlindedAdjudicationError("cases: inconsistent_candidates")
    if (
        not isinstance(submission_manifest_digests, Mapping)
        or set(submission_manifest_digests) != first_identity_set
    ):
        raise BlindedAdjudicationError("submission_manifest_digests: invalid_keys")
    manifests = dict(submission_manifest_digests)
    for digest in manifests.values():
        _require_digest(digest, "submission_manifest_digest")

    base_order = tuple(
        sorted(
            first_identity_set,
            key=lambda identity: (
                _keyed_digest(key, b"candidate-order-v1", identity),
                identity,
            ),
        )
    )
    ordered_cases = tuple(
        sorted(
            normalized_cases,
            key=lambda case: (
                _keyed_digest(key, b"case-order-v1", case.case_ref),
                case.case_ref,
            ),
        )
    )

    entries: list[IdentityMappingEntry] = []
    candidate_count = len(base_order)
    for case_index, case in enumerate(ordered_cases):
        rotation = case_index % candidate_count
        identity_order = base_order[rotation:] + base_order[:rotation]
        assignments = tuple(
            IdentityAssignment(
                alias=_alias(slot),
                candidate_identity=identity,
                submission_manifest_digest=manifests[identity],
                output_digest=_digest_text(case.candidate_outputs[identity]),
            )
            for slot, identity in enumerate(identity_order)
        )
        entries.append(
            IdentityMappingEntry(case_ref=case.case_ref, assignments=assignments)
        )

    payload = _mapping_payload(packet_set_ref, holdout_commitment_digest, entries)
    commitment = _mapping_commitment(payload, key)
    mapping = SealedIdentityMapping(
        packet_set_ref=packet_set_ref,
        holdout_commitment_digest=holdout_commitment_digest,
        entries=tuple(entries),
        mapping_commitment=commitment,
    )
    if not validate_packet_balance(mapping).balanced:
        raise BlindedAdjudicationError("mapping: unbalanced")

    case_by_ref = {case.case_ref: case for case in ordered_cases}
    packets = tuple(
        BlindedAdjudicationPacket(
            packet_set_ref=packet_set_ref,
            case_ref=entry.case_ref,
            holdout_commitment_digest=holdout_commitment_digest,
            mapping_commitment=commitment,
            source_evidence=tuple(case_by_ref[entry.case_ref].source_evidence),
            candidates=tuple(
                BlindedCandidate(
                    alias=assignment.alias,
                    content=case_by_ref[entry.case_ref].candidate_outputs[
                        assignment.candidate_identity
                    ],
                )
                for assignment in entry.assignments
            ),
            rubric=normalized_rubric,
            conflict_of_interest=conflict_of_interest,
        )
        for entry in entries
    )
    return packets, mapping


def verify_sealed_identity_mapping(
    packets: Sequence[BlindedAdjudicationPacket],
    mapping: SealedIdentityMapping,
    randomization_key: bytes,
) -> MappingAuditResult:
    """Verify the private mapping and public packets without exposing labels."""
    key = _require_key(randomization_key)
    if not isinstance(mapping, SealedIdentityMapping):
        raise BlindedAdjudicationError("mapping: invalid_mapping")
    normalized_packets = tuple(packets)
    if not normalized_packets or not all(
        isinstance(packet, BlindedAdjudicationPacket) for packet in normalized_packets
    ):
        return MappingAuditResult(False, (AUDIT_INVALID_PACKET_SET,), 0)

    reasons: list[str] = []
    payload = _mapping_payload(
        mapping.packet_set_ref,
        mapping.holdout_commitment_digest,
        mapping.entries,
    )
    expected_commitment = _mapping_commitment(payload, key)
    if not hmac.compare_digest(expected_commitment, mapping.mapping_commitment):
        reasons.append(AUDIT_COMMITMENT_MISMATCH)

    if len(normalized_packets) != len(mapping.entries):
        reasons.append(AUDIT_MAPPING_MISMATCH)
    else:
        for packet, entry in zip(normalized_packets, mapping.entries):
            packet_shape_matches = (
                packet.packet_set_ref == mapping.packet_set_ref
                and packet.holdout_commitment_digest
                == mapping.holdout_commitment_digest
                and packet.mapping_commitment == mapping.mapping_commitment
                and packet.case_ref == entry.case_ref
                and len(packet.candidates) == len(entry.assignments)
            )
            if packet_shape_matches:
                packet_shape_matches = all(
                    candidate.alias == assignment.alias
                    and hmac.compare_digest(
                        _digest_text(candidate.content), assignment.output_digest
                    )
                    for candidate, assignment in zip(
                        packet.candidates, entry.assignments
                    )
                )
            if not packet_shape_matches:
                reasons.append(AUDIT_MAPPING_MISMATCH)
                break

    unique_reasons = tuple(dict.fromkeys(reasons))
    return MappingAuditResult(
        valid=not unique_reasons,
        reason_codes=unique_reasons or (AUDIT_VALID,),
        packet_count=len(normalized_packets),
    )
