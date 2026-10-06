"""Local, value-free chart-abstraction producer over Journey records."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from openmed.clinical.journey import JourneySnapshot
from openmed.clinical.journey_contracts import (
    ClinicalArtifact,
    ClinicalFact,
    ConflictSet,
    EvidenceLocator,
    canonical_digest,
)

from .abstraction_evidence import (
    AbstractionEvidenceChain,
    AbstractionEvidenceError,
    AbstractionEvidenceIssue,
    ChartAbstractionEvidence,
    ReviewerState,
    SourceKind,
    SourceLocation,
    TransformationKind,
    _bounded_tuple,
    _validate_digest,
    _validate_field_id,
)

_OPAQUE_ID = re.compile(r"[a-z][a-z0-9_]{0,31}_[A-Za-z0-9_-]{16,128}")


@dataclass(frozen=True, slots=True)
class AbstractionFieldBinding:
    """Caller-declared field, candidate facts and extraction kind.

    Args:
        field_id: Developer-authored abstraction schema field identifier.
        fact_ids: Opaque candidate IDs; multiple candidates fail closed.
        transformation_kind: Explicit rule or model declaration, never inferred.
    """

    field_id: str
    fact_ids: tuple[str, ...] = field(repr=False)
    transformation_kind: TransformationKind

    def __post_init__(self) -> None:
        _validate_field_id(self.field_id)
        ids = _bounded_tuple(self.fact_ids, field_name="fact_ids", maximum=10_000)
        if any(type(item) is not str or not _OPAQUE_ID.fullmatch(item) for item in ids):
            raise AbstractionEvidenceError("invalid_fact_id", "fact_ids")
        if len(set(ids)) != len(ids):
            raise AbstractionEvidenceError("duplicate_fact", "fact_ids")
        if type(self.transformation_kind) is not TransformationKind:
            raise AbstractionEvidenceError("invalid_transformation_kind")
        object.__setattr__(self, "fact_ids", tuple(sorted(ids)))


@dataclass(frozen=True, slots=True)
class AbstractionReviewReceipt:
    """Explicit trusted-local review decision bound to an unapproved chain.

    Args:
        field_id: Developer-authored field reviewed by the application.
        pending_chain_digest: Digest of the producer's PENDING chain. This binds
            the snapshot, full fact record, source spans and transformations.
        reviewer_state: Explicit decision; the adapter does not issue approvals
            or authenticate reviewers. Only trusted review code may supply it.
    """

    field_id: str
    pending_chain_digest: str
    reviewer_state: ReviewerState

    def __post_init__(self) -> None:
        _validate_field_id(self.field_id)
        _validate_digest(self.pending_chain_digest, "pending_chain_digest")
        if type(self.reviewer_state) is not ReviewerState:
            raise AbstractionEvidenceError("invalid_reviewer_state")


def build_journey_abstraction_evidence(
    snapshot: JourneySnapshot,
    fields: Iterable[AbstractionFieldBinding],
    *,
    facts: Iterable[ClinicalFact],
    locators: Iterable[EvidenceLocator],
    artifacts: Iterable[ClinicalArtifact],
    source_kinds: Mapping[str, SourceKind],
    conflicts: Iterable[ConflictSet] = (),
    review_receipts: Iterable[AbstractionReviewReceipt] = (),
) -> ChartAbstractionEvidence:
    """Produce chains and persistent coverage blockers from local records.

    Args:
        snapshot: Subject-bound point-in-time Journey pointer. The caller must
            supply records read at this revision, including all relevant conflicts.
        fields: Explicit field-to-fact bindings; no clinical inference is performed.
        facts: Immutable Journey facts at the declared snapshot.
        locators: Available evidence locators; every fact evidence ID is checked.
        artifacts: Available artifacts used to resolve source content hashes.
        source_kinds: Trusted artifact-origin declarations. Unknown origins block;
            neither artifact type nor fact status is treated as proof of origin.
        conflicts: Snapshot conflict sets. Open conflicts block mapped facts.
        review_receipts: Explicit decisions from the trusted local review boundary.

    Returns:
        Existing evidence contract with only digests, offsets, field IDs, closed
        codes and counts. Use evaluate/finalize with the required schema fields.

    Raises:
        AbstractionEvidenceError: For malformed or duplicate input declarations,
            with controlled diagnostics that never echo input content.
    """
    if type(snapshot) is not JourneySnapshot:
        raise AbstractionEvidenceError("invalid_snapshot")
    bindings = _index(fields, AbstractionFieldBinding, "field_id")
    fact_by_id = _index(facts, ClinicalFact, "fact_id")
    locator_by_id = _index(locators, EvidenceLocator, "locator_id")
    artifact_by_id = _index(artifacts, ClinicalArtifact, "artifact_id")
    conflict_by_id = _index(conflicts, ConflictSet, "conflict_id")
    receipts = _index(review_receipts, AbstractionReviewReceipt, "field_id")
    if not isinstance(source_kinds, Mapping) or any(
        type(key) is not str
        or not _OPAQUE_ID.fullmatch(key)
        or type(value) is not SourceKind
        for key, value in source_kinds.items()
    ):
        raise AbstractionEvidenceError("invalid_source_kinds")
    if set(receipts) - set(bindings):
        raise AbstractionEvidenceError("unknown_review_field")
    chains = []
    issues: set[AbstractionEvidenceIssue] = set()

    for field_id, binding in sorted(bindings.items()):

        def block(code: str) -> None:
            issues.add(AbstractionEvidenceIssue(code, field_id))

        if not binding.fact_ids or any(
            item not in fact_by_id for item in binding.fact_ids
        ):
            block("missing_field_evidence")
        if len(binding.fact_ids) > 1:
            block("conflicting_facts")
        if len(binding.fact_ids) != 1 or binding.fact_ids[0] not in fact_by_id:
            continue
        fact = fact_by_id[binding.fact_ids[0]]
        if fact.subject_id != snapshot.subject_id:
            block("subject_mismatch")
        if any(
            fact.fact_id in item.fact_ids and item.status == "open"
            for item in conflict_by_id.values()
        ):
            block("conflicting_facts")

        sources: set[SourceLocation] = set()
        transforms = []
        for evidence_id in fact.evidence_ids:
            locator = locator_by_id.get(evidence_id)
            if locator is None:
                block("missing_locator_evidence")
                continue
            if locator.location_type != "text_span":
                block("non_text_locator_evidence")
                continue
            artifact = artifact_by_id.get(locator.artifact_id)
            if artifact is None:
                block("missing_artifact_evidence")
                continue
            if artifact.subject_id not in {None, snapshot.subject_id}:
                block("subject_mismatch")
            kind = source_kinds.get(artifact.artifact_id)
            if kind is None:
                block("source_kind_undeclared")
                continue
            sources.add(
                SourceLocation(
                    artifact.content_hash,
                    locator.location["start"],
                    locator.location["end"],
                    kind,
                )
            )
            transforms.append(
                canonical_digest(
                    {
                        "locator": locator.to_dict(),
                        "artifact": artifact.to_dict(),
                    }
                )
            )
        if fact.parent_fact_ids and not any(
            source.kind is SourceKind.CLINICAL_RECORD for source in sources
        ):
            block("derived_only_evidence")

        chain = AbstractionEvidenceChain(
            field_id=field_id,
            source_locations=sources,
            normalized_fact_digest=canonical_digest(fact),
            transformation_kind=binding.transformation_kind,
            transformation_digest=canonical_digest(
                {
                    "snapshot": canonical_digest(snapshot.to_dict()),
                    "fact_derivation": fact.derivation_hash,
                    "evidence_steps": sorted(transforms),
                }
            ),
            uncertainty=1.0 if fact.confidence is None else 1.0 - fact.confidence,
            reviewer_state=ReviewerState.PENDING,
        )
        receipt = receipts.get(field_id)
        if receipt is not None:
            if receipt.pending_chain_digest != chain.chain_digest:
                block("review_receipt_mismatch")
            else:
                chain = AbstractionEvidenceChain(
                    field_id=chain.field_id,
                    source_locations=chain.source_locations,
                    normalized_fact_digest=chain.normalized_fact_digest,
                    transformation_kind=chain.transformation_kind,
                    transformation_digest=chain.transformation_digest,
                    uncertainty=chain.uncertainty,
                    reviewer_state=receipt.reviewer_state,
                )
        chains.append(chain)
    return ChartAbstractionEvidence(chains, blocking_issues=issues)


def _index(records: Iterable[Any], record_type: type, id_field: str) -> dict[str, Any]:
    values = _bounded_tuple(records, field_name=id_field, maximum=10_000)
    result = {}
    for record in values:
        if type(record) is not record_type:
            raise AbstractionEvidenceError("invalid_record", id_field)
        key = getattr(record, id_field)
        if key in result:
            raise AbstractionEvidenceError("duplicate_record", id_field)
        result[key] = record
    return result


__all__ = [
    "AbstractionFieldBinding",
    "AbstractionReviewReceipt",
    "build_journey_abstraction_evidence",
]
