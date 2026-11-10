"""Value-free lineage for staged FHIR-to-OMOP writes.

The crosswalk joins hashed FHIR resource identities and element paths to OMOP
row digests. It never stores source values or target row values. Verification
is deterministic and local, and approval fails closed when source or target
coverage is incomplete, vocabulary evidence is missing, or a transform is
declared lossy.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..omop.mutation_batch import (
    OmopApprovalBinding,
    OmopBatchCommitter,
    OmopCommitResult,
    OmopMutation,
    OmopMutationBatch,
    OmopMutationPreview,
)
from ..omop.vocabulary_write_gate import VocabularyMappingProvenance

FHIR_OMOP_WRITE_LINEAGE_SCHEMA = "openmed.interop.lineage.fhir_omop_writes.v1"

_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
_BARE_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_FHIR_PATH_RE = re.compile(
    r"[A-Z][A-Za-z0-9]*(?:\.[A-Za-z][A-Za-z0-9]*(?:\[(?:x|[0-9]+)\])?)*"
)
_REASON_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_RULE_RE = re.compile(r"[a-z][a-z0-9_.:-]{0,127}")
_VOCABULARY_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,63}")


class FhirOmopLineageError(ValueError):
    """A value-free lineage validation or approval error."""

    def __init__(
        self,
        code: str,
        field_name: str | None = None,
        *,
        report: FhirOmopLineageReport | None = None,
    ) -> None:
        self.code = code
        self.field_name = field_name
        self.report = report
        message = code if field_name is None else f"{field_name}: {code}"
        super().__init__(message)


@dataclass(frozen=True, slots=True, order=True)
class FhirElementReference:
    """A FHIR element identified without retaining a resource identifier."""

    resource_digest: str
    element_path: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resource_digest",
            _normalize_resource_digest(self.resource_digest),
        )
        object.__setattr__(
            self,
            "element_path",
            _validate_pattern(self.element_path, "element_path", _FHIR_PATH_RE),
        )

    @property
    def reference_digest(self) -> str:
        """Return a stable digest for this resource-and-path reference."""

        return _digest(
            {
                "element_path": self.element_path,
                "resource_digest": self.resource_digest,
            }
        )

    def to_dict(self) -> dict[str, str]:
        """Return review-safe reference metadata."""

        return {
            "element_path": self.element_path,
            "reference_digest": self.reference_digest,
        }


@dataclass(frozen=True, slots=True, order=True)
class VocabularyLineageEvidence:
    """Digest-only evidence for a vocabulary-backed transform."""

    mapping_digest: str
    target_vocabulary_id: str
    snapshot_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mapping_digest",
            _validate_digest(self.mapping_digest, "mapping_digest"),
        )
        object.__setattr__(
            self,
            "target_vocabulary_id",
            _validate_pattern(
                self.target_vocabulary_id,
                "target_vocabulary_id",
                _VOCABULARY_RE,
            ),
        )
        object.__setattr__(
            self,
            "snapshot_digest",
            _validate_digest(self.snapshot_digest, "snapshot_digest"),
        )

    @classmethod
    def from_mapping(
        cls,
        mapping: VocabularyMappingProvenance,
        *,
        snapshot_digest: str,
    ) -> VocabularyLineageEvidence:
        """Extract value-free evidence from vocabulary mapping provenance."""

        if type(mapping) is not VocabularyMappingProvenance:
            raise FhirOmopLineageError("invalid_mapping", "mapping")
        return cls(
            mapping_digest=mapping.mapping_digest,
            target_vocabulary_id=mapping.target_vocabulary_id,
            snapshot_digest=snapshot_digest,
        )

    def to_dict(self) -> dict[str, str]:
        """Return deterministic evidence metadata."""

        return {
            "mapping_digest": self.mapping_digest,
            "snapshot_digest": self.snapshot_digest,
            "target_vocabulary_id": self.target_vocabulary_id,
        }


@dataclass(frozen=True, slots=True, init=False)
class FhirOmopLineageLink:
    """A many-to-many source-element to target-row transform link."""

    sources: tuple[FhirElementReference, ...]
    transform_rule: str
    target_row_digests: tuple[str, ...]
    vocabulary_evidence: tuple[VocabularyLineageEvidence, ...]
    loss_reason: str | None

    def __init__(
        self,
        sources: Iterable[FhirElementReference],
        transform_rule: str,
        target_row_digests: Iterable[str],
        *,
        vocabulary_evidence: Iterable[VocabularyLineageEvidence] = (),
        loss_reason: str | None = None,
    ) -> None:
        normalized_sources = _normalize_sources(sources, allow_empty=False)
        normalized_targets = _normalize_digests(
            target_row_digests,
            "target_row_digests",
        )
        normalized_evidence = _normalize_evidence(vocabulary_evidence)
        normalized_loss = (
            None
            if loss_reason is None
            else _validate_pattern(loss_reason, "loss_reason", _REASON_RE)
        )
        if not normalized_targets and normalized_loss is None:
            raise FhirOmopLineageError("target_required", "target_row_digests")
        object.__setattr__(self, "sources", normalized_sources)
        object.__setattr__(
            self,
            "transform_rule",
            _validate_pattern(transform_rule, "transform_rule", _RULE_RE),
        )
        object.__setattr__(self, "target_row_digests", normalized_targets)
        object.__setattr__(self, "vocabulary_evidence", normalized_evidence)
        object.__setattr__(self, "loss_reason", normalized_loss)

    @property
    def is_lossy(self) -> bool:
        """Return whether the transform explicitly declares information loss."""

        return self.loss_reason is not None

    @property
    def link_digest(self) -> str:
        """Return a stable digest over the complete value-free link."""

        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic, value-free link metadata."""

        return {
            "is_lossy": self.is_lossy,
            "loss_reason": self.loss_reason,
            "sources": [source.to_dict() for source in self.sources],
            "target_row_digests": list(self.target_row_digests),
            "transform_rule": self.transform_rule,
            "vocabulary_evidence": [
                evidence.to_dict() for evidence in self.vocabulary_evidence
            ],
        }


@dataclass(frozen=True, slots=True)
class FhirOmopLineageIssue:
    """One closed, value-free reason that prevents lineage approval."""

    code: str
    link_ordinal: int | None = None
    element_path: str | None = None
    source_reference_digest: str | None = None
    target_row_digest: str | None = None
    loss_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic issue fields."""

        return {
            "code": self.code,
            "element_path": self.element_path,
            "link_ordinal": self.link_ordinal,
            "loss_reason": self.loss_reason,
            "source_reference_digest": self.source_reference_digest,
            "target_row_digest": self.target_row_digest,
        }


@dataclass(frozen=True, slots=True)
class FhirOmopLineageReport:
    """Deterministic coverage and loss report for one staged batch."""

    batch_digest: str
    crosswalk_digest: str
    required_sources_digest: str
    report_digest: str
    issues: tuple[FhirOmopLineageIssue, ...]
    link_count: int
    required_source_count: int
    covered_source_count: int
    target_row_count: int
    covered_target_count: int
    vocabulary_evidence_count: int
    lossy_link_count: int
    schema: str = FHIR_OMOP_WRITE_LINEAGE_SCHEMA

    @property
    def is_approvable(self) -> bool:
        """Return whether lineage is complete and contains no declared loss."""

        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic, value-free report fields."""

        return {
            "batch_digest": self.batch_digest,
            "covered_source_count": self.covered_source_count,
            "covered_target_count": self.covered_target_count,
            "crosswalk_digest": self.crosswalk_digest,
            "is_approvable": self.is_approvable,
            "issues": [issue.to_dict() for issue in self.issues],
            "link_count": self.link_count,
            "lossy_link_count": self.lossy_link_count,
            "report_digest": self.report_digest,
            "required_source_count": self.required_source_count,
            "required_sources_digest": self.required_sources_digest,
            "schema": self.schema,
            "target_row_count": self.target_row_count,
            "vocabulary_evidence_count": self.vocabulary_evidence_count,
        }

    def to_json(self) -> str:
        """Serialize the report with stable key ordering."""

        return _canonical_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class FhirOmopLineageApproval:
    """A staged-batch approval bound to a passing lineage report."""

    batch_approval: OmopApprovalBinding
    lineage_report_digest: str
    crosswalk_digest: str
    required_sources_digest: str

    def __post_init__(self) -> None:
        if type(self.batch_approval) is not OmopApprovalBinding:
            raise FhirOmopLineageError("invalid_batch_approval", "batch_approval")
        _validate_digest(self.lineage_report_digest, "lineage_report_digest")
        _validate_digest(self.crosswalk_digest, "crosswalk_digest")
        _validate_digest(self.required_sources_digest, "required_sources_digest")

    def commit(
        self,
        batch: OmopMutationBatch,
        committer: OmopBatchCommitter,
    ) -> OmopCommitResult:
        """Commit the exact staged batch carried by this lineage approval."""

        if type(batch) is not OmopMutationBatch:
            raise FhirOmopLineageError("invalid_batch", "batch")
        if batch.batch_digest != self.batch_approval.batch_digest:
            raise FhirOmopLineageError("batch_changed", "batch")
        return batch.commit(committer, approval=self.batch_approval)


@dataclass(frozen=True, slots=True, init=False)
class FhirOmopWriteLineage:
    """Verify and bind a many-to-many FHIR-to-OMOP write crosswalk."""

    links: tuple[FhirOmopLineageLink, ...] = field(repr=False)

    def __init__(self, links: Iterable[FhirOmopLineageLink]) -> None:
        try:
            normalized = tuple(links)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            raise FhirOmopLineageError("invalid_links", "links") from None
        if not normalized:
            raise FhirOmopLineageError("empty_crosswalk", "links")
        if any(type(link) is not FhirOmopLineageLink for link in normalized):
            raise FhirOmopLineageError("invalid_link", "links")
        object.__setattr__(self, "links", normalized)

    @property
    def crosswalk_digest(self) -> str:
        """Return a stable digest over the ordered crosswalk."""

        return _digest([link.link_digest for link in self.links])

    def verify(
        self,
        batch: OmopMutationBatch,
        *,
        required_sources: Iterable[FhirElementReference],
    ) -> FhirOmopLineageReport:
        """Verify exact source, target, vocabulary, and loss coverage."""

        if type(batch) is not OmopMutationBatch:
            raise FhirOmopLineageError("invalid_batch", "batch")
        normalized_sources = _normalize_sources(required_sources, allow_empty=False)
        source_links = {
            source.reference_digest for link in self.links for source in link.sources
        }
        target_links: dict[str, list[FhirOmopLineageLink]] = {}
        for link in self.links:
            for row_digest in link.target_row_digests:
                target_links.setdefault(row_digest, []).append(link)

        mutations_by_digest = {
            mutation.row_digest: mutation for mutation in batch.mutations
        }
        issues: list[FhirOmopLineageIssue] = []
        for source in normalized_sources:
            if source.reference_digest not in source_links:
                issues.append(
                    FhirOmopLineageIssue(
                        code="missing_source_lineage",
                        element_path=source.element_path,
                        source_reference_digest=source.reference_digest,
                    )
                )
        for mutation in batch.mutations:
            links = target_links.get(mutation.row_digest, [])
            if not links:
                issues.append(
                    FhirOmopLineageIssue(
                        code="missing_target_lineage",
                        target_row_digest=mutation.row_digest,
                    )
                )
            elif _requires_vocabulary_evidence(mutation) and not any(
                link.vocabulary_evidence for link in links
            ):
                issues.append(
                    FhirOmopLineageIssue(
                        code="missing_vocabulary_evidence",
                        target_row_digest=mutation.row_digest,
                    )
                )
        for ordinal, link in enumerate(self.links):
            for row_digest in link.target_row_digests:
                if row_digest not in mutations_by_digest:
                    issues.append(
                        FhirOmopLineageIssue(
                            code="unknown_target_row",
                            link_ordinal=ordinal,
                            target_row_digest=row_digest,
                        )
                    )
            if link.is_lossy:
                issues.append(
                    FhirOmopLineageIssue(
                        code="lossy_transformation",
                        link_ordinal=ordinal,
                        loss_reason=link.loss_reason,
                    )
                )

        required_sources_digest = _digest(
            [source.reference_digest for source in normalized_sources]
        )
        batch_targets = set(mutations_by_digest)
        covered_sources = sum(
            source.reference_digest in source_links for source in normalized_sources
        )
        covered_targets = len(batch_targets & set(target_links))
        evidence_count = sum(len(link.vocabulary_evidence) for link in self.links)
        payload = {
            "batch_digest": batch.batch_digest,
            "covered_source_count": covered_sources,
            "covered_target_count": covered_targets,
            "crosswalk_digest": self.crosswalk_digest,
            "issues": [issue.to_dict() for issue in issues],
            "link_count": len(self.links),
            "lossy_link_count": sum(link.is_lossy for link in self.links),
            "required_source_count": len(normalized_sources),
            "required_sources_digest": required_sources_digest,
            "schema": FHIR_OMOP_WRITE_LINEAGE_SCHEMA,
            "target_row_count": len(batch.mutations),
            "vocabulary_evidence_count": evidence_count,
        }
        return FhirOmopLineageReport(
            batch_digest=batch.batch_digest,
            crosswalk_digest=self.crosswalk_digest,
            required_sources_digest=required_sources_digest,
            report_digest=_digest(payload),
            issues=tuple(issues),
            link_count=len(self.links),
            required_source_count=len(normalized_sources),
            covered_source_count=covered_sources,
            target_row_count=len(batch.mutations),
            covered_target_count=covered_targets,
            vocabulary_evidence_count=evidence_count,
            lossy_link_count=sum(link.is_lossy for link in self.links),
        )

    def bind_approval(
        self,
        batch: OmopMutationBatch,
        preview: OmopMutationPreview,
        *,
        required_sources: Iterable[FhirElementReference],
        approved_preview_digest: str,
        approval_receipt_digest: str,
    ) -> FhirOmopLineageApproval:
        """Bind approval only after the current crosswalk passes verification."""

        report = self.verify(batch, required_sources=required_sources)
        if not report.is_approvable:
            raise FhirOmopLineageError(
                "lineage_verification_failed",
                report=report,
            )
        batch_approval = batch.bind_approval(
            preview,
            approved_preview_digest=approved_preview_digest,
            approval_receipt_digest=approval_receipt_digest,
        )
        return FhirOmopLineageApproval(
            batch_approval=batch_approval,
            lineage_report_digest=report.report_digest,
            crosswalk_digest=report.crosswalk_digest,
            required_sources_digest=report.required_sources_digest,
        )

    def __repr__(self) -> str:
        return (
            "FhirOmopWriteLineage("
            f"link_count={len(self.links)}, "
            f"crosswalk_digest={self.crosswalk_digest!r})"
        )


def _requires_vocabulary_evidence(mutation: OmopMutation) -> bool:
    if mutation.table == "concept":
        return False
    return any(
        name.endswith("_concept_id") and value not in (None, 0)
        for name, value in mutation.values.items()
    )


def _normalize_sources(
    values: Iterable[FhirElementReference],
    *,
    allow_empty: bool,
) -> tuple[FhirElementReference, ...]:
    try:
        normalized = tuple(values)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise FhirOmopLineageError("invalid_sources", "sources") from None
    if any(type(value) is not FhirElementReference for value in normalized):
        raise FhirOmopLineageError("invalid_source", "sources")
    result = tuple(sorted(set(normalized)))
    if not result and not allow_empty:
        raise FhirOmopLineageError("sources_required", "sources")
    return result


def _normalize_digests(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    try:
        normalized = tuple(_validate_digest(value, field_name) for value in values)
    except (KeyboardInterrupt, SystemExit, FhirOmopLineageError):
        raise
    except Exception:
        raise FhirOmopLineageError("invalid_digests", field_name) from None
    return tuple(sorted(set(normalized)))


def _normalize_evidence(
    values: Iterable[VocabularyLineageEvidence],
) -> tuple[VocabularyLineageEvidence, ...]:
    try:
        normalized = tuple(values)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise FhirOmopLineageError(
            "invalid_vocabulary_evidence",
            "vocabulary_evidence",
        ) from None
    if any(type(value) is not VocabularyLineageEvidence for value in normalized):
        raise FhirOmopLineageError(
            "invalid_vocabulary_evidence",
            "vocabulary_evidence",
        )
    return tuple(sorted(set(normalized)))


def _validate_digest(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise FhirOmopLineageError("invalid_digest", field_name)
    return value


def _normalize_resource_digest(value: Any) -> str:
    if isinstance(value, str) and _BARE_SHA256_RE.fullmatch(value) is not None:
        return f"sha256:{value}"
    return _validate_digest(value, "resource_digest")


def _validate_pattern(value: Any, field_name: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise FhirOmopLineageError("invalid_value", field_name)
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _digest(value: Any) -> str:
    encoded = _canonical_json(value).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


__all__ = [
    "FHIR_OMOP_WRITE_LINEAGE_SCHEMA",
    "FhirElementReference",
    "FhirOmopLineageApproval",
    "FhirOmopLineageError",
    "FhirOmopLineageIssue",
    "FhirOmopLineageLink",
    "FhirOmopLineageReport",
    "FhirOmopWriteLineage",
    "VocabularyLineageEvidence",
]
