"""Fail-closed OMOP writes across vocabulary snapshot boundaries.

The gate consumes terminology-only mapping provenance and a caller-supplied
target snapshot. It performs no network access and exposes only stable digests,
closed reason codes, and vocabulary identifiers in review artifacts. Clinical
source codes, descriptions, and row values are deliberately outside its data
model.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

from .mutation_batch import (
    OmopApprovalBinding,
    OmopBatchCommitter,
    OmopCommitResult,
    OmopMutationBatch,
)

VOCABULARY_WRITE_GATE_SCHEMA = "openmed.interop.omop.vocabulary_write_gate.v1"

_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")


class VocabularyCompatibility(str, Enum):
    """Closed classification set for a mapping at the target site."""

    COMPATIBLE = "compatible"
    REMAP_REQUIRED = "remap_required"
    RETIRED = "retired"


class VocabularyWriteGateError(ValueError):
    """A value-free vocabulary gate validation or write-blocking error."""

    def __init__(
        self,
        code: str,
        field_name: str | None = None,
        *,
        report: VocabularyWriteGateReport | None = None,
    ) -> None:
        self.code = code
        self.field_name = field_name
        self.report = report
        message = code if field_name is None else f"{field_name}: {code}"
        super().__init__(message)


@dataclass(frozen=True, slots=True, repr=False)
class VocabularyMappingProvenance:
    """Terminology-only provenance for one proposed mapped concept.

    This type intentionally has no source code, source description, patient
    identifier, or OMOP row value fields. ``from_mapping`` accepts router
    results but extracts only target vocabulary metadata.
    """

    target_concept_id: int
    target_vocabulary_id: str
    vocabulary_version: str | None

    def __post_init__(self) -> None:
        if isinstance(self.target_concept_id, bool) or not isinstance(
            self.target_concept_id, int
        ):
            raise VocabularyWriteGateError("invalid_integer", "target_concept_id")
        if self.target_concept_id < 0:
            raise VocabularyWriteGateError("negative_integer", "target_concept_id")
        object.__setattr__(
            self,
            "target_vocabulary_id",
            _normalize_text(self.target_vocabulary_id, "target_vocabulary_id", True),
        )
        object.__setattr__(
            self,
            "vocabulary_version",
            _normalize_optional_text(self.vocabulary_version, "vocabulary_version"),
        )

    @classmethod
    def from_mapping(cls, mapping: Any) -> VocabularyMappingProvenance:
        """Extract safe target provenance from a vocabulary-router mapping."""

        try:
            concept_id = mapping.target_concept_id
            vocabulary_id = mapping.target_vocabulary_id
            vocabulary_version = mapping.vocabulary_version
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            raise VocabularyWriteGateError("invalid_mapping", "mapping") from None
        return cls(
            target_concept_id=concept_id,
            target_vocabulary_id=vocabulary_id,
            vocabulary_version=vocabulary_version,
        )

    @property
    def mapping_digest(self) -> str:
        """Return a stable digest over terminology-only mapping provenance."""

        return _digest(
            {
                "target_concept_id": self.target_concept_id,
                "target_vocabulary_id": self.target_vocabulary_id,
                "vocabulary_version": self.vocabulary_version,
            }
        )

    def __repr__(self) -> str:
        return f"VocabularyMappingProvenance(mapping_digest={self.mapping_digest!r})"


@dataclass(frozen=True, slots=True, repr=False)
class VocabularyConcept:
    """Target-snapshot state required to validate one OMOP concept."""

    concept_id: int
    vocabulary_id: str
    standard_concept: str | None = None
    invalid_reason: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.concept_id, bool) or not isinstance(self.concept_id, int):
            raise VocabularyWriteGateError("invalid_integer", "concept_id")
        if self.concept_id <= 0:
            raise VocabularyWriteGateError("non_positive_integer", "concept_id")
        object.__setattr__(
            self,
            "vocabulary_id",
            _normalize_text(self.vocabulary_id, "vocabulary_id"),
        )
        standard = _normalize_optional_text(self.standard_concept, "standard_concept")
        reason = _normalize_optional_text(self.invalid_reason, "invalid_reason")
        object.__setattr__(
            self, "standard_concept", standard.upper() if standard else None
        )
        object.__setattr__(self, "invalid_reason", reason.upper() if reason else None)

    @property
    def is_retired(self) -> bool:
        """Return whether the target snapshot marks this concept invalid."""

        return self.invalid_reason is not None

    @property
    def concept_digest(self) -> str:
        """Return a stable digest over the target concept state."""

        return _digest(
            {
                "concept_id": self.concept_id,
                "invalid_reason": self.invalid_reason,
                "standard_concept": self.standard_concept,
                "vocabulary_id": self.vocabulary_id,
            }
        )

    def __repr__(self) -> str:
        return f"VocabularyConcept(concept_digest={self.concept_digest!r})"


@dataclass(frozen=True, slots=True, repr=False, init=False)
class VocabularySnapshot:
    """Caller-supplied target vocabulary versions and concept states."""

    _version_items: tuple[tuple[str, str], ...] = field(repr=False)
    _concepts: tuple[VocabularyConcept, ...] = field(repr=False)
    _concept_index: Mapping[int, VocabularyConcept] = field(repr=False, compare=False)

    def __init__(
        self,
        vocabulary_versions: Mapping[str, str],
        concepts: Iterable[VocabularyConcept],
    ) -> None:
        if not isinstance(vocabulary_versions, Mapping):
            raise VocabularyWriteGateError(
                "invalid_version_mapping", "vocabulary_versions"
            )
        version_items = tuple(
            sorted(
                (
                    _normalize_text(vocabulary_id, "vocabulary_id"),
                    _normalize_text(version, "vocabulary_version"),
                )
                for vocabulary_id, version in vocabulary_versions.items()
            )
        )
        if len({item[0] for item in version_items}) != len(version_items):
            raise VocabularyWriteGateError(
                "duplicate_vocabulary", "vocabulary_versions"
            )
        if isinstance(concepts, (str, bytes, bytearray, Mapping)):
            raise VocabularyWriteGateError("invalid_concept_collection", "concepts")
        try:
            normalized_concepts = tuple(concepts)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise VocabularyWriteGateError(
                "invalid_concept_collection", "concepts"
            ) from None
        if any(
            type(concept) is not VocabularyConcept for concept in normalized_concepts
        ):
            raise VocabularyWriteGateError("invalid_concept", "concepts")
        concept_index = {concept.concept_id: concept for concept in normalized_concepts}
        if len(concept_index) != len(normalized_concepts):
            raise VocabularyWriteGateError("duplicate_concept", "concepts")
        ordered_concepts = tuple(
            sorted(normalized_concepts, key=lambda concept: concept.concept_id)
        )
        object.__setattr__(self, "_version_items", version_items)
        object.__setattr__(self, "_concepts", ordered_concepts)
        object.__setattr__(self, "_concept_index", MappingProxyType(concept_index))

    @property
    def snapshot_digest(self) -> str:
        """Return a stable digest over target versions and concept states."""

        return _digest(
            {
                "concept_digests": [
                    concept.concept_digest for concept in self._concepts
                ],
                "vocabulary_versions": dict(self._version_items),
            }
        )

    def version_for(self, vocabulary_id: str) -> str | None:
        """Return the target version for a vocabulary identifier."""

        return dict(self._version_items).get(vocabulary_id)

    def concept(self, concept_id: int) -> VocabularyConcept | None:
        """Return target state for a concept identifier, if present."""

        return self._concept_index.get(concept_id)

    def __repr__(self) -> str:
        return f"VocabularySnapshot(snapshot_digest={self.snapshot_digest!r})"


@dataclass(frozen=True, slots=True)
class VocabularyGateDecision:
    """Value-free compatibility result for one proposed mapping."""

    ordinal: int
    mapping_digest: str
    target_vocabulary_id: str
    compatibility: VocabularyCompatibility
    reason: str
    source_version_digest: str | None
    target_version_digest: str | None

    def __post_init__(self) -> None:
        _validate_digest(self.mapping_digest, "mapping_digest")
        if self.source_version_digest is not None:
            _validate_digest(self.source_version_digest, "source_version_digest")
        if self.target_version_digest is not None:
            _validate_digest(self.target_version_digest, "target_version_digest")

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic terminology-only review metadata."""

        return {
            "compatibility": self.compatibility.value,
            "mapping_digest": self.mapping_digest,
            "ordinal": self.ordinal,
            "reason": self.reason,
            "source_version_digest": self.source_version_digest,
            "target_version_digest": self.target_version_digest,
            "target_vocabulary_id": self.target_vocabulary_id,
        }


@dataclass(frozen=True, slots=True)
class VocabularyRemappingRequest:
    """Value-free queue item for a mapping requiring human review."""

    ordinal: int
    mapping_digest: str
    target_vocabulary_id: str
    compatibility: VocabularyCompatibility
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic queue metadata without source or concept values."""

        return {
            "compatibility": self.compatibility.value,
            "mapping_digest": self.mapping_digest,
            "ordinal": self.ordinal,
            "reason": self.reason,
            "target_vocabulary_id": self.target_vocabulary_id,
        }


@dataclass(frozen=True, slots=True)
class VocabularyWriteGateReport:
    """Deterministic, value-free write decision and remapping queue."""

    snapshot_digest: str
    mappings_digest: str
    report_digest: str
    decisions: tuple[VocabularyGateDecision, ...]
    remapping_queue: tuple[VocabularyRemappingRequest, ...]
    classification_counts: tuple[tuple[str, int], ...]
    schema: str = VOCABULARY_WRITE_GATE_SCHEMA

    @property
    def is_compatible(self) -> bool:
        """Return whether every mapping is safe for the target snapshot."""

        return not self.remapping_queue

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-safe report fields."""

        return {
            "classification_counts": dict(self.classification_counts),
            "decisions": [decision.to_dict() for decision in self.decisions],
            "is_compatible": self.is_compatible,
            "mapping_count": len(self.decisions),
            "mappings_digest": self.mappings_digest,
            "remapping_queue": [item.to_dict() for item in self.remapping_queue],
            "report_digest": self.report_digest,
            "schema": self.schema,
            "snapshot_digest": self.snapshot_digest,
        }

    def to_json(self) -> str:
        """Serialize the report with stable key ordering."""

        return _canonical_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class VocabularyWriteGate:
    """Compare mapping provenance to a target snapshot and guard commits."""

    snapshot: VocabularySnapshot

    def __post_init__(self) -> None:
        if type(self.snapshot) is not VocabularySnapshot:
            raise VocabularyWriteGateError("invalid_snapshot", "snapshot")

    def evaluate(
        self,
        mappings: Iterable[VocabularyMappingProvenance],
    ) -> VocabularyWriteGateReport:
        """Classify mappings and build a deterministic remapping queue."""

        normalized = _normalize_mappings(mappings)
        decisions = tuple(
            self._classify(mapping, ordinal)
            for ordinal, mapping in enumerate(normalized)
        )
        queue = tuple(
            VocabularyRemappingRequest(
                ordinal=decision.ordinal,
                mapping_digest=decision.mapping_digest,
                target_vocabulary_id=decision.target_vocabulary_id,
                compatibility=decision.compatibility,
                reason=decision.reason,
            )
            for decision in decisions
            if decision.compatibility is not VocabularyCompatibility.COMPATIBLE
        )
        counts = Counter(decision.compatibility.value for decision in decisions)
        classification_counts = tuple(sorted(counts.items()))
        mappings_digest = _digest([mapping.mapping_digest for mapping in normalized])
        payload = {
            "classification_counts": dict(classification_counts),
            "decisions": [decision.to_dict() for decision in decisions],
            "mappings_digest": mappings_digest,
            "remapping_queue": [item.to_dict() for item in queue],
            "schema": VOCABULARY_WRITE_GATE_SCHEMA,
            "snapshot_digest": self.snapshot.snapshot_digest,
        }
        return VocabularyWriteGateReport(
            snapshot_digest=self.snapshot.snapshot_digest,
            mappings_digest=mappings_digest,
            report_digest=_digest(payload),
            decisions=decisions,
            remapping_queue=queue,
            classification_counts=classification_counts,
        )

    def assert_writable(
        self,
        mappings: Iterable[VocabularyMappingProvenance],
    ) -> VocabularyWriteGateReport:
        """Return a passing report or raise before any write on material drift."""

        report = self.evaluate(mappings)
        if not report.is_compatible:
            raise VocabularyWriteGateError(
                "incompatible_vocabulary_snapshot",
                report=report,
            )
        return report

    def commit(
        self,
        batch: OmopMutationBatch,
        committer: OmopBatchCommitter,
        *,
        approval: OmopApprovalBinding,
        mappings: Iterable[VocabularyMappingProvenance],
    ) -> OmopCommitResult:
        """Commit an approved batch only after a fresh target-snapshot check."""

        self.assert_writable(mappings)
        return batch.commit(committer, approval=approval)

    def _classify(
        self,
        mapping: VocabularyMappingProvenance,
        ordinal: int,
    ) -> VocabularyGateDecision:
        target_version = self.snapshot.version_for(mapping.target_vocabulary_id)
        compatibility = VocabularyCompatibility.COMPATIBLE
        reason = "snapshot_match"
        concept = self.snapshot.concept(mapping.target_concept_id)

        if mapping.target_concept_id == 0:
            compatibility = VocabularyCompatibility.REMAP_REQUIRED
            reason = "unmapped_concept"
        elif not mapping.target_vocabulary_id:
            compatibility = VocabularyCompatibility.REMAP_REQUIRED
            reason = "target_vocabulary_missing"
        elif concept is None:
            compatibility = VocabularyCompatibility.REMAP_REQUIRED
            reason = "concept_not_found"
        elif concept.is_retired:
            compatibility = VocabularyCompatibility.RETIRED
            reason = "concept_retired"
        elif concept.vocabulary_id != mapping.target_vocabulary_id:
            compatibility = VocabularyCompatibility.REMAP_REQUIRED
            reason = "concept_vocabulary_changed"
        elif concept.standard_concept != "S":
            compatibility = VocabularyCompatibility.REMAP_REQUIRED
            reason = "concept_not_standard"
        elif mapping.vocabulary_version is None:
            compatibility = VocabularyCompatibility.REMAP_REQUIRED
            reason = "mapping_version_missing"
        elif target_version is None:
            compatibility = VocabularyCompatibility.REMAP_REQUIRED
            reason = "target_version_missing"
        elif mapping.vocabulary_version != target_version:
            compatibility = VocabularyCompatibility.REMAP_REQUIRED
            reason = "vocabulary_version_changed"

        return VocabularyGateDecision(
            ordinal=ordinal,
            mapping_digest=mapping.mapping_digest,
            target_vocabulary_id=mapping.target_vocabulary_id,
            compatibility=compatibility,
            reason=reason,
            source_version_digest=_optional_digest(mapping.vocabulary_version),
            target_version_digest=_optional_digest(target_version),
        )


def _normalize_mappings(
    mappings: Iterable[VocabularyMappingProvenance],
) -> tuple[VocabularyMappingProvenance, ...]:
    if isinstance(mappings, (str, bytes, bytearray, Mapping)):
        raise VocabularyWriteGateError("invalid_mapping_collection", "mappings")
    try:
        normalized = tuple(mappings)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise VocabularyWriteGateError(
            "invalid_mapping_collection", "mappings"
        ) from None
    if not normalized:
        raise VocabularyWriteGateError("empty_mapping_collection", "mappings")
    if any(type(mapping) is not VocabularyMappingProvenance for mapping in normalized):
        raise VocabularyWriteGateError("invalid_mapping", "mappings")
    return normalized


def _normalize_text(value: Any, field_name: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise VocabularyWriteGateError("invalid_text", field_name)
    normalized = value.strip()
    if not normalized and not allow_empty:
        raise VocabularyWriteGateError("empty_text", field_name)
    return normalized


def _normalize_optional_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = _normalize_text(value, field_name, True)
    return normalized or None


def _validate_digest(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise VocabularyWriteGateError("invalid_digest", field_name)
    return value


def _optional_digest(value: str | None) -> str | None:
    return _digest(value) if value is not None else None


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _digest(value: Any) -> str:
    return (
        "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    )


__all__ = [
    "VOCABULARY_WRITE_GATE_SCHEMA",
    "VocabularyCompatibility",
    "VocabularyConcept",
    "VocabularyGateDecision",
    "VocabularyMappingProvenance",
    "VocabularyRemappingRequest",
    "VocabularySnapshot",
    "VocabularyWriteGate",
    "VocabularyWriteGateError",
    "VocabularyWriteGateReport",
]
