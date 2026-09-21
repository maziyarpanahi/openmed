"""Immutable saved-cohort definitions, executions, and membership evidence.

This module deliberately persists value-free cohort artifacts rather than raw
clinical rows.  Patient, fact, evidence, and time-window references must be
opaque identifiers supplied by the caller.  Definitions and executions are
content bound, so rerunning the same inputs can detect nondeterministic output
instead of silently replacing a previous result.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from importlib import resources
from pathlib import Path
from typing import Any, Final

from openmed.clinical.journey_contracts import (
    canonical_digest,
    canonical_json,
    derived_opaque_id,
)
from openmed.structured.store import (
    AllowAllStoragePolicy,
    StoragePolicy,
    StoreResult,
    StoreState,
)

from .dsl import Expression, PhenotypeDefinition
from .exchange import CohortSourceSnapshot

SAVED_COHORT_SCHEMA_VERSION: Final = "1.0.0"
SAVED_COHORT_COMPATIBILITY_POLICY: Final = "same_major"
SAVED_COHORT_SCHEMA_NAME: Final = "saved_cohort"
SAVED_COHORT_SCHEMA_PACKAGE: Final = "openmed.core.schemas.json"
SAVED_COHORT_ADVISORY: Final = (
    "Saved cohort membership is an analytical result for human review and must "
    "not automatically trigger enrollment, outreach, or clinical action."
)

_CONTROLLED_RE = re.compile(r"^[a-z][a-z0-9_.:/-]{0,127}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_OPAQUE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}_[A-Za-z0-9_-]{16,128}$")


class SavedCohortError(ValueError):
    """Base error for malformed saved-cohort contracts."""


class SavedCohortConflictError(SavedCohortError):
    """Raised when immutable cohort custody or membership conflicts."""


class SavedCohortUnsupportedError(SavedCohortError):
    """Raised when a saved-cohort version or capability is unsupported."""


class MembershipState(str, Enum):
    """Four explicit states for cohort and criterion membership."""

    MET = "met"
    NOT_MET = "not_met"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"


@dataclass(frozen=True, slots=True)
class CohortDefinitionVersion:
    """One immutable, content-addressed phenotype definition version."""

    definition: PhenotypeDefinition = field(repr=False)
    version_id: str | None = None
    schema_version: str = SAVED_COHORT_SCHEMA_VERSION
    compatibility_policy: str = SAVED_COHORT_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        _contract_version(self.schema_version, self.compatibility_policy)
        if not isinstance(self.definition, PhenotypeDefinition):
            raise TypeError("definition must be PhenotypeDefinition")
        expected = derived_opaque_id(
            "cohortdef",
            self.definition.id,
            self.definition_digest,
            self.schema_version,
        )
        if self.version_id is None:
            object.__setattr__(self, "version_id", expected)
        elif self.version_id != expected:
            raise SavedCohortConflictError("definition version identifier differs")

    @property
    def definition_digest(self) -> str:
        """Return a normalized digest of canonical definition bytes."""

        return f"sha256:{self.definition.sha256}"

    @property
    def criterion_ids(self) -> tuple[str, ...]:
        """Return the definition's criterion identifiers in stable order."""

        return tuple(item.id for item in self.definition.criteria())

    def to_dict(self) -> dict[str, Any]:
        """Return the strict persisted representation."""

        return {
            "compatibility_policy": self.compatibility_policy,
            "definition": self.definition.to_dict(),
            "definition_digest": self.definition_digest,
            "schema_version": self.schema_version,
            "version_id": self.version_id,
        }

    def to_json(self) -> str:
        """Return byte-stable JSON."""

        return canonical_json(self.to_dict())

    def to_json_bytes(self) -> bytes:
        """Return canonical UTF-8 bytes."""

        return self.to_json().encode("utf-8")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CohortDefinitionVersion":
        """Parse and verify a strict persisted definition version."""

        data = _mapping(value, "definition version")
        _exact_keys(
            data,
            {
                "compatibility_policy",
                "definition",
                "definition_digest",
                "schema_version",
                "version_id",
            },
            "definition version",
        )
        result = cls(
            definition=PhenotypeDefinition.from_dict(
                _mapping(data["definition"], "definition")
            ),
            version_id=_text(data["version_id"], "version_id"),
            schema_version=_text(data["schema_version"], "schema_version"),
            compatibility_policy=_text(
                data["compatibility_policy"], "compatibility_policy"
            ),
        )
        if data["definition_digest"] != result.definition_digest:
            raise SavedCohortConflictError("definition digest differs")
        return result

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> "CohortDefinitionVersion":
        """Parse canonical or human-formatted JSON."""

        return cls.from_dict(_json_object(value, "definition version"))


@dataclass(frozen=True, slots=True)
class MembershipEvidence:
    """Value-free evidence and time-window references for one criterion."""

    evidence_id: str
    fact_id: str | None = None
    time_window_id: str | None = None
    role: str = "supporting"

    def __post_init__(self) -> None:
        _opaque_id(self.evidence_id, "evidence_id")
        if self.fact_id is not None:
            _opaque_id(self.fact_id, "fact_id")
        if self.time_window_id is not None:
            _opaque_id(self.time_window_id, "time_window_id")
        _controlled(self.role, "evidence role")

    def to_dict(self) -> dict[str, str | None]:
        """Return the value-free reference."""

        return {
            "evidence_id": self.evidence_id,
            "fact_id": self.fact_id,
            "role": self.role,
            "time_window_id": self.time_window_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MembershipEvidence":
        """Parse a strict evidence reference."""

        data = _mapping(value, "membership evidence")
        _exact_keys(
            data,
            {"evidence_id", "fact_id", "role", "time_window_id"},
            "membership evidence",
        )
        return cls(
            evidence_id=_text(data["evidence_id"], "evidence_id"),
            fact_id=_optional_text(data["fact_id"], "fact_id"),
            time_window_id=_optional_text(data["time_window_id"], "time_window_id"),
            role=_text(data["role"], "evidence role"),
        )


@dataclass(frozen=True, slots=True)
class CriterionMembership:
    """One criterion state with only controlled reasons and opaque evidence."""

    criterion_id: str
    state: MembershipState
    evidence: tuple[MembershipEvidence, ...] = ()
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _controlled(self.criterion_id, "criterion_id")
        state = _membership_state(self.state, "criterion membership state")
        object.__setattr__(self, "state", state)
        if any(not isinstance(item, MembershipEvidence) for item in self.evidence):
            raise TypeError("criterion evidence must contain MembershipEvidence")
        evidence = tuple(
            sorted(
                self.evidence,
                key=lambda item: (
                    item.evidence_id,
                    item.fact_id or "",
                    item.time_window_id or "",
                    item.role,
                ),
            )
        )
        if len({canonical_json(item.to_dict()) for item in evidence}) != len(evidence):
            raise SavedCohortConflictError("criterion evidence must be unique")
        reasons = tuple(
            sorted({_controlled(item, "reason code") for item in self.reason_codes})
        )
        if state in {MembershipState.UNKNOWN, MembershipState.CONFLICT} and not reasons:
            raise SavedCohortError(
                "unknown and conflict criteria require a reason code"
            )
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "reason_codes", reasons)

    @property
    def review_required(self) -> bool:
        """Return whether this criterion cannot be operationalized directly."""

        return self.state in {MembershipState.UNKNOWN, MembershipState.CONFLICT}

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic criterion explanation."""

        return {
            "criterion_id": self.criterion_id,
            "evidence": [item.to_dict() for item in self.evidence],
            "reason_codes": list(self.reason_codes),
            "review_required": self.review_required,
            "state": self.state.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CriterionMembership":
        """Parse and verify one criterion state."""

        data = _mapping(value, "criterion membership")
        _exact_keys(
            data,
            {
                "criterion_id",
                "evidence",
                "reason_codes",
                "review_required",
                "state",
            },
            "criterion membership",
        )
        result = cls(
            criterion_id=_text(data["criterion_id"], "criterion_id"),
            state=_membership_state(data["state"], "criterion membership state"),
            evidence=tuple(
                MembershipEvidence.from_dict(_mapping(item, "membership evidence"))
                for item in _sequence(data["evidence"], "evidence")
            ),
            reason_codes=_text_sequence(data["reason_codes"], "reason_codes"),
        )
        if data["review_required"] is not result.review_required:
            raise SavedCohortConflictError("criterion review state differs")
        return result


@dataclass(frozen=True, slots=True)
class CohortMembership:
    """One opaque patient key and its criterion-complete membership result."""

    patient_key: str
    state: MembershipState
    criteria: tuple[CriterionMembership, ...]

    def __post_init__(self) -> None:
        _opaque_id(self.patient_key, "patient_key")
        state = _membership_state(self.state, "cohort membership state")
        object.__setattr__(self, "state", state)
        if any(not isinstance(item, CriterionMembership) for item in self.criteria):
            raise TypeError("criteria must contain CriterionMembership")
        criteria = tuple(sorted(self.criteria, key=lambda item: item.criterion_id))
        identifiers = [item.criterion_id for item in criteria]
        if not identifiers or len(identifiers) != len(set(identifiers)):
            raise SavedCohortConflictError(
                "membership criteria must be non-empty with unique identifiers"
            )
        if state is MembershipState.MET and any(
            item.review_required for item in criteria
        ):
            raise SavedCohortConflictError(
                "unresolved criterion cannot produce eligible membership"
            )
        object.__setattr__(self, "criteria", criteria)

    @property
    def review_required(self) -> bool:
        """Return whether any unresolved criterion requires human review."""

        return any(item.review_required for item in self.criteria)

    @property
    def eligible(self) -> bool:
        """Return eligibility only for an explicit, fully resolved match."""

        return self.state is MembershipState.MET and not self.review_required

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic, PHI-free membership explanation."""

        return {
            "criteria": [item.to_dict() for item in self.criteria],
            "eligible": self.eligible,
            "patient_key": self.patient_key,
            "review_required": self.review_required,
            "state": self.state.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CohortMembership":
        """Parse and verify one membership explanation."""

        data = _mapping(value, "cohort membership")
        _exact_keys(
            data,
            {"criteria", "eligible", "patient_key", "review_required", "state"},
            "cohort membership",
        )
        result = cls(
            patient_key=_text(data["patient_key"], "patient_key"),
            state=_membership_state(data["state"], "cohort membership state"),
            criteria=tuple(
                CriterionMembership.from_dict(_mapping(item, "criterion membership"))
                for item in _sequence(data["criteria"], "criteria")
            ),
        )
        if data["eligible"] is not result.eligible:
            raise SavedCohortConflictError("persisted eligibility differs")
        if data["review_required"] is not result.review_required:
            raise SavedCohortConflictError("persisted review state differs")
        return result


@dataclass(frozen=True, slots=True)
class CohortExecutionManifest:
    """Content-bound inputs for one reproducible cohort execution."""

    definition_version_id: str
    definition_digest: str
    criterion_ids: tuple[str, ...]
    expression: Expression = field(repr=False)
    source_snapshot: CohortSourceSnapshot
    vocabulary_digest: str
    policy_digest: str
    evaluator_version: str
    execution_id: str | None = None
    schema_version: str = SAVED_COHORT_SCHEMA_VERSION
    compatibility_policy: str = SAVED_COHORT_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        _contract_version(self.schema_version, self.compatibility_policy)
        _opaque_id(self.definition_version_id, "definition_version_id")
        _digest(self.definition_digest, "definition_digest")
        criteria = tuple(
            sorted(_controlled(item, "criterion_id") for item in self.criterion_ids)
        )
        if not criteria or len(criteria) != len(set(criteria)):
            raise SavedCohortConflictError(
                "execution criterion identifiers must be non-empty and unique"
            )
        if not isinstance(self.source_snapshot, CohortSourceSnapshot):
            raise TypeError("source_snapshot must be CohortSourceSnapshot")
        if not isinstance(self.expression, Expression):
            raise TypeError("expression must be Expression")
        expression_criteria = {item.id for item in self.expression.iter_criteria()}
        if expression_criteria != set(criteria):
            raise SavedCohortConflictError(
                "execution expression must cover every criterion exactly once"
            )
        _digest(self.vocabulary_digest, "vocabulary_digest")
        _digest(self.policy_digest, "policy_digest")
        _bounded_text(self.evaluator_version, "evaluator_version", 128)
        object.__setattr__(self, "criterion_ids", criteria)
        expected = derived_opaque_id("cohortrun", self.identity_payload)
        if self.execution_id is None:
            object.__setattr__(self, "execution_id", expected)
        elif self.execution_id != expected:
            raise SavedCohortConflictError("execution identifier differs")

    @property
    def identity_payload(self) -> dict[str, Any]:
        """Return exactly the inputs that identify a reproducible run."""

        return {
            "compatibility_policy": self.compatibility_policy,
            "criterion_ids": list(self.criterion_ids),
            "definition_digest": self.definition_digest,
            "definition_version_id": self.definition_version_id,
            "evaluator_version": self.evaluator_version,
            "expression": self.expression.to_dict(),
            "policy_digest": self.policy_digest,
            "schema_version": self.schema_version,
            "source_snapshot": self.source_snapshot.to_dict(),
            "vocabulary_digest": self.vocabulary_digest,
        }

    @property
    def input_digest(self) -> str:
        """Return the canonical digest of all execution inputs."""

        return canonical_digest(self.identity_payload)

    def to_dict(self) -> dict[str, Any]:
        """Return the strict execution manifest."""

        return {
            **self.identity_payload,
            "execution_id": self.execution_id,
            "input_digest": self.input_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CohortExecutionManifest":
        """Parse and verify an execution manifest."""

        data = _mapping(value, "execution manifest")
        _exact_keys(
            data,
            {
                "compatibility_policy",
                "criterion_ids",
                "definition_digest",
                "definition_version_id",
                "evaluator_version",
                "execution_id",
                "expression",
                "input_digest",
                "policy_digest",
                "schema_version",
                "source_snapshot",
                "vocabulary_digest",
            },
            "execution manifest",
        )
        result = cls(
            definition_version_id=_text(
                data["definition_version_id"], "definition_version_id"
            ),
            definition_digest=_text(data["definition_digest"], "definition_digest"),
            criterion_ids=_text_sequence(data["criterion_ids"], "criterion_ids"),
            expression=Expression.from_dict(_mapping(data["expression"], "expression")),
            source_snapshot=CohortSourceSnapshot.from_dict(
                _mapping(data["source_snapshot"], "source_snapshot")
            ),
            vocabulary_digest=_text(data["vocabulary_digest"], "vocabulary_digest"),
            policy_digest=_text(data["policy_digest"], "policy_digest"),
            evaluator_version=_text(data["evaluator_version"], "evaluator_version"),
            execution_id=_text(data["execution_id"], "execution_id"),
            schema_version=_text(data["schema_version"], "schema_version"),
            compatibility_policy=_text(
                data["compatibility_policy"], "compatibility_policy"
            ),
        )
        if data["input_digest"] != result.input_digest:
            raise SavedCohortConflictError("execution input digest differs")
        return result


@dataclass(frozen=True, slots=True)
class CohortExecution:
    """One immutable saved-cohort run and its complete membership evidence."""

    manifest: CohortExecutionManifest
    memberships: tuple[CohortMembership, ...]
    schema_version: str = SAVED_COHORT_SCHEMA_VERSION
    compatibility_policy: str = SAVED_COHORT_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        _contract_version(self.schema_version, self.compatibility_policy)
        if not isinstance(self.manifest, CohortExecutionManifest):
            raise TypeError("manifest must be CohortExecutionManifest")
        if any(not isinstance(item, CohortMembership) for item in self.memberships):
            raise TypeError("memberships must contain CohortMembership")
        memberships = tuple(sorted(self.memberships, key=lambda item: item.patient_key))
        keys = [item.patient_key for item in memberships]
        if len(keys) != len(set(keys)):
            raise SavedCohortConflictError("patient keys must be unique per execution")
        expected_criteria = set(self.manifest.criterion_ids)
        for item in memberships:
            actual = {criterion.criterion_id for criterion in item.criteria}
            if actual != expected_criteria:
                raise SavedCohortConflictError(
                    "membership evidence must cover every criterion exactly once"
                )
            expected_state = evaluate_membership_expression(
                self.manifest.expression,
                item.criteria,
            )
            if item.state is not expected_state:
                raise SavedCohortConflictError(
                    "membership state differs from definition expression"
                )
        object.__setattr__(self, "memberships", memberships)

    @property
    def membership_digest(self) -> str:
        """Return a stable digest over every ordered membership explanation."""

        return canonical_digest([item.to_dict() for item in self.memberships])

    @property
    def execution_digest(self) -> str:
        """Return a stable digest binding run inputs and membership output."""

        return canonical_digest(
            {
                "manifest": self.manifest.to_dict(),
                "membership_digest": self.membership_digest,
            }
        )

    @property
    def review_required(self) -> bool:
        """Return whether the run contains any unknown or conflicting record."""

        return any(item.review_required for item in self.memberships)

    def membership_for(self, patient_key: str) -> CohortMembership | None:
        """Return one opaque-key membership without exposing any source value."""

        _opaque_id(patient_key, "patient_key")
        return next(
            (item for item in self.memberships if item.patient_key == patient_key),
            None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the strict persisted execution."""

        counts = {state.value: 0 for state in MembershipState}
        for item in self.memberships:
            counts[item.state.value] += 1
        return {
            "advisory": SAVED_COHORT_ADVISORY,
            "compatibility_policy": self.compatibility_policy,
            "execution_digest": self.execution_digest,
            "manifest": self.manifest.to_dict(),
            "membership_counts": counts,
            "membership_digest": self.membership_digest,
            "memberships": [item.to_dict() for item in self.memberships],
            "review_required": self.review_required,
            "schema_version": self.schema_version,
        }

    def to_json(self) -> str:
        """Return byte-stable JSON."""

        return canonical_json(self.to_dict())

    def to_json_bytes(self) -> bytes:
        """Return canonical UTF-8 bytes."""

        return self.to_json().encode("utf-8")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CohortExecution":
        """Parse and verify one saved execution."""

        data = _mapping(value, "cohort execution")
        _exact_keys(
            data,
            {
                "advisory",
                "compatibility_policy",
                "execution_digest",
                "manifest",
                "membership_counts",
                "membership_digest",
                "memberships",
                "review_required",
                "schema_version",
            },
            "cohort execution",
        )
        if data["advisory"] != SAVED_COHORT_ADVISORY:
            raise SavedCohortConflictError("cohort advisory differs")
        result = cls(
            manifest=CohortExecutionManifest.from_dict(
                _mapping(data["manifest"], "execution manifest")
            ),
            memberships=tuple(
                CohortMembership.from_dict(_mapping(item, "cohort membership"))
                for item in _sequence(data["memberships"], "memberships")
            ),
            schema_version=_text(data["schema_version"], "schema_version"),
            compatibility_policy=_text(
                data["compatibility_policy"], "compatibility_policy"
            ),
        )
        expected_counts = {state.value: 0 for state in MembershipState}
        for item in result.memberships:
            expected_counts[item.state.value] += 1
        if data["membership_counts"] != expected_counts:
            raise SavedCohortConflictError("persisted membership counts differ")
        if data["membership_digest"] != result.membership_digest:
            raise SavedCohortConflictError("membership digest differs")
        if data["execution_digest"] != result.execution_digest:
            raise SavedCohortConflictError("execution digest differs")
        if data["review_required"] is not result.review_required:
            raise SavedCohortConflictError("execution review state differs")
        return result

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> "CohortExecution":
        """Parse canonical or human-formatted JSON."""

        return cls.from_dict(_json_object(value, "cohort execution"))


@dataclass(frozen=True, slots=True)
class CohortEvaluationContext:
    """Value-free inputs supplied to a deterministic rerun evaluator."""

    definition_version: CohortDefinitionVersion
    manifest: CohortExecutionManifest


CohortEvaluator = Callable[[CohortEvaluationContext], Sequence[CohortMembership]]


def save_cohort_definition(
    definition: PhenotypeDefinition,
) -> CohortDefinitionVersion:
    """Wrap one phenotype in an immutable version contract."""

    return CohortDefinitionVersion(definition=definition)


def build_cohort_execution(
    definition_version: CohortDefinitionVersion,
    *,
    source_snapshot: CohortSourceSnapshot,
    vocabulary_digest: str,
    policy_digest: str,
    evaluator_version: str,
    memberships: Sequence[CohortMembership],
) -> StoreResult[CohortExecution]:
    """Build an execution with typed invalid, unsupported, and conflict states."""

    try:
        if not isinstance(definition_version, CohortDefinitionVersion):
            raise TypeError("definition_version must be CohortDefinitionVersion")
        manifest = CohortExecutionManifest(
            definition_version_id=definition_version.version_id or "",
            definition_digest=definition_version.definition_digest,
            criterion_ids=definition_version.criterion_ids,
            expression=definition_version.definition.expression,
            source_snapshot=source_snapshot,
            vocabulary_digest=vocabulary_digest,
            policy_digest=policy_digest,
            evaluator_version=evaluator_version,
        )
        execution = CohortExecution(manifest=manifest, memberships=tuple(memberships))
    except SavedCohortUnsupportedError:
        return StoreResult.outcome(StoreState.UNSUPPORTED, "cohort_version_unsupported")
    except SavedCohortConflictError:
        return StoreResult.outcome(StoreState.CONFLICT, "cohort_membership_conflict")
    except (SavedCohortError, TypeError, ValueError):
        return StoreResult.outcome(StoreState.FAILURE, "cohort_execution_invalid")
    return StoreResult.success(execution)


class LocalSavedCohortStore:
    """Append-only local persistence for definition versions and executions."""

    def __init__(
        self,
        root: str | Path,
        *,
        policy: StoragePolicy | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve(strict=False)
        if self.root == Path(self.root.anchor):
            raise ValueError("saved cohort root must be bounded")
        self.policy = policy or AllowAllStoragePolicy()
        self._lock = threading.RLock()
        self._prepare()

    def put_definition(
        self, version: CohortDefinitionVersion
    ) -> StoreResult[CohortDefinitionVersion]:
        """Persist one immutable definition version."""

        if not self.policy.allows("write", "cohort_definition"):
            return StoreResult.outcome(StoreState.DENIED, "policy_denied")
        if not isinstance(version, CohortDefinitionVersion):
            return StoreResult.outcome(StoreState.FAILURE, "definition_invalid")
        target = self._definition_path(version.version_id or "")
        return self._put(target, version.to_json_bytes(), version)

    def get_definition(self, version_id: str) -> StoreResult[CohortDefinitionVersion]:
        """Load and integrity-check one immutable definition version."""

        if not self.policy.allows("read", "cohort_definition"):
            return StoreResult.outcome(StoreState.DENIED, "policy_denied")
        try:
            target = self._definition_path(version_id)
        except SavedCohortError:
            return StoreResult.outcome(StoreState.FAILURE, "definition_id_invalid")
        result = self._read(target)
        if not result.ok or result.value is None:
            return StoreResult.outcome(
                result.state, result.code or "definition_read_failed"
            )
        try:
            value = CohortDefinitionVersion.from_json(result.value)
        except SavedCohortUnsupportedError:
            return StoreResult.outcome(
                StoreState.UNSUPPORTED, "cohort_version_unsupported"
            )
        except SavedCohortConflictError:
            return StoreResult.outcome(
                StoreState.CONFLICT, "definition_integrity_failed"
            )
        except (SavedCohortError, TypeError, ValueError):
            return StoreResult.outcome(StoreState.FAILURE, "definition_read_failed")
        return StoreResult.success(value)

    def put_execution(self, execution: CohortExecution) -> StoreResult[CohortExecution]:
        """Persist an execution only when its referenced definition exists."""

        if not self.policy.allows("write", "cohort_execution"):
            return StoreResult.outcome(StoreState.DENIED, "policy_denied")
        if not isinstance(execution, CohortExecution):
            return StoreResult.outcome(StoreState.FAILURE, "execution_invalid")
        version = self.get_definition(execution.manifest.definition_version_id)
        if not version.ok or version.value is None:
            return StoreResult.outcome(
                version.state,
                "definition_not_available"
                if version.state is StoreState.UNKNOWN
                else (version.code or "definition_read_failed"),
            )
        if version.value.definition_digest != execution.manifest.definition_digest:
            return StoreResult.outcome(
                StoreState.CONFLICT, "definition_digest_conflict"
            )
        target = self._execution_path(execution.manifest.execution_id or "")
        return self._put(target, execution.to_json_bytes(), execution)

    def get_execution(self, execution_id: str) -> StoreResult[CohortExecution]:
        """Load and integrity-check one immutable execution."""

        if not self.policy.allows("read", "cohort_execution"):
            return StoreResult.outcome(StoreState.DENIED, "policy_denied")
        try:
            target = self._execution_path(execution_id)
        except SavedCohortError:
            return StoreResult.outcome(StoreState.FAILURE, "execution_id_invalid")
        result = self._read(target)
        if not result.ok or result.value is None:
            return StoreResult.outcome(
                result.state, result.code or "execution_read_failed"
            )
        try:
            value = CohortExecution.from_json(result.value)
        except SavedCohortUnsupportedError:
            return StoreResult.outcome(
                StoreState.UNSUPPORTED, "cohort_version_unsupported"
            )
        except SavedCohortConflictError:
            return StoreResult.outcome(
                StoreState.CONFLICT, "execution_integrity_failed"
            )
        except (SavedCohortError, TypeError, ValueError):
            return StoreResult.outcome(StoreState.FAILURE, "execution_read_failed")
        version = self.get_definition(value.manifest.definition_version_id)
        if not version.ok or version.value is None:
            return StoreResult.outcome(
                version.state,
                "definition_not_available"
                if version.state is StoreState.UNKNOWN
                else (version.code or "definition_read_failed"),
            )
        if version.value.definition_digest != value.manifest.definition_digest:
            return StoreResult.outcome(
                StoreState.CONFLICT, "definition_digest_conflict"
            )
        return StoreResult.success(value)

    def rerun(
        self,
        execution_id: str,
        evaluator: CohortEvaluator,
    ) -> StoreResult[CohortExecution]:
        """Re-evaluate a saved run and prove its membership digest is unchanged."""

        previous = self.get_execution(execution_id)
        if not previous.ok or previous.value is None:
            return previous
        definition = self.get_definition(previous.value.manifest.definition_version_id)
        if not definition.ok or definition.value is None:
            return StoreResult.outcome(
                definition.state,
                definition.code or "definition_read_failed",
            )
        try:
            memberships = evaluator(
                CohortEvaluationContext(
                    definition_version=definition.value,
                    manifest=previous.value.manifest,
                )
            )
            rerun = CohortExecution(
                manifest=previous.value.manifest,
                memberships=tuple(memberships),
            )
        except SavedCohortUnsupportedError:
            return StoreResult.outcome(
                StoreState.UNSUPPORTED, "cohort_version_unsupported"
            )
        except SavedCohortConflictError:
            return StoreResult.outcome(
                StoreState.CONFLICT, "cohort_membership_conflict"
            )
        except Exception:  # noqa: BLE001 - evaluator is caller supplied.
            return StoreResult.outcome(StoreState.FAILURE, "cohort_rerun_failed")
        if rerun.membership_digest != previous.value.membership_digest:
            return StoreResult.outcome(
                StoreState.CONFLICT,
                "cohort_rerun_digest_mismatch",
                value=rerun,
            )
        return StoreResult.success(rerun, created=False)

    def _prepare(self) -> None:
        try:
            self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(self.root, 0o700)
            for name in ("definitions", "executions"):
                directory = self.root / name
                directory.mkdir(mode=0o700, exist_ok=True)
                os.chmod(directory, 0o700)
        except OSError as exc:
            raise RuntimeError("saved cohort store cannot be initialized") from exc

    def _definition_path(self, version_id: str) -> Path:
        _opaque_id(version_id, "version_id")
        if not version_id.startswith("cohortdef_"):
            raise SavedCohortError("version_id has the wrong kind")
        return self.root / "definitions" / f"{version_id}.json"

    def _execution_path(self, execution_id: str) -> Path:
        _opaque_id(execution_id, "execution_id")
        if not execution_id.startswith("cohortrun_"):
            raise SavedCohortError("execution_id has the wrong kind")
        return self.root / "executions" / f"{execution_id}.json"

    def _put(self, target: Path, payload: bytes, value: Any) -> StoreResult[Any]:
        with self._lock:
            existing = self._read(target)
            if existing.ok:
                if existing.value == payload:
                    return StoreResult.success(value, created=False)
                return StoreResult.outcome(
                    StoreState.CONFLICT, "immutable_record_conflict"
                )
            if existing.state is not StoreState.UNKNOWN:
                return StoreResult.outcome(
                    existing.state, existing.code or "record_read_failed"
                )
            try:
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=".saved-cohort-", dir=target.parent
                )
                try:
                    os.fchmod(descriptor, 0o600)
                    with os.fdopen(descriptor, "wb") as stream:
                        stream.write(payload)
                        stream.flush()
                        os.fsync(stream.fileno())
                    try:
                        os.link(temporary_name, target)
                        created = True
                    except FileExistsError:
                        created = False
                    finally:
                        Path(temporary_name).unlink(missing_ok=True)
                except BaseException:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
                    Path(temporary_name).unlink(missing_ok=True)
                    raise
            except OSError:
                return StoreResult.outcome(StoreState.FAILURE, "record_write_failed")
            verified = self._read(target)
            if not verified.ok or verified.value != payload:
                if created:
                    target.unlink(missing_ok=True)
                return StoreResult.outcome(StoreState.CONFLICT, "record_verify_failed")
            return StoreResult.success(value, created=created)

    def _read(self, target: Path) -> StoreResult[bytes]:
        try:
            if target.is_symlink():
                return StoreResult.outcome(StoreState.FAILURE, "record_path_unsafe")
            return StoreResult.success(target.read_bytes())
        except FileNotFoundError:
            return StoreResult.outcome(StoreState.UNKNOWN, "record_not_found")
        except OSError:
            return StoreResult.outcome(StoreState.FAILURE, "record_read_failed")


def load_saved_cohort_schema() -> dict[str, Any]:
    """Load the bundled JSON Schema for saved definition and execution records."""

    text = (
        resources.files(SAVED_COHORT_SCHEMA_PACKAGE)
        .joinpath(f"{SAVED_COHORT_SCHEMA_NAME}.schema.json")
        .read_text(encoding="utf-8")
    )
    value = json.loads(text)
    if not isinstance(value, dict):  # pragma: no cover - packaged invariant
        raise RuntimeError("saved cohort schema must be an object")
    return value


def evaluate_membership_expression(
    expression: Expression,
    criteria: Sequence[CriterionMembership],
) -> MembershipState:
    """Evaluate the cohort expression with conservative four-state logic.

    Conflict and unknown dominate resolved truth values.  This means an
    unresolved branch can never be hidden by a matching ``or`` branch and
    accidentally become eligible without review.
    """

    by_id = {item.criterion_id: item.state for item in criteria}

    def visit(node: Expression) -> MembershipState:
        if node.operator == "criterion":
            if node.criterion is None:  # pragma: no cover - DSL invariant
                raise SavedCohortConflictError("criterion expression is incomplete")
            try:
                return by_id[node.criterion.id]
            except KeyError:
                raise SavedCohortConflictError(
                    "membership is missing expression criterion"
                ) from None
        states = tuple(visit(child) for child in node.children)
        if MembershipState.CONFLICT in states:
            return MembershipState.CONFLICT
        if MembershipState.UNKNOWN in states:
            return MembershipState.UNKNOWN
        if node.operator == "not":
            return (
                MembershipState.NOT_MET
                if states[0] is MembershipState.MET
                else MembershipState.MET
            )
        matched = (
            all(state is MembershipState.MET for state in states)
            if node.operator == "and"
            else any(state is MembershipState.MET for state in states)
        )
        return MembershipState.MET if matched else MembershipState.NOT_MET

    return visit(expression)


def _contract_version(schema_version: str, compatibility_policy: str) -> None:
    if schema_version != SAVED_COHORT_SCHEMA_VERSION:
        raise SavedCohortUnsupportedError("saved cohort schema version is unsupported")
    if compatibility_policy != SAVED_COHORT_COMPATIBILITY_POLICY:
        raise SavedCohortUnsupportedError(
            "saved cohort compatibility policy is unsupported"
        )


def _json_object(value: str | bytes | bytearray, name: str) -> Mapping[str, Any]:
    try:
        data = json.loads(value)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        raise SavedCohortError(f"{name} is not valid JSON") from None
    return _mapping(data, name)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SavedCohortError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise SavedCohortError(f"{name} must be an array")
    return value


def _exact_keys(data: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(data) != expected:
        raise SavedCohortError(f"{name} fields are incompatible")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SavedCohortError(f"{name} must be non-empty text")
    return value


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _text_sequence(value: Any, name: str) -> tuple[str, ...]:
    return tuple(_text(item, name) for item in _sequence(value, name))


def _bounded_text(value: Any, name: str, maximum: int) -> str:
    text = _text(value, name)
    if len(text) > maximum or any(ord(character) < 32 for character in text):
        raise SavedCohortError(f"{name} must be bounded text")
    return text


def _controlled(value: Any, name: str) -> str:
    text = _text(value, name)
    if _CONTROLLED_RE.fullmatch(text) is None:
        raise SavedCohortError(f"{name} must be controlled")
    return text


def _opaque_id(value: Any, name: str) -> str:
    text = _text(value, name)
    if _OPAQUE_ID_RE.fullmatch(text) is None:
        raise SavedCohortError(f"{name} must be opaque")
    return text


def _digest(value: Any, name: str) -> str:
    text = _text(value, name)
    if _DIGEST_RE.fullmatch(text) is None:
        raise SavedCohortError(f"{name} must be a digest")
    return text


def _membership_state(value: Any, name: str) -> MembershipState:
    try:
        return value if isinstance(value, MembershipState) else MembershipState(value)
    except (TypeError, ValueError):
        raise SavedCohortError(f"{name} is unsupported") from None


__all__ = [
    "SAVED_COHORT_ADVISORY",
    "SAVED_COHORT_COMPATIBILITY_POLICY",
    "SAVED_COHORT_SCHEMA_NAME",
    "SAVED_COHORT_SCHEMA_VERSION",
    "CohortDefinitionVersion",
    "CohortEvaluationContext",
    "CohortEvaluator",
    "CohortExecution",
    "CohortExecutionManifest",
    "CohortMembership",
    "CriterionMembership",
    "LocalSavedCohortStore",
    "MembershipEvidence",
    "MembershipState",
    "SavedCohortConflictError",
    "SavedCohortError",
    "SavedCohortUnsupportedError",
    "build_cohort_execution",
    "evaluate_membership_expression",
    "load_saved_cohort_schema",
    "save_cohort_definition",
]
