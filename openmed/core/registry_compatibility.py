"""Offline compatibility checks for model-registry rollback candidates.

The registry can identify a last-green checkpoint without proving that it can
reproduce the current output contract.  This module compares the metadata
needed for that proof.  It never loads a model, reads a checkpoint directory,
or contacts a registry service.

Reports are deliberately safe to persist: model identifiers, policy values,
tokenizer IDs, and lineage values are represented by stable digests rather
than copied into reports or error messages.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias

COMPATIBILITY_REPORT_SCHEMA_VERSION = 1
DECISION_COMPATIBLE = "compatible"
DECISION_BLOCKED = "blocked"

_MISSING = object()
_SEMVER_RE = re.compile(
    r"^v?(?P<major>0|[1-9][0-9]*)"
    r"(?:\.(?P<minor>0|[1-9][0-9]*))?"
    r"(?:\.(?P<patch>0|[1-9][0-9]*))?"
    r"(?:\+[0-9A-Za-z.-]+)?$"
)
_SEMVER_FROM_ID_RE = re.compile(
    r"(?:^|[-_/])v(?P<major>[0-9]+)"
    r"(?:\.(?P<minor>[0-9]+))?"
    r"(?:\.(?P<patch>[0-9]+))?(?:[-_/]|$)",
    re.IGNORECASE,
)
_CONSTRAINT_RE = re.compile(r"^(?P<operator>\^|~=|~|>=|<=|==|!=|>|<|=)?(?P<value>.+)$")
_CONSTRAINT_SPLIT_RE = re.compile(r"\s*,\s*|\s+(?=[<>=~^])")
_WILDCARD_VALUES = frozenset({"*", "x", "X"})
_LINEAGE_FORWARD_RELATIONS = frozenset(
    {"supersedes", "superseded-by", "promotes", "promoted-from", "derived-from"}
)
_LINEAGE_ROLLBACK_RELATIONS = frozenset({"rolled-back-from", "rollback"})


class RegistryCompatibilityError(ValueError):
    """Raised when a compatibility input cannot be represented safely."""


@dataclass(frozen=True, order=True)
class _Version:
    major: int
    minor: int
    patch: int

    def text(self) -> str:
        """Return the normalized SemVer text."""

        return f"{self.major}.{self.minor}.{self.patch}"


@dataclass(frozen=True)
class RegistryLineageEdge:
    """A safe, metadata-only lineage relation between two checkpoints."""

    relation: str
    source: str
    target: str

    def to_dict(self) -> dict[str, str]:
        """Return a safe representation with hashed checkpoint references."""

        return {
            "relation": _safe_lineage_relation(self.relation),
            "source_ref": _safe_digest(self.source),
            "target_ref": _safe_digest(self.target),
        }


@dataclass(frozen=True)
class RegistryCheckpoint:
    """Metadata required to assess one registry checkpoint.

    ``policy_fingerprint``, ``tokenizer_ids``, and ``evidence_schema_versions``
    may be supplied as already-produced metadata or as synthetic structures.
    Structures are hashed during comparison and are never copied into a
    compatibility report.
    """

    model_id: str
    family: str | None = None
    version: str | None = None
    semver_constraint: str | Sequence[str] | None = None
    lineage: Any = field(default_factory=tuple)
    policy_fingerprint: Any = None
    policy_schema_version: Any = None
    tokenizer_ids: Any = None
    evidence_schema_versions: Any = field(default_factory=tuple)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RegistryCheckpoint":
        """Build a checkpoint from common registry metadata field names."""

        if not isinstance(value, Mapping):
            raise RegistryCompatibilityError("checkpoint metadata must be an object")
        nested = value.get("metadata")
        payload = dict(nested) if isinstance(nested, Mapping) else {}
        payload.update(value)
        return cls(
            model_id=_string_or_none(
                _first_value(payload, "model_id", "repo_id", "artifact_id")
            )
            or "",
            family=_string_or_none(_first_value(payload, "family")),
            version=_string_or_none(
                _first_value(
                    payload,
                    "version",
                    "registry_version",
                    "semver",
                    "semantic_version",
                )
            ),
            semver_constraint=_first_value(
                payload,
                "semver_constraint",
                "semver_constraints",
                "version_constraint",
                "required_semver",
            ),
            lineage=_first_value(payload, "lineage", "ancestors", "parents")
            if _first_value(payload, "lineage", "ancestors", "parents") is not _MISSING
            else (),
            policy_fingerprint=_first_value(
                payload,
                "policy_fingerprint",
                "policy_hash",
                "policy_schema_fingerprint",
                "policy",
            ),
            policy_schema_version=_first_value(
                payload,
                "policy_schema_version",
                "policy_version",
            ),
            tokenizer_ids=_first_value(
                payload,
                "tokenizer_ids",
                "tokenizer_id",
                "tokenizer_fingerprint",
                "tokenizer_contract",
                "tokenizer",
            ),
            evidence_schema_versions=_first_value(
                payload,
                "evidence_schema_versions",
                "evidence_schema_version",
                "evidence_schema",
                "schema_versions",
            )
            if _first_value(
                payload,
                "evidence_schema_versions",
                "evidence_schema_version",
                "evidence_schema",
                "schema_versions",
            )
            is not _MISSING
            else (),
        )

    def to_safe_dict(self) -> dict[str, Any]:
        """Return checkpoint metadata without raw identifiers or contracts."""

        normalised = _normalise_checkpoint(self, "checkpoint")
        safe_version = normalised.version
        if safe_version is not None and _SEMVER_RE.fullmatch(safe_version) is None:
            safe_version = _safe_digest(safe_version)
        return {
            "model_ref": _safe_digest(normalised.model_id),
            "family_ref": _safe_digest(normalised.family),
            "version": safe_version,
            "semver_constraint_ref": _safe_digest(normalised.constraints),
            "lineage_refs": [edge.to_dict() for edge in normalised.lineage_edges],
            "policy_fingerprint_ref": normalised.policy_ref,
            "tokenizer_contract_ref": normalised.tokenizer_ref,
            "evidence_schema_ref": _safe_digest(normalised.evidence_versions),
        }


CheckpointInput: TypeAlias = RegistryCheckpoint | Mapping[str, Any] | str


@dataclass(frozen=True)
class RegistryCompatibilityCheck:
    """One deterministic compatibility dimension result."""

    name: str
    passed: bool
    code: str
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        """Return ``pass`` or ``blocked`` for this dimension."""

        return "pass" if self.passed else DECISION_BLOCKED

    @property
    def compatible(self) -> bool:
        """Return whether this dimension is compatible."""

        return self.passed

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible check result."""

        return {
            "status": self.status,
            "passed": self.passed,
            "code": self.code,
            "details": _safe_json_value(self.details),
        }


@dataclass(frozen=True)
class RegistryCompatibilityReport:
    """Deterministic, PHI-safe rollback compatibility report."""

    _check_results: tuple[RegistryCompatibilityCheck, ...]
    schema_version: int = COMPATIBILITY_REPORT_SCHEMA_VERSION

    @property
    def checks(self) -> Mapping[str, RegistryCompatibilityCheck]:
        """Return checks keyed by compatibility dimension."""

        return {check.name: check for check in self._check_results}

    @property
    def check_results(self) -> tuple[RegistryCompatibilityCheck, ...]:
        """Return checks in their stable evaluation order."""

        return self._check_results

    @property
    def blocked_reasons(self) -> tuple[str, ...]:
        """Return stable reason codes for every blocked dimension."""

        return tuple(check.code for check in self._check_results if not check.passed)

    @property
    def reason_codes(self) -> tuple[str, ...]:
        """Alias for :attr:`blocked_reasons`."""

        return self.blocked_reasons

    @property
    def reasons(self) -> tuple[str, ...]:
        """Alias for :attr:`blocked_reasons`."""

        return self.blocked_reasons

    @property
    def compatible(self) -> bool:
        """Return whether every required contract comparison passed."""

        return all(check.passed for check in self._check_results)

    @property
    def is_compatible(self) -> bool:
        """Return whether every required contract comparison passed."""

        return self.compatible

    @property
    def blocked(self) -> bool:
        """Return whether rollback must be blocked."""

        return not self.compatible

    @property
    def decision(self) -> str:
        """Return ``compatible`` or ``blocked``."""

        return DECISION_COMPATIBLE if self.compatible else DECISION_BLOCKED

    @property
    def status(self) -> str:
        """Return the report decision as a status string."""

        return self.decision

    @property
    def fingerprint(self) -> str:
        """Return the deterministic report fingerprint."""

        return _safe_digest(self._payload_without_fingerprint())

    @property
    def report_fingerprint(self) -> str:
        """Alias for :attr:`fingerprint`."""

        return self.fingerprint

    def check(self, name: str) -> RegistryCompatibilityCheck:
        """Return one named check without exposing arbitrary input values."""

        try:
            return self.checks[name]
        except KeyError as exc:
            raise KeyError(f"unknown compatibility check: {name!r}") from exc

    def to_dict(self) -> dict[str, Any]:
        """Return the safe, deterministic report payload."""

        payload = self._payload_without_fingerprint()
        payload["fingerprint"] = self.fingerprint
        return payload

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the report as deterministic JSON."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            indent=indent,
            sort_keys=True,
        )

    def to_markdown(self) -> str:
        """Render a compact safe report suitable for an audit attachment."""

        lines = [
            "# Rollback compatibility report",
            "",
            f"- Decision: `{self.decision}`",
            f"- Fingerprint: `{self.fingerprint}`",
            "",
            "| Check | Status | Code |",
            "| --- | --- | --- |",
        ]
        lines.extend(
            f"| `{check.name}` | `{check.status}` | `{check.code}` |"
            for check in self._check_results
        )
        return "\n".join(lines) + "\n"

    def _payload_without_fingerprint(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "decision": self.decision,
            "status": self.status,
            "compatible": self.compatible,
            "blocked_reasons": list(self.blocked_reasons),
            "checks": {check.name: check.to_dict() for check in self._check_results},
        }

    def __bool__(self) -> bool:
        """Use the report as its compatibility decision in boolean contexts."""

        return self.compatible


# Short aliases make the report useful to callers that use the generic
# compatibility vocabulary while retaining the registry-specific API name.
CompatibilityCheck = RegistryCompatibilityCheck
CompatibilityReport = RegistryCompatibilityReport
RollbackCompatibilityReport = RegistryCompatibilityReport
RegistryCompatibilityInput = RegistryCheckpoint


def build_rollback_compatibility_report(
    current: CheckpointInput | None = None,
    rollback: CheckpointInput | None = None,
    *,
    target: CheckpointInput | None = None,
    candidate: CheckpointInput | None = None,
    registry_state: Mapping[str, Any] | None = None,
    family: str | None = None,
    requirements: Mapping[str, Any] | None = None,
    constraints: Mapping[str, Any] | None = None,
    semver_constraint: str | Sequence[str] | None | object = _MISSING,
    expected_policy_fingerprint: Any = _MISSING,
    expected_tokenizer_ids: Any = _MISSING,
    expected_evidence_schema_versions: Any = _MISSING,
    lineage: Any = _MISSING,
) -> RegistryCompatibilityReport:
    """Compare a current checkpoint with its proposed rollback checkpoint.

    Inputs are local metadata mappings, not model paths.  The evaluator is
    fail-closed: lineage, SemVer, policy, tokenizer, and evidence-schema
    metadata must all be provably compatible.  Optional ``requirements`` or
    ``constraints`` mappings may provide the expected SemVer or contract when
    the current checkpoint does not carry it.

    When ``registry_state`` and ``family`` are supplied, omitted checkpoints
    are derived from that state's ``latest`` and ``last_green`` pointers.  The
    state is used as metadata only; no files or network resources are read.
    """

    if target is not None:
        if current is not None:
            raise RegistryCompatibilityError("provide current or target, not both")
        current = target
    if candidate is not None:
        if rollback is not None:
            raise RegistryCompatibilityError("provide rollback or candidate, not both")
        rollback = candidate

    merged_requirements = _merge_requirement_mappings(requirements, constraints)
    if registry_state is not None:
        current, rollback = _state_checkpoints(
            registry_state,
            family=family,
            current=current,
            rollback=rollback,
        )

    current_checkpoint = _normalise_checkpoint(current, "current")
    rollback_checkpoint = _normalise_checkpoint(rollback, "rollback")

    if semver_constraint is _MISSING:
        semver_constraint = _first_value(
            merged_requirements,
            "semver_constraint",
            "semver_constraints",
            "version_constraint",
            "required_semver",
        )
    if expected_policy_fingerprint is _MISSING:
        expected_policy_fingerprint = _first_value(
            merged_requirements,
            "policy_fingerprint",
            "policy_hash",
            "policy_schema_fingerprint",
        )
    if expected_tokenizer_ids is _MISSING:
        expected_tokenizer_ids = _first_value(
            merged_requirements,
            "tokenizer_ids",
            "tokenizer_id",
            "tokenizer_fingerprint",
            "tokenizer_contract",
        )
    if expected_evidence_schema_versions is _MISSING:
        expected_evidence_schema_versions = _first_value(
            merged_requirements,
            "evidence_schema_versions",
            "evidence_schema_version",
            "evidence_schema",
        )

    checks = (
        _identity_check(current_checkpoint, rollback_checkpoint),
        _family_check(current_checkpoint, rollback_checkpoint, family),
        _lineage_check(
            current_checkpoint,
            rollback_checkpoint,
            explicit_lineage=None if lineage is _MISSING else lineage,
        ),
        _semver_check(
            current_checkpoint,
            rollback_checkpoint,
            explicit_constraint=semver_constraint,
        ),
        _fingerprint_check(
            "policy_fingerprint",
            "POLICY_FINGERPRINT",
            current_checkpoint.policy_ref
            if expected_policy_fingerprint is _MISSING
            else _fingerprint_value(expected_policy_fingerprint),
            rollback_checkpoint.policy_ref,
        ),
        _fingerprint_check(
            "tokenizer_contract",
            "TOKENIZER_CONTRACT",
            current_checkpoint.tokenizer_ref
            if expected_tokenizer_ids is _MISSING
            else _fingerprint_value(expected_tokenizer_ids),
            rollback_checkpoint.tokenizer_ref,
        ),
        _evidence_schema_check(
            current_checkpoint.evidence_versions
            if expected_evidence_schema_versions is _MISSING
            else _normalise_evidence_versions(expected_evidence_schema_versions),
            rollback_checkpoint.evidence_versions,
        ),
    )
    return RegistryCompatibilityReport(checks)


def assess_rollback_compatibility(
    current: CheckpointInput | None = None,
    rollback: CheckpointInput | None = None,
    **kwargs: Any,
) -> RegistryCompatibilityReport:
    """Alias for :func:`build_rollback_compatibility_report`."""

    return build_rollback_compatibility_report(current, rollback, **kwargs)


def compare_rollback_compatibility(
    current: CheckpointInput | None = None,
    rollback: CheckpointInput | None = None,
    **kwargs: Any,
) -> RegistryCompatibilityReport:
    """Alias for :func:`build_rollback_compatibility_report`."""

    return build_rollback_compatibility_report(current, rollback, **kwargs)


def check_rollback_compatibility(
    current: CheckpointInput | None = None,
    rollback: CheckpointInput | None = None,
    **kwargs: Any,
) -> RegistryCompatibilityReport:
    """Alias for :func:`build_rollback_compatibility_report`."""

    return build_rollback_compatibility_report(current, rollback, **kwargs)


def _identity_check(
    current: "_NormalisedCheckpoint", rollback: "_NormalisedCheckpoint"
) -> RegistryCompatibilityCheck:
    if current.input_error or rollback.input_error:
        return _blocked(
            "identity",
            "INPUT_INVALID",
            input_errors=(current.input_error, rollback.input_error),
        )
    if not current.model_id or not rollback.model_id:
        return _blocked(
            "identity",
            "MODEL_ID_MISSING",
            current_present=bool(current.model_id),
            rollback_present=bool(rollback.model_id),
        )
    return _passed(
        "identity",
        "MODEL_IDENTIFIED",
        current_ref=_safe_digest(current.model_id),
        rollback_ref=_safe_digest(rollback.model_id),
    )


def _family_check(
    current: "_NormalisedCheckpoint",
    rollback: "_NormalisedCheckpoint",
    requested_family: str | None,
) -> RegistryCompatibilityCheck:
    if not current.family or not rollback.family:
        return _blocked(
            "family",
            "FAMILY_MISSING",
            current_present=bool(current.family),
            rollback_present=bool(rollback.family),
        )
    if current.family.casefold() != rollback.family.casefold():
        return _blocked(
            "family",
            "FAMILY_MISMATCH",
            current_ref=_safe_digest(current.family),
            rollback_ref=_safe_digest(rollback.family),
        )
    if requested_family and current.family.casefold() != requested_family.casefold():
        return _blocked(
            "family",
            "REQUESTED_FAMILY_MISMATCH",
            requested_ref=_safe_digest(requested_family),
            observed_ref=_safe_digest(current.family),
        )
    return _passed(
        "family",
        "FAMILY_MATCH",
        family_ref=_safe_digest(current.family),
    )


def _lineage_check(
    current: "_NormalisedCheckpoint",
    rollback: "_NormalisedCheckpoint",
    *,
    explicit_lineage: Any,
) -> RegistryCompatibilityCheck:
    if not current.model_id or not rollback.model_id:
        return _blocked("lineage", "LINEAGE_UNPROVEN")
    if current.model_id == rollback.model_id:
        return _passed(
            "lineage",
            "SAME_CHECKPOINT",
            checkpoint_ref=_safe_digest(current.model_id),
        )

    edges = list(current.lineage_edges) + list(rollback.lineage_edges)
    lineage_ids = set(current.lineage_ids) | set(rollback.lineage_ids)
    if explicit_lineage is not None:
        extra_edges, extra_ids = _parse_lineage(explicit_lineage)
        edges.extend(extra_edges)
        lineage_ids.update(extra_ids)

    if rollback.model_id in current.lineage_ids:
        return _passed(
            "lineage",
            "ROLLBACK_ANCESTOR",
            current_ref=_safe_digest(current.model_id),
            rollback_ref=_safe_digest(rollback.model_id),
        )
    if rollback.model_id in lineage_ids and not edges:
        return _passed(
            "lineage",
            "ROLLBACK_ANCESTOR",
            current_ref=_safe_digest(current.model_id),
            rollback_ref=_safe_digest(rollback.model_id),
        )

    parents: dict[str, set[str]] = {}
    for edge in edges:
        relation = edge.relation.casefold()
        if relation in _LINEAGE_FORWARD_RELATIONS:
            parent, child = edge.source, edge.target
        elif relation in _LINEAGE_ROLLBACK_RELATIONS:
            parent, child = edge.target, edge.source
        else:
            continue
        parents.setdefault(child, set()).add(parent)

    reachable = {current.model_id}
    frontier = [current.model_id]
    while frontier:
        child = frontier.pop()
        for parent in sorted(parents.get(child, ())):
            if parent not in reachable:
                reachable.add(parent)
                frontier.append(parent)
    if rollback.model_id in reachable:
        return _passed(
            "lineage",
            "ROLLBACK_ANCESTOR",
            current_ref=_safe_digest(current.model_id),
            rollback_ref=_safe_digest(rollback.model_id),
        )
    return _blocked(
        "lineage",
        "LINEAGE_NOT_ANCESTOR",
        current_ref=_safe_digest(current.model_id),
        rollback_ref=_safe_digest(rollback.model_id),
        lineage_ref=_safe_digest(
            {
                "ids": sorted(lineage_ids),
                "edges": [edge.to_dict() for edge in edges],
            }
        ),
    )


def _semver_check(
    current: "_NormalisedCheckpoint",
    rollback: "_NormalisedCheckpoint",
    *,
    explicit_constraint: Any,
) -> RegistryCompatibilityCheck:
    if current.constraint_error or rollback.constraint_error:
        return _blocked(
            "semver",
            "SEMVER_CONSTRAINT_INVALID",
            constraint_ref=current.constraint_ref or rollback.constraint_ref,
        )
    if not current.version or not rollback.version:
        return _blocked(
            "semver",
            "SEMVER_MISSING",
            current_present=bool(current.version),
            rollback_present=bool(rollback.version),
        )
    try:
        current_version = _parse_version(current.version)
        rollback_version = _parse_version(rollback.version)
    except ValueError:
        return _blocked(
            "semver",
            "SEMVER_INVALID",
            current_ref=_safe_digest(current.version),
            rollback_ref=_safe_digest(rollback.version),
        )

    constraints: list[str] = list(current.constraints) + list(rollback.constraints)
    if explicit_constraint is not _MISSING and explicit_constraint is not None:
        explicit = _normalise_constraints(explicit_constraint)
        if explicit is None:
            return _blocked(
                "semver",
                "SEMVER_CONSTRAINT_INVALID",
                constraint_ref=_safe_digest(explicit_constraint),
            )
        constraints.extend(explicit)

    if not constraints:
        if current_version == rollback_version:
            return _passed(
                "semver",
                "SEMVER_EXACT_MATCH",
                current_version=current_version.text(),
                rollback_version=rollback_version.text(),
            )
        return _blocked(
            "semver",
            "SEMVER_CONSTRAINT_MISSING",
            current_version=current_version.text(),
            rollback_version=rollback_version.text(),
        )

    for constraint in constraints:
        try:
            matches = _satisfies(rollback_version, constraint)
        except ValueError:
            return _blocked(
                "semver",
                "SEMVER_CONSTRAINT_INVALID",
                constraint_ref=_safe_digest(constraint),
            )
        if not matches:
            return _blocked(
                "semver",
                "SEMVER_OUT_OF_RANGE",
                current_version=current_version.text(),
                rollback_version=rollback_version.text(),
                constraint_ref=_safe_digest(constraint),
            )
    return _passed(
        "semver",
        "SEMVER_CONSTRAINT_SATISFIED",
        current_version=current_version.text(),
        rollback_version=rollback_version.text(),
        constraint_ref=_safe_digest(tuple(constraints)),
    )


def _fingerprint_check(
    name: str,
    prefix: str,
    expected: str | None,
    observed: str | None,
) -> RegistryCompatibilityCheck:
    if not expected or not observed:
        return _blocked(
            name,
            f"{prefix}_MISSING",
            expected_present=bool(expected),
            observed_present=bool(observed),
        )
    if expected != observed:
        return _blocked(
            name,
            f"{prefix}_MISMATCH",
            expected_ref=expected,
            observed_ref=observed,
        )
    return _passed(name, f"{prefix}_MATCH", fingerprint_ref=expected)


def _evidence_schema_check(
    expected: tuple[str, ...] | None,
    observed: tuple[str, ...] | None,
) -> RegistryCompatibilityCheck:
    if not expected or not observed:
        return _blocked(
            "evidence_schema",
            "EVIDENCE_SCHEMA_MISSING",
            expected_present=bool(expected),
            observed_present=bool(observed),
        )
    if expected != observed:
        return _blocked(
            "evidence_schema",
            "EVIDENCE_SCHEMA_MISMATCH",
            expected_ref=_safe_digest(expected),
            observed_ref=_safe_digest(observed),
        )
    return _passed(
        "evidence_schema",
        "EVIDENCE_SCHEMA_MATCH",
        schema_ref=_safe_digest(expected),
    )


@dataclass(frozen=True)
class _NormalisedCheckpoint:
    model_id: str | None
    family: str | None
    version: str | None
    constraints: tuple[str, ...]
    constraint_error: bool
    constraint_ref: str | None
    lineage_edges: tuple[RegistryLineageEdge, ...]
    lineage_ids: frozenset[str]
    policy_ref: str | None
    tokenizer_ref: str | None
    evidence_versions: tuple[str, ...] | None
    input_error: str | None = None


def _normalise_checkpoint(
    value: CheckpointInput | None,
    label: str,
) -> _NormalisedCheckpoint:
    if value is None:
        return _NormalisedCheckpoint(
            None,
            None,
            None,
            (),
            False,
            None,
            (),
            frozenset(),
            None,
            None,
            None,
            f"{label}_metadata_missing",
        )
    if isinstance(value, RegistryCheckpoint):
        checkpoint = value
    elif isinstance(value, Mapping):
        try:
            checkpoint = RegistryCheckpoint.from_mapping(value)
        except (TypeError, ValueError):
            return _NormalisedCheckpoint(
                None,
                None,
                None,
                (),
                False,
                None,
                (),
                frozenset(),
                None,
                None,
                None,
                f"{label}_metadata_invalid",
            )
    elif isinstance(value, str):
        checkpoint = RegistryCheckpoint(model_id=value)
    else:
        return _NormalisedCheckpoint(
            None,
            None,
            None,
            (),
            False,
            None,
            (),
            frozenset(),
            None,
            None,
            None,
            f"{label}_metadata_invalid",
        )

    model_id = _identifier_or_none(checkpoint.model_id)
    family = _identifier_or_none(checkpoint.family)
    version = _normalise_version_text(checkpoint.version)
    if version is None and model_id:
        version = _version_from_model_id(model_id)
    constraints = _normalise_constraints(checkpoint.semver_constraint)
    constraint_error = checkpoint.semver_constraint is not None and constraints is None
    constraint_ref = _safe_digest(checkpoint.semver_constraint)
    edges, lineage_ids = _parse_lineage(checkpoint.lineage)
    policy_value = checkpoint.policy_fingerprint
    if policy_value is _MISSING or policy_value is None:
        policy_ref = None
    elif (
        checkpoint.policy_schema_version is _MISSING
        or checkpoint.policy_schema_version is None
    ):
        policy_ref = _fingerprint_value(policy_value)
    else:
        policy_ref = _fingerprint_value(
            {
                "fingerprint": policy_value,
                "schema_version": checkpoint.policy_schema_version,
            }
        )
    tokenizer_ref = _fingerprint_value(checkpoint.tokenizer_ids)
    evidence_versions = _normalise_evidence_versions(
        checkpoint.evidence_schema_versions
    )
    return _NormalisedCheckpoint(
        model_id=model_id,
        family=family,
        version=version,
        constraints=constraints or (),
        constraint_error=constraint_error,
        constraint_ref=constraint_ref,
        lineage_edges=tuple(edges),
        lineage_ids=frozenset(lineage_ids),
        policy_ref=policy_ref,
        tokenizer_ref=tokenizer_ref,
        evidence_versions=evidence_versions,
    )


def _state_checkpoints(
    state: Mapping[str, Any],
    *,
    family: str | None,
    current: CheckpointInput | None,
    rollback: CheckpointInput | None,
) -> tuple[CheckpointInput | None, CheckpointInput | None]:
    if not isinstance(state, Mapping):
        return current, rollback
    families = state.get("families")
    if not isinstance(families, Mapping) or not family:
        return current, rollback
    entry = next(
        (
            value
            for key, value in families.items()
            if str(key).casefold() == family.casefold()
        ),
        None,
    )
    if not isinstance(entry, Mapping):
        return current, rollback
    pointers = entry.get("pointers")
    versions = entry.get("versions")
    if not isinstance(pointers, Mapping):
        return current, rollback
    versions = versions if isinstance(versions, Mapping) else {}
    state_lineage = entry.get("lineage", ())
    metadata_by_id = state.get("checkpoints") or state.get("artifacts") or {}
    if not isinstance(metadata_by_id, Mapping):
        metadata_by_id = {}

    def make_record(
        value: CheckpointInput | None, pointer_name: str
    ) -> CheckpointInput | None:
        if value is not None:
            if isinstance(value, str):
                base = metadata_by_id.get(value)
                payload = dict(base) if isinstance(base, Mapping) else {}
                payload.setdefault("model_id", value)
                payload.setdefault("family", family)
                payload.setdefault("version", versions.get(value))
                if not payload.get("lineage"):
                    payload["lineage"] = state_lineage
                return payload
            return value
        model_id = pointers.get(pointer_name)
        if not isinstance(model_id, str) or not model_id:
            return None
        base = metadata_by_id.get(model_id)
        payload = dict(base) if isinstance(base, Mapping) else {}
        payload.setdefault("model_id", model_id)
        payload.setdefault("family", family)
        payload.setdefault("version", versions.get(model_id))
        if not payload.get("lineage"):
            payload["lineage"] = state_lineage
        return payload

    return make_record(current, "latest"), make_record(rollback, "last_green")


def _parse_lineage(
    value: Any, *, _depth: int = 0
) -> tuple[list[RegistryLineageEdge], set[str]]:
    if value is None or value is _MISSING or _depth > 5:
        return [], set()
    edges: list[RegistryLineageEdge] = []
    identifiers: set[str] = set()
    if isinstance(value, str):
        identifier = _identifier_or_none(value)
        if identifier:
            identifiers.add(identifier)
        return edges, identifiers
    if isinstance(value, Mapping):
        source = _identifier_or_none(
            _first_value(value, "from", "source", "parent", "parent_model_id")
        )
        target = _identifier_or_none(_first_value(value, "to", "target", "child"))
        relation = _identifier_or_none(value.get("relation")) or "supersedes"
        if source and target:
            edges.append(RegistryLineageEdge(relation, source, target))
        for key in ("lineage", "ancestors", "parents", "edges"):
            nested = value.get(key)
            if nested is not None and nested is not value:
                nested_edges, nested_ids = _parse_lineage(nested, _depth=_depth + 1)
                edges.extend(nested_edges)
                identifiers.update(nested_ids)
        return edges, identifiers
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for item in value:
            nested_edges, nested_ids = _parse_lineage(item, _depth=_depth + 1)
            edges.extend(nested_edges)
            identifiers.update(nested_ids)
        return edges, identifiers
    return edges, identifiers


def _normalise_constraints(value: Any) -> tuple[str, ...] | None:
    if value is None or value is _MISSING:
        return ()
    values: tuple[str, ...]
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = tuple(item for item in value if isinstance(item, str))
        if len(values) != len(value):
            return None
    else:
        return None
    normalised = tuple(item.strip() for item in values if item.strip())
    return normalised if len(normalised) == len(values) else None


def _normalise_evidence_versions(value: Any) -> tuple[str, ...] | None:
    if value is None or value is _MISSING:
        return None
    values: tuple[str, ...]
    if isinstance(value, Mapping):
        nested = _first_value(value, "versions", "schema_versions")
        if nested is not _MISSING:
            return _normalise_evidence_versions(nested)
        return (_safe_digest(value),)
    if isinstance(value, str) or isinstance(value, int):
        values = (str(value),)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = tuple(str(item) for item in value)
    else:
        return (_safe_digest(value),)
    values = tuple(item for item in values if item)
    return tuple(sorted(set(values))) or None


def _fingerprint_value(value: Any) -> str | None:
    if value is None or value is _MISSING:
        return None
    if isinstance(value, str) and not value:
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            return None
    return _safe_digest(value)


def _merge_requirement_mappings(
    first: Mapping[str, Any] | None,
    second: Mapping[str, Any] | None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for mapping in (first, second):
        if isinstance(mapping, Mapping):
            merged.update(mapping)
    return merged


def _first_value(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return _MISSING


def _string_or_none(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    return value if isinstance(value, str) else None


def _identifier_or_none(value: Any) -> str | None:
    if not isinstance(value, str) or not value or "\n" in value or "\r" in value:
        return None
    return value


def _normalise_version_text(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _version_from_model_id(model_id: str) -> str | None:
    matches = list(_SEMVER_FROM_ID_RE.finditer(model_id))
    if not matches:
        return None
    match = matches[-1]
    return ".".join(
        (
            match.group("major"),
            match.group("minor") or "0",
            match.group("patch") or "0",
        )
    )


def _parse_version(value: str) -> _Version:
    match = _SEMVER_RE.fullmatch(value.strip())
    if match is None:
        raise ValueError("invalid SemVer")
    return _Version(
        int(match.group("major")),
        int(match.group("minor") or 0),
        int(match.group("patch") or 0),
    )


def _satisfies(version: _Version, expression: str) -> bool:
    clauses = tuple(
        clause for clause in _CONSTRAINT_SPLIT_RE.split(expression.strip()) if clause
    )
    if not clauses:
        raise ValueError("empty constraint")
    return all(_satisfies_clause(version, clause) for clause in clauses)


def _satisfies_clause(version: _Version, clause: str) -> bool:
    match = _CONSTRAINT_RE.fullmatch(clause.strip())
    if match is None:
        raise ValueError("invalid constraint")
    operator = match.group("operator") or ""
    value = match.group("value").strip()
    if not value:
        raise ValueError("invalid constraint")
    if value in _WILDCARD_VALUES and not operator:
        return True

    parts = value.removeprefix("v").split(".")
    if len(parts) > 3:
        raise ValueError("invalid constraint version")
    wildcard_index = next(
        (index for index, part in enumerate(parts) if part in _WILDCARD_VALUES),
        None,
    )
    partial = len(parts) < 3 or wildcard_index is not None
    if wildcard_index is not None:
        if any(part not in _WILDCARD_VALUES for part in parts[wildcard_index:]):
            raise ValueError("wildcards must be trailing")
        parts = parts[:wildcard_index]
    try:
        numbers = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("invalid constraint version") from exc
    if any(number < 0 for number in numbers):
        raise ValueError("invalid constraint version")
    base = _Version(*(numbers + (0,) * (3 - len(numbers))))

    if operator in {"^", "~", "~=", ""} and partial:
        upper = _upper_bound(base, operator, len(numbers))
        lower_matches = version >= base
        upper_matches = version < upper
        if operator == "" and wildcard_index is None and len(numbers) == 3:
            return version == base
        return lower_matches and upper_matches

    if operator in {"", "=", "=="}:
        return version == base
    if operator == "!=":
        return version != base
    if operator == ">":
        return version > base
    if operator == ">=":
        return version >= base
    if operator == "<":
        return version < base
    if operator == "<=":
        return version <= base
    if operator == "^":
        return version >= base and version < _upper_bound(base, operator, len(numbers))
    if operator in {"~", "~="}:
        return version >= base and version < _upper_bound(base, operator, len(numbers))
    raise ValueError("invalid constraint operator")


def _upper_bound(base: _Version, operator: str, component_count: int) -> _Version:
    if operator == "^":
        if base.major > 0:
            return _Version(base.major + 1, 0, 0)
        if base.minor > 0:
            return _Version(0, base.minor + 1, 0)
        return _Version(0, 0, base.patch + 1)
    if component_count <= 1:
        return _Version(base.major + 1, 0, 0)
    return _Version(base.major, base.minor + 1, 0)


def _passed(name: str, code: str, **details: Any) -> RegistryCompatibilityCheck:
    return RegistryCompatibilityCheck(name, True, code, details)


def _blocked(name: str, code: str, **details: Any) -> RegistryCompatibilityCheck:
    return RegistryCompatibilityCheck(name, False, code, details)


def _safe_digest(value: Any) -> str:
    encoded = json.dumps(
        _canonical_value(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _safe_lineage_relation(value: str) -> str:
    relation = value.casefold()
    known = _LINEAGE_FORWARD_RELATIONS | _LINEAGE_ROLLBACK_RELATIONS
    return relation if relation in known else "other"


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_canonical_value(item) for item in value]
    return {"type": type(value).__name__}


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _safe_json_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, tuple):
        return [_safe_json_value(item) for item in value]
    if isinstance(value, list):
        return [_safe_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return _safe_digest(value)


__all__ = [
    "COMPATIBILITY_REPORT_SCHEMA_VERSION",
    "DECISION_BLOCKED",
    "DECISION_COMPATIBLE",
    "CheckpointInput",
    "CompatibilityCheck",
    "CompatibilityReport",
    "RegistryCheckpoint",
    "RegistryCompatibilityCheck",
    "RegistryCompatibilityError",
    "RegistryCompatibilityInput",
    "RegistryCompatibilityReport",
    "RegistryLineageEdge",
    "RollbackCompatibilityReport",
    "assess_rollback_compatibility",
    "build_rollback_compatibility_report",
    "check_rollback_compatibility",
    "compare_rollback_compatibility",
]
