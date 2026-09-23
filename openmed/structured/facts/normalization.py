"""Model-independent, evidence-bound normalization into clinical facts.

Component adapters emit fragments rather than facts.  The normalizer merges
those fragments deterministically, validates every evidence reference, and
creates the common Journey :class:`ClinicalFact` contract.  Exact component
outputs remain available only through an explicit protected representation;
the default serialized result contains their digests and version metadata.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from openmed.clinical.journey_contracts import (
    ClinicalFact,
    EvidenceLocator,
    JourneyContractError,
    canonical_digest,
    canonical_json,
    compute_derivation_hash,
    derived_opaque_id,
)
from openmed.structured.store import StoreResult, StoreState

CLINICAL_FACT_NORMALIZATION_SCHEMA_VERSION = "1.0.0"
CLINICAL_FACT_NORMALIZATION_COMPATIBILITY_POLICY = "same_major"
NORMALIZER_VERSION = "1.0.0"

FACT_FRAGMENT_KINDS = frozenset(
    {
        "assertion",
        "extraction",
        "relation",
        "status",
        "temporality",
        "value",
    }
)
FACT_FIELD_STATES = frozenset(
    {"known", "partial", "unknown", "conflict", "unsupported"}
)
CANONICAL_FACT_FIELDS = frozenset(
    {
        "assertion",
        "attributes",
        "certainty",
        "confidence",
        "effective_time",
        "experiencer",
        "relation_participants",
        "status",
        "unit",
        "value",
    }
)
ASSERTION_VALUES = frozenset(
    {"affirmed", "conditional", "negated", "uncertain", "unknown"}
)
CERTAINTY_VALUES = frozenset(
    {"certain", "probable", "possible", "uncertain", "unknown"}
)
EXPERIENCER_VALUES = frozenset({"patient", "family", "other", "unknown"})
EFFECTIVE_TIME_PRECISIONS = frozenset({"year", "month", "day", "second", "unknown"})

_OPAQUE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}_[A-Za-z0-9_-]{16,128}$")
_CONTROLLED_RE = re.compile(r"^[a-z][a-z0-9_.:/-]{0,127}$")
_SCHEMA_VERSION_RE = re.compile(r"^[1-9][0-9]*\.[0-9]+\.[0-9]+$")
_PARTIAL_DATE_RE = re.compile(
    r"^(?P<year>[0-9]{4})(?:-(?P<month>0[1-9]|1[0-2])"
    r"(?:-(?P<day>0[1-9]|[12][0-9]|3[01]))?)?$"
)
_STATE_PRECEDENCE = {
    "known": 0,
    "unknown": 1,
    "partial": 2,
    "unsupported": 3,
    "conflict": 4,
}


class FactNormalizationError(ValueError):
    """Raised when a normalization input violates the public contract."""


@dataclass(frozen=True, slots=True)
class FactProfileSpec:
    """Validation rules for one common clinical-fact profile."""

    profile: str
    required_fields: frozenset[str]
    allowed_statuses: frozenset[str]

    def __post_init__(self) -> None:
        if _CONTROLLED_RE.fullmatch(self.profile) is None:
            raise FactNormalizationError("profile must be a controlled identifier")
        if (
            not self.required_fields
            or not self.required_fields <= CANONICAL_FACT_FIELDS
        ):
            raise FactNormalizationError("profile contains unsupported required fields")
        if not self.allowed_statuses or "unknown" not in self.allowed_statuses:
            raise FactNormalizationError("profile statuses must include unknown")
        for status in self.allowed_statuses:
            if _CONTROLLED_RE.fullmatch(status) is None:
                raise FactNormalizationError(
                    "profile status must be a controlled identifier"
                )


FACT_PROFILE_SPECS: Mapping[str, FactProfileSpec] = MappingProxyType(
    {
        spec.profile: spec
        for spec in (
            FactProfileSpec(
                profile="condition",
                required_fields=frozenset(
                    {
                        "assertion",
                        "certainty",
                        "experiencer",
                        "status",
                        "value",
                    }
                ),
                allowed_statuses=frozenset(
                    {"active", "history", "inactive", "resolved", "unknown"}
                ),
            ),
            FactProfileSpec(
                profile="medication",
                required_fields=frozenset(
                    {"assertion", "experiencer", "status", "value"}
                ),
                allowed_statuses=frozenset(
                    {
                        "active",
                        "completed",
                        "held",
                        "planned",
                        "stopped",
                        "unknown",
                    }
                ),
            ),
            FactProfileSpec(
                profile="laboratory",
                required_fields=frozenset(
                    {"effective_time", "status", "unit", "value"}
                ),
                allowed_statuses=frozenset(
                    {"amended", "corrected", "final", "preliminary", "unknown"}
                ),
            ),
            FactProfileSpec(
                profile="procedure",
                required_fields=frozenset(
                    {"assertion", "effective_time", "experiencer", "status", "value"}
                ),
                allowed_statuses=frozenset(
                    {"cancelled", "completed", "planned", "unknown"}
                ),
            ),
            FactProfileSpec(
                profile="observation",
                required_fields=frozenset(
                    {"assertion", "effective_time", "experiencer", "status", "value"}
                ),
                allowed_statuses=frozenset(
                    {"amended", "corrected", "final", "preliminary", "unknown"}
                ),
            ),
            FactProfileSpec(
                profile="social_determinant",
                required_fields=frozenset(
                    {
                        "assertion",
                        "certainty",
                        "effective_time",
                        "experiencer",
                        "status",
                        "value",
                    }
                ),
                allowed_statuses=frozenset(
                    {"active", "historical", "inactive", "unknown"}
                ),
            ),
        )
    }
)


@dataclass(frozen=True, slots=True)
class ComponentOutputEnvelope:
    """Exact component output plus path-safe version and digest metadata."""

    component: str
    component_version: str
    output_schema: str
    output_schema_version: str
    output: Any = field(repr=False)
    schema_version: str = CLINICAL_FACT_NORMALIZATION_SCHEMA_VERSION
    compatibility_policy: str = CLINICAL_FACT_NORMALIZATION_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        _require_controlled(self.component, "component")
        _require_version(self.component_version, "component_version")
        _require_controlled(self.output_schema, "output_schema")
        _require_version(self.output_schema_version, "output_schema_version")
        _require_version(self.schema_version, "schema_version")
        if self.schema_version != CLINICAL_FACT_NORMALIZATION_SCHEMA_VERSION:
            raise FactNormalizationError("unsupported component envelope schema")
        if self.compatibility_policy != (
            CLINICAL_FACT_NORMALIZATION_COMPATIBILITY_POLICY
        ):
            raise FactNormalizationError("unsupported compatibility policy")
        try:
            frozen_output = _freeze_json_value(self.output, "component output")
        except (JourneyContractError, TypeError, ValueError):
            raise FactNormalizationError(
                "component output must contain finite JSON values"
            ) from None
        object.__setattr__(self, "output", frozen_output)

    @property
    def output_digest(self) -> str:
        """Return the exact canonical digest for this component output."""

        return canonical_digest(self.output)

    def to_safe_dict(self) -> dict[str, Any]:
        """Return value-free metadata safe for ordinary audit records."""

        return {
            "compatibility_policy": self.compatibility_policy,
            "component": self.component,
            "component_version": self.component_version,
            "output_digest": self.output_digest,
            "output_schema": self.output_schema,
            "output_schema_version": self.output_schema_version,
            "schema_version": self.schema_version,
        }

    def to_protected_dict(self) -> dict[str, Any]:
        """Return exact output for an explicitly protected audit store."""

        return self.to_safe_dict() | {"output": _plain(self.output)}


@dataclass(frozen=True, slots=True)
class RelationParticipant:
    """One typed participant in a clinical relation."""

    role: str
    target_id: str
    target_type: str

    def __post_init__(self) -> None:
        _require_controlled(self.role, "relation role")
        _require_opaque_id(self.target_id, "relation target_id")
        if self.target_type not in {"evidence", "fact"}:
            raise FactNormalizationError("relation target_type is unsupported")
        if not self.target_id.startswith(f"{self.target_type}_"):
            raise FactNormalizationError("relation target type does not match its ID")

    def to_dict(self) -> dict[str, str]:
        """Return deterministic relation-participant metadata."""

        return {
            "role": self.role,
            "target_id": self.target_id,
            "target_type": self.target_type,
        }


@dataclass(frozen=True, slots=True)
class FactFragment:
    """Canonical fields contributed by one versioned component output."""

    kind: str
    component_output: ComponentOutputEnvelope
    fields: Mapping[str, Any] = field(default_factory=dict, repr=False)
    evidence_ids: tuple[str, ...] = ()
    field_states: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in FACT_FRAGMENT_KINDS:
            raise FactNormalizationError("unsupported fact fragment kind")
        fields = _freeze_mapping(self.fields, "fragment fields")
        if not set(fields) <= CANONICAL_FACT_FIELDS:
            raise FactNormalizationError("fragment contains an unsupported fact field")
        states = _freeze_state_mapping(self.field_states)
        if not set(states) <= CANONICAL_FACT_FIELDS:
            raise FactNormalizationError("fragment state names an unsupported field")
        unknown_states = set(states) - set(fields)
        if any(states[name] == "known" for name in unknown_states):
            raise FactNormalizationError("missing fragment fields cannot be known")
        evidence = _opaque_ids(self.evidence_ids, "fragment evidence_ids")
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "field_states", states)
        object.__setattr__(self, "evidence_ids", evidence)

    @property
    def fragment_id(self) -> str:
        """Return deterministic identity without exposing component values."""

        return derived_opaque_id(
            "fragment",
            self.kind,
            self.component_output.component,
            self.component_output.component_version,
            self.component_output.output_digest,
            self.evidence_ids,
            canonical_digest(self.fields),
        )


@dataclass(frozen=True, slots=True)
class FactNormalizationRequest:
    """All fragments and identity context used to construct one clinical fact."""

    subject_id: str
    profile: str
    fragments: tuple[FactFragment, ...]
    encounter_id: str | None = None
    parent_fact_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_opaque_id(self.subject_id, "subject_id")
        if self.encounter_id is not None:
            _require_opaque_id(self.encounter_id, "encounter_id")
        _require_controlled(self.profile, "profile")
        fragments = tuple(self.fragments)
        if not fragments:
            raise FactNormalizationError("fact normalization requires fragments")
        if len({fragment.fragment_id for fragment in fragments}) != len(fragments):
            raise FactNormalizationError("fact fragments must be unique")
        object.__setattr__(self, "fragments", fragments)
        object.__setattr__(
            self,
            "parent_fact_ids",
            _opaque_ids(self.parent_fact_ids, "parent_fact_ids"),
        )


@dataclass(frozen=True, slots=True)
class NormalizedClinicalFact:
    """Clinical fact plus source-output custody and explicit field states."""

    fact: ClinicalFact
    component_outputs: tuple[ComponentOutputEnvelope, ...] = field(repr=False)
    field_states: Mapping[str, str]
    normalization_state: str
    schema_version: str = CLINICAL_FACT_NORMALIZATION_SCHEMA_VERSION
    compatibility_policy: str = CLINICAL_FACT_NORMALIZATION_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        outputs = tuple(self.component_outputs)
        if not outputs:
            raise FactNormalizationError("normalized facts require component outputs")
        output_keys = {
            (
                output.component,
                output.component_version,
                output.output_schema,
                output.output_schema_version,
                output.output_digest,
            )
            for output in outputs
        }
        if len(output_keys) != len(outputs):
            raise FactNormalizationError("component outputs must be unique")
        states = _freeze_state_mapping(self.field_states)
        if self.normalization_state not in {
            "success",
            "partial",
            "unknown",
            "unsupported",
        }:
            raise FactNormalizationError("unsupported normalization state")
        if self.schema_version != CLINICAL_FACT_NORMALIZATION_SCHEMA_VERSION:
            raise FactNormalizationError("unsupported normalized fact schema")
        if self.compatibility_policy != (
            CLINICAL_FACT_NORMALIZATION_COMPATIBILITY_POLICY
        ):
            raise FactNormalizationError("unsupported compatibility policy")
        object.__setattr__(self, "component_outputs", outputs)
        object.__setattr__(self, "field_states", states)

    def to_dict(self) -> dict[str, Any]:
        """Return a value-free-output public record; fact values remain protected."""

        return {
            "compatibility_policy": self.compatibility_policy,
            "component_outputs": [
                output.to_safe_dict() for output in self.component_outputs
            ],
            "fact": self.fact.to_dict(),
            "field_states": dict(self.field_states),
            "normalization_state": self.normalization_state,
            "schema_version": self.schema_version,
        }

    def protected_component_outputs(self) -> tuple[dict[str, Any], ...]:
        """Return exact outputs for an explicitly protected audit store."""

        return tuple(output.to_protected_dict() for output in self.component_outputs)


@dataclass(frozen=True, slots=True)
class MappingFactAdapter:
    """Adapt one mapping-shaped component output through explicit field paths."""

    component: str
    component_version: str
    output_schema: str
    output_schema_version: str
    kind: str
    field_paths: Mapping[str, str]

    def __post_init__(self) -> None:
        _require_controlled(self.component, "component")
        _require_version(self.component_version, "component_version")
        _require_controlled(self.output_schema, "output_schema")
        _require_version(self.output_schema_version, "output_schema_version")
        if self.kind not in FACT_FRAGMENT_KINDS:
            raise FactNormalizationError("unsupported adapter fragment kind")
        paths: dict[str, str] = {}
        for field_name, path in self.field_paths.items():
            if field_name not in CANONICAL_FACT_FIELDS:
                raise FactNormalizationError("adapter maps an unsupported fact field")
            if not isinstance(path, str) or not path or path.startswith("."):
                raise FactNormalizationError("adapter field paths must be dotted names")
            parts = path.split(".")
            if any(_CONTROLLED_RE.fullmatch(part) is None for part in parts):
                raise FactNormalizationError("adapter field path is invalid")
            paths[field_name] = path
        if not paths:
            raise FactNormalizationError("mapping adapter requires field paths")
        object.__setattr__(self, "field_paths", MappingProxyType(paths))

    def adapt(
        self,
        output: Mapping[str, Any],
        *,
        evidence_ids: Sequence[str],
        field_states: Mapping[str, str] | None = None,
    ) -> StoreResult[FactFragment]:
        """Return a fragment without inferring fields missing from the output."""

        if not isinstance(output, Mapping):
            return StoreResult.outcome(StoreState.UNSUPPORTED, "component_not_mapping")
        try:
            envelope = ComponentOutputEnvelope(
                component=self.component,
                component_version=self.component_version,
                output_schema=self.output_schema,
                output_schema_version=self.output_schema_version,
                output=output,
            )
            fields: dict[str, Any] = {}
            for field_name, path in self.field_paths.items():
                found, value = _mapping_path(output, path)
                if found:
                    fields[field_name] = value
            fragment = FactFragment(
                kind=self.kind,
                component_output=envelope,
                fields=fields,
                evidence_ids=tuple(evidence_ids),
                field_states=field_states or {},
            )
        except FactNormalizationError:
            return StoreResult.outcome(StoreState.FAILURE, "component_contract_invalid")
        return StoreResult.success(fragment)


class ClinicalFactNormalizer:
    """Merge heterogeneous fragments into one evidence-bound clinical fact."""

    def normalize(
        self,
        request: FactNormalizationRequest,
        *,
        evidence_by_id: Mapping[str, EvidenceLocator],
    ) -> StoreResult[NormalizedClinicalFact]:
        """Normalize one request without inventing absent or unsupported fields."""

        profile = FACT_PROFILE_SPECS.get(request.profile)
        if profile is None:
            return StoreResult.outcome(
                StoreState.UNSUPPORTED, "fact_profile_unsupported"
            )
        evidence_result = _validate_evidence(request.fragments, evidence_by_id)
        if not evidence_result.ok:
            return StoreResult.outcome(
                evidence_result.state,
                evidence_result.code or "evidence_invalid",
            )
        assert evidence_result.value is not None
        evidence_ids = evidence_result.value
        merged = _merge_fragments(request.fragments)
        if not merged.ok:
            return StoreResult.outcome(
                merged.state,
                merged.code or "component_merge_failed",
            )
        assert merged.value is not None
        fields, field_states = merged.value
        status = fields.get("status", "unknown")
        if not isinstance(status, str) or status not in profile.allowed_statuses:
            return StoreResult.outcome(
                StoreState.UNSUPPORTED, "fact_status_unsupported"
            )
        state_result = _complete_field_states(profile, fields, field_states)
        fields, states = state_result
        if "conflict" in states.values():
            return StoreResult.outcome(StoreState.CONFLICT, "component_state_conflict")
        try:
            effective_time = _normalize_effective_time(fields.get("effective_time"))
        except FactNormalizationError:
            return StoreResult.outcome(
                StoreState.UNSUPPORTED, "effective_time_unsupported"
            )
        try:
            assertion = _controlled_axis(
                fields.get("assertion", "unknown"), ASSERTION_VALUES, "assertion"
            )
            certainty = _controlled_axis(
                fields.get("certainty", "unknown"), CERTAINTY_VALUES, "certainty"
            )
            experiencer = _controlled_axis(
                fields.get("experiencer", "unknown"),
                EXPERIENCER_VALUES,
                "experiencer",
            )
        except FactNormalizationError:
            return StoreResult.outcome(StoreState.UNSUPPORTED, "fact_axis_unsupported")
        try:
            participants = _normalize_participants(
                fields.get("relation_participants", ()),
                evidence_ids=evidence_ids,
                parent_fact_ids=request.parent_fact_ids,
            )
        except FactNormalizationError:
            return StoreResult.outcome(
                StoreState.CONFLICT, "relation_provenance_invalid"
            )
        try:
            attributes = _normalization_attributes(
                fields.get("attributes"),
                assertion=assertion,
                certainty=certainty,
                experiencer=experiencer,
                participants=participants,
                fragments=request.fragments,
                field_states=states,
            )
            confidence = _confidence(fields.get("confidence"))
            unit = _optional_unit(fields.get("unit"))
        except FactNormalizationError:
            return StoreResult.outcome(StoreState.UNSUPPORTED, "fact_value_unsupported")
        try:
            component_outputs = tuple(
                sorted(
                    (fragment.component_output for fragment in request.fragments),
                    key=lambda output: (
                        output.output_digest,
                        output.component,
                        output.component_version,
                        output.output_schema,
                        output.output_schema_version,
                    ),
                )
            )
            output_digests = tuple(output.output_digest for output in component_outputs)
            component_provenance = tuple(
                output.to_safe_dict() for output in component_outputs
            )
            identity_fields = dict(fields)
            identity_fields["effective_time"] = dict(effective_time)
            identity_fields["relation_participants"] = [
                participant.to_dict() for participant in participants
            ]
            normalized_fields_digest = canonical_digest(identity_fields)
            derivation_hash = compute_derivation_hash(
                "clinical.fact.normalizer",
                NORMALIZER_VERSION,
                evidence_ids,
                configuration={
                    "component_outputs": component_provenance,
                    "field_states": dict(states),
                    "normalized_fields_digest": normalized_fields_digest,
                    "parent_fact_ids": request.parent_fact_ids,
                    "profile": profile.profile,
                    "schema_version": CLINICAL_FACT_NORMALIZATION_SCHEMA_VERSION,
                },
            )
            fact_id = derived_opaque_id(
                "fact",
                request.subject_id,
                request.encounter_id,
                profile.profile,
                component_provenance,
                evidence_ids,
                request.parent_fact_ids,
                normalized_fields_digest,
            )
            fact = ClinicalFact(
                fact_id=fact_id,
                subject_id=request.subject_id,
                encounter_id=request.encounter_id,
                fact_type=profile.profile,
                value=fields.get("value"),
                status=status,
                evidence_ids=evidence_ids,
                derivation_hash=derivation_hash,
                parent_fact_ids=request.parent_fact_ids,
                effective_time=effective_time,
                unit=unit,
                confidence=confidence,
                attributes=attributes,
            )
        except (JourneyContractError, TypeError, ValueError):
            return StoreResult.outcome(StoreState.FAILURE, "fact_contract_invalid")
        normalization_state = _normalization_state(states)
        normalized = NormalizedClinicalFact(
            fact=fact,
            component_outputs=component_outputs,
            field_states=states,
            normalization_state=normalization_state.value,
        )
        if normalization_state is StoreState.SUCCESS:
            return StoreResult.success(normalized)
        code = {
            StoreState.PARTIAL: "fact_partial",
            StoreState.UNKNOWN: "fact_unknown",
            StoreState.UNSUPPORTED: "fact_value_unsupported",
        }[normalization_state]
        return StoreResult.outcome(normalization_state, code, value=normalized)


def load_clinical_fact_normalization_schema() -> dict[str, Any]:
    """Load the bundled public normalization-result JSON Schema."""

    path = (
        Path(__file__).resolve().parents[2]
        / "core"
        / "schemas"
        / "json"
        / "clinical_fact_normalization.schema.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_evidence(
    fragments: Sequence[FactFragment],
    evidence_by_id: Mapping[str, EvidenceLocator],
) -> StoreResult[tuple[str, ...]]:
    ids = tuple(
        sorted(
            {
                evidence_id
                for fragment in fragments
                for evidence_id in fragment.evidence_ids
            }
        )
    )
    if not ids:
        return StoreResult.outcome(StoreState.CONFLICT, "evidence_required")
    for evidence_id in ids:
        locator = evidence_by_id.get(evidence_id)
        if locator is None or locator.locator_id != evidence_id:
            return StoreResult.outcome(StoreState.CONFLICT, "evidence_not_found")
    return StoreResult.success(ids)


def _merge_fragments(
    fragments: Sequence[FactFragment],
) -> StoreResult[tuple[dict[str, Any], Mapping[str, str]]]:
    values: dict[str, Any] = {}
    states: dict[str, str] = {}
    attributes: dict[str, Any] = {}
    participants: dict[str, Any] = {}
    for fragment in sorted(fragments, key=lambda item: item.fragment_id):
        for field_name, state in fragment.field_states.items():
            states[field_name] = _merge_state(states.get(field_name), state)
        for field_name, value in fragment.fields.items():
            if field_name == "attributes":
                if not isinstance(value, Mapping):
                    return StoreResult.outcome(
                        StoreState.UNSUPPORTED, "attributes_not_mapping"
                    )
                for key, item in value.items():
                    if not isinstance(key, str):
                        return StoreResult.outcome(
                            StoreState.UNSUPPORTED, "attribute_key_invalid"
                        )
                    if key in attributes and not _same_json(attributes[key], item):
                        return StoreResult.outcome(
                            StoreState.CONFLICT, "component_field_conflict"
                        )
                    attributes[key] = item
                continue
            if field_name == "relation_participants":
                if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                    return StoreResult.outcome(
                        StoreState.UNSUPPORTED, "relation_participants_invalid"
                    )
                for item in value:
                    try:
                        participant = _participant_from_value(item)
                    except FactNormalizationError:
                        return StoreResult.outcome(
                            StoreState.UNSUPPORTED, "relation_participants_invalid"
                        )
                    participants[canonical_json(participant.to_dict())] = participant
                continue
            if field_name in values and not _same_json(values[field_name], value):
                return StoreResult.outcome(
                    StoreState.CONFLICT, "component_field_conflict"
                )
            values[field_name] = value
            states.setdefault(field_name, "known")
    if attributes:
        values["attributes"] = attributes
        states.setdefault("attributes", "known")
    if participants:
        values["relation_participants"] = tuple(participants.values())
        states.setdefault("relation_participants", "known")
    return StoreResult.success((values, MappingProxyType(dict(sorted(states.items())))))


def _complete_field_states(
    profile: FactProfileSpec,
    fields: dict[str, Any],
    states: Mapping[str, str],
) -> tuple[dict[str, Any], Mapping[str, str]]:
    completed = dict(states)
    for field_name in profile.required_fields:
        if field_name not in fields:
            completed[field_name] = _merge_state(completed.get(field_name), "unknown")
    if "value" in fields and fields["value"] is None:
        completed["value"] = _merge_state(completed.get("value"), "unknown")
    if fields.get("status") == "unknown":
        completed["status"] = _merge_state(completed.get("status"), "unknown")
    return fields, MappingProxyType(dict(sorted(completed.items())))


def _normalization_state(states: Mapping[str, str]) -> StoreState:
    values = set(states.values())
    if "conflict" in values:
        return StoreState.CONFLICT
    if "unsupported" in values:
        return StoreState.UNSUPPORTED
    if "partial" in values:
        return StoreState.PARTIAL
    if "unknown" in values:
        return StoreState.UNKNOWN
    return StoreState.SUCCESS


def _normalization_attributes(
    raw_attributes: Any,
    *,
    assertion: str,
    certainty: str,
    experiencer: str,
    participants: tuple[RelationParticipant, ...],
    fragments: Sequence[FactFragment],
    field_states: Mapping[str, str],
) -> Mapping[str, Any]:
    attributes = dict(_freeze_mapping(raw_attributes or {}, "attributes"))
    collision = {
        "assertion",
        "certainty",
        "component_outputs",
        "experiencer",
        "field_states",
        "normalizer_version",
        "relation_participants",
    }.intersection(attributes)
    if collision:
        raise FactNormalizationError("attributes collide with normalization metadata")
    attributes.update(
        {
            "assertion": assertion,
            "certainty": certainty,
            "component_outputs": [
                fragment.component_output.to_safe_dict()
                for fragment in sorted(
                    fragments,
                    key=lambda item: (
                        item.component_output.output_digest,
                        item.component_output.component,
                        item.component_output.component_version,
                        item.component_output.output_schema,
                        item.component_output.output_schema_version,
                    ),
                )
            ],
            "experiencer": experiencer,
            "field_states": dict(field_states),
            "normalizer_version": NORMALIZER_VERSION,
            "relation_participants": [
                participant.to_dict() for participant in participants
            ],
        }
    )
    return MappingProxyType(attributes)


def _normalize_effective_time(value: Any) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({"precision": "unknown"})
    if not isinstance(value, Mapping):
        raise FactNormalizationError("effective_time must be an object")
    allowed = {"end", "precision", "start"}
    if set(value) - allowed:
        raise FactNormalizationError("effective_time contains unsupported fields")
    precision = value.get("precision", "unknown")
    if precision not in EFFECTIVE_TIME_PRECISIONS:
        raise FactNormalizationError("effective_time precision is unsupported")
    start = _optional_temporal(value.get("start"), precision)
    end = _optional_temporal(value.get("end"), precision)
    if precision != "unknown" and start is None and end is None:
        raise FactNormalizationError("known effective_time requires a boundary")
    if (
        start is not None
        and end is not None
        and _temporal_sort_key(start) > _temporal_sort_key(end)
    ):
        raise FactNormalizationError("effective_time start follows end")
    result: dict[str, Any] = {"precision": precision}
    if start is not None:
        result["start"] = start
    if end is not None:
        result["end"] = end
    return MappingProxyType(result)


def _optional_temporal(value: Any, precision: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > 64:
        raise FactNormalizationError("effective_time boundary must be text")
    if precision == "second":
        candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            raise FactNormalizationError(
                "effective_time timestamp is invalid"
            ) from None
        if parsed.tzinfo is None:
            raise FactNormalizationError("effective_time timestamp needs a timezone")
        return value
    match = _PARTIAL_DATE_RE.fullmatch(value)
    if match is None:
        raise FactNormalizationError("effective_time partial date is invalid")
    year = int(match.group("year"))
    month = int(match.group("month") or 1)
    day = int(match.group("day") or 1)
    try:
        date(year, month, day)
    except ValueError:
        raise FactNormalizationError("effective_time partial date is invalid") from None
    expected_parts = {"year": 1, "month": 2, "day": 3, "unknown": None}[precision]
    if expected_parts is not None and len(value.split("-")) != expected_parts:
        raise FactNormalizationError("effective_time boundary precision mismatches")
    return value


def _temporal_sort_key(value: str) -> str:
    if "T" in value:
        candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
        return datetime.fromisoformat(candidate).isoformat()
    parts = value.split("-")
    return "-".join(parts + ["01"] * (3 - len(parts)))


def _normalize_participants(
    value: Any,
    *,
    evidence_ids: Sequence[str],
    parent_fact_ids: Sequence[str],
) -> tuple[RelationParticipant, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FactNormalizationError("relation_participants must be a sequence")
    participants = tuple(_participant_from_value(item) for item in value)
    allowed = set(evidence_ids) | set(parent_fact_ids)
    if any(participant.target_id not in allowed for participant in participants):
        raise FactNormalizationError("relation participant target is not in provenance")
    return tuple(
        sorted(
            participants,
            key=lambda item: (item.role, item.target_type, item.target_id),
        )
    )


def _participant_from_value(value: Any) -> RelationParticipant:
    if isinstance(value, RelationParticipant):
        return value
    if not isinstance(value, Mapping) or set(value) != {
        "role",
        "target_id",
        "target_type",
    }:
        raise FactNormalizationError("relation participant is invalid")
    return RelationParticipant(
        role=value["role"],
        target_id=value["target_id"],
        target_type=value["target_type"],
    )


def _mapping_path(value: Mapping[str, Any], path: str) -> tuple[bool, Any]:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _freeze_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FactNormalizationError(f"{name} must be an object")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise FactNormalizationError(f"{name} keys must be strings")
        try:
            result[key] = _freeze_json_value(item, name)
        except (JourneyContractError, TypeError, ValueError):
            raise FactNormalizationError(
                f"{name} must contain finite JSON values"
            ) from None
    return MappingProxyType(dict(sorted(result.items())))


def _freeze_json_value(value: Any, name: str) -> Any:
    try:
        decoded = json.loads(canonical_json(value))
    except (JourneyContractError, TypeError, ValueError):
        raise FactNormalizationError(
            f"{name} must contain finite JSON values"
        ) from None
    return _deep_freeze(decoded)


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in sorted(value.items())}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _freeze_state_mapping(value: Any) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise FactNormalizationError("field_states must be an object")
    result: dict[str, str] = {}
    for key, state in value.items():
        if not isinstance(key, str) or state not in FACT_FIELD_STATES:
            raise FactNormalizationError("field_states contains an invalid state")
        result[key] = state
    return MappingProxyType(dict(sorted(result.items())))


def _merge_state(current: str | None, incoming: str) -> str:
    if current is None:
        return incoming
    return max((current, incoming), key=lambda value: _STATE_PRECEDENCE[value])


def _same_json(left: Any, right: Any) -> bool:
    try:
        return canonical_json(left) == canonical_json(right)
    except (JourneyContractError, TypeError, ValueError):
        return False


def _confidence(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FactNormalizationError("confidence must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise FactNormalizationError("confidence must be between zero and one")
    return number


def _optional_unit(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > 128:
        raise FactNormalizationError("unit must be non-empty text")
    return value.strip()


def _controlled_axis(value: Any, allowed: frozenset[str], name: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise FactNormalizationError(f"{name} is unsupported")
    return value


def _require_controlled(value: Any, name: str) -> None:
    if not isinstance(value, str) or _CONTROLLED_RE.fullmatch(value) is None:
        raise FactNormalizationError(f"{name} must be a controlled identifier")


def _require_version(value: Any, name: str) -> None:
    if not isinstance(value, str) or _SCHEMA_VERSION_RE.fullmatch(value) is None:
        raise FactNormalizationError(f"{name} must be a semantic version")


def _require_opaque_id(value: Any, name: str) -> None:
    if not isinstance(value, str) or _OPAQUE_ID_RE.fullmatch(value) is None:
        raise FactNormalizationError(f"{name} must be an opaque identifier")


def _opaque_ids(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise FactNormalizationError(f"{name} must be a sequence")
    result = tuple(values)
    for value in result:
        _require_opaque_id(value, name)
    if len(set(result)) != len(result):
        raise FactNormalizationError(f"{name} must not contain duplicates")
    return result


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


__all__ = [
    "CLINICAL_FACT_NORMALIZATION_COMPATIBILITY_POLICY",
    "CLINICAL_FACT_NORMALIZATION_SCHEMA_VERSION",
    "FACT_FIELD_STATES",
    "FACT_FRAGMENT_KINDS",
    "FACT_PROFILE_SPECS",
    "NORMALIZER_VERSION",
    "ClinicalFactNormalizer",
    "ComponentOutputEnvelope",
    "FactFragment",
    "FactNormalizationError",
    "FactNormalizationRequest",
    "FactProfileSpec",
    "MappingFactAdapter",
    "NormalizedClinicalFact",
    "RelationParticipant",
    "load_clinical_fact_normalization_schema",
]
