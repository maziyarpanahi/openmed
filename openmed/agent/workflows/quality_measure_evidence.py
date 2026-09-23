"""Signed, value-free evidence for reproducible quality-measure results.

The packet binds measure metadata, terminology snapshots, an input projection,
the measurement period, exclusions, implementation metadata, a calculation
trace, and aggregate results. Patient-level values and identifiers are outside
the contract. Comparison emits only closed drift categories and public
component identifiers.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Final, cast

QUALITY_MEASURE_EVIDENCE_SCHEMA: Final = (
    "openmed.agent.workflows.quality_measure_evidence.v1"
)
QUALITY_MEASURE_DRIFT_SCHEMA: Final = "openmed.agent.workflows.quality_measure_drift.v1"
QUALITY_MEASURE_SIGNATURE_ALGORITHM: Final = "hmac-sha256"

_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
_SIGNATURE_RE = re.compile(r"hmac-sha256:[0-9a-f]{64}")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}")
_VERSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,127}")
_TIMESTAMP_RE = re.compile(
    r"(?:[0-9]{4})-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])"
    r"T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z"
)
_MAX_ITEMS: Final = 10_000
_MAX_COUNT: Final = (1 << 63) - 1
_PACKET_FIELDS: Final = frozenset(
    {
        "schema",
        "measure_definition",
        "value_sets",
        "input_projection",
        "time_boundaries",
        "exclusions",
        "implementation",
        "calculation_trace",
        "result",
        "key_id",
        "signature_algorithm",
        "signature",
    }
)


class QualityMeasureEvidenceError(ValueError):
    """A value-free packet validation or verification error."""

    def __init__(self, code: str, field_name: str | None = None) -> None:
        self.code = code
        self.field_name = field_name
        message = code if field_name is None else f"{field_name}: {code}"
        super().__init__(message)


class DriftSource(str, Enum):
    """Closed semantic sources of quality-measure drift."""

    MEASURE_DEFINITION = "measure_definition"
    VALUE_SET = "value_set"
    INPUT_PROJECTION = "input_projection"
    TIME_BOUNDARIES = "time_boundaries"
    EXCLUSION_DEFINITION = "exclusion_definition"
    EXCLUSION_EVIDENCE = "exclusion_evidence"
    IMPLEMENTATION = "implementation"
    CALCULATION_LOGIC = "calculation_logic"
    CALCULATION_OUTPUT = "calculation_output"


class DriftChange(str, Enum):
    """Closed change kinds for one drift source."""

    ADDED = "added"
    REMOVED = "removed"
    CHANGED = "changed"


@dataclass(frozen=True, slots=True, order=True)
class MeasureDefinitionEvidence:
    """Public measure identity plus a digest of its exact definition."""

    measure_id: str
    version: str
    definition_digest: str

    def __post_init__(self) -> None:
        _validate_identifier(self.measure_id, "measure_id")
        _validate_version(self.version, "version")
        _validate_digest(self.definition_digest, "definition_digest")

    def to_dict(self) -> dict[str, str]:
        """Return canonical measure-definition evidence."""

        return {
            "definition_digest": self.definition_digest,
            "measure_id": self.measure_id,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MeasureDefinitionEvidence:
        """Restore definition evidence from an exact mapping."""

        data = _exact_mapping(
            value, {"definition_digest", "measure_id", "version"}, "definition"
        )
        return cls(
            cast(str, data["measure_id"]),
            cast(str, data["version"]),
            cast(str, data["definition_digest"]),
        )


@dataclass(frozen=True, slots=True, order=True)
class ValueSetEvidence:
    """Version and expansion digest for one public value-set identifier."""

    value_set_id: str
    version: str
    expansion_digest: str

    def __post_init__(self) -> None:
        _validate_identifier(self.value_set_id, "value_set_id")
        _validate_version(self.version, "version")
        _validate_digest(self.expansion_digest, "expansion_digest")

    def to_dict(self) -> dict[str, str]:
        """Return canonical value-set evidence."""

        return {
            "expansion_digest": self.expansion_digest,
            "value_set_id": self.value_set_id,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ValueSetEvidence:
        """Restore value-set evidence from an exact mapping."""

        data = _exact_mapping(
            value, {"expansion_digest", "value_set_id", "version"}, "value_set"
        )
        return cls(
            cast(str, data["value_set_id"]),
            cast(str, data["version"]),
            cast(str, data["expansion_digest"]),
        )


@dataclass(frozen=True, slots=True, order=True)
class InputProjectionEvidence:
    """Digest-only binding to locally retained projected input rows."""

    schema_digest: str
    projection_digest: str
    record_count: int

    def __post_init__(self) -> None:
        _validate_digest(self.schema_digest, "schema_digest")
        _validate_digest(self.projection_digest, "projection_digest")
        _validate_count(self.record_count, "record_count")

    def to_dict(self) -> dict[str, str | int]:
        """Return input projection metadata without projected values."""

        return {
            "projection_digest": self.projection_digest,
            "record_count": self.record_count,
            "schema_digest": self.schema_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> InputProjectionEvidence:
        """Restore input evidence from an exact mapping."""

        data = _exact_mapping(
            value,
            {"projection_digest", "record_count", "schema_digest"},
            "input_projection",
        )
        return cls(
            cast(str, data["schema_digest"]),
            cast(str, data["projection_digest"]),
            cast(int, data["record_count"]),
        )


@dataclass(frozen=True, slots=True, order=True)
class MeasureTimeBoundaries:
    """Inclusive start and exclusive end of the population measurement period."""

    start: str
    end: str

    def __post_init__(self) -> None:
        _validate_timestamp(self.start, "start")
        _validate_timestamp(self.end, "end")
        if self.end <= self.start:
            raise QualityMeasureEvidenceError("invalid_time_range", "end")

    def to_dict(self) -> dict[str, str]:
        """Return canonical UTC period boundaries."""

        return {"end": self.end, "start": self.start}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MeasureTimeBoundaries:
        """Restore time boundaries from an exact mapping."""

        data = _exact_mapping(value, {"end", "start"}, "time_boundaries")
        return cls(cast(str, data["start"]), cast(str, data["end"]))


@dataclass(frozen=True, slots=True, order=True)
class ExclusionEvidence:
    """Definition and aggregate outcome evidence for one exclusion."""

    exclusion_id: str
    definition_digest: str
    evidence_digest: str
    excluded_count: int

    def __post_init__(self) -> None:
        _validate_identifier(self.exclusion_id, "exclusion_id")
        _validate_digest(self.definition_digest, "definition_digest")
        _validate_digest(self.evidence_digest, "evidence_digest")
        _validate_count(self.excluded_count, "excluded_count")

    def to_dict(self) -> dict[str, str | int]:
        """Return exclusion metadata without patient-level membership."""

        return {
            "definition_digest": self.definition_digest,
            "evidence_digest": self.evidence_digest,
            "excluded_count": self.excluded_count,
            "exclusion_id": self.exclusion_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ExclusionEvidence:
        """Restore exclusion evidence from an exact mapping."""

        data = _exact_mapping(
            value,
            {
                "definition_digest",
                "evidence_digest",
                "excluded_count",
                "exclusion_id",
            },
            "exclusion",
        )
        return cls(
            cast(str, data["exclusion_id"]),
            cast(str, data["definition_digest"]),
            cast(str, data["evidence_digest"]),
            cast(int, data["excluded_count"]),
        )


@dataclass(frozen=True, slots=True, order=True)
class MeasureImplementationEvidence:
    """Versioned implementation identity and executable/configuration digest."""

    implementation_id: str
    version: str
    implementation_digest: str

    def __post_init__(self) -> None:
        _validate_identifier(self.implementation_id, "implementation_id")
        _validate_version(self.version, "version")
        _validate_digest(self.implementation_digest, "implementation_digest")

    def to_dict(self) -> dict[str, str]:
        """Return canonical implementation metadata."""

        return {
            "implementation_digest": self.implementation_digest,
            "implementation_id": self.implementation_id,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MeasureImplementationEvidence:
        """Restore implementation evidence from an exact mapping."""

        data = _exact_mapping(
            value,
            {"implementation_digest", "implementation_id", "version"},
            "implementation",
        )
        return cls(
            cast(str, data["implementation_id"]),
            cast(str, data["version"]),
            cast(str, data["implementation_digest"]),
        )


@dataclass(frozen=True, slots=True, order=True)
class CalculationTraceStep:
    """One ordered, digest-addressed measure-calculation step."""

    step_id: str
    operation_id: str
    logic_digest: str
    input_digest: str
    output_digest: str
    output_count: int

    def __post_init__(self) -> None:
        _validate_identifier(self.step_id, "step_id")
        _validate_identifier(self.operation_id, "operation_id")
        _validate_digest(self.logic_digest, "logic_digest")
        _validate_digest(self.input_digest, "input_digest")
        _validate_digest(self.output_digest, "output_digest")
        _validate_count(self.output_count, "output_count")

    def logic_dict(self) -> dict[str, str]:
        """Return only the step fields that define calculation semantics."""

        return {
            "logic_digest": self.logic_digest,
            "operation_id": self.operation_id,
            "step_id": self.step_id,
        }

    def output_dict(self) -> dict[str, str | int]:
        """Return only the step fields that bind inputs and aggregate outputs."""

        return {
            "input_digest": self.input_digest,
            "output_count": self.output_count,
            "output_digest": self.output_digest,
            "step_id": self.step_id,
        }

    def to_dict(self) -> dict[str, str | int]:
        """Return the complete value-free trace step."""

        return {
            "input_digest": self.input_digest,
            "logic_digest": self.logic_digest,
            "operation_id": self.operation_id,
            "output_count": self.output_count,
            "output_digest": self.output_digest,
            "step_id": self.step_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CalculationTraceStep:
        """Restore a trace step from an exact mapping."""

        data = _exact_mapping(
            value,
            {
                "input_digest",
                "logic_digest",
                "operation_id",
                "output_count",
                "output_digest",
                "step_id",
            },
            "calculation_trace",
        )
        return cls(
            cast(str, data["step_id"]),
            cast(str, data["operation_id"]),
            cast(str, data["logic_digest"]),
            cast(str, data["input_digest"]),
            cast(str, data["output_digest"]),
            cast(int, data["output_count"]),
        )


@dataclass(frozen=True, slots=True, order=True)
class MeasureResultEvidence:
    """Aggregate measure counts plus a digest of the exact local result."""

    denominator_count: int
    numerator_count: int
    exclusion_count: int
    result_digest: str

    def __post_init__(self) -> None:
        _validate_count(self.denominator_count, "denominator_count")
        _validate_count(self.numerator_count, "numerator_count")
        _validate_count(self.exclusion_count, "exclusion_count")
        if self.numerator_count > self.denominator_count:
            raise QualityMeasureEvidenceError("count_out_of_range", "numerator_count")
        _validate_digest(self.result_digest, "result_digest")

    def to_dict(self) -> dict[str, str | int]:
        """Return aggregate result evidence."""

        return {
            "denominator_count": self.denominator_count,
            "exclusion_count": self.exclusion_count,
            "numerator_count": self.numerator_count,
            "result_digest": self.result_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MeasureResultEvidence:
        """Restore result evidence from an exact mapping."""

        data = _exact_mapping(
            value,
            {
                "denominator_count",
                "exclusion_count",
                "numerator_count",
                "result_digest",
            },
            "result",
        )
        return cls(
            cast(int, data["denominator_count"]),
            cast(int, data["numerator_count"]),
            cast(int, data["exclusion_count"]),
            cast(str, data["result_digest"]),
        )


@dataclass(frozen=True, slots=True, repr=False, init=False)
class QualityMeasureEvidencePacket:
    """Canonical signed evidence for one quality-measure calculation."""

    measure_definition: MeasureDefinitionEvidence
    value_sets: tuple[ValueSetEvidence, ...]
    input_projection: InputProjectionEvidence
    time_boundaries: MeasureTimeBoundaries
    exclusions: tuple[ExclusionEvidence, ...]
    implementation: MeasureImplementationEvidence
    calculation_trace: tuple[CalculationTraceStep, ...]
    result: MeasureResultEvidence
    key_id: str
    signature: str
    schema: str
    signature_algorithm: str

    def __init__(
        self,
        measure_definition: MeasureDefinitionEvidence,
        value_sets: Iterable[ValueSetEvidence],
        input_projection: InputProjectionEvidence,
        time_boundaries: MeasureTimeBoundaries,
        exclusions: Iterable[ExclusionEvidence],
        implementation: MeasureImplementationEvidence,
        calculation_trace: Iterable[CalculationTraceStep],
        result: MeasureResultEvidence,
        key_id: str,
        signature: str,
        schema: str = QUALITY_MEASURE_EVIDENCE_SCHEMA,
        signature_algorithm: str = QUALITY_MEASURE_SIGNATURE_ALGORITHM,
    ) -> None:
        if type(measure_definition) is not MeasureDefinitionEvidence:
            raise QualityMeasureEvidenceError(
                "invalid_definition", "measure_definition"
            )
        if type(input_projection) is not InputProjectionEvidence:
            raise QualityMeasureEvidenceError("invalid_projection", "input_projection")
        if type(time_boundaries) is not MeasureTimeBoundaries:
            raise QualityMeasureEvidenceError("invalid_boundaries", "time_boundaries")
        if type(implementation) is not MeasureImplementationEvidence:
            raise QualityMeasureEvidenceError(
                "invalid_implementation", "implementation"
            )
        if type(result) is not MeasureResultEvidence:
            raise QualityMeasureEvidenceError("invalid_result", "result")
        if schema != QUALITY_MEASURE_EVIDENCE_SCHEMA:
            raise QualityMeasureEvidenceError("invalid_schema", "schema")
        if signature_algorithm != QUALITY_MEASURE_SIGNATURE_ALGORITHM:
            raise QualityMeasureEvidenceError(
                "invalid_signature_algorithm", "signature_algorithm"
            )
        _validate_identifier(key_id, "key_id")
        if type(signature) is not str or _SIGNATURE_RE.fullmatch(signature) is None:
            raise QualityMeasureEvidenceError("invalid_signature", "signature")
        normalized_value_sets = _normalized_unique(
            value_sets, ValueSetEvidence, "value_sets", "value_set_id"
        )
        normalized_exclusions = _normalized_unique(
            exclusions, ExclusionEvidence, "exclusions", "exclusion_id"
        )
        trace = _bounded_tuple(calculation_trace, "calculation_trace")
        if not trace or any(type(item) is not CalculationTraceStep for item in trace):
            raise QualityMeasureEvidenceError("invalid_trace", "calculation_trace")
        step_ids = tuple(step.step_id for step in trace)
        if len(set(step_ids)) != len(step_ids):
            raise QualityMeasureEvidenceError("duplicate_item", "calculation_trace")
        object.__setattr__(self, "measure_definition", measure_definition)
        object.__setattr__(self, "value_sets", normalized_value_sets)
        object.__setattr__(self, "input_projection", input_projection)
        object.__setattr__(self, "time_boundaries", time_boundaries)
        object.__setattr__(self, "exclusions", normalized_exclusions)
        object.__setattr__(self, "implementation", implementation)
        object.__setattr__(self, "calculation_trace", trace)
        object.__setattr__(self, "result", result)
        object.__setattr__(self, "key_id", key_id)
        object.__setattr__(self, "signature", signature)
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "signature_algorithm", signature_algorithm)

    @property
    def packet_digest(self) -> str:
        """Return a stable digest of signed content, excluding the signature."""

        return _digest(self.signing_payload())

    def signing_payload(self) -> dict[str, Any]:
        """Return the complete canonical payload covered by the signature."""

        return {
            "calculation_trace": [step.to_dict() for step in self.calculation_trace],
            "exclusions": [item.to_dict() for item in self.exclusions],
            "implementation": self.implementation.to_dict(),
            "input_projection": self.input_projection.to_dict(),
            "key_id": self.key_id,
            "measure_definition": self.measure_definition.to_dict(),
            "result": self.result.to_dict(),
            "schema": self.schema,
            "signature_algorithm": self.signature_algorithm,
            "time_boundaries": self.time_boundaries.to_dict(),
            "value_sets": [item.to_dict() for item in self.value_sets],
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical signed packet."""

        payload = self.signing_payload()
        payload["signature"] = self.signature
        return payload

    def to_json(self) -> str:
        """Serialize the packet as compact canonical JSON."""

        return _canonical_json(self.to_dict())

    def verify(self, signing_key: bytes | bytearray) -> bool:
        """Verify the packet signature locally and fail closed on mismatch."""

        expected = _signature(self.signing_payload(), signing_key)
        if not hmac.compare_digest(self.signature, expected):
            raise QualityMeasureEvidenceError("signature_mismatch", "signature")
        return True

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> QualityMeasureEvidencePacket:
        """Restore a packet while rejecting omitted or unsigned fields."""

        data = _exact_mapping(value, _PACKET_FIELDS, "packet")
        value_sets = _mapping_sequence(data["value_sets"], "value_sets")
        exclusions = _mapping_sequence(data["exclusions"], "exclusions")
        trace = _mapping_sequence(data["calculation_trace"], "calculation_trace")
        return cls(
            MeasureDefinitionEvidence.from_dict(
                _as_mapping(data["measure_definition"], "measure_definition")
            ),
            (ValueSetEvidence.from_dict(item) for item in value_sets),
            InputProjectionEvidence.from_dict(
                _as_mapping(data["input_projection"], "input_projection")
            ),
            MeasureTimeBoundaries.from_dict(
                _as_mapping(data["time_boundaries"], "time_boundaries")
            ),
            (ExclusionEvidence.from_dict(item) for item in exclusions),
            MeasureImplementationEvidence.from_dict(
                _as_mapping(data["implementation"], "implementation")
            ),
            (CalculationTraceStep.from_dict(item) for item in trace),
            MeasureResultEvidence.from_dict(_as_mapping(data["result"], "result")),
            cast(str, data["key_id"]),
            cast(str, data["signature"]),
            cast(str, data["schema"]),
            cast(str, data["signature_algorithm"]),
        )

    @classmethod
    def from_json(
        cls, serialized: str | bytes | bytearray
    ) -> QualityMeasureEvidencePacket:
        """Restore a packet from strict JSON."""

        if not isinstance(serialized, (str, bytes, bytearray)):
            raise QualityMeasureEvidenceError("invalid_json", "packet")
        try:
            value = json.loads(serialized, object_pairs_hook=_reject_duplicate_pairs)
        except QualityMeasureEvidenceError:
            raise
        except (UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            raise QualityMeasureEvidenceError("invalid_json", "packet") from None
        return cls.from_dict(_as_mapping(value, "packet"))

    def __repr__(self) -> str:
        return f"QualityMeasureEvidencePacket(packet_digest={self.packet_digest!r})"


@dataclass(frozen=True, slots=True, order=True)
class QualityMeasureDrift:
    """One value-free semantic difference between two packets."""

    source: DriftSource
    change: DriftChange
    component_id: str

    def __post_init__(self) -> None:
        if type(self.source) is not DriftSource:
            raise QualityMeasureEvidenceError("invalid_drift_source", "source")
        if type(self.change) is not DriftChange:
            raise QualityMeasureEvidenceError("invalid_drift_change", "change")
        _validate_identifier(self.component_id, "component_id")

    def to_dict(self) -> dict[str, str]:
        """Return closed drift metadata."""

        return {
            "change": self.change.value,
            "component_id": self.component_id,
            "source": self.source.value,
        }


@dataclass(frozen=True, slots=True)
class QualityMeasureDriftReport:
    """Deterministic comparison report with no patient-level values."""

    baseline_packet_digest: str
    current_packet_digest: str
    result_changed: bool
    drift: tuple[QualityMeasureDrift, ...]
    report_digest: str
    schema: str = QUALITY_MEASURE_DRIFT_SCHEMA

    def __post_init__(self) -> None:
        _validate_digest(self.baseline_packet_digest, "baseline_packet_digest")
        _validate_digest(self.current_packet_digest, "current_packet_digest")
        if type(self.result_changed) is not bool:
            raise QualityMeasureEvidenceError("invalid_boolean", "result_changed")
        if type(self.drift) is not tuple or any(
            type(item) is not QualityMeasureDrift for item in self.drift
        ):
            raise QualityMeasureEvidenceError("invalid_drift", "drift")
        if self.drift != tuple(sorted(set(self.drift))):
            raise QualityMeasureEvidenceError("drift_not_sorted_unique", "drift")
        _validate_digest(self.report_digest, "report_digest")
        if self.schema != QUALITY_MEASURE_DRIFT_SCHEMA:
            raise QualityMeasureEvidenceError("invalid_schema", "schema")

    @property
    def is_equivalent(self) -> bool:
        """Return whether signed calculation content is semantically identical."""

        return not self.drift and not self.result_changed

    def to_dict(self) -> dict[str, Any]:
        """Return the value-free report."""

        return {
            "baseline_packet_digest": self.baseline_packet_digest,
            "current_packet_digest": self.current_packet_digest,
            "drift": [item.to_dict() for item in self.drift],
            "is_equivalent": self.is_equivalent,
            "report_digest": self.report_digest,
            "result_changed": self.result_changed,
            "schema": self.schema,
        }

    def to_json(self) -> str:
        """Serialize the report as compact canonical JSON."""

        return _canonical_json(self.to_dict())


def build_quality_measure_evidence_packet(
    measure_definition: MeasureDefinitionEvidence,
    value_sets: Iterable[ValueSetEvidence],
    input_projection: InputProjectionEvidence,
    time_boundaries: MeasureTimeBoundaries,
    exclusions: Iterable[ExclusionEvidence],
    implementation: MeasureImplementationEvidence,
    calculation_trace: Iterable[CalculationTraceStep],
    result: MeasureResultEvidence,
    *,
    key_id: str,
    signing_key: bytes | bytearray,
) -> QualityMeasureEvidencePacket:
    """Build and sign a canonical packet using a caller-managed local key."""

    placeholder = f"{QUALITY_MEASURE_SIGNATURE_ALGORITHM}:" + "0" * 64
    packet = QualityMeasureEvidencePacket(
        measure_definition,
        value_sets,
        input_projection,
        time_boundaries,
        exclusions,
        implementation,
        calculation_trace,
        result,
        key_id,
        placeholder,
    )
    return QualityMeasureEvidencePacket(
        packet.measure_definition,
        packet.value_sets,
        packet.input_projection,
        packet.time_boundaries,
        packet.exclusions,
        packet.implementation,
        packet.calculation_trace,
        packet.result,
        packet.key_id,
        _signature(packet.signing_payload(), signing_key),
    )


def compare_quality_measure_evidence(
    baseline: QualityMeasureEvidencePacket,
    current: QualityMeasureEvidencePacket,
) -> QualityMeasureDriftReport:
    """Identify semantic result-drift sources without patient-level values."""

    if (
        type(baseline) is not QualityMeasureEvidencePacket
        or type(current) is not QualityMeasureEvidencePacket
    ):
        raise QualityMeasureEvidenceError("invalid_packet", "packet")
    drift: list[QualityMeasureDrift] = []
    if baseline.measure_definition != current.measure_definition:
        drift.append(
            QualityMeasureDrift(
                DriftSource.MEASURE_DEFINITION,
                DriftChange.CHANGED,
                current.measure_definition.measure_id,
            )
        )
    _compare_named(
        baseline.value_sets,
        current.value_sets,
        "value_set_id",
        DriftSource.VALUE_SET,
        drift,
    )
    if baseline.input_projection != current.input_projection:
        drift.append(
            QualityMeasureDrift(
                DriftSource.INPUT_PROJECTION, DriftChange.CHANGED, "input_projection"
            )
        )
    if baseline.time_boundaries != current.time_boundaries:
        drift.append(
            QualityMeasureDrift(
                DriftSource.TIME_BOUNDARIES, DriftChange.CHANGED, "measurement_period"
            )
        )
    _compare_exclusions(baseline.exclusions, current.exclusions, drift)
    if baseline.implementation != current.implementation:
        drift.append(
            QualityMeasureDrift(
                DriftSource.IMPLEMENTATION,
                DriftChange.CHANGED,
                current.implementation.implementation_id,
            )
        )
    _compare_trace(baseline.calculation_trace, current.calculation_trace, drift)
    normalized = tuple(sorted(set(drift)))
    result_changed = baseline.result != current.result
    report_payload = {
        "baseline_packet_digest": baseline.packet_digest,
        "current_packet_digest": current.packet_digest,
        "drift": [item.to_dict() for item in normalized],
        "result_changed": result_changed,
        "schema": QUALITY_MEASURE_DRIFT_SCHEMA,
    }
    return QualityMeasureDriftReport(
        baseline.packet_digest,
        current.packet_digest,
        result_changed,
        normalized,
        _digest(report_payload),
    )


def _compare_named(
    baseline: tuple[Any, ...],
    current: tuple[Any, ...],
    identifier_field: str,
    source: DriftSource,
    output: list[QualityMeasureDrift],
) -> None:
    before = {getattr(item, identifier_field): item for item in baseline}
    after = {getattr(item, identifier_field): item for item in current}
    for component_id in sorted(before.keys() | after.keys()):
        if component_id not in before:
            change = DriftChange.ADDED
        elif component_id not in after:
            change = DriftChange.REMOVED
        elif before[component_id] != after[component_id]:
            change = DriftChange.CHANGED
        else:
            continue
        output.append(QualityMeasureDrift(source, change, component_id))


def _compare_exclusions(
    baseline: tuple[ExclusionEvidence, ...],
    current: tuple[ExclusionEvidence, ...],
    output: list[QualityMeasureDrift],
) -> None:
    before = {item.exclusion_id: item for item in baseline}
    after = {item.exclusion_id: item for item in current}
    for component_id in sorted(before.keys() | after.keys()):
        if component_id not in before:
            output.append(
                QualityMeasureDrift(
                    DriftSource.EXCLUSION_DEFINITION, DriftChange.ADDED, component_id
                )
            )
        elif component_id not in after:
            output.append(
                QualityMeasureDrift(
                    DriftSource.EXCLUSION_DEFINITION, DriftChange.REMOVED, component_id
                )
            )
        else:
            before_item = before[component_id]
            after_item = after[component_id]
            if before_item.definition_digest != after_item.definition_digest:
                output.append(
                    QualityMeasureDrift(
                        DriftSource.EXCLUSION_DEFINITION,
                        DriftChange.CHANGED,
                        component_id,
                    )
                )
            if (
                before_item.evidence_digest != after_item.evidence_digest
                or before_item.excluded_count != after_item.excluded_count
            ):
                output.append(
                    QualityMeasureDrift(
                        DriftSource.EXCLUSION_EVIDENCE,
                        DriftChange.CHANGED,
                        component_id,
                    )
                )


def _compare_trace(
    baseline: tuple[CalculationTraceStep, ...],
    current: tuple[CalculationTraceStep, ...],
    output: list[QualityMeasureDrift],
) -> None:
    before_ids = tuple(step.step_id for step in baseline)
    after_ids = tuple(step.step_id for step in current)
    before = {step.step_id: step for step in baseline}
    after = {step.step_id: step for step in current}
    for component_id in sorted(before.keys() | after.keys()):
        if component_id not in before:
            output.append(
                QualityMeasureDrift(
                    DriftSource.CALCULATION_LOGIC, DriftChange.ADDED, component_id
                )
            )
        elif component_id not in after:
            output.append(
                QualityMeasureDrift(
                    DriftSource.CALCULATION_LOGIC, DriftChange.REMOVED, component_id
                )
            )
        else:
            if before[component_id].logic_dict() != after[component_id].logic_dict():
                output.append(
                    QualityMeasureDrift(
                        DriftSource.CALCULATION_LOGIC,
                        DriftChange.CHANGED,
                        component_id,
                    )
                )
            if before[component_id].output_dict() != after[component_id].output_dict():
                output.append(
                    QualityMeasureDrift(
                        DriftSource.CALCULATION_OUTPUT,
                        DriftChange.CHANGED,
                        component_id,
                    )
                )
    if before_ids != after_ids and set(before_ids) == set(after_ids):
        output.append(
            QualityMeasureDrift(
                DriftSource.CALCULATION_LOGIC,
                DriftChange.CHANGED,
                "trace_order",
            )
        )


def _normalized_unique(
    values: Iterable[Any], expected_type: type[Any], field_name: str, key: str
) -> tuple[Any, ...]:
    items = _bounded_tuple(values, field_name)
    if any(type(item) is not expected_type for item in items):
        raise QualityMeasureEvidenceError("invalid_item", field_name)
    normalized = tuple(sorted(items, key=lambda item: getattr(item, key)))
    identifiers = tuple(getattr(item, key) for item in normalized)
    if len(set(identifiers)) != len(identifiers):
        raise QualityMeasureEvidenceError("duplicate_item", field_name)
    return normalized


def _bounded_tuple(values: Iterable[Any], field_name: str) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes, bytearray, Mapping)):
        raise QualityMeasureEvidenceError("invalid_collection", field_name)
    try:
        iterator = iter(values)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise QualityMeasureEvidenceError("invalid_collection", field_name) from None
    items: list[Any] = []
    try:
        for item in iterator:
            if len(items) >= _MAX_ITEMS:
                raise QualityMeasureEvidenceError("too_many_items", field_name)
            items.append(item)
    except (QualityMeasureEvidenceError, KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise QualityMeasureEvidenceError("invalid_collection", field_name) from None
    return tuple(items)


def _validate_identifier(value: Any, field_name: str) -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise QualityMeasureEvidenceError("invalid_identifier", field_name)
    return value


def _validate_version(value: Any, field_name: str) -> str:
    if type(value) is not str or _VERSION_RE.fullmatch(value) is None:
        raise QualityMeasureEvidenceError("invalid_version", field_name)
    return value


def _validate_digest(value: Any, field_name: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise QualityMeasureEvidenceError("invalid_digest", field_name)
    return value


def _validate_timestamp(value: Any, field_name: str) -> str:
    if type(value) is not str or _TIMESTAMP_RE.fullmatch(value) is None:
        raise QualityMeasureEvidenceError("invalid_timestamp", field_name)
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        raise QualityMeasureEvidenceError("invalid_timestamp", field_name) from None
    return value


def _validate_count(value: Any, field_name: str) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_COUNT:
        raise QualityMeasureEvidenceError("invalid_count", field_name)
    return value


def _exact_mapping(
    value: Mapping[str, Any], fields: set[str] | frozenset[str], field_name: str
) -> Mapping[str, Any]:
    data = _as_mapping(value, field_name)
    if set(data) != set(fields) or any(type(key) is not str for key in data):
        raise QualityMeasureEvidenceError("invalid_fields", field_name)
    return data


def _as_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QualityMeasureEvidenceError("invalid_mapping", field_name)
    return value


def _mapping_sequence(value: Any, field_name: str) -> tuple[Mapping[str, Any], ...]:
    values = _bounded_tuple(value, field_name)
    return tuple(_as_mapping(item, field_name) for item in values)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise QualityMeasureEvidenceError("duplicate_json_key", "packet")
        value[key] = item
    return value


def _key_bytes(value: bytes | bytearray) -> bytes:
    if not isinstance(value, (bytes, bytearray)) or len(value) < 16:
        raise QualityMeasureEvidenceError("invalid_signing_key", "signing_key")
    return bytes(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value: Any) -> str:
    return f"sha256:{hashlib.sha256(_canonical_json(value).encode()).hexdigest()}"


def _signature(value: Any, signing_key: bytes | bytearray) -> str:
    digest = hmac.new(
        _key_bytes(signing_key), _canonical_json(value).encode(), hashlib.sha256
    ).hexdigest()
    return f"{QUALITY_MEASURE_SIGNATURE_ALGORITHM}:{digest}"


__all__ = [
    "QUALITY_MEASURE_DRIFT_SCHEMA",
    "QUALITY_MEASURE_EVIDENCE_SCHEMA",
    "QUALITY_MEASURE_SIGNATURE_ALGORITHM",
    "CalculationTraceStep",
    "DriftChange",
    "DriftSource",
    "ExclusionEvidence",
    "InputProjectionEvidence",
    "MeasureDefinitionEvidence",
    "MeasureImplementationEvidence",
    "MeasureResultEvidence",
    "MeasureTimeBoundaries",
    "QualityMeasureDrift",
    "QualityMeasureDriftReport",
    "QualityMeasureEvidenceError",
    "QualityMeasureEvidencePacket",
    "ValueSetEvidence",
    "build_quality_measure_evidence_packet",
    "compare_quality_measure_evidence",
]
