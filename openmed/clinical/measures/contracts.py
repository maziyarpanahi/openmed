"""Deterministic, value-free contracts for clinical measure evaluation."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from openmed.clinical.journey_contracts import canonical_digest, canonical_json

MEASURE_SCHEMA_VERSION: Final = "1.0.0"
MEASURE_COMPATIBILITY_POLICY: Final = "same_major"
MEASURE_ADVISORY: Final = (
    "Measure output is deterministic decision-support evidence for governed "
    "review; it is not a diagnosis, treatment, order, outreach, or payment action."
)

_CONTROLLED_RE = re.compile(r"^[a-z][a-z0-9_.:/-]{0,127}$")
_OPAQUE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}_[A-Za-z0-9_-]{16,128}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_VERSION_RE = re.compile(r"^[1-9][0-9]*\.[0-9]+\.[0-9]+$")
_TIMESTAMP_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})$"
)


class MeasureContractError(ValueError):
    """Raised when a measure artifact violates its public contract."""


class MeasureConflictError(MeasureContractError):
    """Raised when immutable measure custody or semantics conflict."""


class MeasureUnsupportedError(MeasureContractError):
    """Raised when a measure language, feature, or version is unsupported."""


class MeasureLanguage(str, Enum):
    """Supported execution contract families."""

    NATIVE = "native"
    ELM_JSON = "elm_json"


class PopulationKind(str, Enum):
    """Standard measure population roles."""

    INITIAL_POPULATION = "initial_population"
    DENOMINATOR = "denominator"
    DENOMINATOR_EXCLUSION = "denominator_exclusion"
    DENOMINATOR_EXCEPTION = "denominator_exception"
    NUMERATOR = "numerator"
    MEASURE_OBSERVATION = "measure_observation"


class PopulationState(str, Enum):
    """Explicit result state for one population expression."""

    MET = "met"
    NOT_MET = "not_met"
    UNKNOWN = "unknown"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class MeasureTimeWindow:
    """Inclusive UTC measurement-period boundaries."""

    start: str
    end: str

    def __post_init__(self) -> None:
        start = _timestamp(self.start, "measurement period start")
        end = _timestamp(self.end, "measurement period end")
        if _time_key(end) < _time_key(start):
            raise MeasureConflictError("measurement period end precedes start")

    @property
    def digest(self) -> str:
        """Return the exact measurement-period digest."""

        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, str]:
        """Return canonical time boundaries."""

        return {"end": self.end, "start": self.start}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MeasureTimeWindow":
        """Parse strict time boundaries."""

        data = _mapping(value, "measurement period")
        _exact_keys(data, {"end", "start"}, "measurement period")
        return cls(start=_text(data["start"], "start"), end=_text(data["end"], "end"))


@dataclass(frozen=True, slots=True)
class ValueSetBinding:
    """Pinned value-set identity without licensed vocabulary contents."""

    value_set_id: str
    version: str
    digest: str

    def __post_init__(self) -> None:
        _controlled(self.value_set_id, "value_set_id")
        _text(self.version, "value-set version")
        _digest(self.digest, "value-set digest")

    def to_dict(self) -> dict[str, str]:
        """Return value-set custody."""

        return {
            "digest": self.digest,
            "value_set_id": self.value_set_id,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ValueSetBinding":
        """Parse one strict value-set binding."""

        data = _mapping(value, "value-set binding")
        _exact_keys(data, {"digest", "value_set_id", "version"}, "value-set binding")
        return cls(**{key: _text(data[key], key) for key in data})


@dataclass(frozen=True, slots=True)
class MeasurePopulationDefinition:
    """One named population expression in a versioned measure."""

    population_id: str
    kind: PopulationKind
    expression_ref: str

    def __post_init__(self) -> None:
        _controlled(self.population_id, "population_id")
        object.__setattr__(
            self, "kind", _enum(self.kind, PopulationKind, "population kind")
        )
        _controlled(self.expression_ref, "expression_ref")

    def to_dict(self) -> dict[str, str]:
        """Return the population definition."""

        return {
            "expression_ref": self.expression_ref,
            "kind": self.kind.value,
            "population_id": self.population_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MeasurePopulationDefinition":
        """Parse one strict population definition."""

        data = _mapping(value, "population definition")
        _exact_keys(
            data, {"expression_ref", "kind", "population_id"}, "population definition"
        )
        return cls(
            population_id=_text(data["population_id"], "population_id"),
            kind=_enum(data["kind"], PopulationKind, "population kind"),
            expression_ref=_text(data["expression_ref"], "expression_ref"),
        )


@dataclass(frozen=True, slots=True)
class MeasureDefinition:
    """Content-addressed native or ELM JSON measure definition."""

    measure_id: str
    version: str
    language: MeasureLanguage
    populations: tuple[MeasurePopulationDefinition, ...]
    value_sets: tuple[ValueSetBinding, ...] = ()
    library_id: str | None = None
    library_digest: str | None = None
    schema_version: str = MEASURE_SCHEMA_VERSION
    compatibility_policy: str = MEASURE_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        _contract_version(self.schema_version, self.compatibility_policy)
        _opaque_id(self.measure_id, "measure_id")
        _semantic_version(self.version, "measure version")
        language = _enum(self.language, MeasureLanguage, "measure language")
        raw_populations = tuple(self.populations)
        if any(
            not isinstance(item, MeasurePopulationDefinition)
            for item in raw_populations
        ):
            raise TypeError("populations must contain MeasurePopulationDefinition")
        populations = tuple(
            sorted(raw_populations, key=lambda item: item.population_id)
        )
        if not populations or len({item.population_id for item in populations}) != len(
            populations
        ):
            raise MeasureConflictError(
                "population identifiers must be non-empty and unique"
            )
        raw_value_sets = tuple(self.value_sets)
        if any(not isinstance(item, ValueSetBinding) for item in raw_value_sets):
            raise TypeError("value_sets must contain ValueSetBinding")
        value_sets = tuple(sorted(raw_value_sets, key=lambda item: item.value_set_id))
        if len({item.value_set_id for item in value_sets}) != len(value_sets):
            raise MeasureConflictError("value-set identifiers must be unique")
        if language is MeasureLanguage.ELM_JSON:
            if self.library_id is None or self.library_digest is None:
                raise MeasureConflictError("ELM JSON measures require a pinned library")
            _controlled(self.library_id, "library_id")
            _digest(self.library_digest, "library_digest")
        elif self.library_id is not None or self.library_digest is not None:
            raise MeasureConflictError("native measures cannot declare an ELM library")
        object.__setattr__(self, "language", language)
        object.__setattr__(self, "populations", populations)
        object.__setattr__(self, "value_sets", value_sets)

    @property
    def definition_digest(self) -> str:
        """Return the canonical semantic definition digest."""

        return canonical_digest(self.to_dict())

    @property
    def version_id(self) -> str:
        """Return a deterministic opaque definition-version identifier."""

        return _derived_id("measureversion", self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return the complete semantic definition."""

        return {
            "compatibility_policy": self.compatibility_policy,
            "language": self.language.value,
            "library_digest": self.library_digest,
            "library_id": self.library_id,
            "measure_id": self.measure_id,
            "populations": [item.to_dict() for item in self.populations],
            "schema_version": self.schema_version,
            "value_sets": [item.to_dict() for item in self.value_sets],
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MeasureDefinition":
        """Parse one strict measure definition."""

        data = _mapping(value, "measure definition")
        _exact_keys(
            data,
            {
                "compatibility_policy",
                "language",
                "library_digest",
                "library_id",
                "measure_id",
                "populations",
                "schema_version",
                "value_sets",
                "version",
            },
            "measure definition",
        )
        return cls(
            measure_id=_text(data["measure_id"], "measure_id"),
            version=_text(data["version"], "version"),
            language=_enum(data["language"], MeasureLanguage, "measure language"),
            populations=tuple(
                MeasurePopulationDefinition.from_dict(_mapping(item, "population"))
                for item in _sequence(data["populations"], "populations")
            ),
            value_sets=tuple(
                ValueSetBinding.from_dict(_mapping(item, "value set"))
                for item in _sequence(data["value_sets"], "value_sets")
            ),
            library_id=_optional_text(data["library_id"], "library_id"),
            library_digest=_optional_text(data["library_digest"], "library_digest"),
            schema_version=_text(data["schema_version"], "schema_version"),
            compatibility_policy=_text(
                data["compatibility_policy"], "compatibility_policy"
            ),
        )


@dataclass(frozen=True, slots=True)
class MeasureEngineIdentity:
    """Pinned evaluator identity included in every result."""

    engine_id: str
    version: str
    artifact_digest: str
    execution_mode: str

    def __post_init__(self) -> None:
        _controlled(self.engine_id, "engine_id")
        _semantic_version(self.version, "engine version")
        _digest(self.artifact_digest, "engine artifact digest")
        if self.execution_mode not in {"native", "subprocess", "service"}:
            raise MeasureUnsupportedError("engine execution mode is unsupported")

    @property
    def digest(self) -> str:
        """Return a stable evaluator identity digest."""

        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, str]:
        """Return pinned evaluator metadata."""

        return {
            "artifact_digest": self.artifact_digest,
            "engine_id": self.engine_id,
            "execution_mode": self.execution_mode,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MeasureEngineIdentity":
        """Parse strict evaluator metadata."""

        data = _mapping(value, "engine identity")
        _exact_keys(
            data,
            {"artifact_digest", "engine_id", "execution_mode", "version"},
            "engine identity",
        )
        return cls(**{key: _text(data[key], key) for key in data})


@dataclass(frozen=True, slots=True)
class MeasureEvidence:
    """Value-free fact and evidence custody for one result."""

    fact_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    derivation_digests: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.derivation_digests) != len(self.fact_ids):
            raise MeasureConflictError("derivation digests must cover every fact")
        pairs = tuple(
            sorted(
                (
                    _opaque_id(fact_id, "fact_ids"),
                    _digest(derivation, "derivation digest"),
                )
                for fact_id, derivation in zip(
                    self.fact_ids, self.derivation_digests, strict=True
                )
            )
        )
        if len({fact_id for fact_id, _ in pairs}) != len(pairs):
            raise MeasureConflictError("fact_ids must be unique")
        fact_ids = tuple(fact_id for fact_id, _ in pairs)
        evidence_ids = _opaque_values(self.evidence_ids, "evidence_ids")
        derivations = tuple(derivation for _, derivation in pairs)
        object.__setattr__(self, "fact_ids", fact_ids)
        object.__setattr__(self, "evidence_ids", evidence_ids)
        object.__setattr__(self, "derivation_digests", derivations)

    @property
    def digest(self) -> str:
        """Return the evidence-custody digest."""

        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return value-free evidence custody."""

        return {
            "derivation_digests": list(self.derivation_digests),
            "evidence_ids": list(self.evidence_ids),
            "fact_ids": list(self.fact_ids),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MeasureEvidence":
        """Parse strict value-free evidence custody."""

        data = _mapping(value, "measure evidence")
        _exact_keys(
            data, {"derivation_digests", "evidence_ids", "fact_ids"}, "measure evidence"
        )
        return cls(
            fact_ids=_text_sequence(data["fact_ids"], "fact_ids"),
            evidence_ids=_text_sequence(data["evidence_ids"], "evidence_ids"),
            derivation_digests=_text_sequence(
                data["derivation_digests"], "derivation_digests"
            ),
        )


@dataclass(frozen=True, slots=True)
class CalculationTraceStep:
    """Value-free population calculation trace step."""

    step_id: str
    expression_ref: str
    state: PopulationState
    input_digest: str
    evidence_digest: str
    reason_code: str

    def __post_init__(self) -> None:
        _controlled(self.step_id, "step_id")
        _controlled(self.expression_ref, "expression_ref")
        object.__setattr__(
            self, "state", _enum(self.state, PopulationState, "population state")
        )
        _digest(self.input_digest, "input_digest")
        _digest(self.evidence_digest, "evidence_digest")
        _controlled(self.reason_code, "reason_code")

    def to_dict(self) -> dict[str, str]:
        """Return value-free trace metadata."""

        return {
            "evidence_digest": self.evidence_digest,
            "expression_ref": self.expression_ref,
            "input_digest": self.input_digest,
            "reason_code": self.reason_code,
            "state": self.state.value,
            "step_id": self.step_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CalculationTraceStep":
        """Parse one strict calculation step."""

        data = _mapping(value, "calculation trace step")
        _exact_keys(
            data,
            {
                "evidence_digest",
                "expression_ref",
                "input_digest",
                "reason_code",
                "state",
                "step_id",
            },
            "calculation trace step",
        )
        return cls(
            step_id=_text(data["step_id"], "step_id"),
            expression_ref=_text(data["expression_ref"], "expression_ref"),
            state=_enum(data["state"], PopulationState, "population state"),
            input_digest=_text(data["input_digest"], "input_digest"),
            evidence_digest=_text(data["evidence_digest"], "evidence_digest"),
            reason_code=_text(data["reason_code"], "reason_code"),
        )


@dataclass(frozen=True, slots=True)
class PopulationResult:
    """One explicit population state with evidence or typed failure."""

    population_id: str
    kind: PopulationKind
    state: PopulationState
    evidence: MeasureEvidence
    reason_code: str
    error_code: str | None = None

    def __post_init__(self) -> None:
        _controlled(self.population_id, "population_id")
        object.__setattr__(
            self, "kind", _enum(self.kind, PopulationKind, "population kind")
        )
        object.__setattr__(
            self, "state", _enum(self.state, PopulationState, "population state")
        )
        if not isinstance(self.evidence, MeasureEvidence):
            raise TypeError("evidence must be MeasureEvidence")
        _controlled(self.reason_code, "reason_code")
        if self.state is PopulationState.ERROR:
            if self.error_code is None:
                raise MeasureConflictError("error results require an error code")
            _controlled(self.error_code, "error_code")
        elif self.error_code is not None:
            raise MeasureConflictError("non-error results cannot carry an error code")

    @property
    def digest(self) -> str:
        """Return the exact population-result digest."""

        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return one value-free population result."""

        return {
            "error_code": self.error_code,
            "evidence": self.evidence.to_dict(),
            "kind": self.kind.value,
            "population_id": self.population_id,
            "reason_code": self.reason_code,
            "state": self.state.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PopulationResult":
        """Parse one strict population result."""

        data = _mapping(value, "population result")
        _exact_keys(
            data,
            {"error_code", "evidence", "kind", "population_id", "reason_code", "state"},
            "population result",
        )
        return cls(
            population_id=_text(data["population_id"], "population_id"),
            kind=_enum(data["kind"], PopulationKind, "population kind"),
            state=_enum(data["state"], PopulationState, "population state"),
            evidence=MeasureEvidence.from_dict(_mapping(data["evidence"], "evidence")),
            reason_code=_text(data["reason_code"], "reason_code"),
            error_code=_optional_text(data["error_code"], "error_code"),
        )


@dataclass(frozen=True, slots=True)
class MeasureSubjectResult:
    """One subject's versioned population results and value-free trace."""

    result_id: str
    subject_id: str
    definition_version_id: str
    definition_digest: str
    source_snapshot_id: str
    source_snapshot_digest: str
    measurement_period: MeasureTimeWindow
    engine: MeasureEngineIdentity
    input_projection_digest: str
    value_set_digests: Mapping[str, str]
    populations: tuple[PopulationResult, ...]
    trace: tuple[CalculationTraceStep, ...]
    evaluated_at: str
    schema_version: str = MEASURE_SCHEMA_VERSION
    compatibility_policy: str = MEASURE_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        _contract_version(self.schema_version, self.compatibility_policy)
        for name in (
            "result_id",
            "subject_id",
            "definition_version_id",
            "source_snapshot_id",
        ):
            _opaque_id(getattr(self, name), name)
        for name in (
            "definition_digest",
            "source_snapshot_digest",
            "input_projection_digest",
        ):
            _digest(getattr(self, name), name)
        if not isinstance(self.measurement_period, MeasureTimeWindow):
            raise TypeError("measurement_period must be MeasureTimeWindow")
        if not isinstance(self.engine, MeasureEngineIdentity):
            raise TypeError("engine must be MeasureEngineIdentity")
        values = _digest_mapping(self.value_set_digests, "value_set_digests")
        raw_populations = tuple(self.populations)
        if any(not isinstance(item, PopulationResult) for item in raw_populations):
            raise TypeError("populations must contain PopulationResult")
        populations = tuple(
            sorted(raw_populations, key=lambda item: item.population_id)
        )
        if not populations or len({item.population_id for item in populations}) != len(
            populations
        ):
            raise MeasureConflictError(
                "population results must be non-empty and unique"
            )
        raw_trace = tuple(self.trace)
        if any(not isinstance(item, CalculationTraceStep) for item in raw_trace):
            raise TypeError("trace must contain CalculationTraceStep")
        trace = tuple(sorted(raw_trace, key=lambda item: item.step_id))
        if len(trace) != len(populations):
            raise MeasureConflictError("trace must cover every population result")
        if {item.step_id for item in trace} != {
            item.population_id for item in populations
        }:
            raise MeasureConflictError("trace steps must match population identifiers")
        _timestamp(self.evaluated_at, "evaluated_at")
        object.__setattr__(self, "value_set_digests", MappingProxyType(values))
        object.__setattr__(self, "populations", populations)
        object.__setattr__(self, "trace", trace)

    @property
    def result_digest(self) -> str:
        """Return the digest of the complete subject result."""

        return canonical_digest(self.identity_payload)

    @property
    def identity_payload(self) -> dict[str, Any]:
        """Return all fields covered by ``result_digest``."""

        return {
            "compatibility_policy": self.compatibility_policy,
            "definition_digest": self.definition_digest,
            "definition_version_id": self.definition_version_id,
            "engine": self.engine.to_dict(),
            "evaluated_at": self.evaluated_at,
            "input_projection_digest": self.input_projection_digest,
            "measurement_period": self.measurement_period.to_dict(),
            "populations": [item.to_dict() for item in self.populations],
            "result_id": self.result_id,
            "schema_version": self.schema_version,
            "source_snapshot_digest": self.source_snapshot_digest,
            "source_snapshot_id": self.source_snapshot_id,
            "subject_id": self.subject_id,
            "trace": [item.to_dict() for item in self.trace],
            "value_set_digests": dict(self.value_set_digests),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the complete value-free subject result."""

        return {
            "advisory": MEASURE_ADVISORY,
            "artifact_type": "measure_subject_result",
            "result_digest": self.result_digest,
            **self.identity_payload,
        }

    def to_json(self) -> str:
        """Return canonical subject-result JSON."""

        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MeasureSubjectResult":
        """Parse and verify a persisted subject result."""

        data = _mapping(value, "measure subject result")
        expected = set(cls._identity_keys()) | {
            "advisory",
            "artifact_type",
            "result_digest",
        }
        _exact_keys(data, expected, "measure subject result")
        if data["artifact_type"] != "measure_subject_result":
            raise MeasureUnsupportedError("measure artifact type is unsupported")
        if data["advisory"] != MEASURE_ADVISORY:
            raise MeasureConflictError("measure advisory differs")
        result = cls(
            result_id=_text(data["result_id"], "result_id"),
            subject_id=_text(data["subject_id"], "subject_id"),
            definition_version_id=_text(
                data["definition_version_id"], "definition_version_id"
            ),
            definition_digest=_text(data["definition_digest"], "definition_digest"),
            source_snapshot_id=_text(data["source_snapshot_id"], "source_snapshot_id"),
            source_snapshot_digest=_text(
                data["source_snapshot_digest"], "source_snapshot_digest"
            ),
            measurement_period=MeasureTimeWindow.from_dict(
                _mapping(data["measurement_period"], "measurement_period")
            ),
            engine=MeasureEngineIdentity.from_dict(_mapping(data["engine"], "engine")),
            input_projection_digest=_text(
                data["input_projection_digest"], "input_projection_digest"
            ),
            value_set_digests={
                _text(key, "value_set_id"): _text(item, "value_set_digest")
                for key, item in _mapping(
                    data["value_set_digests"], "value_set_digests"
                ).items()
            },
            populations=tuple(
                PopulationResult.from_dict(_mapping(item, "population result"))
                for item in _sequence(data["populations"], "populations")
            ),
            trace=tuple(
                CalculationTraceStep.from_dict(_mapping(item, "trace step"))
                for item in _sequence(data["trace"], "trace")
            ),
            evaluated_at=_text(data["evaluated_at"], "evaluated_at"),
            schema_version=_text(data["schema_version"], "schema_version"),
            compatibility_policy=_text(
                data["compatibility_policy"], "compatibility_policy"
            ),
        )
        if data["result_digest"] != result.result_digest:
            raise MeasureConflictError("measure result digest differs")
        return result

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> "MeasureSubjectResult":
        """Parse canonical or human-formatted subject-result JSON."""

        return cls.from_dict(_json_object(value, "measure subject result"))

    @staticmethod
    def _identity_keys() -> tuple[str, ...]:
        return (
            "compatibility_policy",
            "definition_digest",
            "definition_version_id",
            "engine",
            "evaluated_at",
            "input_projection_digest",
            "measurement_period",
            "populations",
            "result_id",
            "schema_version",
            "source_snapshot_digest",
            "source_snapshot_id",
            "subject_id",
            "trace",
            "value_set_digests",
        )


@dataclass(frozen=True, slots=True)
class MeasureRunResult:
    """Deterministic run with patient results and a counts-only summary."""

    run_id: str
    definition: MeasureDefinition
    source_snapshot_id: str
    source_snapshot_digest: str
    measurement_period: MeasureTimeWindow
    engine: MeasureEngineIdentity
    input_projection_digest: str
    subject_results: tuple[MeasureSubjectResult, ...]
    evaluated_at: str

    def __post_init__(self) -> None:
        _opaque_id(self.run_id, "run_id")
        if not isinstance(self.definition, MeasureDefinition):
            raise TypeError("definition must be MeasureDefinition")
        _opaque_id(self.source_snapshot_id, "source_snapshot_id")
        _digest(self.source_snapshot_digest, "source_snapshot_digest")
        if not isinstance(self.measurement_period, MeasureTimeWindow):
            raise TypeError("measurement_period must be MeasureTimeWindow")
        if not isinstance(self.engine, MeasureEngineIdentity):
            raise TypeError("engine must be MeasureEngineIdentity")
        _digest(self.input_projection_digest, "input_projection_digest")
        raw_results = tuple(self.subject_results)
        if any(not isinstance(item, MeasureSubjectResult) for item in raw_results):
            raise TypeError("subject_results must contain MeasureSubjectResult")
        results = tuple(sorted(raw_results, key=lambda item: item.subject_id))
        if len({item.subject_id for item in results}) != len(results):
            raise MeasureConflictError("subject results must be unique")
        expected_populations = {
            item.population_id: item.kind for item in self.definition.populations
        }
        expected_value_sets = {
            item.value_set_id: item.digest for item in self.definition.value_sets
        }
        for result in results:
            if (
                result.definition_version_id != self.definition.version_id
                or result.definition_digest != self.definition.definition_digest
                or result.source_snapshot_id != self.source_snapshot_id
                or result.source_snapshot_digest != self.source_snapshot_digest
                or result.measurement_period != self.measurement_period
                or result.engine != self.engine
                or result.input_projection_digest != self.input_projection_digest
                or dict(result.value_set_digests) != expected_value_sets
                or {item.population_id: item.kind for item in result.populations}
                != expected_populations
                or result.evaluated_at != self.evaluated_at
            ):
                raise MeasureConflictError("subject result custody differs from run")
        _timestamp(self.evaluated_at, "evaluated_at")
        object.__setattr__(self, "subject_results", results)

    @property
    def semantic_fingerprint(self) -> str:
        """Return the complete execution-semantics fingerprint."""

        return canonical_digest(
            {
                "definition_digest": self.definition.definition_digest,
                "engine_digest": self.engine.digest,
                "input_projection_digest": self.input_projection_digest,
                "measurement_period_digest": self.measurement_period.digest,
                "source_snapshot_digest": self.source_snapshot_digest,
                "value_set_digests": {
                    item.value_set_id: item.digest
                    for item in self.definition.value_sets
                },
            }
        )

    @property
    def run_digest(self) -> str:
        """Return the complete run digest."""

        return canonical_digest(self.to_dict())

    def safe_summary(self) -> dict[str, Any]:
        """Return counts only, suitable for logs and aggregate evidence."""

        counts: Counter[tuple[str, str]] = Counter()
        for subject in self.subject_results:
            for population in subject.populations:
                counts[(population.population_id, population.state.value)] += 1
        population_counts: dict[str, dict[str, int]] = {}
        for definition_population in self.definition.populations:
            population_counts[definition_population.population_id] = {
                state.value: counts[(definition_population.population_id, state.value)]
                for state in PopulationState
            }
        return {
            "advisory": MEASURE_ADVISORY,
            "definition_digest": self.definition.definition_digest,
            "definition_version_id": self.definition.version_id,
            "engine_digest": self.engine.digest,
            "evaluated_at": self.evaluated_at,
            "population_counts": population_counts,
            "run_id": self.run_id,
            "semantic_fingerprint": self.semantic_fingerprint,
            "subject_count": len(self.subject_results),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the full run artifact; use ``safe_summary`` for logs."""

        return {
            "advisory": MEASURE_ADVISORY,
            "artifact_type": "measure_run_result",
            "definition": self.definition.to_dict(),
            "engine": self.engine.to_dict(),
            "evaluated_at": self.evaluated_at,
            "input_projection_digest": self.input_projection_digest,
            "measurement_period": self.measurement_period.to_dict(),
            "run_id": self.run_id,
            "safe_summary": self.safe_summary(),
            "source_snapshot_digest": self.source_snapshot_digest,
            "source_snapshot_id": self.source_snapshot_id,
            "subject_results": [item.to_dict() for item in self.subject_results],
        }

    def to_json(self) -> str:
        """Return canonical run JSON."""

        return canonical_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class MeasureDriftReport:
    """Counts-only comparison between two measure runs."""

    baseline_run_id: str
    candidate_run_id: str
    semantic_drift: bool
    result_drift: bool
    reason_codes: tuple[str, ...]
    population_count_deltas: Mapping[str, Mapping[str, int]]

    def __post_init__(self) -> None:
        _opaque_id(self.baseline_run_id, "baseline_run_id")
        _opaque_id(self.candidate_run_id, "candidate_run_id")
        if type(self.semantic_drift) is not bool or type(self.result_drift) is not bool:
            raise MeasureContractError("drift flags must be boolean")
        reasons = tuple(
            sorted({_controlled(item, "reason code") for item in self.reason_codes})
        )
        deltas: dict[str, Mapping[str, int]] = {}
        for population_id, values in sorted(self.population_count_deltas.items()):
            _controlled(population_id, "population_id")
            parsed = {
                _controlled(state, "population state"): _integer(delta, "count delta")
                for state, delta in sorted(
                    _mapping(values, "population count delta").items()
                )
            }
            deltas[population_id] = MappingProxyType(parsed)
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "population_count_deltas", MappingProxyType(deltas))

    def to_dict(self) -> dict[str, Any]:
        """Return the counts-only drift report."""

        return {
            "baseline_run_id": self.baseline_run_id,
            "candidate_run_id": self.candidate_run_id,
            "population_count_deltas": {
                key: dict(value) for key, value in self.population_count_deltas.items()
            },
            "reason_codes": list(self.reason_codes),
            "result_drift": self.result_drift,
            "semantic_drift": self.semantic_drift,
        }


def compare_measure_runs(
    baseline: MeasureRunResult, candidate: MeasureRunResult
) -> MeasureDriftReport:
    """Detect execution-semantic and aggregate-result drift without patient data."""

    if not isinstance(baseline, MeasureRunResult) or not isinstance(
        candidate, MeasureRunResult
    ):
        raise TypeError("measure run comparison requires MeasureRunResult values")
    baseline_counts = baseline.safe_summary()["population_counts"]
    candidate_counts = candidate.safe_summary()["population_counts"]
    population_ids = sorted(set(baseline_counts) | set(candidate_counts))
    deltas: dict[str, dict[str, int]] = {}
    for population_id in population_ids:
        states = sorted(
            set(baseline_counts.get(population_id, {}))
            | set(candidate_counts.get(population_id, {}))
        )
        deltas[population_id] = {
            state: int(candidate_counts.get(population_id, {}).get(state, 0))
            - int(baseline_counts.get(population_id, {}).get(state, 0))
            for state in states
        }
    semantic_drift = baseline.semantic_fingerprint != candidate.semantic_fingerprint
    result_drift = any(delta for values in deltas.values() for delta in values.values())
    reasons = []
    if baseline.definition.definition_digest != candidate.definition.definition_digest:
        reasons.append("definition_changed")
    if baseline.engine.digest != candidate.engine.digest:
        reasons.append("engine_changed")
    if baseline.measurement_period != candidate.measurement_period:
        reasons.append("measurement_period_changed")
    if baseline.source_snapshot_digest != candidate.source_snapshot_digest:
        reasons.append("source_snapshot_changed")
    if baseline.input_projection_digest != candidate.input_projection_digest:
        reasons.append("input_projection_changed")
    if result_drift:
        reasons.append("population_counts_changed")
    return MeasureDriftReport(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        semantic_drift=semantic_drift,
        result_drift=result_drift,
        reason_codes=tuple(reasons),
        population_count_deltas=deltas,
    )


def _contract_version(schema_version: str, compatibility_policy: str) -> None:
    if schema_version != MEASURE_SCHEMA_VERSION:
        raise MeasureUnsupportedError("measure schema version is unsupported")
    if compatibility_policy != MEASURE_COMPATIBILITY_POLICY:
        raise MeasureUnsupportedError("measure compatibility policy is unsupported")


def _derived_id(prefix: str, *materials: Any) -> str:
    return f"{prefix}_{canonical_digest(list(materials)).removeprefix('sha256:')[:32]}"


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MeasureContractError(f"{name} must be a mapping")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise MeasureContractError(f"{name} must be a sequence")
    return value


def _exact_keys(data: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(data) != expected:
        raise MeasureContractError(f"{name} fields differ")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise MeasureContractError(f"{name} must be non-empty text")
    return value


def _optional_text(value: Any, name: str) -> str | None:
    return None if value is None else _text(value, name)


def _text_sequence(value: Any, name: str) -> tuple[str, ...]:
    return tuple(_text(item, name) for item in _sequence(value, name))


def _integer(value: Any, name: str) -> int:
    if type(value) is not int:
        raise MeasureContractError(f"{name} must be an integer")
    return value


def _controlled(value: Any, name: str) -> str:
    text = _text(value, name)
    if _CONTROLLED_RE.fullmatch(text) is None:
        raise MeasureContractError(f"{name} must be controlled")
    return text


def _opaque_id(value: Any, name: str) -> str:
    text = _text(value, name)
    if _OPAQUE_ID_RE.fullmatch(text) is None:
        raise MeasureContractError(f"{name} must be opaque")
    return text


def _opaque_values(value: Sequence[str], name: str) -> tuple[str, ...]:
    values = tuple(sorted({_opaque_id(item, name) for item in value}))
    if len(values) != len(value):
        raise MeasureConflictError(f"{name} must be unique")
    return values


def _digest(value: Any, name: str) -> str:
    text = _text(value, name)
    if _DIGEST_RE.fullmatch(text) is None:
        raise MeasureContractError(f"{name} must be a digest")
    return text


def _digest_mapping(value: Mapping[str, str], name: str) -> dict[str, str]:
    data = _mapping(value, name)
    return {
        _controlled(key, "value_set_id"): _digest(item, "value_set_digest")
        for key, item in sorted(data.items())
    }


def _semantic_version(value: Any, name: str) -> str:
    text = _text(value, name)
    if _VERSION_RE.fullmatch(text) is None:
        raise MeasureContractError(f"{name} must be semantic")
    return text


def _timestamp(value: Any, name: str) -> str:
    text = _text(value, name)
    if _TIMESTAMP_RE.fullmatch(text) is None:
        raise MeasureContractError(f"{name} must be an RFC 3339 timestamp")
    return text


def _time_key(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    try:
        return value if isinstance(value, enum_type) else enum_type(value)
    except (TypeError, ValueError):
        raise MeasureUnsupportedError(f"{name} is unsupported") from None


def _json_object(value: str | bytes | bytearray, name: str) -> Mapping[str, Any]:
    try:
        parsed = json.loads(value)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError):
        raise MeasureContractError(f"{name} must be valid JSON") from None
    return _mapping(parsed, name)


__all__ = [
    "MEASURE_ADVISORY",
    "MEASURE_COMPATIBILITY_POLICY",
    "MEASURE_SCHEMA_VERSION",
    "CalculationTraceStep",
    "MeasureConflictError",
    "MeasureContractError",
    "MeasureDefinition",
    "MeasureDriftReport",
    "MeasureEngineIdentity",
    "MeasureEvidence",
    "MeasureLanguage",
    "MeasurePopulationDefinition",
    "MeasureRunResult",
    "MeasureSubjectResult",
    "MeasureTimeWindow",
    "MeasureUnsupportedError",
    "PopulationKind",
    "PopulationResult",
    "PopulationState",
    "ValueSetBinding",
    "compare_measure_runs",
]
