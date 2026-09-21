"""Pinned optional subprocess and service adapters for CQL/ELM evaluation.

OpenMed validates and records evaluator inputs and outputs; it does not parse or
reimplement the CQL language. Patient projections are accepted only as a
request payload and are never retained in returned artifacts, logs, exceptions,
or representations.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from openmed.clinical.journey_contracts import (
    canonical_digest,
    canonical_json,
    derived_opaque_id,
)
from openmed.clinical.measures.contracts import (
    CalculationTraceStep,
    MeasureConflictError,
    MeasureDefinition,
    MeasureEngineIdentity,
    MeasureEvidence,
    MeasureLanguage,
    MeasureRunResult,
    MeasureSubjectResult,
    MeasureTimeWindow,
    MeasureUnsupportedError,
    PopulationResult,
    PopulationState,
)
from openmed.structured.store import StoreResult, StoreState

CQL_ELM_BRIDGE_SCHEMA_VERSION = "1.0.0"
_MAX_REQUEST_BYTES = 16 * 1024 * 1024
_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
_OPAQUE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}_[A-Za-z0-9_-]{16,128}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_TIMESTAMP_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})$"
)


class CqlElmTransport(Protocol):
    """Injected configured-service transport; core performs no network access."""

    def __call__(
        self, endpoint: str, payload: bytes, timeout_seconds: float
    ) -> bytes: ...


@dataclass(frozen=True, slots=True, repr=False)
class CqlElmEvaluationRequest:
    """Pinned ELM request whose protected projection is excluded from repr."""

    definition: MeasureDefinition
    elm_library: Mapping[str, Any] = field(repr=False)
    input_projection: Mapping[str, Any] = field(repr=False)
    source_snapshot_id: str
    source_snapshot_digest: str
    measurement_period: MeasureTimeWindow
    evaluated_at: str
    engine: MeasureEngineIdentity
    required_features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.definition, MeasureDefinition):
            raise TypeError("definition must be MeasureDefinition")
        if self.definition.language is not MeasureLanguage.ELM_JSON:
            raise MeasureUnsupportedError(
                "CQL/ELM adapters require an ELM JSON measure"
            )
        if not isinstance(self.measurement_period, MeasureTimeWindow):
            raise TypeError("measurement_period must be MeasureTimeWindow")
        if not isinstance(self.engine, MeasureEngineIdentity):
            raise TypeError("engine must be MeasureEngineIdentity")
        if self.engine.execution_mode not in {"subprocess", "service"}:
            raise MeasureConflictError("external measure engine mode differs")
        _opaque_id(self.source_snapshot_id, "source_snapshot_id")
        _digest(self.source_snapshot_digest, "source_snapshot_digest")
        _timestamp(self.evaluated_at, "evaluated_at")
        library = _json_mapping(self.elm_library, "elm_library")
        projection = _json_mapping(self.input_projection, "input_projection")
        if not projection:
            raise MeasureConflictError("input projection must contain a subject")
        for subject_id in projection:
            _opaque_id(subject_id, "input projection subject_id")
        if canonical_digest(library) != self.definition.library_digest:
            raise MeasureConflictError("ELM library digest differs from definition")
        features = tuple(sorted(set(self.required_features)))
        if len(features) != len(self.required_features):
            raise MeasureConflictError("required features must be unique")
        for feature in features:
            _controlled(feature, "required feature")
        object.__setattr__(self, "elm_library", library)
        object.__setattr__(self, "input_projection", projection)
        object.__setattr__(self, "required_features", features)

    @property
    def request_id(self) -> str:
        """Return the content-derived request identifier."""

        return derived_opaque_id(
            "measurerequest",
            self.definition.definition_digest,
            self.source_snapshot_digest,
            self.measurement_period.digest,
            self.engine.digest,
            self.input_projection_digest,
            self.evaluated_at,
        )

    @property
    def input_projection_digest(self) -> str:
        """Return a digest without exposing the protected projection."""

        return canonical_digest(self.input_projection)

    def to_runner_dict(self) -> dict[str, Any]:
        """Return the private wire payload for the explicitly selected evaluator."""

        return {
            "definition": self.definition.to_dict(),
            "elm_library": dict(self.elm_library),
            "engine": self.engine.to_dict(),
            "evaluated_at": self.evaluated_at,
            "input_projection": dict(self.input_projection),
            "input_projection_digest": self.input_projection_digest,
            "measurement_period": self.measurement_period.to_dict(),
            "request_id": self.request_id,
            "required_features": list(self.required_features),
            "schema_version": CQL_ELM_BRIDGE_SCHEMA_VERSION,
            "source_snapshot_digest": self.source_snapshot_digest,
            "source_snapshot_id": self.source_snapshot_id,
        }

    def safe_summary(self) -> dict[str, Any]:
        """Return request custody with no subject identifiers or values."""

        return {
            "definition_digest": self.definition.definition_digest,
            "engine_digest": self.engine.digest,
            "input_projection_digest": self.input_projection_digest,
            "measurement_period_digest": self.measurement_period.digest,
            "request_id": self.request_id,
            "required_features": list(self.required_features),
            "source_snapshot_digest": self.source_snapshot_digest,
        }


@dataclass(frozen=True, slots=True)
class CqlElmAdapterConfig:
    """Bounded adapter configuration and declared feature support."""

    supported_features: tuple[str, ...]
    timeout_seconds: float = 60.0
    max_request_bytes: int = _MAX_REQUEST_BYTES
    max_response_bytes: int = _MAX_RESPONSE_BYTES

    def __post_init__(self) -> None:
        features = tuple(sorted(set(self.supported_features)))
        if len(features) != len(self.supported_features):
            raise MeasureConflictError("supported features must be unique")
        for feature in features:
            _controlled(feature, "supported feature")
        if isinstance(self.timeout_seconds, bool) or not isinstance(
            self.timeout_seconds, (int, float)
        ):
            raise TypeError("timeout_seconds must be numeric")
        timeout = float(self.timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0 or timeout > 900:
            raise ValueError("timeout_seconds must be finite and within 900 seconds")
        for value, name in (
            (self.max_request_bytes, "max_request_bytes"),
            (self.max_response_bytes, "max_response_bytes"),
        ):
            if type(value) is not int or value < 1024 or value > 64 * 1024 * 1024:
                raise ValueError(f"{name} is outside the bounded range")
        object.__setattr__(self, "supported_features", features)
        object.__setattr__(self, "timeout_seconds", timeout)


class SubprocessCqlElmAdapter:
    """Invoke one caller-installed evaluator without a shell or stderr capture."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        config: CqlElmAdapterConfig,
        cwd: str | Path | None = None,
    ) -> None:
        command_tuple = tuple(command)
        if not command_tuple or any(
            not isinstance(item, str) or not item for item in command_tuple
        ):
            raise ValueError("command must be a non-empty argv sequence")
        if any("\x00" in item for item in command_tuple):
            raise ValueError("command contains a null byte")
        if not isinstance(config, CqlElmAdapterConfig):
            raise TypeError("config must be CqlElmAdapterConfig")
        self._command = command_tuple
        self._config = config
        self._cwd = None if cwd is None else Path(cwd)

    def evaluate(
        self, request: CqlElmEvaluationRequest
    ) -> StoreResult[MeasureRunResult]:
        """Evaluate one private request through the configured local process."""

        denied = _preflight(request, self._config, expected_mode="subprocess")
        if denied is not None:
            return denied
        payload = canonical_json(request.to_runner_dict()).encode("utf-8")
        if len(payload) > self._config.max_request_bytes:
            return StoreResult.outcome(StoreState.DENIED, "cql_elm_request_too_large")
        try:
            completed = subprocess.run(  # noqa: S603 - explicit caller-selected argv
                self._command,
                input=payload,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=self._config.timeout_seconds,
                cwd=self._cwd,
                env=_minimal_environment(),
            )
        except subprocess.TimeoutExpired:
            return StoreResult.outcome(StoreState.FAILURE, "cql_elm_evaluator_timeout")
        except (OSError, subprocess.SubprocessError):
            return StoreResult.outcome(
                StoreState.FAILURE, "cql_elm_evaluator_unavailable"
            )
        if completed.returncode != 0:
            return StoreResult.outcome(StoreState.FAILURE, "cql_elm_evaluator_failed")
        return _parse_response(completed.stdout, request, self._config)


class ServiceCqlElmAdapter:
    """Use an explicitly injected service transport; no core network client exists."""

    def __init__(
        self,
        endpoint: str,
        transport: CqlElmTransport,
        *,
        config: CqlElmAdapterConfig,
    ) -> None:
        if not isinstance(endpoint, str) or not endpoint.startswith(
            ("https://", "http://127.0.0.1", "http://localhost")
        ):
            raise ValueError("endpoint must use HTTPS or an explicit loopback address")
        if not callable(transport):
            raise TypeError("transport must be callable")
        if not isinstance(config, CqlElmAdapterConfig):
            raise TypeError("config must be CqlElmAdapterConfig")
        self._endpoint = endpoint
        self._transport = transport
        self._config = config

    def evaluate(
        self, request: CqlElmEvaluationRequest
    ) -> StoreResult[MeasureRunResult]:
        """Evaluate through the caller's explicitly configured transport."""

        denied = _preflight(request, self._config, expected_mode="service")
        if denied is not None:
            return denied
        payload = canonical_json(request.to_runner_dict()).encode("utf-8")
        if len(payload) > self._config.max_request_bytes:
            return StoreResult.outcome(StoreState.DENIED, "cql_elm_request_too_large")
        try:
            response = self._transport(
                self._endpoint, payload, self._config.timeout_seconds
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            return StoreResult.outcome(StoreState.FAILURE, "cql_elm_service_failed")
        if not isinstance(response, bytes):
            return StoreResult.outcome(StoreState.FAILURE, "cql_elm_response_invalid")
        return _parse_response(response, request, self._config)


def _preflight(
    request: CqlElmEvaluationRequest,
    config: CqlElmAdapterConfig,
    *,
    expected_mode: str,
) -> StoreResult[MeasureRunResult] | None:
    if not isinstance(request, CqlElmEvaluationRequest):
        raise TypeError("request must be CqlElmEvaluationRequest")
    if request.engine.execution_mode != expected_mode:
        return StoreResult.outcome(StoreState.CONFLICT, "cql_elm_engine_mode_conflict")
    if not set(request.required_features) <= set(config.supported_features):
        return StoreResult.outcome(
            StoreState.UNSUPPORTED, "cql_elm_feature_unsupported"
        )
    return None


def _parse_response(
    payload: bytes,
    request: CqlElmEvaluationRequest,
    config: CqlElmAdapterConfig,
) -> StoreResult[MeasureRunResult]:
    if len(payload) > config.max_response_bytes:
        return StoreResult.outcome(StoreState.DENIED, "cql_elm_response_too_large")
    try:
        raw = json.loads(payload)
        data = _mapping(raw, "CQL/ELM response")
        _exact_keys(
            data,
            {"engine", "request_id", "schema_version", "subject_results"},
            "CQL/ELM response",
        )
        if data["schema_version"] != CQL_ELM_BRIDGE_SCHEMA_VERSION:
            return StoreResult.outcome(
                StoreState.UNSUPPORTED, "cql_elm_response_version_unsupported"
            )
        if data["request_id"] != request.request_id:
            return StoreResult.outcome(
                StoreState.CONFLICT, "cql_elm_request_identity_conflict"
            )
        response_engine = MeasureEngineIdentity.from_dict(
            _mapping(data["engine"], "engine")
        )
        if response_engine != request.engine:
            return StoreResult.outcome(
                StoreState.CONFLICT, "cql_elm_engine_identity_conflict"
            )
        raw_subjects = _sequence(data["subject_results"], "subject_results")
        subjects = tuple(_subject_result(item, request) for item in raw_subjects)
        if len({item.subject_id for item in subjects}) != len(subjects):
            return StoreResult.outcome(StoreState.CONFLICT, "cql_elm_subject_duplicate")
        run = MeasureRunResult(
            run_id=derived_opaque_id(
                "measurerun",
                request.request_id,
                [item.result_digest for item in subjects],
            ),
            definition=request.definition,
            source_snapshot_id=request.source_snapshot_id,
            source_snapshot_digest=request.source_snapshot_digest,
            measurement_period=request.measurement_period,
            engine=request.engine,
            input_projection_digest=request.input_projection_digest,
            subject_results=subjects,
            evaluated_at=request.evaluated_at,
        )
    except MeasureUnsupportedError:
        return StoreResult.outcome(
            StoreState.UNSUPPORTED, "cql_elm_response_feature_unsupported"
        )
    except (
        MeasureConflictError,
        TypeError,
        ValueError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        return StoreResult.outcome(StoreState.CONFLICT, "cql_elm_response_invalid")
    return StoreResult.success(run, created=True)


def _subject_result(
    value: Any, request: CqlElmEvaluationRequest
) -> MeasureSubjectResult:
    data = _mapping(value, "subject result")
    _exact_keys(data, {"populations", "subject_id"}, "subject result")
    subject_id = _text(data["subject_id"], "subject_id")
    if subject_id not in request.input_projection:
        raise MeasureConflictError(
            "evaluator returned a subject outside the projection"
        )
    definitions = {item.population_id: item for item in request.definition.populations}
    populations: list[PopulationResult] = []
    trace: list[CalculationTraceStep] = []
    subject_input_digest = canonical_digest(request.input_projection[subject_id])
    for raw_population in _sequence(data["populations"], "populations"):
        population_data = _mapping(raw_population, "population result")
        _exact_keys(
            population_data,
            {
                "derivation_digests",
                "error_code",
                "evidence_ids",
                "fact_ids",
                "population_id",
                "reason_code",
                "state",
            },
            "population result",
        )
        population_id = _text(population_data["population_id"], "population_id")
        definition = definitions.get(population_id)
        if definition is None:
            raise MeasureConflictError("evaluator returned an unknown population")
        evidence = MeasureEvidence(
            fact_ids=_text_sequence(population_data["fact_ids"], "fact_ids"),
            evidence_ids=_text_sequence(
                population_data["evidence_ids"], "evidence_ids"
            ),
            derivation_digests=_text_sequence(
                population_data["derivation_digests"], "derivation_digests"
            ),
        )
        state = PopulationState(_text(population_data["state"], "state"))
        reason_code = _controlled(population_data["reason_code"], "reason_code")
        result = PopulationResult(
            population_id=population_id,
            kind=definition.kind,
            state=state,
            evidence=evidence,
            reason_code=reason_code,
            error_code=(
                None
                if population_data["error_code"] is None
                else _controlled(population_data["error_code"], "error_code")
            ),
        )
        populations.append(result)
        trace.append(
            CalculationTraceStep(
                step_id=population_id,
                expression_ref=definition.expression_ref,
                state=state,
                input_digest=canonical_digest(
                    {
                        "population_id": population_id,
                        "subject_input_digest": subject_input_digest,
                    }
                ),
                evidence_digest=evidence.digest,
                reason_code=reason_code,
            )
        )
    if {item.population_id for item in populations} != set(definitions):
        raise MeasureConflictError("evaluator omitted a population")
    return MeasureSubjectResult(
        result_id=derived_opaque_id(
            "measureresult",
            request.request_id,
            subject_id,
            [item.digest for item in populations],
        ),
        subject_id=subject_id,
        definition_version_id=request.definition.version_id,
        definition_digest=request.definition.definition_digest,
        source_snapshot_id=request.source_snapshot_id,
        source_snapshot_digest=request.source_snapshot_digest,
        measurement_period=request.measurement_period,
        engine=request.engine,
        input_projection_digest=request.input_projection_digest,
        value_set_digests={
            item.value_set_id: item.digest for item in request.definition.value_sets
        },
        populations=tuple(populations),
        trace=tuple(trace),
        evaluated_at=request.evaluated_at,
    )


def _minimal_environment() -> dict[str, str]:
    allowed = ("PATH", "LANG", "LC_ALL", "SYSTEMROOT", "WINDIR")
    return {key: os.environ[key] for key in allowed if key in os.environ}


def _json_mapping(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    try:
        normalized = json.loads(canonical_json(value))
    except (TypeError, ValueError):
        raise ValueError(f"{name} must contain canonical JSON values") from None
    if not isinstance(normalized, dict):
        raise ValueError(f"{name} must be a mapping")
    return _freeze_json(normalized)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a sequence")
    return value


def _exact_keys(data: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(data) != expected:
        raise ValueError(f"{name} fields differ")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be non-empty text")
    return value


def _controlled(value: Any, name: str) -> str:
    text = _text(value, name)
    if (
        not text[0].islower()
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_.:/-"
            for character in text
        )
        or len(text) > 128
    ):
        raise ValueError(f"{name} must be controlled")
    return text


def _opaque_id(value: Any, name: str) -> str:
    text = _text(value, name)
    if _OPAQUE_ID_RE.fullmatch(text) is None:
        raise ValueError(f"{name} must be opaque")
    return text


def _digest(value: Any, name: str) -> str:
    text = _text(value, name)
    if _DIGEST_RE.fullmatch(text) is None:
        raise ValueError(f"{name} must be a digest")
    return text


def _timestamp(value: Any, name: str) -> str:
    text = _text(value, name)
    if _TIMESTAMP_RE.fullmatch(text) is None:
        raise ValueError(f"{name} must be an RFC 3339 timestamp")
    return text


def _text_sequence(value: Any, name: str) -> tuple[str, ...]:
    return tuple(_text(item, name) for item in _sequence(value, name))


__all__ = [
    "CQL_ELM_BRIDGE_SCHEMA_VERSION",
    "CqlElmAdapterConfig",
    "CqlElmEvaluationRequest",
    "CqlElmTransport",
    "ServiceCqlElmAdapter",
    "SubprocessCqlElmAdapter",
]
