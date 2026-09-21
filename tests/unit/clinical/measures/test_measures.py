"""Deterministic native and external clinical measure tests."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace

import pytest
from jsonschema.validators import validator_for

from openmed.clinical.journey_contracts import ClinicalFact, canonical_digest
from openmed.clinical.measures import (
    MeasureDefinition,
    MeasureEngineIdentity,
    MeasureLanguage,
    MeasurePopulationDefinition,
    MeasureSubjectResult,
    MeasureTimeWindow,
    NativePopulationRule,
    PopulationKind,
    PopulationState,
    ValueSetBinding,
    compare_measure_runs,
    evaluate_native_measure,
    load_measure_schema,
)
from openmed.interop.cql import (
    CqlElmAdapterConfig,
    CqlElmEvaluationRequest,
    ServiceCqlElmAdapter,
    SubprocessCqlElmAdapter,
)
from openmed.structured.store import StoreState

SUBJECT_A = "patient_measurepatient01"
SUBJECT_B = "patient_measurepatient02"
SNAPSHOT_ID = "snapshot_measuresnapshot1"
SNAPSHOT_DIGEST = "sha256:" + "1" * 64
EVALUATED = "2026-01-02T03:04:05Z"
PERIOD = MeasureTimeWindow(start="2025-01-01T00:00:00Z", end="2025-12-31T23:59:59Z")


def _native_definition() -> MeasureDefinition:
    return MeasureDefinition(
        measure_id="measure_syntheticmeasure01",
        version="1.0.0",
        language=MeasureLanguage.NATIVE,
        populations=(
            MeasurePopulationDefinition(
                population_id="denominator",
                kind=PopulationKind.DENOMINATOR,
                expression_ref="native:denominator",
            ),
            MeasurePopulationDefinition(
                population_id="exclusion",
                kind=PopulationKind.DENOMINATOR_EXCLUSION,
                expression_ref="native:exclusion",
            ),
            MeasurePopulationDefinition(
                population_id="exception",
                kind=PopulationKind.DENOMINATOR_EXCEPTION,
                expression_ref="native:exception",
            ),
            MeasurePopulationDefinition(
                population_id="initial",
                kind=PopulationKind.INITIAL_POPULATION,
                expression_ref="native:initial",
            ),
            MeasurePopulationDefinition(
                population_id="numerator",
                kind=PopulationKind.NUMERATOR,
                expression_ref="native:numerator",
            ),
        ),
        value_sets=(
            ValueSetBinding(
                value_set_id="synthetic_conditions",
                version="2026.1",
                digest="sha256:" + "2" * 64,
            ),
        ),
    )


def _rules() -> tuple[NativePopulationRule, ...]:
    return (
        NativePopulationRule(
            population_id="initial",
            fact_types=("encounter",),
            allowed_statuses=("complete",),
        ),
        NativePopulationRule(
            population_id="denominator",
            fact_types=("condition",),
            allowed_statuses=("active",),
        ),
        NativePopulationRule(
            population_id="exclusion",
            fact_types=("exclusion",),
            allowed_statuses=("active",),
        ),
        NativePopulationRule(
            population_id="exception",
            fact_types=("exception",),
            allowed_statuses=("active",),
        ),
        NativePopulationRule(
            population_id="numerator",
            fact_types=("procedure", "medication"),
            allowed_statuses=("complete", "active"),
            match_mode="all",
        ),
    )


def _fact(
    suffix: str,
    *,
    subject_id: str = SUBJECT_A,
    fact_type: str,
    status: str,
    value: str | None = None,
) -> ClinicalFact:
    return ClinicalFact(
        fact_id=f"fact_{suffix * 16}",
        subject_id=subject_id,
        fact_type=fact_type,
        value=value or f"protected-{suffix}-value",
        status=status,
        evidence_ids=(f"evidence_{suffix * 16}",),
        derivation_hash=canonical_digest({"suffix": suffix}),
    )


def _native_run(*, include_numerator: bool = True):
    facts = [
        _fact("a", fact_type="encounter", status="complete"),
        _fact("b", fact_type="condition", status="active"),
        _fact("c", fact_type="exclusion", status="inactive"),
        _fact("d", fact_type="exception", status="unknown"),
        _fact("e", fact_type="procedure", status="complete"),
        _fact(
            "g",
            subject_id=SUBJECT_B,
            fact_type="encounter",
            status="complete",
        ),
    ]
    if include_numerator:
        facts.append(_fact("f", fact_type="medication", status="active"))
    return evaluate_native_measure(
        _native_definition(),
        _rules(),
        reversed(facts),
        subject_ids=(SUBJECT_B, SUBJECT_A),
        source_snapshot_id=SNAPSHOT_ID,
        source_snapshot_digest=SNAPSHOT_DIGEST,
        measurement_period=PERIOD,
        evaluated_at=EVALUATED,
    )


def _elm_definition(library: dict[str, object]) -> MeasureDefinition:
    return MeasureDefinition(
        measure_id="measure_syntheticmeasure02",
        version="1.0.0",
        language=MeasureLanguage.ELM_JSON,
        populations=(
            MeasurePopulationDefinition(
                population_id="denominator",
                kind=PopulationKind.DENOMINATOR,
                expression_ref="denominator_expression",
            ),
            MeasurePopulationDefinition(
                population_id="numerator",
                kind=PopulationKind.NUMERATOR,
                expression_ref="numerator_expression",
            ),
        ),
        library_id="synthetic_measure_library",
        library_digest=canonical_digest(library),
    )


def _engine(mode: str) -> MeasureEngineIdentity:
    return MeasureEngineIdentity(
        engine_id="synthetic.elm.engine",
        version="2.1.0",
        artifact_digest="sha256:" + "3" * 64,
        execution_mode=mode,
    )


def _request(mode: str = "service") -> CqlElmEvaluationRequest:
    library = {"library": {"identifier": {"id": "Synthetic"}}}
    return CqlElmEvaluationRequest(
        definition=_elm_definition(library),
        elm_library=library,
        input_projection={SUBJECT_A: {"condition": "protected-diabetes", "score": 42}},
        source_snapshot_id=SNAPSHOT_ID,
        source_snapshot_digest=SNAPSHOT_DIGEST,
        measurement_period=PERIOD,
        evaluated_at=EVALUATED,
        engine=_engine(mode),
        required_features=("retrieve", "valueset_membership"),
    )


def _response(request: CqlElmEvaluationRequest) -> bytes:
    return json.dumps(
        {
            "engine": request.engine.to_dict(),
            "request_id": request.request_id,
            "schema_version": "1.0.0",
            "subject_results": [
                {
                    "populations": [
                        {
                            "derivation_digests": ["sha256:" + "4" * 64],
                            "error_code": None,
                            "evidence_ids": ["evidence_externalresult01"],
                            "fact_ids": ["fact_externalresult0001"],
                            "population_id": "denominator",
                            "reason_code": "expression_true",
                            "state": "met",
                        },
                        {
                            "derivation_digests": [],
                            "error_code": None,
                            "evidence_ids": [],
                            "fact_ids": [],
                            "population_id": "numerator",
                            "reason_code": "expression_false",
                            "state": "not_met",
                        },
                    ],
                    "subject_id": SUBJECT_A,
                }
            ],
        },
        sort_keys=True,
    ).encode()


def test_native_measure_reproduces_population_states_and_safe_summary() -> None:
    result = _native_run()

    assert result.state is StoreState.SUCCESS
    assert result.value is not None
    first = result.value.subject_results[0]
    states = {item.population_id: item.state for item in first.populations}
    assert states == {
        "denominator": PopulationState.MET,
        "exception": PopulationState.UNKNOWN,
        "exclusion": PopulationState.NOT_MET,
        "initial": PopulationState.MET,
        "numerator": PopulationState.MET,
    }
    assert first == MeasureSubjectResult.from_json(first.to_json())
    serialized = result.value.to_json()
    assert "protected-" not in serialized
    safe = json.dumps(result.value.safe_summary(), sort_keys=True)
    assert SUBJECT_A not in safe
    assert SUBJECT_B not in safe
    assert "protected" not in safe
    assert result.value.safe_summary()["subject_count"] == 2

    schema = load_measure_schema()
    validator = validator_for(schema)
    validator.check_schema(schema)
    assert not tuple(validator(schema).iter_errors(_native_definition().to_dict()))
    assert not tuple(validator(schema).iter_errors(first.to_dict()))


def test_native_measure_order_is_deterministic_and_unsupported_language_is_explicit() -> (
    None
):
    first = _native_run()
    second = _native_run()
    assert first.value is not None and second.value is not None
    assert first.value.to_json() == second.value.to_json()

    library = {"library": {"identifier": {"id": "Synthetic"}}}
    unsupported = evaluate_native_measure(
        _elm_definition(library),
        _rules(),
        (),
        subject_ids=(SUBJECT_A,),
        source_snapshot_id=SNAPSHOT_ID,
        source_snapshot_digest=SNAPSHOT_DIGEST,
        measurement_period=PERIOD,
        evaluated_at=EVALUATED,
    )
    assert unsupported.state is StoreState.UNSUPPORTED
    assert unsupported.code == "native_measure_language_unsupported"


def test_counts_only_drift_report_detects_semantic_and_result_change() -> None:
    baseline = _native_run(include_numerator=True)
    candidate = _native_run(include_numerator=False)
    assert baseline.value is not None and candidate.value is not None

    drift = compare_measure_runs(baseline.value, candidate.value)
    assert drift.semantic_drift
    assert drift.result_drift
    assert "input_projection_changed" in drift.reason_codes
    assert "population_counts_changed" in drift.reason_codes
    serialized = json.dumps(drift.to_dict(), sort_keys=True)
    assert SUBJECT_A not in serialized
    assert SUBJECT_B not in serialized


def test_service_adapter_is_explicit_and_returns_value_free_results() -> None:
    request = _request()
    calls: list[tuple[str, bytes, float]] = []

    def transport(endpoint: str, payload: bytes, timeout: float) -> bytes:
        calls.append((endpoint, payload, timeout))
        return _response(request)

    result = ServiceCqlElmAdapter(
        "https://measure.example.invalid/evaluate",
        transport,
        config=CqlElmAdapterConfig(
            supported_features=("retrieve", "valueset_membership")
        ),
    ).evaluate(request)

    assert result.state is StoreState.SUCCESS
    assert result.value is not None
    assert result.value.subject_results[0].populations[0].state is PopulationState.MET
    assert len(calls) == 1
    assert b"protected-diabetes" in calls[0][1]
    assert "protected-diabetes" not in json.dumps(request.safe_summary())
    assert "protected-diabetes" not in result.value.to_json()
    assert "protected-diabetes" not in repr(request)


def test_unsupported_external_feature_fails_before_transport() -> None:
    request = _request()
    called = False

    def transport(endpoint: str, payload: bytes, timeout: float) -> bytes:
        nonlocal called
        called = True
        return b"{}"

    result = ServiceCqlElmAdapter(
        "http://127.0.0.1:8080/evaluate",
        transport,
        config=CqlElmAdapterConfig(supported_features=("retrieve",)),
    ).evaluate(request)
    assert result.state is StoreState.UNSUPPORTED
    assert result.code == "cql_elm_feature_unsupported"
    assert not called


def test_subprocess_adapter_uses_argv_without_shell_or_stderr_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request("subprocess")
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed.update(kwargs)
        return subprocess.CompletedProcess(command, 0, stdout=_response(request))

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = SubprocessCqlElmAdapter(
        ("synthetic-elm-runner", "--json"),
        config=CqlElmAdapterConfig(
            supported_features=("retrieve", "valueset_membership")
        ),
    ).evaluate(request)

    assert result.state is StoreState.SUCCESS
    assert observed["command"] == ("synthetic-elm-runner", "--json")
    assert "shell" not in observed
    assert observed["stderr"] is subprocess.DEVNULL
    assert isinstance(observed["env"], dict)


def test_external_response_identity_and_population_coverage_fail_closed() -> None:
    request = _request()
    wrong_engine = replace(request.engine, version="2.2.0")
    payload = json.loads(_response(request))
    payload["engine"] = wrong_engine.to_dict()

    result = ServiceCqlElmAdapter(
        "https://measure.example.invalid/evaluate",
        lambda endpoint, body, timeout: json.dumps(payload).encode(),
        config=CqlElmAdapterConfig(
            supported_features=("retrieve", "valueset_membership")
        ),
    ).evaluate(request)
    assert result.state is StoreState.CONFLICT
    assert result.code == "cql_elm_engine_identity_conflict"


def test_elm_library_digest_and_result_digest_tampering_are_rejected() -> None:
    request = _request()
    with pytest.raises(Exception, match="ELM library digest differs"):
        replace(request, elm_library={"library": {"changed": True}})

    native = _native_run()
    assert native.value is not None
    payload = native.value.subject_results[0].to_dict()
    payload["result_digest"] = "sha256:" + "0" * 64
    with pytest.raises(Exception, match="result digest differs"):
        MeasureSubjectResult.from_dict(payload)
