"""Offline replay of sustained resource failures and protected output withholding."""

import json
import socket
from dataclasses import replace

import pytest

from openmed.multimodal.capability_matrix import (
    ModalityCapability,
    ModalityProfile,
    ProviderCapabilityEnvelope,
)
from openmed.multimodal.runtime.thermal_gate import (
    EdgeBudget,
    GateReason,
    LocalMode,
    MemoryPressure,
    PlatformObservation,
    ThermalGate,
    ThermalGateError,
    ThermalState,
    vlm_adapter_digest,
)
from openmed.multimodal.streaming.privacy.withholding_buffer import (
    BufferState,
    PhiRedactionSpan,
    PhiResolution,
    PhiScreenDeclaration,
    PhiScreenResult,
    SegmentPhiDecision,
    WithholdingBuffer,
    WithholdingPolicy,
)
from openmed.multimodal.vision.providers.base import (
    ImageInput,
    LocalProviderMetadata,
    Prompt,
    RuntimeBackend,
    RuntimeMetadata,
    VLMRequest,
    VLMResult,
)

PRIVATE = "SYNTHETIC-PATIENT-1234"


class Clock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now

    def advance(self, milliseconds):
        self.now += milliseconds / 1000


class Sensors:
    def __init__(self, clock):
        self.clock = clock
        self.thermal = ThermalState.NOMINAL
        self.pressure = MemoryPressure.NORMAL
        self.available = 1000
        self.delay_ms = 0
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return PlatformObservation(
            observed_at=self.clock() - self.delay_ms / 1000,
            thermal=self.thermal,
            memory_pressure=self.pressure,
            available_bytes=self.available,
        )


class Provider:
    def __init__(self, clock, digit, delay=20):
        self.clock = clock
        self.metadata = LocalProviderMetadata(
            backend=RuntimeBackend.SYNTHETIC,
            model_digest=digit * 64,
            preprocessing_digest="f" * 64,
        )
        self.calls = 0
        self.delay = delay
        self.effect = lambda: None

    def infer(self, request):
        self.calls += 1
        self.clock.advance(self.delay)
        self.effect()
        return VLMResult(
            request_digest=request.digest,
            runtime=RuntimeMetadata(
                provider=self.metadata, duration_ms=0, output_tokens=1
            ),
            text=PRIVATE,
        )


def capability(modality="image", **kwargs):
    return ProviderCapabilityEnvelope(
        provider_id="local",
        capability_version="1.0.0",
        capabilities=(ModalityCapability(modality, **kwargs),),
    )


def make_gate(*, budget=None, modes=None, profile=None):
    clock = Clock()
    sensors = Sensors(clock)
    providers = (Provider(clock, "a"), Provider(clock, "b"))
    default = tuple(
        LocalMode(
            adapter_digest=vlm_adapter_digest(p.metadata),
            qualification_digest="c" * 64,
            capabilities=capability(),
            required_available_bytes=size,
            max_thermal=thermal,
            max_memory_pressure=pressure,
        )
        for p, size, thermal, pressure in (
            (providers[0], 800, ThermalState.NOMINAL, MemoryPressure.NORMAL),
            (providers[1], 200, ThermalState.SERIOUS, MemoryPressure.WARNING),
        )
    )
    gate = ThermalGate(
        profile or ModalityProfile("task", "image"),
        modes or default,
        budget
        or EdgeBudget(max_latency_ms=100, max_queue_ms=50, max_output_age_ms=140),
        sensors,
        clock=clock,
    )
    return gate, clock, sensors, providers


def request():
    return VLMRequest(
        image=ImageInput(
            data=b"synthetic-image", media_type="image/png", width=1, height=1
        ),
        prompt=Prompt(text=PRIVATE),
    )


def run(gate, clock, providers, *, queued_at=None):
    return gate.run_vlm(
        providers, request(), queued_at=clock() if queued_at is None else queued_at
    )


def test_real_provider_dispatch_and_aggregate_report_do_not_expose_payloads():
    gate, clock, _, providers = make_gate()
    result = run(gate, clock, providers)
    assert result.reason is GateReason.READY and result.value.text == PRIVATE
    assert result.value.request_digest == request().digest
    assert [p.calls for p in providers] == [1, 0]
    assert PRIVATE not in repr(result) + result.report_json
    assert (
        "model_digest" not in result.report_json
        and "request_digest" not in result.report_json
    )
    assert json.loads(result.report_json)["modes"][0]["p95_ms"] == pytest.approx(20)


@pytest.mark.parametrize("change", ["thermal", "memory", "pressure"])
def test_only_explicit_qualified_modes_receive_original_input(change):
    gate, clock, sensors, providers = make_gate()
    if change == "thermal":
        sensors.thermal = ThermalState.SERIOUS
    elif change == "memory":
        sensors.available = 400
    else:
        sensors.pressure = MemoryPressure.WARNING
    outcome = run(gate, clock, providers)
    assert outcome.reason is GateReason.READY and outcome.mode_index == 1
    assert [p.calls for p in providers] == [0, 1]
    sensors.thermal, sensors.available, sensors.pressure = (
        ThermalState.NOMINAL,
        1000,
        MemoryPressure.NORMAL,
    )
    assert (
        run(gate, clock, providers).mode_index == 1
    )  # No oscillating automatic upgrade.


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("thermal", None),
        ("pressure", None),
        ("available", None),
        ("thermal", ThermalState.CRITICAL),
        ("pressure", MemoryPressure.CRITICAL),
        ("available", 1),
        ("delay_ms", 501),
        ("delay_ms", -1),
    ],
)
def test_unknown_stale_future_or_critical_observations_never_execute(attribute, value):
    gate, clock, sensors, providers = make_gate()
    setattr(sensors, attribute, value)
    result = run(gate, clock, providers)
    assert result.value is None and result.reason is not GateReason.READY
    assert sum(p.calls for p in providers) == 0


def test_observation_replay_is_rejected_even_inside_freshness_window():
    gate, clock, sensors, providers = make_gate()
    assert run(gate, clock, providers).value is not None
    sensors.delay_ms = 1
    assert run(gate, clock, providers).reason is GateReason.OBSERVATION
    assert providers[0].calls == 1


@pytest.mark.parametrize("age_ms", [51, -1])
def test_queue_age_withholds_without_trying_faster_modes(age_ms):
    gate, clock, _, providers = make_gate()
    outcome = run(gate, clock, providers, queued_at=clock() - age_ms / 1000)
    assert outcome.reason is GateReason.QUEUE_EXPIRED
    assert sum(p.calls for p in providers) == 0


def test_slow_execution_is_measured_independently_of_provider_and_never_retried():
    gate, clock, _, providers = make_gate()
    providers[0].delay = 110
    result = run(gate, clock, providers)
    assert result.value is None and result.reason is GateReason.LATENCY
    assert providers[1].calls == 0
    result = run(gate, clock, providers)
    assert result.reason is GateReason.READY and result.mode_index == 1
    report = json.loads(result.report_json)
    assert report["modes"][0]["p95_ms"] == pytest.approx(110)
    assert report["mode_switches"] == 1


def test_queue_plus_execution_age_withholds_result_even_if_single_latency_passes():
    gate, clock, _, providers = make_gate()
    providers[0].delay = 95
    result = run(gate, clock, providers, queued_at=clock() - 0.049)
    assert result.value is None and result.reason is GateReason.OUTPUT_EXPIRED


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("thermal", ThermalState.CRITICAL),
        ("available", 0),
        ("pressure", MemoryPressure.CRITICAL),
        ("thermal", None),
        ("delay_ms", 501),
    ],
)
def test_resources_are_checked_again_after_actual_inference(attribute, value):
    gate, clock, sensors, providers = make_gate()
    providers[0].effect = lambda: setattr(sensors, attribute, value)
    result = run(gate, clock, providers)
    assert result.value is None and result.reason is not GateReason.READY
    assert providers[0].calls == 1 and providers[1].calls == 0


def test_regressing_clock_is_terminal_and_never_releases_a_result():
    gate, clock, _, providers = make_gate()
    providers[0].effect = lambda: clock.advance(-100)
    result = run(gate, clock, providers)
    assert result.value is None and result.reason is GateReason.CLOCK
    clock.advance(500)
    assert run(gate, clock, providers).reason is GateReason.CLOSED


def test_sensor_and_provider_errors_are_content_free_and_no_context_escapes():
    gate, clock, _, providers = make_gate()

    def fail():
        raise RuntimeError(PRIVATE)

    providers[0].effect = fail
    outcome = run(gate, clock, providers)
    assert outcome.reason is GateReason.PROVIDER and PRIVATE not in repr(outcome)
    assert run(gate, clock, providers).mode_index == 1
    gate._observe = fail
    outcome = run(gate, clock, providers)
    assert outcome.reason is GateReason.OBSERVATION and PRIVATE not in repr(outcome)


@pytest.mark.parametrize("when", ["before", "during"])
def test_changed_provider_pin_withholds(when):
    gate, clock, _, providers = make_gate()

    def mutate():
        providers[0].metadata = replace(providers[0].metadata, model_digest="d" * 64)

    if when == "before":
        mutate()
    else:
        providers[0].effect = mutate
    assert run(gate, clock, providers).reason is GateReason.PROVIDER
    assert providers[0].calls == (when == "during")


def test_bounded_rolling_history_and_operation_limit():
    budget = EdgeBudget(
        max_latency_ms=100,
        max_queue_ms=50,
        max_output_age_ms=140,
        window=3,
        max_operations=5,
    )
    gate, clock, _, providers = make_gate(budget=budget)
    for duration in (90, 80, 10, 20, 30):
        providers[0].delay = duration
        assert run(gate, clock, providers).reason is GateReason.READY
    report = json.loads(gate.to_json())
    assert report["modes"][0]["samples"] == 3
    assert report["modes"][0]["p95_ms"] == pytest.approx(30)
    assert run(gate, clock, providers).reason is GateReason.LIMIT
    assert providers[0].calls == 5


def test_last_mode_slow_abstains_without_restarting_bad_history():
    gate, clock, sensors, providers = make_gate()
    sensors.available = 300
    providers[1].delay = 150
    assert run(gate, clock, providers).value is None
    providers[1].delay = 1
    assert run(gate, clock, providers).reason is GateReason.RESOURCES
    assert providers[1].calls == 1


def test_capabilities_needing_transformation_are_not_silently_admitted():
    _, _, _, providers = make_gate()
    mode = LocalMode(
        adapter_digest=vlm_adapter_digest(providers[0].metadata),
        qualification_digest="c" * 64,
        capabilities=capability(transformable_features=("source-coordinates",)),
        required_available_bytes=1,
    )
    with pytest.raises(ThermalGateError):
        make_gate(
            modes=(mode,),
            profile=ModalityProfile("task", "image", ("source-coordinates",)),
        )


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -1, True, PRIVATE, 10**1000]
)
def test_invalid_numeric_configuration_and_clock_are_not_echoed(value):
    with pytest.raises(ThermalGateError) as error:
        EdgeBudget(max_latency_ms=value, max_queue_ms=50, max_output_age_ms=140)
    assert PRIVATE not in str(error.value)
    gate, clock, _, providers = make_gate()
    clock.now = value
    # Enqueue time remains a valid host timestamp; malformed clock closes admission.
    outcome = run(gate, clock, providers, queued_at=10)
    assert outcome.value is None and outcome.reason is GateReason.CLOCK


def test_reentrant_sensor_cannot_execute_nested_work():
    gate, clock, sensors, providers = make_gate()

    def observe():
        run(gate, clock, providers)
        return sensors()

    gate._observe = observe
    assert run(gate, clock, providers).reason is GateReason.OBSERVATION
    assert sum(p.calls for p in providers) == 0


class Screener:
    declaration = PhiScreenDeclaration(artifact_digest="e" * 64)

    def __init__(self, effect=lambda: None):
        self.effect = effect
        self.resolution = PhiResolution.RESOLVED

    def screen(self, request):
        self.effect()
        return PhiScreenResult(
            request_digest=request.request_digest,
            decisions=tuple(
                SegmentPhiDecision(
                    segment_ref=s.segment_ref,
                    revision=s.revision,
                    resolution=self.resolution,
                    redactions=(PhiRedactionSpan(0, len(s.text)),),
                )
                for s in request.segments
            ),
        )


def speech_gate():
    clock = Clock()
    sensors = Sensors(clock)
    screener = Screener()
    buffer = WithholdingBuffer(
        "0" * 32,
        screener,
        clock=clock,
        policy=WithholdingPolicy(trailing_segments=0),
    )
    buffer.append("1" * 32, PRIVATE)
    buffer.finalize("1" * 32, expected_revision=0)
    mode = LocalMode(
        adapter_digest=screener.declaration.artifact_digest,
        qualification_digest="c" * 64,
        capabilities=capability("audio"),
        required_available_bytes=100,
    )
    gate = ThermalGate(
        ModalityProfile("speech", "audio"),
        (mode,),
        EdgeBudget(max_latency_ms=100, max_queue_ms=50, max_output_age_ms=140),
        sensors,
        clock=clock,
    )
    return gate, clock, sensors, screener, buffer


def test_speech_release_runs_existing_phi_screen_and_preserves_redaction_lineage():
    gate, clock, _, _, buffer = speech_gate()
    result = gate.release_speech(buffer, queued_at=clock())
    assert result.reason is GateReason.READY
    assert result.value[0].text == "[REDACTED]" and result.value[0].mappings[0].redacted
    assert PRIVATE not in repr(result) + result.report_json
    assert json.loads(result.value[0].lineage_json)["finalized"]


def test_speech_output_and_pending_context_are_discarded_when_screening_overheats():
    gate, clock, sensors, screener, buffer = speech_gate()
    screener.effect = lambda: setattr(sensors, "thermal", ThermalState.CRITICAL)
    result = gate.release_speech(buffer, queued_at=clock())
    assert result.value is None and result.reason is GateReason.RESOURCES
    assert buffer.state is BufferState.CLOSED
    assert json.loads(buffer.to_json())["pending"] == []


def test_thermal_hold_does_not_reset_original_speech_deadline_or_replay_text():
    gate, clock, sensors, _, buffer = speech_gate()
    sensors.thermal = ThermalState.CRITICAL
    assert gate.release_speech(buffer, queued_at=clock()).value is None
    sensors.thermal = ThermalState.NOMINAL
    assert gate.release_speech(buffer, queued_at=clock()).value is None
    assert buffer.state is BufferState.CLOSED


def test_unresolved_phi_and_provisional_context_still_withhold():
    gate, clock, _, screener, buffer = speech_gate()
    screener.resolution = PhiResolution.CRITICAL_UNRESOLVED
    result = gate.release_speech(buffer, queued_at=clock())
    assert result.value == ()  # Runtime clearance never overrides privacy hold.
    assert json.loads(buffer.to_json())["pending"]
    assert PRIVATE not in result.report_json


def test_stream_buffer_deadline_is_enforced_even_when_runtime_budget_allows_call():
    gate, clock, _, _, buffer = speech_gate()
    clock.advance(5001)
    result = gate.release_speech(buffer, queued_at=clock())
    assert result.value is None and buffer.state is BufferState.ABSTAINED


def test_speech_slow_screen_has_no_released_value():
    gate, clock, _, screener, buffer = speech_gate()
    screener.effect = lambda: clock.advance(150)
    assert gate.release_speech(buffer, queued_at=clock()).value is None
    assert buffer.state is BufferState.CLOSED


def test_offline_replay_is_deterministic(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket, "socket", forbidden)
    reports = []
    for _ in range(2):
        gate, clock, sensors, providers = make_gate()
        local = [run(gate, clock, providers).report_json]
        sensors.available = 300
        local.append(run(gate, clock, providers).report_json)
        providers[1].delay = 150
        local.append(run(gate, clock, providers).report_json)
        reports.append(local)
    assert reports[0] == reports[1]


def test_sensor_outage_does_not_erase_slow_call_history_or_reuse_bad_mode():
    gate, clock, sensors, providers = make_gate()
    providers[0].delay = 110
    providers[0].effect = lambda: setattr(sensors, "thermal", None)
    assert run(gate, clock, providers).reason is GateReason.OBSERVATION
    assert json.loads(gate.to_json())["modes"][0]["p95_ms"] == pytest.approx(110)
    sensors.thermal = ThermalState.NOMINAL
    assert run(gate, clock, providers).mode_index == 1
    assert providers[0].calls == 1
