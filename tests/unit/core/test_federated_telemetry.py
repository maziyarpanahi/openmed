"""Focused tests for the value-free federated round telemetry family."""

from __future__ import annotations

import inspect
import socket

import pytest

from openmed.core.federated_telemetry import (
    FEDERATED_PHASE_STATUS_VALUES,
    FEDERATED_TELEMETRY_MINIMUM_GROUP_SIZE,
    FederatedRoundTelemetry,
    FederatedTelemetryError,
    FederatedUpdateBand,
    UnapprovedFederatedValueError,
    band_update_count,
)
from openmed.core.no_phi_telemetry import (
    FEDERATED_PHASE_LATENCY_NAME,
    FEDERATED_PHASE_VALUES,
    FEDERATED_REASON_CODE_VALUES,
    FEDERATED_UPDATE_BAND_VALUES,
    PIPELINE_LATENCY_NAME,
    PIPELINE_STATUS_VALUES,
    CounterName,
    NoPHITelemetryExporter,
)
from openmed.training.federated_metrics import FederatedParticipantCountBand
from openmed.training.federated_round import FederatedRoundState
from openmed.training.federated_status import (
    DEFAULT_FEDERATED_MINIMUM_GROUP_SIZE,
    FederatedRoundReasonCode,
)

_UPDATES = CounterName.FEDERATED_PHASE_UPDATES.value
_TRANSITIONS = CounterName.FEDERATED_PHASE_TRANSITIONS.value
_REJECTIONS = CounterName.FEDERATED_PHASE_REJECTIONS.value
_DIMENSION_VOCABULARIES = {
    "phase": set(FEDERATED_PHASE_VALUES),
    "status": set(FEDERATED_PHASE_STATUS_VALUES),
    "reason_code": set(FEDERATED_REASON_CODE_VALUES),
    "update_band": set(FEDERATED_UPDATE_BAND_VALUES),
}


def _samples(payload: dict, name: str) -> list[dict]:
    return [item for item in payload["counters"] if item["name"] == name]


def _sample(payload: dict, name: str) -> dict:
    return next(iter(_samples(payload, name)))


def test_phase_transitions_and_rejections_use_closed_dimensions() -> None:
    recorder = FederatedRoundTelemetry()

    recorder.record_phase_transition(phase="preflight")
    recorder.record_phase_transition(phase="held", status="rejected")
    recorder.record_rejection(reason_code="quorum_not_met", phase="held")

    payload = recorder.export()
    transitions = _samples(payload, _TRANSITIONS)
    assert [item["value"] for item in transitions] == [1, 1]
    assert {tuple(sorted(item["dimensions"].items())) for item in transitions} == {
        (("phase", "held"), ("status", "rejected")),
        (("phase", "preflight"), ("status", "success")),
    }

    rejection = _sample(payload, _REJECTIONS)
    assert rejection["value"] == 1
    assert rejection["dimensions"] == {
        "phase": "held",
        "reason_code": "quorum_not_met",
    }

    for sample in payload["counters"]:
        for dimension, value in sample["dimensions"].items():
            assert value in _DIMENSION_VOCABULARIES[dimension]


def test_banded_update_counts_never_release_exact_small_groups() -> None:
    recorder = FederatedRoundTelemetry()

    for count in (1, 2, 3, 4):
        recorder.record_update_count(count=count, phase="aggregating")
    recorder.record_update_count(count=7, phase="aggregating")

    samples = _samples(recorder.export(), _UPDATES)
    assert sorted(item["value"] for item in samples) == [1, 4]
    assert 7 not in {item["value"] for item in samples}
    assert {item["dimensions"]["update_band"] for item in samples} == {
        FederatedUpdateBand.SUPPRESSED.value,
        FederatedUpdateBand.MINIMUM_TO_UNDER_DOUBLE.value,
    }
    suppressed = next(
        item
        for item in samples
        if item["dimensions"]["update_band"] == FederatedUpdateBand.SUPPRESSED.value
    )
    assert suppressed["dimensions"] == {
        "phase": "aggregating",
        "update_band": FederatedUpdateBand.SUPPRESSED.value,
    }
    # Four rounds were observed; the sub-group magnitude never leaves the band.
    assert suppressed["value"] == 4


def test_exact_release_requires_the_minimum_group_size() -> None:
    recorder = FederatedRoundTelemetry()

    with pytest.raises(
        UnapprovedFederatedValueError,
        match="below the minimum group must be suppressed",
    ):
        recorder.record_update_count(count=2, phase="aggregating", exact=True)

    assert recorder.export()["counters"] == []

    recorder.record_update_count(count=5, phase="aggregating", exact=True)
    released = _sample(recorder.export(), _UPDATES)
    assert released["value"] == 5
    assert released["dimensions"] == {"phase": "aggregating"}

    with pytest.raises(FederatedTelemetryError, match="exact update flag is invalid"):
        recorder.record_update_count(count=5, phase="aggregating", exact=1)  # type: ignore[arg-type]
    with pytest.raises(FederatedTelemetryError, match="non-negative integer"):
        recorder.record_update_count(count=-1, phase="aggregating")

    recorder.record_update_count(count=0, phase="preflight")
    assert _samples(recorder.export(), _UPDATES) == [released]


@pytest.mark.parametrize(
    "value",
    [
        "site-7f3a91c2d4e5",
        "/var/lib/openmed/round-3",
        "sha256:" + "0" * 64,
        "11111111-2222-3333-4444-555555555555",
        "Person",
        "PREFLIGHT",
    ],
)
def test_identifying_values_are_refused_without_being_echoed(value: str) -> None:
    recorder = FederatedRoundTelemetry()

    with pytest.raises(UnapprovedFederatedValueError) as excinfo:
        recorder.record_phase_transition(phase=value)
    assert value not in str(excinfo.value)

    with pytest.raises(UnapprovedFederatedValueError):
        recorder.record_update_count(count=9, phase=value)
    with pytest.raises(UnapprovedFederatedValueError):
        recorder.record_rejection(reason_code=value, phase="held")
    with pytest.raises(UnapprovedFederatedValueError):
        recorder.record_phase_transition(phase="preflight", status=value)
    with pytest.raises(UnapprovedFederatedValueError):
        recorder.observe_phase_latency(phase=value, seconds=1.0)

    assert recorder.export()["counters"] == []
    assert recorder.export()["latencies"] == []


@pytest.mark.parametrize("value", [None, 5, b"planned", ["planned"], ("planned",)])
def test_non_string_values_are_refused(value: object) -> None:
    recorder = FederatedRoundTelemetry()

    with pytest.raises(UnapprovedFederatedValueError):
        recorder.record_phase_transition(phase=value)  # type: ignore[arg-type]


def test_export_is_deterministic_and_never_opens_network_sockets(monkeypatch) -> None:
    def deny_socket(*args: object, **kwargs: object) -> None:
        raise AssertionError("federated telemetry must not open a network socket")

    monkeypatch.setattr(socket, "socket", deny_socket)

    recorder = FederatedRoundTelemetry()
    recorder.record_phase_transition(phase="collecting")
    recorder.observe_phase_latency(phase="collecting", seconds=0.5)
    first = recorder.export_json()

    assert recorder.export_json() == first
    assert recorder.export() == recorder.export()
    assert "openmed_federated_phase_latency_seconds_count" in (
        recorder.render_prometheus()
    )


def test_recorder_exposes_no_transport_configuration() -> None:
    signature = inspect.signature(FederatedRoundTelemetry)
    assert set(signature.parameters) == {"exporter", "minimum_group_size"}

    recorder = FederatedRoundTelemetry()
    assert not hasattr(recorder, "endpoint")
    assert not hasattr(recorder, "collector")
    assert not hasattr(recorder, "url")


def test_phase_latency_requires_exactly_one_unit() -> None:
    recorder = FederatedRoundTelemetry()

    with pytest.raises(FederatedTelemetryError, match="multiple units"):
        recorder.observe_phase_latency(
            phase="aggregating", seconds=1.0, milliseconds=1.0
        )
    with pytest.raises(FederatedTelemetryError, match="requires a measurement"):
        recorder.observe_phase_latency(phase="aggregating")

    recorder.observe_phase_latency(phase="aggregating", milliseconds=12.5)
    latency = recorder.export()["latencies"][0]
    assert latency["name"] == FEDERATED_PHASE_LATENCY_NAME
    assert latency["sum_seconds"] == pytest.approx(0.0125)
    assert latency["dimensions"] == {"phase": "aggregating", "status": "success"}


def test_federated_and_pipeline_histograms_stay_separate() -> None:
    recorder = FederatedRoundTelemetry()
    recorder.observe_phase_latency(phase="evaluating", seconds=2.0)
    recorder.exporter.observe_latency_seconds(0.25)

    names = {item["name"] for item in recorder.export()["latencies"]}
    assert names == {FEDERATED_PHASE_LATENCY_NAME, PIPELINE_LATENCY_NAME}

    rendered = recorder.render_prometheus()
    assert "# TYPE openmed_federated_phase_latency_seconds histogram" in rendered
    assert "# TYPE openmed_pipeline_latency_seconds histogram" in rendered


def test_recorder_validates_construction_and_delegates_to_its_exporter() -> None:
    with pytest.raises(FederatedTelemetryError, match="no-PHI telemetry exporter"):
        FederatedRoundTelemetry(exporter="https://collector.invalid")  # type: ignore[arg-type]
    with pytest.raises(FederatedTelemetryError, match="greater than one"):
        FederatedRoundTelemetry(minimum_group_size=1)

    exporter = NoPHITelemetryExporter()
    recorder = FederatedRoundTelemetry(exporter=exporter, minimum_group_size=10)
    assert recorder.exporter is exporter
    assert recorder.minimum_group_size == 10

    recorder.record_phase_transition(phase="promoted")
    assert _sample(exporter.export(), _TRANSITIONS)["value"] == 1
    assert recorder.snapshot().counters[0].name is (
        CounterName.FEDERATED_PHASE_TRANSITIONS
    )
    recorder.clear()
    assert recorder.export()["counters"] == []


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (0, FederatedUpdateBand.SUPPRESSED),
        (4, FederatedUpdateBand.SUPPRESSED),
        (5, FederatedUpdateBand.MINIMUM_TO_UNDER_DOUBLE),
        (9, FederatedUpdateBand.MINIMUM_TO_UNDER_DOUBLE),
        (10, FederatedUpdateBand.DOUBLE_TO_UNDER_FOURFOLD),
        (19, FederatedUpdateBand.DOUBLE_TO_UNDER_FOURFOLD),
        (20, FederatedUpdateBand.FOURFOLD_OR_MORE),
        (2_000, FederatedUpdateBand.FOURFOLD_OR_MORE),
    ],
)
def test_band_update_count_matches_the_shared_banding(
    count: int, expected: FederatedUpdateBand
) -> None:
    assert band_update_count(count) is expected
    assert band_update_count(count, minimum_group_size=5).value == expected.value


def test_band_update_count_honours_a_custom_minimum_group_size() -> None:
    assert band_update_count(9, minimum_group_size=10) is FederatedUpdateBand.SUPPRESSED
    assert (
        band_update_count(10, minimum_group_size=10)
        is FederatedUpdateBand.MINIMUM_TO_UNDER_DOUBLE
    )
    with pytest.raises(FederatedTelemetryError, match="greater than one"):
        band_update_count(3, minimum_group_size=1)
    with pytest.raises(FederatedTelemetryError, match="non-negative integer"):
        band_update_count(True)  # type: ignore[arg-type]


def test_federated_vocabularies_track_the_training_contracts() -> None:
    assert FEDERATED_PHASE_VALUES[-1] == "other"
    assert FEDERATED_PHASE_VALUES[:-1] == tuple(
        state.value for state in FederatedRoundState
    )
    assert FEDERATED_REASON_CODE_VALUES[-1] == "other"
    assert FEDERATED_REASON_CODE_VALUES[:-1] == tuple(
        code.value for code in FederatedRoundReasonCode
    )
    assert FEDERATED_UPDATE_BAND_VALUES[:-1] == tuple(
        band.value for band in FederatedParticipantCountBand
    )
    assert (
        tuple(band.value for band in FederatedUpdateBand)
        == (FEDERATED_UPDATE_BAND_VALUES[:-1])
    )
    assert FEDERATED_PHASE_STATUS_VALUES == PIPELINE_STATUS_VALUES
    assert (
        FEDERATED_TELEMETRY_MINIMUM_GROUP_SIZE
        == DEFAULT_FEDERATED_MINIMUM_GROUP_SIZE
        == 5
    )
