from __future__ import annotations

import json
import traceback
import urllib.request
from dataclasses import replace
from typing import Any

import pytest

from openmed.agent.workflows import (
    QUALITY_MEASURE_EVIDENCE_SCHEMA,
    CalculationTraceStep,
    DriftChange,
    DriftSource,
    ExclusionEvidence,
    InputProjectionEvidence,
    MeasureDefinitionEvidence,
    MeasureImplementationEvidence,
    MeasureResultEvidence,
    MeasureTimeBoundaries,
    QualityMeasureDriftReport,
    QualityMeasureEvidenceError,
    QualityMeasureEvidencePacket,
    ValueSetEvidence,
    build_quality_measure_evidence_packet,
    compare_quality_measure_evidence,
)

DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64
DIGEST_C = "sha256:" + "c" * 64
DIGEST_D = "sha256:" + "d" * 64
DIGEST_E = "sha256:" + "e" * 64
DIGEST_F = "sha256:" + "f" * 64
DIGEST_1 = "sha256:" + "1" * 64
DIGEST_2 = "sha256:" + "2" * 64
DIGEST_3 = "sha256:" + "3" * 64
DIGEST_4 = "sha256:" + "4" * 64
SIGNING_KEY = b"synthetic-unit-signing-key"


def _definition(
    *, version: str = "2026.1", digest: str = DIGEST_A
) -> MeasureDefinitionEvidence:
    return MeasureDefinitionEvidence("CMS-example", version, digest)


def _value_sets() -> tuple[ValueSetEvidence, ...]:
    return (
        ValueSetEvidence("valueset.numerator", "2026.1", DIGEST_C),
        ValueSetEvidence("valueset.denominator", "2026.1", DIGEST_B),
    )


def _projection(*, digest: str = DIGEST_D, count: int = 240) -> InputProjectionEvidence:
    return InputProjectionEvidence(DIGEST_C, digest, count)


def _boundaries(*, end: str = "2027-01-01T00:00:00Z") -> MeasureTimeBoundaries:
    return MeasureTimeBoundaries("2026-01-01T00:00:00Z", end)


def _exclusions() -> tuple[ExclusionEvidence, ...]:
    return (ExclusionEvidence("hospice", DIGEST_E, DIGEST_F, 3),)


def _implementation(
    *, version: str = "1.0.0", digest: str = DIGEST_1
) -> MeasureImplementationEvidence:
    return MeasureImplementationEvidence("openmed.quality.example", version, digest)


def _trace(
    *, logic_digest: str = DIGEST_2, output_digest: str = DIGEST_3
) -> tuple[CalculationTraceStep, ...]:
    return (
        CalculationTraceStep(
            "denominator",
            "filter",
            logic_digest,
            DIGEST_D,
            output_digest,
            120,
        ),
        CalculationTraceStep(
            "numerator", "aggregate", DIGEST_3, output_digest, DIGEST_4, 84
        ),
    )


def _result(*, numerator: int = 84, digest: str = DIGEST_4) -> MeasureResultEvidence:
    return MeasureResultEvidence(120, numerator, 3, digest)


def _packet(
    *,
    definition: MeasureDefinitionEvidence | None = None,
    value_sets: tuple[ValueSetEvidence, ...] | None = None,
    projection: InputProjectionEvidence | None = None,
    boundaries: MeasureTimeBoundaries | None = None,
    exclusions: tuple[ExclusionEvidence, ...] | None = None,
    implementation: MeasureImplementationEvidence | None = None,
    trace: tuple[CalculationTraceStep, ...] | None = None,
    result: MeasureResultEvidence | None = None,
    key_id: str = "quality-evidence-2026",
    signing_key: bytes = SIGNING_KEY,
) -> QualityMeasureEvidencePacket:
    return build_quality_measure_evidence_packet(
        _definition() if definition is None else definition,
        _value_sets() if value_sets is None else value_sets,
        _projection() if projection is None else projection,
        _boundaries() if boundaries is None else boundaries,
        _exclusions() if exclusions is None else exclusions,
        _implementation() if implementation is None else implementation,
        _trace() if trace is None else trace,
        _result() if result is None else result,
        key_id=key_id,
        signing_key=signing_key,
    )


def test_packet_captures_complete_signed_reproducibility_evidence() -> None:
    packet = _packet()

    assert packet.schema == QUALITY_MEASURE_EVIDENCE_SCHEMA
    assert packet.verify(SIGNING_KEY)
    assert packet.packet_digest.startswith("sha256:")
    assert packet.signature.startswith("hmac-sha256:")
    assert packet.to_dict()["measure_definition"] == {
        "definition_digest": DIGEST_A,
        "measure_id": "CMS-example",
        "version": "2026.1",
    }
    assert packet.to_dict()["time_boundaries"] == {
        "end": "2027-01-01T00:00:00Z",
        "start": "2026-01-01T00:00:00Z",
    }
    assert packet.to_dict()["result"]["numerator_count"] == 84
    assert [item["step_id"] for item in packet.to_dict()["calculation_trace"]] == [
        "denominator",
        "numerator",
    ]


def test_packet_is_deterministic_and_normalizes_unordered_components() -> None:
    first = _packet()
    second = _packet(
        value_sets=tuple(reversed(_value_sets())),
        exclusions=tuple(reversed(_exclusions())),
    )

    assert first == second
    assert first.to_json() == second.to_json()
    assert [item.value_set_id for item in first.value_sets] == [
        "valueset.denominator",
        "valueset.numerator",
    ]


def test_round_trip_preserves_packet_and_signature() -> None:
    packet = _packet()

    restored = QualityMeasureEvidencePacket.from_json(packet.to_json())

    assert restored == packet
    assert restored.verify(SIGNING_KEY)


def test_signature_detects_any_signed_field_mutation() -> None:
    packet = _packet()
    mutated = packet.to_dict()
    mutated["result"]["numerator_count"] = 85
    restored = QualityMeasureEvidencePacket.from_dict(mutated)

    with pytest.raises(QualityMeasureEvidenceError) as caught:
        restored.verify(SIGNING_KEY)

    assert caught.value.code == "signature_mismatch"
    assert "85" not in str(caught.value)


def test_compare_identifies_semantic_drift_without_copying_values() -> None:
    baseline = _packet()
    current_value_sets = (
        ValueSetEvidence("valueset.denominator", "2026.2", DIGEST_1),
        ValueSetEvidence("valueset.new", "2026.1", DIGEST_2),
    )
    current_exclusions = (ExclusionEvidence("hospice", DIGEST_1, DIGEST_2, 4),)
    current = _packet(
        definition=_definition(version="2026.2", digest=DIGEST_1),
        value_sets=current_value_sets,
        projection=_projection(digest=DIGEST_1, count=241),
        boundaries=_boundaries(end="2027-02-01T00:00:00Z"),
        exclusions=current_exclusions,
        implementation=_implementation(version="1.1.0", digest=DIGEST_2),
        trace=_trace(logic_digest=DIGEST_3, output_digest=DIGEST_1),
        result=_result(numerator=85, digest=DIGEST_2),
    )

    report = compare_quality_measure_evidence(baseline, current)
    changes = {(item.source, item.change, item.component_id) for item in report.drift}

    assert report.result_changed
    assert not report.is_equivalent
    assert (
        DriftSource.MEASURE_DEFINITION,
        DriftChange.CHANGED,
        "CMS-example",
    ) in changes
    assert (
        DriftSource.VALUE_SET,
        DriftChange.CHANGED,
        "valueset.denominator",
    ) in changes
    assert (
        DriftSource.VALUE_SET,
        DriftChange.REMOVED,
        "valueset.numerator",
    ) in changes
    assert (
        DriftSource.VALUE_SET,
        DriftChange.ADDED,
        "valueset.new",
    ) in changes
    assert (
        DriftSource.INPUT_PROJECTION,
        DriftChange.CHANGED,
        "input_projection",
    ) in changes
    assert (
        DriftSource.TIME_BOUNDARIES,
        DriftChange.CHANGED,
        "measurement_period",
    ) in changes
    assert (
        DriftSource.EXCLUSION_DEFINITION,
        DriftChange.CHANGED,
        "hospice",
    ) in changes
    assert (
        DriftSource.EXCLUSION_EVIDENCE,
        DriftChange.CHANGED,
        "hospice",
    ) in changes
    assert (
        DriftSource.IMPLEMENTATION,
        DriftChange.CHANGED,
        "openmed.quality.example",
    ) in changes
    assert (
        DriftSource.CALCULATION_LOGIC,
        DriftChange.CHANGED,
        "denominator",
    ) in changes
    assert (
        DriftSource.CALCULATION_OUTPUT,
        DriftChange.CHANGED,
        "denominator",
    ) in changes

    serialized = report.to_json()
    for private_value in (
        "2027-02-01T00:00:00Z",
        DIGEST_1,
        current.signature,
        '"record_count":241',
        '"numerator_count":85',
    ):
        assert private_value not in serialized


def test_compare_equivalent_content_ignores_signature_bytes() -> None:
    baseline = _packet(signing_key=b"first-synthetic-signing-key")
    current = _packet(signing_key=b"second-synthetic-signing-key")

    report = compare_quality_measure_evidence(baseline, current)

    assert report.is_equivalent
    assert report.drift == ()
    assert not report.result_changed


def test_trace_order_change_is_explicit_semantic_drift() -> None:
    baseline = _packet()
    current = _packet(trace=tuple(reversed(_trace())))

    report = compare_quality_measure_evidence(baseline, current)

    assert any(
        item.source is DriftSource.CALCULATION_LOGIC
        and item.component_id == "trace_order"
        for item in report.drift
    )


@pytest.mark.parametrize(
    ("factory", "code"),
    [
        (
            lambda: MeasureDefinitionEvidence("unsafe id", "1.0", DIGEST_A),
            "invalid_identifier",
        ),
        (
            lambda: ValueSetEvidence("safe", "unsafe version", DIGEST_A),
            "invalid_version",
        ),
        (lambda: InputProjectionEvidence("bad", DIGEST_A, 1), "invalid_digest"),
        (
            lambda: MeasureTimeBoundaries("2026-01-01", "2027-01-01T00:00:00Z"),
            "invalid_timestamp",
        ),
        (
            lambda: MeasureTimeBoundaries(
                "2027-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
            ),
            "invalid_time_range",
        ),
        (lambda: MeasureResultEvidence(2, 3, 0, DIGEST_A), "count_out_of_range"),
    ],
)
def test_invalid_metadata_fails_closed(factory: Any, code: str) -> None:
    with pytest.raises(QualityMeasureEvidenceError) as caught:
        factory()

    assert caught.value.code == code


def test_duplicate_components_and_trace_steps_fail_closed() -> None:
    value_set = _value_sets()[0]
    step = _trace()[0]

    with pytest.raises(QualityMeasureEvidenceError, match="duplicate_item"):
        _packet(value_sets=(value_set, value_set))
    with pytest.raises(QualityMeasureEvidenceError, match="duplicate_item"):
        _packet(trace=(step, step))


def test_parser_rejects_unknown_unsigned_fields() -> None:
    payload = _packet().to_dict()
    payload["patient_name"] = "Synthetic Person"

    with pytest.raises(QualityMeasureEvidenceError) as caught:
        QualityMeasureEvidencePacket.from_dict(payload)

    assert caught.value.code == "invalid_fields"
    assert "Synthetic Person" not in str(caught.value)


def test_parser_rejects_duplicate_json_fields() -> None:
    serialized = (
        _packet()
        .to_json()
        .replace(
            '"schema":"openmed.agent.workflows.quality_measure_evidence.v1",',
            '"schema":"openmed.agent.workflows.quality_measure_evidence.v1",'
            '"schema":"openmed.agent.workflows.quality_measure_evidence.v1",',
        )
    )

    with pytest.raises(QualityMeasureEvidenceError) as caught:
        QualityMeasureEvidencePacket.from_json(serialized)

    assert caught.value.code == "duplicate_json_key"


def test_packet_repr_report_and_errors_do_not_leak_sensitive_values() -> None:
    sentinel = "Synthetic Person / patient-123 /tmp/private.csv"
    packet = _packet()
    report = compare_quality_measure_evidence(packet, packet)

    assert sentinel not in repr(packet)
    assert packet.signature not in repr(packet)
    assert DIGEST_D not in report.to_json()
    with pytest.raises(QualityMeasureEvidenceError) as caught:
        MeasureDefinitionEvidence(sentinel, "1.0", DIGEST_A)
    assert sentinel not in str(caught.value)
    assert sentinel not in "".join(traceback.format_exception(caught.value))

    with pytest.raises(QualityMeasureEvidenceError) as caught:
        QualityMeasureDriftReport(
            DIGEST_A,
            DIGEST_B,
            False,
            (sentinel,),  # type: ignore[arg-type]
            DIGEST_C,
        )
    assert sentinel not in str(caught.value)


def test_signing_requires_caller_managed_key_with_minimum_length() -> None:
    with pytest.raises(QualityMeasureEvidenceError) as caught:
        _packet(signing_key=b"short")

    assert caught.value.code == "invalid_signing_key"
    assert "short" not in str(caught.value)


def test_build_compare_and_verify_perform_no_network_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def deny_network(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(urllib.request, "urlopen", deny_network)

    first = _packet()
    second = QualityMeasureEvidencePacket.from_json(first.to_json())

    assert second.verify(SIGNING_KEY)
    assert compare_quality_measure_evidence(first, second).is_equivalent


def test_report_digest_binds_closed_drift_metadata() -> None:
    report = compare_quality_measure_evidence(
        _packet(),
        _packet(result=_result(numerator=85, digest=DIGEST_1)),
    )
    rendered = json.loads(report.to_json())

    assert report.result_changed
    assert report.drift == ()
    assert rendered["report_digest"].startswith("sha256:")
    assert replace(report, report_digest=DIGEST_A) != report
