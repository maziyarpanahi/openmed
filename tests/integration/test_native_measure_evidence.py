"""Offline native evaluation through signed evidence and drift comparison."""

import socket
import urllib.request
from dataclasses import replace

import pytest

from openmed.agent.workflows import (
    DriftSource,
    QualityMeasureEvidencePacket,
    build_native_measure_evidence_packet,
    compare_quality_measure_evidence,
)
from openmed.clinical.journey_contracts import ClinicalFact, canonical_digest
from openmed.clinical.measures import (
    MeasureDefinition,
    MeasureLanguage,
    MeasurePopulationDefinition,
    MeasureTimeWindow,
    NativePopulationRule,
    PopulationKind,
    compare_measure_runs,
    evaluate_native_measure,
)


@pytest.mark.integration
def test_native_evaluation_signing_round_trip_and_drift_are_offline(
    monkeypatch, capsys
):
    def deny_io(*args, **kwargs):
        pytest.fail("Unexpected network operation")

    monkeypatch.setattr(socket, "create_connection", deny_io)
    monkeypatch.setattr(socket.socket, "connect", deny_io)
    monkeypatch.setattr(urllib.request, "urlopen", deny_io)
    subject = "patient_syntheticpatient01"
    definition = MeasureDefinition(
        "measure_syntheticmeasure01",
        "1.0.0",
        MeasureLanguage.NATIVE,
        (
            MeasurePopulationDefinition(
                "denominator", PopulationKind.DENOMINATOR, "native:denominator"
            ),
            MeasurePopulationDefinition(
                "numerator", PopulationKind.NUMERATOR, "native:numerator"
            ),
        ),
    )
    facts = tuple(
        ClinicalFact(
            fact_id=f"fact_syntheticfact{index:04d}",
            subject_id=subject,
            fact_type=name,
            value="SYNTHETIC protected person 555-010-9999",
            status="active",
            evidence_ids=(f"evidence_syntheticevidence{index:04d}",),
            derivation_hash=canonical_digest({"synthetic": index}),
        )
        for index, name in enumerate(("denominator", "numerator"))
    )
    rules = tuple(
        NativePopulationRule(name, (name,), ("active",))
        for name in ("denominator", "numerator")
    )

    def evaluate(rules):
        result = evaluate_native_measure(
            definition,
            rules,
            facts,
            subject_ids=(subject,),
            source_snapshot_id="snapshot_syntheticsnapshot1",
            source_snapshot_digest=canonical_digest({"synthetic_snapshot": 1}),
            measurement_period=MeasureTimeWindow(
                "2026-01-01T00:00:00Z", "2026-12-31T23:59:59Z"
            ),
            evaluated_at="2027-01-01T00:00:00Z",
        )
        assert result.value is not None
        return result.value

    baseline = evaluate(rules)
    candidate = evaluate((rules[0], replace(rules[1], minimum_matches=2)))
    key = b"synthetic-local-evidence-key"
    key_requests = []

    def provider(key_id):
        key_requests.append(key_id)
        return key

    packets = [
        build_native_measure_evidence_packet(
            run,
            input_schema_digest=canonical_digest({"synthetic_input_schema": "1.0.0"}),
            key_id="synthetic.local.key",
            key_provider=provider,
        )
        for run in (baseline, candidate)
    ]
    restored = [QualityMeasureEvidencePacket.from_json(p.to_json()) for p in packets]
    assert all(p.verify(key) for p in restored)
    assert restored == packets
    assert key_requests == ["synthetic.local.key"] * 2
    native = compare_measure_runs(baseline, candidate)
    evidence = compare_quality_measure_evidence(*restored)
    assert native.result_drift == evidence.result_changed == True
    assert set(native.reason_codes) == {
        "input_projection_changed",
        "population_counts_changed",
    }
    # Native rule configuration is committed in inputs, not recoverable as logic.
    assert {d.source for d in evidence.drift} == {
        DriftSource.INPUT_PROJECTION,
        DriftSource.CALCULATION_OUTPUT,
    }
    for packet in restored:
        assert subject not in packet.to_json()
        assert facts[0].value not in packet.to_json()
    assert capsys.readouterr() == ("", "")
