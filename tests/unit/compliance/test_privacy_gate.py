"""Focused tests for the deterministic privacy release-gate aggregator."""

from __future__ import annotations

import json

import pytest

from openmed.compliance import (
    GateStatus,
    PrivacyGateResult,
    PrivacyReleaseGateRecord,
    ReleaseDecision,
    aggregate_privacy_gates,
    render_privacy_release_gate,
)
from openmed.core.audit import stable_hash


def test_aggregate_uses_explicit_state_precedence_and_counts() -> None:
    gates = [
        PrivacyGateResult("dependency", GateStatus.WAIVED, waiver_code="approved"),
        PrivacyGateResult("evidence", GateStatus.WARNING, finding_count=2),
        PrivacyGateResult("risk", GateStatus.BLOCKING, finding_count=1),
    ]

    record = aggregate_privacy_gates(gates)

    assert record.decision is ReleaseDecision.BLOCKED
    assert record.released is False
    assert dict(record.counts) == {"blocking": 1, "warning": 1, "waived": 1}
    assert [gate.gate for gate in record.gates] == [
        "dependency",
        "evidence",
        "risk",
    ]
    assert record.gate_fingerprints["risk"].startswith("sha256:")
    assert record.to_dict()["gates"][0]["waiver_code"] == "approved"


def test_warning_is_released_only_when_no_warning_or_blocking_gate_remains() -> None:
    warning = aggregate_privacy_gates(
        [PrivacyGateResult("policy", "warning", finding_count=1)]
    )
    waived = aggregate_privacy_gates(
        [PrivacyGateResult("policy", "waived", waiver_code="reviewed")]
    )

    assert warning.decision is ReleaseDecision.WARNING
    assert waived.decision is ReleaseDecision.RELEASED
    assert dict(waived.counts) == {"blocking": 0, "warning": 0, "waived": 1}


def test_rendering_is_stable_and_counts_only() -> None:
    sensitive = "synthetic patient value 12345"
    raw_result = {
        "gate": "risk",
        "status": "blocking",
        "findings": [{"value": sensitive, "detail": "do not export"}],
        "details": {"raw_value": sensitive},
    }
    first = aggregate_privacy_gates(
        [raw_result, {"gate": "policy", "status": "waived", "findings": []}]
    )
    second = aggregate_privacy_gates(
        [{"gate": "policy", "status": "waived", "findings": []}, raw_result]
    )

    rendered = render_privacy_release_gate(first)
    assert rendered == second.to_json()
    assert rendered == first.to_json()
    assert sensitive not in rendered
    payload = json.loads(rendered)
    assert set(payload) == {
        "counts",
        "decision",
        "fingerprint",
        "gates",
        "report_type",
        "schema_version",
    }
    assert all(
        set(gate) <= {"fingerprint", "finding_count", "gate", "status"}
        for gate in payload["gates"]
    )


def test_fingerprints_are_deterministic_and_round_trip_without_trusting_input() -> None:
    gate = PrivacyGateResult(
        "risk",
        "blocking",
        finding_count=2,
        fingerprint=stable_hash({"synthetic": "summary"}),
    )
    record = aggregate_privacy_gates([gate])
    payload = record.to_dict()
    payload["fingerprint"] = stable_hash({"tampered": True})

    rebuilt = PrivacyReleaseGateRecord.from_dict(payload)

    assert rebuilt.to_json() == record.to_json()
    assert rebuilt.fingerprint == record.fingerprint


def test_invalid_inputs_do_not_echo_untrusted_values() -> None:
    sensitive = "synthetic patient value 98765"

    with pytest.raises(ValueError, match="stable identifier") as error:
        PrivacyGateResult(sensitive, "blocking")
    assert sensitive not in str(error.value)

    with pytest.raises(ValueError, match="sha256 digest") as error:
        PrivacyGateResult("risk", "blocking", fingerprint=sensitive)
    assert sensitive not in str(error.value)

    with pytest.raises(ValueError, match="blocking, warning, or waived") as error:
        PrivacyGateResult("risk", sensitive)
    assert sensitive not in str(error.value)


def test_gate_result_rejects_duplicate_names_and_raw_findings_are_only_counted() -> (
    None
):
    result = PrivacyGateResult.from_findings(
        "risk",
        "blocking",
        [{"sensitive": "synthetic patient value"}, {"sensitive": "second"}],
    )

    assert result.finding_count == 2
    assert result.to_dict()["fingerprint"].startswith("sha256:")
    with pytest.raises(ValueError, match="unique"):
        aggregate_privacy_gates([result, result])


def test_empty_and_non_sized_findings_fail_closed() -> None:
    with pytest.raises(ValueError, match="at least one"):
        aggregate_privacy_gates([])
    with pytest.raises(TypeError, match="sized collection"):
        PrivacyGateResult.from_findings("risk", "blocking", iter([1]))

    sensitive = "synthetic patient value from len"

    class SensitiveSized:
        def __len__(self) -> int:
            raise RuntimeError(sensitive)

    with pytest.raises(TypeError, match="countable collection") as error:
        PrivacyGateResult.from_findings("risk", "blocking", SensitiveSized())
    assert sensitive not in str(error.value)
