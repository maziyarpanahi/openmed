"""Offline regressions for the empty-evidence summary refusal gate."""

from __future__ import annotations

import json
import socket

import pytest

from openmed.clinical import (
    SUMMARY_EMPTY_EVIDENCE_REFUSAL_CODE,
    SummaryEmptyEvidenceError,
    SummaryEmptyEvidenceRefusal,
    build_summary_empty_evidence_refusal,
    guard_summary_generation,
    require_summary_evidence,
)

PRIVATE_SENTINEL = "SYNTHETIC_SOURCE_VALUE"


def test_empty_input_refuses_without_invoking_the_generator() -> None:
    calls: list[tuple[object, ...]] = []

    def generator(evidence: tuple[object, ...]) -> str:
        calls.append(evidence)
        return "synthetic summary"

    result = guard_summary_generation([], generator)

    assert isinstance(result, SummaryEmptyEvidenceRefusal)
    assert result.refusal_code == SUMMARY_EMPTY_EVIDENCE_REFUSAL_CODE
    assert result.to_dict()["status"] == "refused"
    assert result.to_dict()["input_evidence_count"] == 0
    assert result.to_dict()["approved_evidence_count"] == 0
    assert result.to_dict()["excluded_evidence_count"] == 0
    assert calls == []


def test_wholly_invalid_evidence_returns_counts_only_and_no_source_text() -> None:
    result = guard_summary_generation(
        [
            None,
            {"approved": True, "valid": False, "text": PRIVATE_SENTINEL},
            {"approved": False, "text": PRIVATE_SENTINEL},
            PRIVATE_SENTINEL,
        ],
        lambda _evidence: "must not be generated",
    )

    assert isinstance(result, SummaryEmptyEvidenceRefusal)
    assert result.to_dict()["input_evidence_count"] == 4
    assert result.to_dict()["approved_evidence_count"] == 0
    assert result.to_dict()["excluded_evidence_count"] == 4
    serialized = result.to_json()
    assert PRIVATE_SENTINEL not in serialized
    assert PRIVATE_SENTINEL not in str(result)
    assert json.loads(serialized) == result.to_dict()


def test_mixed_evidence_only_passes_surviving_records_to_the_generator() -> None:
    approved = {"approved": True, "text": "SYNTHETIC_APPROVED_VALUE"}
    rejected = {"approved": False, "text": PRIVATE_SENTINEL}
    calls: list[tuple[object, ...]] = []

    def generator(evidence: tuple[object, ...]) -> int:
        calls.append(evidence)
        return len(evidence)

    result = guard_summary_generation([rejected, approved], generator)

    assert result == 1
    assert len(calls) == 1
    assert calls[0] == (approved,)
    assert calls[0][0] is approved
    assert rejected not in calls[0]


def test_refusal_is_deterministic_for_equivalent_input_orderings() -> None:
    first = [
        {"approved": False, "text": "SYNTHETIC_REJECTED_A"},
        {"approved": True, "valid": False, "text": "SYNTHETIC_INVALID_B"},
    ]
    second = list(reversed(first))

    first_refusal = build_summary_empty_evidence_refusal(first)
    second_refusal = build_summary_empty_evidence_refusal(second)

    assert first_refusal is not None
    assert second_refusal is not None
    assert first_refusal.to_json() == second_refusal.to_json()


def test_strict_boundary_raises_a_fixed_code_exception_without_source_text() -> None:
    with pytest.raises(SummaryEmptyEvidenceError) as caught:
        require_summary_evidence([{"approved": False, "text": PRIVATE_SENTINEL}])

    assert caught.value.refusal_code == SUMMARY_EMPTY_EVIDENCE_REFUSAL_CODE
    assert str(caught.value) == ("summary_refused_empty_approved_evidence")
    assert PRIVATE_SENTINEL not in str(caught.value)
    assert PRIVATE_SENTINEL not in json.dumps(caught.value.to_dict())


def test_empty_path_performs_no_network_call(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_socket(*args: object, **kwargs: object) -> None:
        raise AssertionError("empty-evidence refusal must remain local")

    monkeypatch.setattr(socket, "socket", fail_socket)

    result = guard_summary_generation(
        None,
        lambda _evidence: (_ for _ in ()).throw(
            AssertionError("generator should not run")
        ),
    )

    assert isinstance(result, SummaryEmptyEvidenceRefusal)
