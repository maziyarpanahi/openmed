"""Offline tests for FHIR update preconditions and redacted review reports."""

from __future__ import annotations

import socket

import pytest

from openmed.interop.fhir.concurrency_guard import (
    FHIRWriteConflict,
    VersionEvidence,
    guard_update,
    require_no_server_conflict,
    summarize_conflict,
)

_MARKER = "private-marker"


def _resource(version: str = "4", timestamp: str = "2026-01-01T00:00:00Z") -> dict:
    return {
        "resourceType": "Observation",
        "id": "synthetic-resource",
        "meta": {"versionId": version, "lastUpdated": timestamp},
        "status": "final",
        "note": {"text": _MARKER},
    }


def test_update_requires_both_read_evidence_fields_and_sends_if_match() -> None:
    expected = VersionEvidence.from_resource(_resource())
    assert guard_update(
        expected, VersionEvidence.from_resource(_resource())
    ).if_match == ('W/"4"')
    for observed in (
        VersionEvidence.from_resource(_resource(version="5")),
        VersionEvidence.from_resource(_resource(timestamp="2026-01-02T00:00:00Z")),
    ):
        with pytest.raises(FHIRWriteConflict) as error:
            guard_update(expected, observed)
        assert error.value.reason_code == "stale_evidence"
        assert _MARKER not in repr(error.value)


@pytest.mark.parametrize(
    "meta",
    [
        {},
        {"versionId": "4"},
        {"lastUpdated": "2026-01-01T00:00:00Z"},
        {"versionId": "4\r\nprivate-marker", "lastUpdated": "2026-01-01T00:00:00Z"},
        {"versionId": "4", "lastUpdated": "private-marker"},
        {"versionId": "4", "lastUpdated": "2026-01-01T00:00:00"},
    ],
)
def test_missing_or_invalid_evidence_fails_closed_without_echo(meta: dict) -> None:
    resource = _resource()
    resource["meta"] = meta
    with pytest.raises(ValueError) as error:
        VersionEvidence.from_resource(resource)
    assert _MARKER not in str(error.value)


def test_precondition_and_conflict_representations_are_redacted() -> None:
    evidence = VersionEvidence("private-marker", "2026-01-01T00:00:00Z")
    precondition = guard_update(evidence, evidence)
    assert _MARKER not in repr(evidence)
    assert _MARKER not in repr(precondition)
    assert precondition.if_match == 'W/"private-marker"'


@pytest.mark.parametrize(
    "status, reason",
    [
        (409, "server_conflict"),
        (412, "precondition_failed"),
        (428, "precondition_required"),
    ],
)
def test_server_precondition_failures_are_typed_and_value_free(
    status: int, reason: str
) -> None:
    with pytest.raises(FHIRWriteConflict) as error:
        require_no_server_conflict(status)
    assert error.value.reason_code == reason
    assert _MARKER not in str(error.value)
    require_no_server_conflict(200)


def test_three_way_summary_reports_only_counts_and_never_merges() -> None:
    base = _resource()
    current = _resource(version="5")
    proposed = _resource()
    current["note"]["text"] = "server-private-marker"
    proposed["note"]["text"] = "proposal-private-marker"
    current["status"] = "amended"
    proposed["status"] = "amended"
    summary = summarize_conflict(base, current, proposed)
    assert (
        summary.server_changes,
        summary.proposed_changes,
        summary.overlapping_changes,
        summary.divergent_changes,
        summary.already_applied_changes,
    ) == (2, 2, 2, 1, 1)
    assert "private-marker" not in repr(summary)
    assert base["note"]["text"] == _MARKER
    assert proposed["note"]["text"] == "proposal-private-marker"


def test_nonoverlapping_change_and_identity_mismatch() -> None:
    base = _resource()
    current = _resource(version="5")
    current["status"] = "amended"
    proposed = _resource()
    proposed["note"]["text"] = "proposal-private-marker"
    summary = summarize_conflict(base, current, proposed)
    assert summary.overlapping_changes == 0
    proposed["id"] = "other-private-marker"
    with pytest.raises(ValueError) as error:
        summarize_conflict(base, current, proposed)
    assert "private-marker" not in str(error.value)


def test_parent_deletion_overlaps_a_nested_proposal() -> None:
    base = _resource()
    current = _resource(version="5")
    current.pop("note")
    proposed = _resource()
    proposed["note"]["text"] = "proposal-private-marker"
    summary = summarize_conflict(base, current, proposed)
    assert summary.overlapping_changes == 1
    assert summary.divergent_changes == 1


def test_guard_is_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_socket(*args, **kwargs):
        raise AssertionError("unexpected network access")

    monkeypatch.setattr(socket, "socket", fail_socket)
    evidence = VersionEvidence.from_resource(_resource())
    guard_update(evidence, evidence)
    summarize_conflict(_resource(), _resource(), _resource())
