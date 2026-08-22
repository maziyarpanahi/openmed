"""Focused tests for the privacy-waiver lifecycle ledger."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from openmed.compliance import (
    InvalidWaiverIdentifierError,
    InvalidWaiverTransitionError,
    WaiverEventType,
    WaiverLedger,
    WaiverState,
    render_active_state_counts,
)


def test_ledger_records_the_complete_synthetic_lifecycle() -> None:
    ledger = WaiverLedger()

    ledger.create("wvr_001", "pol_privacy_001")
    ledger.approve("wvr_001")
    ledger.create("wvr_002", "pol_privacy_001")
    ledger.approve("wvr_002")
    ledger.supersede("wvr_002", replacement_waiver_id="wvr_003")
    ledger.create("wvr_004", "pol_privacy_001")
    ledger.approve("wvr_004")
    ledger.revoke("wvr_004")
    ledger.create("wvr_005", "pol_privacy_001")
    ledger.approve("wvr_005")
    ledger.expire("wvr_005")

    assert [event.event_type for event in ledger.events] == [
        WaiverEventType.CREATE,
        WaiverEventType.APPROVE,
        WaiverEventType.CREATE,
        WaiverEventType.APPROVE,
        WaiverEventType.SUPERSEDE,
        WaiverEventType.CREATE,
        WaiverEventType.APPROVE,
        WaiverEventType.REVOKE,
        WaiverEventType.CREATE,
        WaiverEventType.APPROVE,
        WaiverEventType.EXPIRE,
    ]
    assert [event.sequence for event in ledger.events] == list(range(11))
    assert ledger.current_state("wvr_001") is WaiverState.ACTIVE
    assert ledger.current_state("wvr_002") is WaiverState.SUPERSEDED
    assert ledger.current_state("wvr_004") is WaiverState.REVOKED
    assert ledger.current_state("wvr_005") is WaiverState.EXPIRED
    assert ledger.events[4].superseded_by == "wvr_003"


def test_invalid_transitions_are_rejected_without_mutating_the_ledger() -> None:
    ledger = WaiverLedger()
    ledger.create("wvr_001", "pol_privacy_001")

    with pytest.raises(InvalidWaiverTransitionError):
        ledger.revoke("wvr_001")
    assert len(ledger) == 1

    ledger.approve("wvr_001")
    with pytest.raises(InvalidWaiverTransitionError):
        ledger.approve("wvr_001")
    with pytest.raises(InvalidWaiverTransitionError):
        ledger.expire("wvr_001", "pol_other_001")
    assert len(ledger) == 2

    with pytest.raises(InvalidWaiverTransitionError):
        ledger.create("wvr_001", "pol_privacy_001")
    assert len(ledger) == 2


def test_identifiers_and_errors_do_not_carry_raw_sensitive_values() -> None:
    untrusted = "synthetic sensitive finding text"
    ledger = WaiverLedger()

    with pytest.raises(InvalidWaiverIdentifierError) as error:
        ledger.create(untrusted, "pol_privacy_001")

    assert untrusted not in str(error.value)
    assert ledger.events == ()

    ledger.create("wvr_001", "pol_privacy_001")
    payload = json.dumps(ledger.to_dict(), sort_keys=True)
    assert untrusted not in payload
    assert "finding" not in payload
    assert "identity" not in payload


def test_state_counts_and_rendering_are_deterministic_and_aggregate_only() -> None:
    def build() -> WaiverLedger:
        result = WaiverLedger()
        result.create("wvr_001", "pol_privacy_001")
        result.approve("wvr_001")
        result.create("wvr_002", "pol_privacy_001")
        result.approve("wvr_002")
        result.expire("wvr_002")
        result.create("wvr_003", "pol_privacy_001")
        return result

    first = build()
    second = build()
    expected = {
        "pending": 1,
        "active": 1,
        "superseded": 0,
        "revoked": 0,
        "expired": 1,
    }

    assert first.state_counts() == expected
    assert first.active_state_counts() == expected
    assert first.render_active_state_counts() == second.render_active_state_counts()
    assert render_active_state_counts(first) == first.render_active_state_counts()
    assert json.loads(first.render_active_state_counts()) == expected
    assert "wvr_001" not in first.render_active_state_counts()


def test_event_records_are_frozen_and_json_round_trip_is_stable() -> None:
    ledger = WaiverLedger()
    ledger.create("wvr_001", "pol_privacy_001")
    ledger.approve("wvr_001")

    with pytest.raises(FrozenInstanceError):
        ledger.events[0].state = WaiverState.ACTIVE  # type: ignore[misc]

    restored = WaiverLedger.from_json(ledger.to_json())
    assert restored.to_json() == ledger.to_json()
    assert restored.events == ledger.events
