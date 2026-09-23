"""Offline, synthetic FHIR Subscription checkpoint contracts."""

from __future__ import annotations

import sqlite3

import pytest

from openmed.interop.fhir.subscription_checkpoint import (
    SubscriptionCheckpoint,
    SubscriptionCheckpointError,
)

_SECRET = b"synthetic-test-key-32-bytes-long-000000"


def _event(
    sequence: int,
    *,
    notification: str | None = None,
    resource: str | None = None,
    version: str | None = None,
) -> dict:
    return {
        "subscription_id": "synthetic-subscription",
        "notification_id": notification or f"synthetic-notification-{sequence}",
        "resource_id": resource or f"synthetic-resource-{sequence}",
        "resource_version": version or f"synthetic-version-{sequence}",
        "sequence": sequence,
    }


def test_claim_commit_retry_and_restart(tmp_path):
    path = tmp_path / "checkpoint.sqlite"
    event = _event(0)
    with SubscriptionCheckpoint(path, secret=_SECRET) as store:
        assert store.claim(**event).status == "claimed"
        assert store.claim(**event).reason == "unconfirmed"
    with SubscriptionCheckpoint(path, secret=_SECRET) as store:
        assert store.claim(**event).reason == "unconfirmed"
        store.commit(**event)  # Only after durable downstream acknowledgment.
        assert store.claim(**event).status == "duplicate"
        with pytest.raises(SubscriptionCheckpointError, match="invalid_claim"):
            store.commit(**event)
    with pytest.raises(SubscriptionCheckpointError, match="incompatible_key"):
        SubscriptionCheckpoint(path, secret=b"another-32-byte-synthetic-test-key-000")


def test_two_local_readers_cannot_claim_same_delivery(tmp_path):
    path = tmp_path / "checkpoint.sqlite"
    event = _event(0)
    with (
        SubscriptionCheckpoint(path, secret=_SECRET) as first,
        SubscriptionCheckpoint(path, secret=_SECRET) as second,
    ):
        assert first.claim(**event).status == "claimed"
        assert second.claim(**event).reason == "unconfirmed"
        first.commit(**event)
        assert second.claim(**event).status == "duplicate"


def test_gap_replay_and_conflicts(tmp_path):
    with SubscriptionCheckpoint(
        tmp_path / "checkpoint.sqlite", secret=_SECRET
    ) as store:
        later = _event(1)
        assert store.claim(**later).reason == "gap"
        with pytest.raises(SubscriptionCheckpointError, match="invalid_claim"):
            store.commit(**later)
        first = _event(0)
        assert store.claim(**first).status == "claimed"
        assert store.claim(**_event(0, notification="other")).reason == "conflict"
        assert (
            store.claim(**_event(2, notification=first["notification_id"])).reason
            == "conflict"
        )
        store.commit(**first)
        assert store.claim(**later).reason == "gap"
        assert store.claim(**later, replay_gap=True).reason == "gap_replay"
        store.commit(**later)
        assert store.claim(**later).status == "duplicate"


def test_gap_with_conflicting_delivery_needs_review(tmp_path):
    with SubscriptionCheckpoint(
        tmp_path / "checkpoint.sqlite", secret=_SECRET
    ) as store:
        later = _event(1)
        assert store.claim(**later).reason == "gap"
        assert store.claim(**_event(1, notification="conflicting")).reason == "conflict"
        first = _event(0)
        store.claim(**first)
        records = store.quarantines(first["subscription_id"])
        assert {record.reason for record in records} == {
            "gap",
            "conflict",
            "unconfirmed",
        }
        conflict = next(record for record in records if record.reason == "conflict")
        store.discard_quarantine(first["subscription_id"], conflict.event_digest)
        store.commit(**first)
        assert store.claim(**later, replay_gap=True).status == "claimed"
        with pytest.raises(SubscriptionCheckpointError, match="missing_record"):
            store.discard_quarantine(first["subscription_id"], conflict.event_digest)


def test_resource_version_dedup_advances_order(tmp_path):
    with SubscriptionCheckpoint(
        tmp_path / "checkpoint.sqlite", secret=_SECRET
    ) as store:
        first = _event(0)
        store.claim(**first)
        store.commit(**first)
        repeated = _event(
            1, resource=first["resource_id"], version=first["resource_version"]
        )
        decision = store.claim(**repeated)
        assert (decision.status, decision.reason) == ("duplicate", "resource_version")
        following = _event(2)
        assert store.claim(**following).status == "claimed"
        store.commit(**following)


def test_one_based_subscription_sequence(tmp_path):
    with SubscriptionCheckpoint(
        tmp_path / "checkpoint.sqlite", secret=_SECRET, initial_sequence=1
    ) as store:
        first = _event(1)
        assert store.claim(**first).status == "claimed"
        store.commit(**first)
        assert store.claim(**_event(3)).reason == "gap"


def test_bounded_history_and_quarantine(tmp_path):
    with SubscriptionCheckpoint(
        tmp_path / "checkpoint.sqlite",
        secret=_SECRET,
        window=1,
        max_gap=1,
        max_quarantine=1,
    ) as store:
        for sequence in range(3):
            event = _event(sequence)
            assert store.claim(**event).status == "claimed"
            store.commit(**event)
        assert store.claim(**_event(0)).reason == "outside_window"
        assert store.claim(**_event(9)).reason == "capacity"
        assert store.claim(**_event(4)).reason == "capacity"
        assert store.claim(**_event(3)).status == "claimed"


def test_checkpoint_contains_no_raw_identifiers_or_resource_body(tmp_path):
    path = tmp_path / "checkpoint.sqlite"
    event = _event(
        0,
        notification="secret-notification",
        resource="secret-resource",
        version="secret-version",
    )
    with SubscriptionCheckpoint(path, secret=_SECRET) as store:
        assert "secret" not in repr(store)
        assert "secret" not in repr(store.claim(**event))
        store.commit(**event)
    with sqlite3.connect(path) as db:
        rows = repr(db.execute("SELECT * FROM streams").fetchall())
        rows += repr(db.execute("SELECT * FROM deliveries").fetchall())
    for value in event.values():
        if isinstance(value, str):
            assert value not in rows
    assert "secret-notification" not in path.read_bytes().decode(
        "utf-8", errors="ignore"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("subscription_id", ""),
        ("notification_id", "sensitive\nidentifier" * 100),
        ("resource_version", None),
        ("sequence", True),
        ("sequence", -1),
    ],
)
def test_invalid_input_fails_without_echo(tmp_path, field, value):
    event = _event(0)
    event[field] = value
    with SubscriptionCheckpoint(
        tmp_path / "checkpoint.sqlite", secret=_SECRET
    ) as store:
        with pytest.raises(SubscriptionCheckpointError) as error:
            store.claim(**event)
    assert "sensitive" not in str(error.value)


def test_no_network_needed_and_distinct_streams(tmp_path, monkeypatch):
    def blocked(*_args, **_kwargs):
        raise AssertionError("network attempted")

    monkeypatch.setattr("socket.socket", blocked)
    with SubscriptionCheckpoint(
        tmp_path / "checkpoint.sqlite", secret=_SECRET
    ) as store:
        first = _event(0)
        assert store.claim(**first).status == "claimed"
        store.commit(**first)
        first["subscription_id"] = "different-synthetic-subscription"
        assert store.claim(**first).status == "claimed"
