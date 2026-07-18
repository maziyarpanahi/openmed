"""Tests for the minimal hash-chained append-only audit log."""

from __future__ import annotations

import pytest

from openmed.compliance import (
    AuditRecord,
    AuditSink,
    HashChainAuditLog,
)


def test_append_returns_sequential_records():
    log = HashChainAuditLog()

    first = log.append("dsar.export", {"entries": 2})
    second = log.append("dsar.export", {"entries": 5})

    assert isinstance(first, AuditRecord)
    assert (first.sequence, second.sequence) == (0, 1)
    assert first.event_type == "dsar.export"
    assert first.payload == {"entries": 2}


def test_records_form_a_hash_chain():
    log = HashChainAuditLog()
    a = log.append("dsar.export", {"n": 1})
    b = log.append("dsar.erasure_preview", {"n": 2})

    # Genesis record links to a fixed empty-chain anchor; each subsequent record
    # commits to its predecessor's hash.
    assert a.previous_hash == HashChainAuditLog.GENESIS_HASH
    assert b.previous_hash == a.record_hash
    assert a.record_hash != b.record_hash


def test_verify_detects_tampering():
    log = HashChainAuditLog()
    log.append("dsar.export", {"n": 1})
    log.append("dsar.export", {"n": 2})

    assert log.verify() is True

    # Mutating a recorded payload must break the chain.
    object.__setattr__(log.records[0], "payload", {"n": 999})
    assert log.verify() is False


def test_hashing_is_deterministic_across_logs():
    log_a = HashChainAuditLog()
    log_b = HashChainAuditLog()
    for log in (log_a, log_b):
        log.append("dsar.export", {"subject": "abc", "entries": 3})

    assert log_a.records[0].record_hash == log_b.records[0].record_hash


def test_append_rejects_non_mapping_payload():
    log = HashChainAuditLog()
    with pytest.raises(TypeError):
        log.append("dsar.export", ["not", "a", "mapping"])  # type: ignore[arg-type]


def test_hashchain_log_is_an_audit_sink():
    # The concrete log satisfies the injected AuditSink protocol.
    def use_sink(sink: AuditSink) -> AuditRecord:
        return sink.append("dsar.export", {"ok": True})

    record = use_sink(HashChainAuditLog())
    assert record.sequence == 0


def test_to_payload_round_trips_the_chain():
    log = HashChainAuditLog()
    log.append("dsar.export", {"n": 1})
    log.append("dsar.export", {"n": 2})

    payload = log.to_payload()
    assert [r["sequence"] for r in payload["records"]] == [0, 1]
    assert (
        payload["records"][1]["previous_hash"] == payload["records"][0]["record_hash"]
    )
