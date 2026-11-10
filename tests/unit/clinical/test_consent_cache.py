"""Focused synthetic tests for consent-revocation cache invalidation."""

from __future__ import annotations

import json

from openmed.clinical.consent_cache import (
    CONSENT_CACHE_EVENT_TYPE,
    ConsentCache,
    fingerprint_consent_revision,
    fingerprint_consent_scope,
)

SYNTHETIC_SCOPE = "summary"
SYNTHETIC_REVISION = "receipt-v1"
SYNTHETIC_CACHE_KEY = "synthetic-cache-key"
SYNTHETIC_OUTPUT = {"derived": "synthetic-output"}


def test_fingerprints_are_deterministic_without_returning_raw_inputs() -> None:
    first = fingerprint_consent_scope(("analytics", "summary"))
    second = fingerprint_consent_scope({"summary", "analytics"})

    assert first == second
    assert first.startswith("sha256:")
    assert SYNTHETIC_SCOPE not in first
    assert SYNTHETIC_REVISION not in fingerprint_consent_revision(
        {"revision": SYNTHETIC_REVISION}
    )


def test_entry_metadata_contains_fingerprints_but_not_key_or_output() -> None:
    cache = ConsentCache()
    assert cache.put(
        SYNTHETIC_CACHE_KEY,
        SYNTHETIC_OUTPUT,
        scope=SYNTHETIC_SCOPE,
        revision=SYNTHETIC_REVISION,
    )

    entry = cache.get_entry(
        SYNTHETIC_CACHE_KEY,
        scope=SYNTHETIC_SCOPE,
        revision=SYNTHETIC_REVISION,
    )
    assert entry is not None
    assert entry.value == SYNTHETIC_OUTPUT
    metadata = entry.to_dict()
    serialized = json.dumps(metadata, sort_keys=True)

    assert metadata["consent_scope_fingerprint"] == fingerprint_consent_scope(
        SYNTHETIC_SCOPE
    )
    assert metadata["consent_revision_fingerprint"] == fingerprint_consent_revision(
        SYNTHETIC_REVISION
    )
    assert SYNTHETIC_CACHE_KEY not in serialized
    assert "synthetic-output" not in serialized
    assert SYNTHETIC_CACHE_KEY not in repr(entry)
    assert "synthetic-output" not in repr(entry)


def test_revocation_invalidates_only_the_matching_scope_and_revision() -> None:
    cache = ConsentCache()
    cache.put("key-one", "output-one", scope="summary", revision="v1")
    cache.put("key-two", "output-two", scope="summary", revision="v2")
    cache.put("key-three", "output-three", scope="billing", revision="v1")

    event = cache.revoke(scope="summary", revision="v1")

    assert event.invalidated_count == 1
    assert cache.get("key-one", scope="summary", revision="v1") is None
    assert cache.get("key-two", scope="summary", revision="v2") == "output-two"
    assert cache.get("key-three", scope="billing", revision="v1") == "output-three"
    assert event.to_dict() == {
        "event_type": CONSENT_CACHE_EVENT_TYPE,
        "invalidated_count": 1,
    }


def test_revocation_tombstone_blocks_reinsertion_of_old_revision() -> None:
    cache = ConsentCache()
    cache.put("key", "old-output", scope="summary", revision="v1")

    cache.revoke("summary", "v1")

    assert cache.put("key", "stale-output", scope="summary", revision="v1") is False
    assert cache.is_revoked("summary", "v1")
    assert cache.put("key", "new-output", scope="summary", revision="v2") is True
    assert cache.get("key") == "new-output"


def test_scope_revocation_invalidates_current_revisions_and_audits_counts_only() -> (
    None
):
    cache = ConsentCache()
    cache.put("key-one", "output-one", scope="summary", revision="v1")
    cache.put("key-two", "output-two", scope="summary", revision="v2")
    cache.put("key-three", "output-three", scope="billing", revision="v1")

    event = cache.revoke(scope="summary")

    assert event.count == 2
    assert len(cache) == 1
    assert cache.get("key-three", scope="billing", revision="v1") == "output-three"
    assert (
        cache.put("key-one", "replayed-output", scope="summary", revision="v1") is False
    )
    assert (
        cache.put("key-two", "replayed-output", scope="summary", revision="v2") is False
    )

    audit = cache.audit_log()
    assert audit[-1] == {
        "event_type": CONSENT_CACHE_EVENT_TYPE,
        "invalidated_count": 2,
    }
    assert all(
        set(event_payload) == {"event_type", "invalidated_count"}
        for event_payload in audit
    )
    serialized = json.dumps(audit, sort_keys=True)
    assert "output-one" not in serialized
    assert "key-one" not in serialized


def test_repeated_revocation_is_deterministic_and_reports_zero_after_first_pass() -> (
    None
):
    cache = ConsentCache()
    cache.put("key", "output", scope=SYNTHETIC_SCOPE, revision=SYNTHETIC_REVISION)

    first = cache.revoke(SYNTHETIC_SCOPE, SYNTHETIC_REVISION)
    second = cache.revoke(SYNTHETIC_SCOPE, SYNTHETIC_REVISION)

    assert first.invalidated_count == 1
    assert second.invalidated_count == 0
    assert cache.audit_events[-2:] == (first, second)
