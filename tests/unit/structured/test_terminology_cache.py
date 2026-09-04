"""Focused tests for the deterministic terminology cache."""

from __future__ import annotations

import socket

import pytest

from openmed.structured.terminology_cache import (
    StaleTerminologyError,
    TerminologyCache,
    TerminologyCacheError,
    TerminologyProvenanceError,
    compute_terminology_fingerprint,
    terminology_response_fingerprint,
)


def test_response_fingerprint_is_stable_for_mapping_order() -> None:
    first = {"codes": [{"code": "SYN-001", "display": "Synthetic finding"}]}
    second = {"codes": [{"display": "Synthetic finding", "code": "SYN-001"}]}

    assert terminology_response_fingerprint(first) == terminology_response_fingerprint(
        second
    )
    assert compute_terminology_fingerprint(
        first,
        vocabulary="synthetic-vocabulary",
        release="2026.01",
        source="local-fixture",
    ) == compute_terminology_fingerprint(
        second,
        vocabulary="synthetic-vocabulary",
        release="2026.01",
        source="local-fixture",
    )


def test_cache_attaches_provenance_and_detaches_response() -> None:
    response = {"codes": [{"code": "SYN-001", "display": "Synthetic finding"}]}
    cache = TerminologyCache()

    entry = cache.put(
        "synthetic-vocabulary",
        "2026.01",
        response,
        source="local-fixture",
    )
    response["codes"].append({"code": "SYN-002"})

    cached = cache.get("synthetic-vocabulary", "2026.01")
    assert cached is not None
    assert cached.response == {
        "codes": [{"code": "SYN-001", "display": "Synthetic finding"}]
    }
    assert cached.source == "local-fixture"
    assert cached.provenance.response_fingerprint.startswith("sha256:")
    assert cached.fingerprint.startswith("sha256:")
    assert cached.to_dict() == entry.to_dict()
    assert "Synthetic finding" not in repr(cached)
    assert "Synthetic finding" not in repr(cache)
    assert "Synthetic finding" not in str(cache.report())


def test_release_mismatch_refuses_stale_entry_without_computing() -> None:
    cache = TerminologyCache()
    cache.put(
        "synthetic-vocabulary",
        "2026.01",
        {"codes": ["SYN-001"]},
        source="local-fixture",
    )
    compute_calls: list[str] = []

    with pytest.raises(StaleTerminologyError):
        cache.get("synthetic-vocabulary", "2026.02", source="local-fixture")
    with pytest.raises(StaleTerminologyError):
        cache.get_or_compute(
            "synthetic-vocabulary",
            "2026.02",
            lambda: compute_calls.append("called") or {"codes": ["SYN-002"]},
            source="local-fixture",
        )

    assert compute_calls == []


def test_source_mismatch_refuses_cached_entry() -> None:
    cache = TerminologyCache()
    cache.put(
        "synthetic-vocabulary",
        "2026.01",
        {"codes": ["SYN-001"]},
        source="local-fixture",
    )

    with pytest.raises(TerminologyProvenanceError):
        cache.get("synthetic-vocabulary", "2026.01", source="other-fixture")


def test_cache_hit_does_not_call_compute_and_miss_is_local() -> None:
    cache = TerminologyCache()
    cache.put("synthetic-vocabulary", "2026.01", {"codes": ["SYN-001"]})
    compute_calls: list[str] = []

    hit = cache.get_or_compute(
        "synthetic-vocabulary",
        "2026.01",
        lambda: compute_calls.append("hit") or {"codes": ["unexpected"]},
    )
    miss = cache.get_or_compute(
        "another-synthetic-vocabulary",
        "2026.02",
        lambda: compute_calls.append("miss") or {"codes": ["SYN-002"]},
    )

    assert hit.response == {"codes": ["SYN-001"]}
    assert miss.response == {"codes": ["SYN-002"]}
    assert compute_calls == ["miss"]


def test_invalid_response_errors_do_not_echo_input() -> None:
    cache = TerminologyCache()
    raw_value = "synthetic-sensitive-value"

    with pytest.raises(TerminologyCacheError) as exc_info:
        cache.put(
            "synthetic-vocabulary",
            "2026.01",
            {"invalid": object(), "raw": raw_value},
        )

    assert raw_value not in str(exc_info.value)


def test_cache_has_no_network_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_socket(*args: object, **kwargs: object) -> socket.socket:
        raise AssertionError("terminology cache must not open a socket")

    monkeypatch.setattr(socket, "socket", fail_socket)
    cache = TerminologyCache()
    entry = cache.put("synthetic-vocabulary", "2026.01", {"codes": ["SYN-001"]})

    assert cache.get_response("synthetic-vocabulary", "2026.01") == entry.response
