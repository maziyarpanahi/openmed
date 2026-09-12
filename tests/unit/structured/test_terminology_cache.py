"""Focused tests for the deterministic terminology cache."""

from __future__ import annotations

import socket
from collections.abc import Iterator, Mapping

import pytest

from openmed.structured.terminology_cache import (
    StaleTerminologyError,
    TerminologyCache,
    TerminologyCacheEntry,
    TerminologyCacheError,
    TerminologyCacheKey,
    TerminologyProvenance,
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


@pytest.mark.parametrize(
    ("replacement", "source"),
    [
        ({"codes": ["SYN-002"]}, "local-fixture"),
        ({"codes": ["SYN-001"]}, "other-fixture"),
    ],
)
def test_exact_key_refuses_conflicting_provenance(
    replacement: dict[str, list[str]], source: str
) -> None:
    cache = TerminologyCache()
    original = cache.put(
        "synthetic-vocabulary",
        "2026.01",
        {"codes": ["SYN-001"]},
        source="local-fixture",
    )

    with pytest.raises(TerminologyProvenanceError):
        cache.put(
            "synthetic-vocabulary",
            "2026.01",
            replacement,
            source=source,
        )

    assert cache.get("synthetic-vocabulary", "2026.01") == original


def test_exact_duplicate_put_is_idempotent() -> None:
    cache = TerminologyCache()
    first = cache.put(
        "synthetic-vocabulary",
        "2026.01",
        {"codes": ["SYN-001"]},
        source="local-fixture",
    )

    second = cache.put(
        "synthetic-vocabulary",
        "2026.01",
        {"codes": ["SYN-001"]},
        source="local-fixture",
    )

    assert second is first
    assert len(cache) == 1


def test_constructor_refuses_duplicate_key_with_different_provenance() -> None:
    first = TerminologyCacheEntry.from_response(
        vocabulary="synthetic-vocabulary",
        release="2026.01",
        response={"codes": ["SYN-001"]},
    )
    second = TerminologyCacheEntry.from_response(
        vocabulary="synthetic-vocabulary",
        release="2026.01",
        response={"codes": ["SYN-002"]},
    )

    with pytest.raises(TerminologyProvenanceError):
        TerminologyCache([first, second])


def test_provenance_rejects_fingerprint_that_does_not_match_metadata() -> None:
    key = TerminologyCacheKey("synthetic-vocabulary", "2026.01")
    response_fingerprint = terminology_response_fingerprint({"codes": ["SYN-001"]})

    with pytest.raises(TerminologyProvenanceError):
        TerminologyProvenance(
            key=key,
            source="local-fixture",
            response_fingerprint=response_fingerprint,
            fingerprint="sha256:" + "0" * 64,
        )


def test_multiple_releases_coexist_and_missing_release_is_stale() -> None:
    cache = TerminologyCache()
    cache.put("synthetic-vocabulary", "2026.01", {"codes": ["SYN-001"]})
    cache.put("synthetic-vocabulary", "2026.02", {"codes": ["SYN-002"]})

    assert cache.get_response("synthetic-vocabulary", "2026.01") == {
        "codes": ["SYN-001"]
    }
    assert cache.get_response("synthetic-vocabulary", "2026.02") == {
        "codes": ["SYN-002"]
    }
    with pytest.raises(StaleTerminologyError) as exc_info:
        cache.get("synthetic-vocabulary", "2026.03")
    assert [key.release for key in exc_info.value.cached_keys] == [
        "2026.01",
        "2026.02",
    ]


def test_report_is_deterministic_and_excludes_responses() -> None:
    cache = TerminologyCache()
    cache.put("z-vocabulary", "2", {"raw": "SYN-SENSITIVE-2"})
    cache.put("a-vocabulary", "1", {"raw": "SYN-SENSITIVE-1"})

    report = cache.report()

    assert [item["key"]["vocabulary"] for item in report["entries"]] == [
        "a-vocabulary",
        "z-vocabulary",
    ]
    assert "SYN-SENSITIVE" not in str(report)


def test_cyclic_and_overdeep_responses_raise_bounded_safe_errors() -> None:
    cyclic: list[object] = []
    cyclic.append(cyclic)
    nested: object = "SYN-SENSITIVE"
    for _ in range(65):
        nested = [nested]

    for response in (cyclic, nested):
        with pytest.raises(TerminologyCacheError) as exc_info:
            terminology_response_fingerprint(response)
        assert "SYN-SENSITIVE" not in str(exc_info.value)
        assert len(str(exc_info.value)) < 100


class _ExplodingMapping(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        raise RuntimeError("SYN-SENSITIVE")

    def __iter__(self) -> Iterator[str]:
        raise RuntimeError("SYN-SENSITIVE")

    def __len__(self) -> int:
        return 1


def test_hostile_mapping_error_is_sanitized() -> None:
    with pytest.raises(TerminologyCacheError) as exc_info:
        terminology_response_fingerprint(_ExplodingMapping())

    assert "SYN-SENSITIVE" not in str(exc_info.value)
