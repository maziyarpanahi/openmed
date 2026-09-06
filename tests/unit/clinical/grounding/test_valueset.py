"""Synthetic coverage for local and delegated ValueSet expansion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from openmed.clinical.grounding import (
    ExpansionProvenance,
    LexicalConcept,
    TerminologySnapshot,
    TerminologySnapshotCache,
    ValueSetExpansionCache,
    ValueSetExpansionConfigurationError,
    ValueSetExpansionEngine,
    ValueSetExpansionPolicyError,
    ValueSetExpansionResponseError,
    ValueSetExpansionUnsupportedError,
    VocabConcept,
    VocabularyIndex,
    VocabularyLoaderRegistry,
    expand_valueset,
)

SYSTEM = "http://human-phenotype-ontology.org"
VALUESET_URL = "https://example.org/fhir/ValueSet/selected-findings"


def _index(*, reversed_order: bool = False) -> VocabularyIndex:
    concepts = [
        VocabConcept(
            system="hpo",
            code="SYN-001",
            preferred_term="Synthetic finding one",
        ),
        VocabConcept(
            system="hpo",
            code="SYN-002",
            preferred_term="Synthetic finding two",
        ),
        VocabConcept(
            system="hpo",
            code="SYN-003",
            preferred_term="Synthetic finding three",
        ),
    ]
    if reversed_order:
        concepts.reverse()
    return VocabularyIndex("hpo", concepts)


def _snapshot(*, reversed_order: bool = False) -> TerminologySnapshot:
    index = _index(reversed_order=reversed_order)
    return TerminologySnapshot(
        index=index,
        system_uri=SYSTEM,
        release_version="2026.09",
        content_hash=index.content_hash,
    )


def _local_valueset() -> dict[str, Any]:
    return {
        "resourceType": "ValueSet",
        "url": VALUESET_URL,
        "version": "1.0.0",
        "compose": {
            "include": [{"system": SYSTEM, "version": "2026.09"}],
            "exclude": [
                {
                    "system": SYSTEM,
                    "version": "2026.09",
                    "concept": [{"code": "SYN-003"}],
                }
            ],
        },
    }


class _Response:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        return self._payload


class _Client:
    def __init__(self, *responses: dict[str, Any]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def get(
        self,
        url: str,
        *,
        params: dict[str, str],
        headers: dict[str, str],
    ) -> _Response:
        self.requests.append({"url": url, "params": params, "headers": headers})
        return _Response(self.responses.pop(0))


def _remote_valueset(
    *,
    system: str = SYSTEM,
    codes: tuple[str, ...] = ("SYN-001", "SYN-002"),
    version: str = "1.0.0",
) -> dict[str, Any]:
    return {
        "resourceType": "ValueSet",
        "url": VALUESET_URL,
        "version": version,
        "expansion": {
            "identifier": "urn:uuid:00000000-0000-0000-0000-000000000569",
            "total": len(codes),
            "offset": 0,
            "contains": [
                {"system": system, "version": "2026.09", "code": code} for code in codes
            ],
        },
    }


def test_extensional_valueset_expands_loaded_free_vocabulary_offline() -> None:
    result = expand_valueset(
        _local_valueset(),
        vocabularies={SYSTEM: _snapshot()},
    )

    assert result.members == frozenset({"SYN-001", "SYN-002"})
    assert tuple(result) == ("SYN-001", "SYN-002")
    assert result.version == "1.0.0"
    assert result.provenance == ExpansionProvenance(
        source_kind="local",
        method="local-extensional",
        version="1.0.0",
        request_sha256=result.provenance.request_sha256,
        response_sha256=result.provenance.response_sha256,
        valueset_url=VALUESET_URL,
        vocabulary_versions=((SYSTEM, "2026.09"),),
    )
    assert [coding.to_dict() for coding in result.codings] == [
        {"system": SYSTEM, "code": "SYN-001", "version": "2026.09"},
        {"system": SYSTEM, "code": "SYN-002", "version": "2026.09"},
    ]


def test_local_expansion_is_reproducible_across_vocabulary_order() -> None:
    first = expand_valueset(
        _local_valueset(),
        vocabularies={SYSTEM: _snapshot()},
    )
    second = expand_valueset(
        _local_valueset(),
        vocabularies={SYSTEM: _snapshot(reversed_order=True)},
    )

    assert first.to_dict() == second.to_dict()
    assert first.provenance.request_sha256.startswith("sha256:")
    assert first.provenance.response_sha256.startswith("sha256:")


def test_local_complete_fhir_expansion_flattens_nested_members() -> None:
    valueset = {
        "resourceType": "ValueSet",
        "url": VALUESET_URL,
        "version": "2",
        "expansion": {
            "total": 3,
            "contains": [
                {
                    "system": SYSTEM,
                    "code": "group",
                    "abstract": True,
                    "contains": [
                        {"code": "SYN-002"},
                        {"code": "SYN-001"},
                    ],
                }
            ],
        },
    }

    result = expand_valueset(valueset)

    assert result.members == frozenset({"SYN-001", "SYN-002"})
    assert all(member.system == SYSTEM for member in result.codings)


def test_local_code_filter_uses_loaded_vocabulary() -> None:
    valueset = {
        "resourceType": "ValueSet",
        "url": VALUESET_URL,
        "version": "filters-v1",
        "compose": {
            "include": [
                {
                    "system": SYSTEM,
                    "filter": [{"property": "code", "op": "regex", "value": r"002$"}],
                }
            ]
        },
    }

    result = expand_valueset(valueset, vocabularies={SYSTEM: _snapshot()})

    assert result.members == frozenset({"SYN-002"})


def test_local_expansion_accepts_the_free_vocabulary_loader_registry() -> None:
    class SyntheticLoader:
        system_uri = SYSTEM
        redistributable = True
        release_version = "2026.09"

        def load(self) -> dict[str, LexicalConcept]:
            return {
                "Synthetic finding one": LexicalConcept(
                    SYSTEM,
                    "SYN-001",
                    "Synthetic finding one",
                ),
                "Synthetic finding two": LexicalConcept(
                    SYSTEM,
                    "SYN-002",
                    "Synthetic finding two",
                ),
            }

    registry = VocabularyLoaderRegistry()
    registry.register(SyntheticLoader())
    valueset = _local_valueset()
    valueset["compose"]["exclude"][0]["concept"][0]["code"] = "SYN-999"

    result = expand_valueset(valueset, vocabulary_registry=registry)

    assert result.members == frozenset({"SYN-001", "SYN-002"})
    assert result.provenance.vocabulary_versions == ((SYSTEM, "2026.09"),)


def test_local_hierarchy_filter_requires_remote_delegation() -> None:
    valueset = {
        "resourceType": "ValueSet",
        "version": "1",
        "compose": {
            "include": [
                {
                    "system": SYSTEM,
                    "filter": [
                        {"property": "concept", "op": "is-a", "value": "SYN-001"}
                    ],
                }
            ]
        },
    }

    with pytest.raises(
        ValueSetExpansionUnsupportedError,
        match="filter operator requires terminology-server delegation",
    ):
        expand_valueset(valueset, vocabularies={SYSTEM: _snapshot()})


def test_ecl_is_delegated_to_configured_fhir_expand_endpoint() -> None:
    ecl = "*"
    response = _remote_valueset(system="http://snomed.info/sct")
    client = _Client(response)
    engine = ValueSetExpansionEngine(
        "https://terminology.example/fhir",
        client=client,
        bearer_token="secret-token",
    )

    result = engine.expand_valueset(ecl, version="1.0.0")

    assert result.members == frozenset({"SYN-001", "SYN-002"})
    assert result.provenance.method == "remote-ecl-delegation"
    assert result.provenance.restricted is True
    assert result.provenance.version == "1.0.0"
    assert result.provenance.expansion_identifier.endswith("569")
    assert client.requests == [
        {
            "url": "https://terminology.example/fhir/ValueSet/$expand",
            "params": {
                "url": f"http://snomed.info/sct?fhir_vs=ecl/{ecl}",
                "system-version": "1.0.0",
                "count": "1000",
            },
            "headers": {
                "Accept": "application/fhir+json, application/json",
                "Authorization": "Bearer secret-token",
            },
        }
    ]


def test_canonical_valueset_url_uses_fhir_expand() -> None:
    client = _Client(_remote_valueset())

    result = expand_valueset(
        f"{VALUESET_URL}|1.0.0",
        endpoint="https://terminology.example/fhir/ValueSet/$expand",
        client=client,
    )

    assert result.members == frozenset({"SYN-001", "SYN-002"})
    assert result.provenance.source_kind == "valueset-url"
    assert client.requests[0]["params"]["url"] == VALUESET_URL
    assert client.requests[0]["params"]["valueSetVersion"] == "1.0.0"


def test_remote_expand_follows_bounded_fhir_pagination() -> None:
    first = _remote_valueset(codes=("SYN-001",))
    first["expansion"]["total"] = 2
    second = _remote_valueset(codes=("SYN-002",))
    second["expansion"]["total"] = 2
    second["expansion"]["offset"] = 1
    client = _Client(first, second)

    result = expand_valueset(
        VALUESET_URL,
        version="1.0.0",
        endpoint="https://terminology.example/fhir",
        client=client,
    )

    assert result.members == frozenset({"SYN-001", "SYN-002"})
    assert client.requests[1]["params"]["offset"] == "1"


def test_remote_expansion_requires_an_explicit_endpoint() -> None:
    with pytest.raises(
        ValueSetExpansionConfigurationError,
        match="caller-supplied terminology endpoint",
    ):
        expand_valueset(VALUESET_URL, version="1")

    with pytest.raises(
        ValueSetExpansionConfigurationError,
        match="caller-supplied terminology endpoint",
    ):
        expand_valueset("*", version="1")


def test_remote_operation_outcome_raises_safe_structured_error() -> None:
    client = _Client({"resourceType": "OperationOutcome", "issue": []})

    with pytest.raises(
        ValueSetExpansionResponseError,
        match="FHIR OperationOutcome",
    ):
        expand_valueset(
            VALUESET_URL,
            version="1",
            endpoint="https://terminology.example/fhir",
            client=client,
        )


def test_free_expansion_cache_is_versioned_and_reuses_snapshot_root(
    tmp_path: Path,
) -> None:
    cache = ValueSetExpansionCache(tmp_path / "terminology-snapshots")
    engine = ValueSetExpansionEngine(
        vocabularies={SYSTEM: _snapshot()},
        cache=cache,
    )

    first = engine.expand_valueset(_local_valueset())
    second = engine.expand_valueset(_local_valueset())

    assert first.provenance.cache_hit is False
    assert second.provenance.cache_hit is True
    assert first.provenance.response_sha256 == second.provenance.response_sha256
    assert tuple(cache.cache_dir.glob("*/expansion.json"))


def test_expansion_cache_can_share_a_terminology_snapshot_cache_root(
    tmp_path: Path,
) -> None:
    snapshot_cache = TerminologySnapshotCache(tmp_path / "shared")
    engine = ValueSetExpansionEngine(
        vocabularies={SYSTEM: _snapshot()},
        cache=snapshot_cache,
    )

    engine.expand_valueset(_local_valueset())

    assert tuple((snapshot_cache.cache_dir / "expansions").glob("*/expansion.json"))


def test_restricted_remote_codes_are_not_cached_without_double_opt_in(
    tmp_path: Path,
) -> None:
    ecl = "*"
    response = _remote_valueset(system="http://snomed.info/sct")
    cache = ValueSetExpansionCache(tmp_path / "default-cache")
    client = _Client(response, response)
    engine = ValueSetExpansionEngine(
        "https://terminology.example/fhir",
        client=client,
        cache=cache,
        bearer_token="secret-token",
    )

    engine.expand_valueset(ecl, version="1.0.0")
    engine.expand_valueset(ecl, version="1.0.0")

    assert len(client.requests) == 2
    assert not cache.cache_dir.exists()


def test_restricted_cache_requires_explicit_policy_and_never_stores_raw_ecl(
    tmp_path: Path,
) -> None:
    ecl = "*"
    response = _remote_valueset(system="http://snomed.info/sct")
    cache = ValueSetExpansionCache(
        tmp_path / "allowed-cache",
        allow_restricted=True,
    )
    client = _Client(response)
    engine = ValueSetExpansionEngine(
        "https://terminology.example/fhir",
        client=client,
        cache=cache,
    )

    first = engine.expand_valueset(ecl, version="1.0.0")
    second = engine.expand_valueset(ecl, version="1.0.0")

    assert first.provenance.cache_hit is False
    assert second.provenance.cache_hit is True
    assert len(client.requests) == 1
    cache_text = "\n".join(
        path.read_text(encoding="utf-8") for path in cache.cache_dir.glob("*/*.json")
    )
    assert ecl not in cache_text
    assert "secret-token" not in cache_text


def test_direct_restricted_cache_store_is_fail_closed(tmp_path: Path) -> None:
    client = _Client(_remote_valueset(system="http://snomed.info/sct"))
    result = expand_valueset(
        "*",
        version="1.0.0",
        endpoint="https://terminology.example/fhir",
        client=client,
    )

    with pytest.raises(
        ValueSetExpansionPolicyError,
        match="allow_restricted=True",
    ):
        ValueSetExpansionCache(tmp_path / "cache").store("*", result)


def test_cache_manifest_contains_only_hashes_and_versioned_metadata(
    tmp_path: Path,
) -> None:
    cache = ValueSetExpansionCache(tmp_path / "cache")
    ValueSetExpansionEngine(
        vocabularies={SYSTEM: _snapshot()},
        cache=cache,
    ).expand_valueset(_local_valueset())

    manifest_path = next(cache.cache_dir.glob("*/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["version"] == "1.0.0"
    assert manifest["source_sha256"].startswith("sha256:")
    assert VALUESET_URL not in manifest_path.read_text(encoding="utf-8")
    artifact_path = next(cache.cache_dir.glob("*/expansion.json"))
    assert "Synthetic finding" not in artifact_path.read_text(encoding="utf-8")
