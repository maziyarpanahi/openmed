from __future__ import annotations

from openmed.interop.omop_concept_resolver import (
    ConceptResolution,
    UsagiConceptResolver,
)


def _concept(
    concept_id: int,
    code: str,
    vocabulary_id: str,
    domain_id: str,
    *,
    standard_concept: str | None = "S",
    **extra: object,
) -> dict[str, object]:
    return {
        "concept_id": concept_id,
        "concept_code": code,
        "vocabulary_id": vocabulary_id,
        "domain_id": domain_id,
        "standard_concept": standard_concept,
        **extra,
    }


def test_usagi_mapping_wins_and_preserves_athena_metadata() -> None:
    resolver = UsagiConceptResolver(
        {"LOCAL:SRC-1": 1001},
        {
            "LOCAL": {
                "SRC-1": _concept(
                    9001,
                    "SRC-1",
                    "LOCAL",
                    "Condition",
                    standard_concept=None,
                )
            },
            "SNOMED": {"S-1001": _concept(1001, "S-1001", "SNOMED", "Condition")},
        },
    )

    resolution = resolver.resolve("LOCAL", "SRC-1")

    assert resolution == ConceptResolution(
        1001,
        9001,
        "S",
        True,
        vocabulary_id="SNOMED",
        domain_id="Condition",
    )


def test_unmapped_source_returns_zero_and_source_concept_id() -> None:
    resolver = UsagiConceptResolver(
        {},
        {
            "LOCAL": {
                "SRC-UNKNOWN": _concept(
                    9100,
                    "SRC-UNKNOWN",
                    "LOCAL",
                    "Condition",
                    standard_concept=None,
                )
            }
        },
    )

    resolution = resolver.resolve("LOCAL", "SRC-UNKNOWN")

    assert resolution.concept_id == 0
    assert resolution.source_concept_id == 9100
    assert resolution.standard_concept is None
    assert resolution.mapped is False


def test_athena_maps_to_fallback_carries_target_metadata() -> None:
    resolver = UsagiConceptResolver(
        {},
        {
            "LOCAL": {
                "SRC-2": _concept(
                    9200,
                    "SRC-2",
                    "LOCAL",
                    "Condition",
                    standard_concept=None,
                    **{"Maps to": 1002},
                )
            },
            "SNOMED": {"S-1002": _concept(1002, "S-1002", "SNOMED", "Condition")},
        },
    )

    resolution = resolver.resolve("LOCAL", "SRC-2")

    assert resolution.concept_id == 1002
    assert resolution.source_concept_id == 9200
    assert resolution.standard_concept == "S"
    assert resolution.vocabulary_id == "SNOMED"
    assert resolution.domain_id == "Condition"
    assert resolution.mapped is True


def test_provenance_and_mapping_hash_are_stable_and_exposed() -> None:
    athena_a = {
        "SNOMED": {"S-1001": _concept(1001, "S-1001", "SNOMED", "Condition")},
        "_meta": {
            "provenance": {"user_supplied": True, "bundled": False},
            "source": "synthetic-athena",
        },
    }
    athena_b = {
        "_meta": {
            "source": "synthetic-athena",
            "provenance": {"bundled": False, "user_supplied": True},
        },
        "SNOMED": {"S-1001": _concept(1001, "S-1001", "SNOMED", "Condition")},
    }

    first = UsagiConceptResolver({"LOCAL:SRC-1": 1001}, athena_a)
    second = UsagiConceptResolver({"LOCAL:SRC-1": 1001}, athena_b)

    assert first.mapping_hash.startswith("sha256:")
    assert first.mapping_hash == second.mapping_hash
    assert first.mapping_set_hash == first.mapping_hash
    assert first.provenance_hash == first.mapping_hash
    assert first.vocabulary_version == first.mapping_hash
    assert first.provenance["mapping_hash"] == first.mapping_hash
    assert first.provenance["athena"]["provenance"] == {
        "user_supplied": True,
        "bundled": False,
    }
    assert first.provenance["usagi"]["user_supplied"] is True


def test_callable_adapter_returns_id_for_exporter_and_none_for_miss() -> None:
    resolver = UsagiConceptResolver({"LOCAL:SRC-1": 1001})

    assert resolver("LOCAL", "SRC-1") == resolver.resolve("LOCAL", "SRC-1")
    assert resolver({"candidates": ({"system": "LOCAL", "code": "SRC-1"},)}) == 1001
    assert resolver({"candidates": ({"system": "LOCAL", "code": "MISSING"},)}) is None
