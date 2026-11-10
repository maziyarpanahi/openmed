from __future__ import annotations

from pathlib import Path

from openmed.interop.athena import AthenaVocabularyIndex
from openmed.interop.omop import (
    UNMAPPED_CONCEPT_ID,
    SourceToConceptMapping,
    VocabularyRouter,
    domain_cdm_table,
    route_domain,
)

FIXTURES = Path(__file__).with_name("fixtures")


def _concept(
    concept_id: int,
    concept_code: str,
    domain_id: str,
    vocabulary_id: str,
    *,
    standard: str | None = "S",
) -> dict[str, object]:
    return {
        "concept_id": concept_id,
        "concept_name": f"Synthetic {concept_code}",
        "domain_id": domain_id,
        "vocabulary_id": vocabulary_id,
        "concept_class_id": "Synthetic",
        "standard_concept": standard,
        "concept_code": concept_code,
    }


def _vocabulary() -> AthenaVocabularyIndex:
    """Synthetic Athena index spanning every routable OMOP domain."""

    return {
        "SNOTEST": {
            "C-100": _concept(1001, "C-100", "Condition", "SNOTEST"),
            "P-100": _concept(1004, "P-100", "Procedure", "SNOTEST"),
            "O-100": _concept(1005, "O-100", "Observation", "SNOTEST"),
        },
        "RXTEST": {"D-100": _concept(2001, "D-100", "Drug", "RXTEST")},
        "LABTEST": {"M-100": _concept(3001, "M-100", "Measurement", "LABTEST")},
        "LOCAL": {
            # Non-standard source code that must be mapped via Usagi.
            "SRC-COND": _concept(9001, "SRC-COND", "Condition", "LOCAL", standard=None),
        },
        "_meta": {"vocabulary_ids": ["LOCAL", "RXTEST", "SNOTEST"]},
    }


def _usagi() -> dict[str, int]:
    return {
        "LOCAL:SRC-COND": 1001,
        "LOCAL:SRC-DRUG": 2001,
        "SRC-NOVOCAB": 7777,
    }


def _router() -> VocabularyRouter:
    return VocabularyRouter(
        _vocabulary(),
        _usagi(),
        vocabulary_version="SYNTHETIC 2026-01",
    )


def test_fixture_concepts_route_to_expected_cdm_domain_tables() -> None:
    router = _router()

    expected = {
        ("C-100", "SNOTEST"): ("Condition", "condition_occurrence", 1001),
        ("D-100", "RXTEST"): ("Drug", "drug_exposure", 2001),
        ("M-100", "LABTEST"): ("Measurement", "measurement", 3001),
        ("P-100", "SNOTEST"): ("Procedure", "procedure_occurrence", 1004),
        ("O-100", "SNOTEST"): ("Observation", "observation", 1005),
    }

    for (code, vocab), (domain, table, concept_id) in expected.items():
        mapping = router.route(code, source_vocabulary_id=vocab)
        assert mapping.domain == domain
        assert mapping.cdm_table == table
        assert mapping.target_concept_id == concept_id
        assert mapping.is_mapped
        assert domain_cdm_table(domain) == table


def test_non_standard_source_routes_through_usagi_to_standard_concept() -> None:
    router = _router()

    mapping = router.route(
        "SRC-COND",
        source_vocabulary_id="LOCAL",
        source_code_description="local condition alpha",
    )

    assert mapping.mapping_status == "mapped"
    assert mapping.source_concept_id == 9001
    assert mapping.target_concept_id == 1001
    assert mapping.target_vocabulary_id == "SNOTEST"
    assert mapping.domain == "Condition"
    assert mapping.cdm_table == "condition_occurrence"
    assert mapping.standard_concept == "S"


def test_vocabulary_version_provenance_present_for_every_mapped_row() -> None:
    router = _router()

    spans = [
        {"code": "C-100", "vocabulary_id": "SNOTEST"},
        {"code": "D-100", "vocabulary_id": "RXTEST"},
        {"source_code": "SRC-COND", "source_vocabulary_id": "LOCAL"},
    ]
    mappings = router.route_all(spans)

    assert [m.is_mapped for m in mappings] == [True, True, True]
    for mapping in mappings:
        assert mapping.vocabulary_version == "SYNTHETIC 2026-01"
        assert mapping.to_source_to_concept_map_row()["vocabulary_version"] == (
            "SYNTHETIC 2026-01"
        )


def test_per_vocabulary_version_mapping_is_resolved_by_target_vocabulary() -> None:
    router = VocabularyRouter(
        _vocabulary(),
        _usagi(),
        vocabulary_version={"SNOTEST": "SNO 2026", "RXTEST": "RX 2025", "": "DEF"},
    )

    assert router.route("C-100", source_vocabulary_id="SNOTEST").vocabulary_version == (
        "SNO 2026"
    )
    assert router.route("D-100", source_vocabulary_id="RXTEST").vocabulary_version == (
        "RX 2025"
    )
    # Falls back to the default key when neither target nor source has a version.
    assert router.route("M-100", source_vocabulary_id="LABTEST").vocabulary_version == (
        "DEF"
    )


def test_missing_vocabulary_match_stays_source_only_with_zero_concept() -> None:
    router = _router()

    mapping = router.route(
        "UNKNOWN-1",
        source_vocabulary_id="LOCAL",
        domain_hint="Condition",
        source_code_description="mystery finding",
    )

    assert mapping.mapping_status == "source_only"
    assert mapping.is_mapped is False
    assert mapping.source_concept_id == UNMAPPED_CONCEPT_ID
    assert mapping.target_concept_id == UNMAPPED_CONCEPT_ID
    assert mapping.target_vocabulary_id == "UNMAPPED"
    assert mapping.standard_concept is None
    # Source metadata is preserved and the domain hint still routes to a table.
    assert mapping.source_code == "UNKNOWN-1"
    assert mapping.source_vocabulary_id == "LOCAL"
    assert mapping.source_code_description == "mystery finding"
    assert mapping.domain == "Condition"
    assert mapping.cdm_table == "condition_occurrence"

    row = mapping.to_source_to_concept_map_row()
    assert row["target_concept_id"] == UNMAPPED_CONCEPT_ID
    assert row["invalid_reason"] == "UNMAPPED"
    assert row["source_code"] == "UNKNOWN-1"


def test_never_fabricates_concept_id_without_usagi_or_standard_source() -> None:
    router = VocabularyRouter(_vocabulary())  # no Usagi mapping supplied

    mapping = router.route("SRC-COND", source_vocabulary_id="LOCAL")

    assert mapping.target_concept_id == UNMAPPED_CONCEPT_ID
    assert mapping.mapping_status == "source_only"
    # The non-standard source concept id is still preserved as provenance.
    assert mapping.source_concept_id == 9001


def test_usagi_key_without_source_vocabulary_is_resolved() -> None:
    router = _router()

    mapping = router.route("SRC-NOVOCAB")

    assert mapping.target_concept_id == 7777
    assert mapping.mapping_status == "mapped"


def test_usagi_target_absent_from_index_does_not_mislabel_target_vocabulary() -> None:
    # Usagi resolves a real concept_id that is not present in the supplied
    # Athena index: the id is kept, but the target vocabulary is unknown and
    # must not be echoed as the source vocabulary.
    router = VocabularyRouter(
        _vocabulary(),
        {"LOCAL:SRC-ABSENT": 8888},
        vocabulary_version="SYNTHETIC 2026-01",
    )

    mapping = router.route("SRC-ABSENT", source_vocabulary_id="LOCAL")

    assert mapping.target_concept_id == 8888
    assert mapping.is_mapped
    assert mapping.target_vocabulary_id == ""  # unknown, NOT "LOCAL"


def test_source_to_concept_map_row_shape_is_stable() -> None:
    mapping = _router().route("C-100", source_vocabulary_id="SNOTEST")

    assert set(mapping.to_source_to_concept_map_row()) == {
        "source_code",
        "source_concept_id",
        "source_vocabulary_id",
        "source_code_description",
        "target_concept_id",
        "target_vocabulary_id",
        "valid_start_date",
        "valid_end_date",
        "invalid_reason",
        "vocabulary_version",
    }


def test_route_domain_and_table_helpers() -> None:
    assert route_domain("Condition") == "Condition"
    assert route_domain("diagnosis") == "Condition"
    assert route_domain("lab value") == "Measurement"
    assert route_domain("anatomy") is None
    assert route_domain(None) is None
    assert domain_cdm_table("Drug") == "drug_exposure"


def test_from_athena_loads_shared_fixture_and_routes_via_usagi() -> None:
    router = VocabularyRouter.from_athena(
        FIXTURES,
        FIXTURES / "usagi_export.csv",
        vocabulary_version="ATHENA-FIXTURE",
    )

    # Standard concept present directly in the Athena CONCEPT.csv fixture.
    direct = router.route("T-100", source_vocabulary_id="TESTVOCAB")
    assert direct.target_concept_id == 1001
    assert direct.domain == "Condition"
    assert direct.cdm_table == "condition_occurrence"

    # Local source code mapped to a standard concept through the Usagi export.
    mapped = router.route("SRC-B", source_vocabulary_id="LOCAL")
    assert mapped.target_concept_id == 2001
    assert mapped.domain == "Drug"
    assert mapped.cdm_table == "drug_exposure"
    assert mapped.vocabulary_version == "ATHENA-FIXTURE"


def test_mapping_is_json_serializable_and_frozen() -> None:
    mapping = _router().route("C-100", source_vocabulary_id="SNOTEST")

    assert isinstance(mapping, SourceToConceptMapping)
    payload = mapping.to_dict()
    assert payload["source_code"] == "C-100"
    assert payload["domain"] == "Condition"
