"""Focused offline tests for the user-supplied UMLS crosswalk."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from openmed.clinical.grounding import RestrictedVocabularyError
from openmed.clinical.grounding.crosswalk import (
    CrosswalkCandidate,
    UMLSCodeCrosswalk,
    UMLSCrosswalk,
    crosswalk,
)


def _write_synthetic_source(root: Path) -> tuple[Path, Path]:
    mrconso = root / "MRCONSO.RRF"
    mrmap = root / "MRMAP.RRF"
    mrconso.write_text(
        "CUI|LAT|SAB|CODE|STR\n"
        "C-SYN-ICD|ENG|ICD10CM|E11.9|Synthetic diabetes\n"
        "C-SYN-SCT-A|ENG|SNOMEDCT_US|44054006|Synthetic diabetes concept A\n"
        "C-SYN-SCT-B|ENG|SNOMEDCT_US|73211009|Synthetic diabetes concept B\n",
        encoding="utf-8",
    )
    mrmap.write_text(
        "source_system|source_code|target_system|target_code|map_rule|"
        "map_priority|map_advice\n"
        "ICD10CM|E11.9|SNOMEDCT_US|73211009|RULE-SECOND|20|CHECK\n"
        "ICD10CM|E11.9|SNOMEDCT_US|44054006|RULE-FIRST|10|USE\n"
        "ICD10CM|E11.9|SNOMEDCT_US|44054006|RULE-DUPLICATE|30|SKIP\n",
        encoding="utf-8",
    )
    return mrconso, mrmap


def test_crosswalk_requires_a_user_configured_local_umls_source() -> None:
    with pytest.raises(RestrictedVocabularyError, match="user-configured"):
        crosswalk("E11.9", "ICD10CM", "SNOMEDCT")


def test_one_to_many_maps_are_priority_ordered_and_provenanced(tmp_path: Path) -> None:
    _write_synthetic_source(tmp_path)

    engine = UMLSCrosswalk(tmp_path)
    candidates = engine.crosswalk("E11.9", "ICD-10-CM", "SNOMED CT")

    assert all(isinstance(candidate, CrosswalkCandidate) for candidate in candidates)
    assert [candidate.target_code for candidate in candidates] == [
        "44054006",
        "73211009",
    ]
    assert [candidate.map_priority for candidate in candidates] == [10, 20]
    assert candidates[0].provenance["map_rule"] == "RULE-FIRST"
    assert candidates[0].provenance["map_advice"] == "USE"
    assert candidates[0].provenance["data_source"] == "user-supplied-local"
    assert candidates[0].to_dict()["target_system"] == "SNOMEDCT"


def test_reverse_crosswalk_is_local_and_keeps_one_to_many_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mrconso, mrmap = _write_synthetic_source(tmp_path)

    def fail_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("crosswalk attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    candidates = crosswalk(
        "73211009",
        "SNOMEDCT",
        "ICD10CM",
        mrconso_path=mrconso,
        mrmap_path=mrmap,
    )

    assert [candidate.code for candidate in candidates] == ["E11.9"]
    assert candidates[0].source_system == "SNOMEDCT"
    assert candidates[0].target_system == "ICD10CM"
    assert candidates[0].map_rule == "RULE-SECOND"


def test_standard_rrf_positions_are_supported(tmp_path: Path) -> None:
    mrconso = tmp_path / "MRCONSO.RRF"
    mrmap = tmp_path / "MRMAP.RRF"
    conso_row = "|".join(
        [
            "C000001",
            "ENG",
            "P",
            "L000001",
            "PF",
            "S000001",
            "Y",
            "A000001",
            "",
            "",
            "",
            "ICD10CM",
            "PT",
            "E11.9",
            "Synthetic diabetes",
            "",
            "",
        ]
    )
    target_row = "|".join(
        [
            "C000002",
            "ENG",
            "P",
            "L000002",
            "PF",
            "S000002",
            "Y",
            "A000002",
            "",
            "",
            "",
            "SNOMEDCT_US",
            "PT",
            "44054006",
            "Synthetic SNOMED concept",
            "",
            "",
        ]
    )
    mrconso.write_text(f"{conso_row}\n{target_row}\n", encoding="utf-8")
    map_values = [
        "C000001",
        "MAPSET",
        "T1",
        "STYPE",
        "E11.9",
        "ATN",
        "",
        "ICD10CM",
        "",
        "",
        "",
        "",
        "",
        "SNOMEDCT:44054006",
        "RULE-EQUIVALENT",
        "TARGET-REVIEW",
    ]
    mrmap.write_text("|".join(map_values) + "\n", encoding="utf-8")

    candidates = UMLSCodeCrosswalk(tmp_path).crosswalk("E119", "ICD10CM", "SCT")

    assert [candidate.target_code for candidate in candidates] == ["44054006"]
    assert candidates[0].map_rule == "RULE-EQUIVALENT"
    assert candidates[0].provenance["source_cui"] == "C000001"


def test_unsupported_systems_fail_closed(tmp_path: Path) -> None:
    _write_synthetic_source(tmp_path)
    with pytest.raises(ValueError, match="ICD-10-CM or SNOMED CT"):
        UMLSCrosswalk(tmp_path).crosswalk("E11.9", "LOINC", "SNOMEDCT")
