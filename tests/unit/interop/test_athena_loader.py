from __future__ import annotations

import os
from pathlib import Path

import pytest

from openmed.interop.athena import load_athena_vocab, load_usagi_mapping

FIXTURES = Path(__file__).with_name("fixtures")
CONCEPT_CSV = FIXTURES / "CONCEPT.csv"
USAGI_CSV = FIXTURES / "usagi_export.csv"


def test_load_athena_vocab_builds_concept_alias_index() -> None:
    index = load_athena_vocab(FIXTURES)

    record = index["TESTVOCAB"]["T-100"]
    assert record["concept_id"] == 1001
    assert record["concept_name"] == "Example condition alpha"
    assert record["domain_id"] == "Condition"
    assert record["vocabulary_id"] == "TESTVOCAB"
    assert record["standard_concept"] == "S"
    assert record["concept_code"] == "T-100"
    assert record["synonyms"] == ["Alpha condition", "Condition alpha"]
    assert record["aliases"] == [
        "Example condition alpha",
        "Alpha condition",
        "Condition alpha",
    ]


def test_load_athena_vocab_metadata_records_user_supplied_provenance() -> None:
    index = load_athena_vocab(FIXTURES)

    meta = index["_meta"]
    assert meta["vocabulary_ids"] == ["RXTEST", "TESTVOCAB"]
    assert meta["concept_count"] == 3
    assert meta["synonym_count"] == 3
    assert "CC BY-SA 4.0" in meta["license"]
    assert "restricted vocabularies" in meta["license"]
    assert meta["provenance"]["user_supplied"] is True
    assert meta["provenance"]["bundled"] is False
    assert meta["provenance"]["concept_file"].endswith("CONCEPT.csv")


def test_load_athena_vocab_can_skip_synonyms() -> None:
    index = load_athena_vocab(FIXTURES, include_synonyms=False)

    record = index["TESTVOCAB"]["T-100"]
    assert record["synonyms"] == []
    assert record["aliases"] == ["Example condition alpha"]
    assert index["_meta"]["synonym_count"] == 0
    assert index["_meta"]["provenance"]["synonym_file"] is None


def test_load_athena_vocab_filters_by_vocabulary_id() -> None:
    index = load_athena_vocab(FIXTURES, vocabulary_ids=["TESTVOCAB"])

    assert set(index) == {"TESTVOCAB", "_meta"}
    assert set(index["TESTVOCAB"]) == {"T-100", "T-200"}
    assert index["_meta"]["vocabulary_ids"] == ["TESTVOCAB"]


def test_load_athena_vocab_accepts_direct_concept_csv_path() -> None:
    from_directory = load_athena_vocab(FIXTURES)
    from_file = load_athena_vocab(CONCEPT_CSV)

    assert from_file["TESTVOCAB"]["T-100"] == from_directory["TESTVOCAB"]["T-100"]


def test_load_athena_vocab_raises_for_missing_concept_csv(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="CONCEPT.csv"):
        load_athena_vocab(tmp_path)


def test_load_athena_vocab_raises_for_malformed_concept_csv(tmp_path: Path) -> None:
    (tmp_path / "CONCEPT.csv").write_text("concept_id\tconcept_name\n1001\tBad\n")

    with pytest.raises(ValueError, match="missing required columns"):
        load_athena_vocab(tmp_path)


def test_load_usagi_mapping_parses_approved_rows() -> None:
    mapping = load_usagi_mapping(USAGI_CSV)

    assert mapping == {
        "LOCAL:SRC-A": 1001,
        "LOCAL:SRC-B": 2001,
        "SRC-NOVOCAB": 3001,
    }


def test_load_usagi_mapping_filters_by_equivalence() -> None:
    mapping = load_usagi_mapping(USAGI_CSV, min_equivalence="EQUIVALENT")

    assert mapping == {
        "LOCAL:SRC-A": 1001,
        "SRC-NOVOCAB": 3001,
    }


def test_load_usagi_mapping_can_include_unapproved_rows() -> None:
    mapping = load_usagi_mapping(USAGI_CSV, approved_only=False)

    assert mapping["LOCAL:SRC-PENDING"] == 4001
    assert "LOCAL:SRC-ZERO" not in mapping


def test_load_usagi_mapping_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_usagi_mapping(tmp_path / "missing.csv")


def test_load_usagi_mapping_raises_for_missing_required_columns(tmp_path: Path) -> None:
    path = tmp_path / "usagi.csv"
    path.write_text("sourceName,conceptName\nsource,target\n")

    with pytest.raises(ValueError, match="missing one of"):
        load_usagi_mapping(path)


def test_openmed_package_does_not_bundle_vocabulary_csvs() -> None:
    import openmed

    package_root = Path(openmed.__file__).parent
    forbidden_names = {
        "CONCEPT.csv",
        "CONCEPT_SYNONYM.csv",
        "CONCEPT_RELATIONSHIP.csv",
        "CONCEPT_ANCESTOR.csv",
        "VOCABULARY.csv",
        "DOMAIN.csv",
        "CONCEPT_CLASS.csv",
        "DRUG_STRENGTH.csv",
        "usagi_export.csv",
    }

    found = []
    for root, _dirs, files in os.walk(package_root):
        for file_name in files:
            if file_name in forbidden_names:
                found.append(os.path.join(root, file_name))

    assert found == []
