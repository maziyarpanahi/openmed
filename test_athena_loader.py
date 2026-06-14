"""
Unit tests for openmed.interop.athena
--------------------------------------
Covers:
  - load_athena_vocab: concept + alias index shape, synonym attachment,
    vocabulary_id filtering, _meta provenance, missing-file errors.
  - load_usagi_mapping: approved-only filtering, zero-concept skip,
    unchecked-row skip, min_equivalence filter, composite key format.
  - Guard: no vocabulary CSV data is bundled inside the openmed package tree.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from openmed.interop.athena import load_athena_vocab, load_usagi_mapping

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures"
CONCEPT_CSV = FIXTURES / "CONCEPT.csv"
SYNONYM_CSV = FIXTURES / "CONCEPT_SYNONYM.csv"
USAGI_CSV = FIXTURES / "usagi_export.csv"


# ---------------------------------------------------------------------------
# load_athena_vocab tests
# ---------------------------------------------------------------------------

class TestLoadAthenaVocab:
    def test_basic_index_shape(self):
        index = load_athena_vocab(FIXTURES)
        # Two SNOMED entries + one RxNorm entry should be present
        assert "SNOMED" in index
        assert "RxNorm" in index
        # _meta sentinel
        assert "_meta" in index

    def test_concept_entry_keys(self):
        index = load_athena_vocab(FIXTURES)
        entry = index["SNOMED"]["73211009"]
        required_keys = {
            "concept_id", "concept_name", "domain_id", "vocabulary_id",
            "concept_class_id", "standard_concept", "concept_code", "synonyms",
        }
        assert required_keys.issubset(entry.keys())

    def test_concept_id_type_and_value(self):
        index = load_athena_vocab(FIXTURES)
        entry = index["SNOMED"]["73211009"]
        assert entry["concept_id"] == 201826
        assert isinstance(entry["concept_id"], int)

    def test_concept_name(self):
        index = load_athena_vocab(FIXTURES)
        assert index["SNOMED"]["73211009"]["concept_name"] == "Type 2 diabetes mellitus"

    def test_synonyms_attached(self):
        index = load_athena_vocab(FIXTURES, include_synonyms=True)
        synonyms = index["SNOMED"]["73211009"]["synonyms"]
        assert isinstance(synonyms, list)
        assert len(synonyms) == 2
        assert "T2DM" in synonyms
        assert "Diabetes mellitus type 2" in synonyms

    def test_synonyms_skipped_when_disabled(self):
        index = load_athena_vocab(FIXTURES, include_synonyms=False)
        assert index["SNOMED"]["73211009"]["synonyms"] == []

    def test_vocabulary_id_filter(self):
        index = load_athena_vocab(FIXTURES, vocabulary_ids=["SNOMED"])
        assert "SNOMED" in index
        assert "RxNorm" not in index

    def test_meta_provenance(self):
        index = load_athena_vocab(FIXTURES)
        meta = index["_meta"]
        assert "source" in meta
        assert "vocabulary_ids" in meta
        assert "licence" in meta
        assert "CC BY-SA" in meta["licence"]
        assert "SNOMED" in meta["vocabulary_ids"]
        assert "RxNorm" in meta["vocabulary_ids"]

    def test_direct_file_path_resolves_to_parent(self):
        # Passing CONCEPT.csv directly should work the same as passing its dir
        index_from_file = load_athena_vocab(CONCEPT_CSV)
        index_from_dir = load_athena_vocab(FIXTURES)
        assert set(index_from_file["SNOMED"].keys()) == set(index_from_dir["SNOMED"].keys())

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_athena_vocab(tmp_path / "nonexistent")

    def test_missing_concept_csv_raises(self, tmp_path):
        # Directory exists but contains no CONCEPT.csv
        (tmp_path / "CONCEPT_SYNONYM.csv").touch()
        with pytest.raises(FileNotFoundError, match="CONCEPT.csv"):
            load_athena_vocab(tmp_path)

    def test_no_synonyms_file_does_not_error(self, tmp_path):
        # Copy only CONCEPT.csv, omit CONCEPT_SYNONYM.csv
        import shutil
        shutil.copy(CONCEPT_CSV, tmp_path / "CONCEPT.csv")
        index = load_athena_vocab(tmp_path, include_synonyms=True)
        # Should succeed; synonyms lists remain empty
        assert index["SNOMED"]["73211009"]["synonyms"] == []


# ---------------------------------------------------------------------------
# load_usagi_mapping tests
# ---------------------------------------------------------------------------

class TestLoadUsagiMapping:
    def test_basic_mapping_shape(self):
        mapping = load_usagi_mapping(USAGI_CSV)
        assert isinstance(mapping, dict)

    def test_approved_entries_present(self):
        mapping = load_usagi_mapping(USAGI_CSV)
        # E11.9 and I10 are APPROVED with non-zero concept_id
        assert ":E11.9" in " ".join(mapping.keys()) or "E11.9" in mapping
        # We use sourceVocabularyId-less fixture, so keys are plain source codes
        # Check via any key that ends with E11.9
        keys = list(mapping.keys())
        assert any("E11.9" in k for k in keys)
        assert any("I10" in k for k in keys)

    def test_approved_concept_ids_correct(self):
        mapping = load_usagi_mapping(USAGI_CSV)
        e11_key = next(k for k in mapping if "E11.9" in k)
        i10_key = next(k for k in mapping if k.endswith("I10") or k == "I10")
        assert mapping[e11_key] == 201826
        assert mapping[i10_key] == 4195694

    def test_zero_concept_id_excluded(self):
        # Z99.9 maps to concept_id 0 → must be skipped
        mapping = load_usagi_mapping(USAGI_CSV)
        assert not any("Z99.9" in k for k in mapping)

    def test_unchecked_rows_excluded(self):
        # D63.0 has mappingStatus UNCHECKED → must be skipped
        mapping = load_usagi_mapping(USAGI_CSV)
        assert not any("D63.0" in k for k in mapping)

    def test_min_equivalence_filter(self):
        # Only EQUIVALENT rows: E11.9 and I10 qualify; Z99.9 (INEXACT) is
        # already skipped for concept_id=0 anyway
        mapping = load_usagi_mapping(USAGI_CSV, min_equivalence="EQUIVALENT")
        assert any("E11.9" in k for k in mapping)
        assert any("I10" in k for k in mapping)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_usagi_mapping(tmp_path / "nonexistent.csv")


# ---------------------------------------------------------------------------
# Guard: no vocabulary data is bundled in the package
# ---------------------------------------------------------------------------

class TestNoBundledVocabData:
    """Ensure that no Athena / OMOP vocabulary CSVs are shipped inside openmed."""

    FORBIDDEN_NAMES = {
        "CONCEPT.csv",
        "CONCEPT_SYNONYM.csv",
        "CONCEPT_RELATIONSHIP.csv",
        "CONCEPT_ANCESTOR.csv",
        "VOCABULARY.csv",
        "DOMAIN.csv",
        "CONCEPT_CLASS.csv",
        "DRUG_STRENGTH.csv",
    }

    def test_no_vocab_csvs_in_package(self):
        import openmed
        package_root = Path(openmed.__file__).parent
        found = []
        for root, _dirs, files in os.walk(package_root):
            for fname in files:
                if fname in self.FORBIDDEN_NAMES:
                    found.append(os.path.join(root, fname))
        assert found == [], (
            "Vocabulary CSV files must NEVER be bundled inside the openmed package. "
            f"Found: {found}"
        )
