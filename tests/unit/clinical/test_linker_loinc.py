"""Tests for the LOINC laboratory-test linker (issue #2092)."""

from __future__ import annotations

from pathlib import Path

import pytest

from openmed.clinical.grounding import (
    Candidate,
    VocabLoader,
    VocabSource,
    VocabularyIndex,
    available_linkers,
    get_linker,
)
from openmed.clinical.grounding.linkers.loinc import LoincLinker

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "clinical"
    / "grounding"
    / "loinc_sample.jsonl"
)


@pytest.fixture
def linker() -> LoincLinker:
    return LoincLinker(_load_fixture_index())


def _load_fixture_index() -> VocabularyIndex:
    loader = VocabLoader(registry={"loinc": VocabSource(system="loinc", path=FIXTURE)})
    return loader.get_index("loinc")


class TestLoincLinker:
    def test_lab_name_links_to_expected_loinc_code(self, linker):
        candidates = linker.link("hemoglobin a1c")
        assert candidates
        top = candidates[0]
        assert isinstance(top, Candidate)
        assert top.system == "LOINC"
        assert top.code == "4548-4"
        assert top.display == "Hemoglobin A1c/Hemoglobin.total in Blood"
        assert top.score == pytest.approx(1.0)

    def test_synonym_links_to_expected_code(self, linker):
        assert linker.link("HbA1c")[0].code == "4548-4"

    def test_deterministic_across_runs(self, linker):
        assert linker.link("hemoglobin a1c") == linker.link("hemoglobin a1c")

    def test_lab_test_gate_blocks_non_lab_label(self, linker):
        assert linker.link("hemoglobin a1c", canonical_label="DATE") == []
        assert linker.link("hemoglobin a1c", canonical_label="LAB_TEST")


class TestRegistryAndReuse:
    def test_registered_under_loinc(self):
        assert "loinc" in available_linkers()
        assert get_linker("loinc") is LoincLinker

    def test_reuses_shared_matching_base(self):
        from openmed.clinical.grounding.linkers.base import VocabLinker

        assert issubclass(LoincLinker, VocabLinker)
