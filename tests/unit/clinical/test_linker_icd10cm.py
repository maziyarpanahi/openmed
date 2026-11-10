"""Tests for the ICD-10-CM diagnosis linker (issue #268)."""

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
from openmed.clinical.grounding.linkers.icd10cm import Icd10cmLinker

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "clinical"
    / "grounding"
    / "icd10cm_sample.jsonl"
)


@pytest.fixture
def linker() -> Icd10cmLinker:
    return Icd10cmLinker(_load_fixture_index())


def _load_fixture_index() -> VocabularyIndex:
    loader = VocabLoader(
        registry={"icd10cm": VocabSource(system="icd10cm", path=FIXTURE)}
    )
    return loader.get_index("icd10cm")


class TestIcd10cmLinker:
    def test_diagnosis_links_to_leaf_code_with_dotted_format(self, linker):
        candidates = linker.link("type 2 diabetes mellitus")
        assert candidates
        top = candidates[0]
        assert isinstance(top, Candidate)
        assert top.system == "ICD10CM"
        # Leaf (billable) code preferred over the E11 category, dotted formatting.
        assert top.code == "E11.9"

    def test_synonym_links_to_expected_code(self, linker):
        assert linker.link("high blood pressure")[0].code == "I10"

    def test_dotted_formatting_for_longer_codes(self, linker):
        candidates = linker.link("type 2 diabetes mellitus with hyperglycemia")
        assert candidates[0].code == "E11.65"

    def test_deterministic_across_runs(self, linker):
        assert linker.link("pneumonia") == linker.link("pneumonia")

    def test_condition_gate_blocks_non_condition_label(self, linker):
        assert linker.link("pneumonia", canonical_label="DATE") == []
        assert linker.link("pneumonia", canonical_label="disease")


class TestRegistryAndReuse:
    def test_registered_under_icd10cm(self):
        assert "icd10cm" in available_linkers()
        assert get_linker("icd10cm") is Icd10cmLinker

    def test_reuses_shared_matching_base(self):
        from openmed.clinical.grounding.linkers.base import VocabLinker
        from openmed.clinical.grounding.linkers.rxnorm import RxNormLinker

        # Both linkers share the base implementation rather than duplicating it.
        assert issubclass(Icd10cmLinker, VocabLinker)
        assert issubclass(RxNormLinker, VocabLinker)
