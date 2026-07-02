"""Tests for the HPO phenotype linker (issue #297)."""

from __future__ import annotations

from pathlib import Path

import pytest

from openmed.clinical.grounding import (
    Candidate,
    available_linkers,
    get_linker,
    load_vocab,
)
from openmed.clinical.grounding.linkers.hpo import HpoLinker

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "clinical"
    / "grounding"
    / "hpo_sample.jsonl"
)


@pytest.fixture
def linker() -> HpoLinker:
    return HpoLinker(load_vocab(FIXTURE))


class TestHpoLinker:
    def test_phenotype_links_to_expected_hp_code(self, linker):
        candidates = linker.link("seizure")
        assert candidates
        top = candidates[0]
        assert isinstance(top, Candidate)
        assert top.system == "HPO"
        assert top.code == "HP:0001250"
        assert top.score == pytest.approx(1.0)

    def test_synonym_links_to_expected_code(self, linker):
        assert linker.link("pyrexia")[0].code == "HP:0001945"

    def test_hp_code_format_is_normalized(self, linker):
        # Fixture stores bare/short/prefixed codes; output is always HP:nnnnnnn.
        assert linker.link("headache")[0].code == "HP:0002315"
        assert linker.link("ataxia")[0].code == "HP:0001251"
        assert linker.link("fever")[0].code == "HP:0001945"

    def test_deterministic_across_runs(self, linker):
        assert linker.link("cough") == linker.link("cough")

    def test_condition_gate(self, linker):
        assert linker.link("seizure", canonical_label="DATE") == []
        assert linker.link("seizure", canonical_label="finding")


class TestRegistryAndReuse:
    def test_registered_under_hpo(self):
        assert "hpo" in available_linkers()
        assert get_linker("hpo") is HpoLinker

    def test_reuses_shared_matching_base(self):
        from openmed.clinical.grounding.linkers.base import VocabLinker

        assert issubclass(HpoLinker, VocabLinker)
