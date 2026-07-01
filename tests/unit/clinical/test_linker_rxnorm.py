"""Tests for the RxNorm approximate-match linker and grounding scaffolding (#267)."""

from __future__ import annotations

from pathlib import Path

import pytest

from openmed.clinical.grounding import (
    Candidate,
    available_linkers,
    get_linker,
    load_vocab,
)
from openmed.clinical.grounding.linkers.rxnorm import RxNormLinker

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "clinical"
    / "grounding"
    / "rxnorm_sample.jsonl"
)


@pytest.fixture
def linker() -> RxNormLinker:
    return RxNormLinker(load_vocab(FIXTURE))


class TestVocabLoader:
    def test_loads_from_path(self):
        vocab = load_vocab(FIXTURE)
        assert vocab.exact("aspirin")

    def test_loads_from_in_memory_rows(self):
        vocab = load_vocab(
            [{"code": "1", "display": "widget", "synonyms": ["widget", "gadget"]}]
        )
        assert vocab.exact("gadget")


class TestExactMatch:
    def test_link_exact_returns_expected_rxcui_top1(self, linker):
        candidates = linker.link("aspirin")
        assert candidates
        top = candidates[0]
        assert isinstance(top, Candidate)
        assert top.system == "RXNORM"
        assert top.code == "1191"
        assert top.display == "aspirin"
        assert top.score == pytest.approx(1.0)

    def test_link_matches_brand_synonym(self, linker):
        candidates = linker.link("Tylenol")
        assert candidates[0].code == "161"

    def test_out_of_vocabulary_returns_empty_or_low_score(self, linker):
        candidates = linker.link("qwertabcdrug")
        assert candidates == [] or all(c.score < 0.5 for c in candidates)

    def test_deterministic_across_runs(self, linker):
        first = linker.link("acetaminophen")
        second = linker.link("acetaminophen")
        assert first == second


class TestApproximateMatch:
    def test_misspelling_links_within_threshold(self, linker):
        pytest.importorskip("rapidfuzz")
        candidates = linker.link("asprin")  # missing 'a'
        assert candidates
        assert candidates[0].code == "1191"
        assert candidates[0].score < 1.0

    def test_top_k_limits_results(self, linker):
        pytest.importorskip("rapidfuzz")
        candidates = linker.link("metformin", top_k=2)
        assert len(candidates) <= 2


class TestMedicationGate:
    def test_gate_blocks_non_medication_label(self, linker):
        assert linker.link("aspirin", canonical_label="DATE") == []

    def test_gate_allows_drug_alias(self, linker):
        # 'drug' normalizes to MEDICATION via the canonical label taxonomy.
        assert linker.link("aspirin", canonical_label="drug")


class TestRegistry:
    def test_rxnorm_linker_is_discoverable(self):
        assert "rxnorm" in available_linkers()
        assert get_linker("rxnorm") is RxNormLinker

    def test_unknown_system_raises(self):
        with pytest.raises(KeyError):
            get_linker("does-not-exist")
