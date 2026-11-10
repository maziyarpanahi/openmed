"""Tests for synthetic concept normalization (OM-613)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from openmed.clinical.normalization import (
    SYNTHETIC_CONCEPTS,
    SYNTHETIC_GOLD_SET,
    BackendIdentity,
    ConceptNormalizationCache,
    ConceptNormalizer,
    SyntheticTerminologyBackend,
    TerminologyConcept,
    evaluate_normalization_gold,
    generate_query_variants,
    make_normalization_cache_key,
    normalize_surface,
)


class CountingBackend(SyntheticTerminologyBackend):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lookup_calls = 0
        self.candidate_calls = 0

    def lookup(self, normalized):
        self.lookup_calls += 1
        return super().lookup(normalized)

    def candidates(self, tokens):
        self.candidate_calls += 1
        return super().candidates(tokens)

    @property
    def call_count(self):
        return self.lookup_calls + self.candidate_calls


def test_synthetic_backend_exact_lookup_and_metadata():
    backend = SyntheticTerminologyBackend()

    concepts = backend.lookup(normalize_surface("Aster pyrexia"))

    assert concepts[0].code == "SYN-COND-001"
    assert concepts[0].system_uri.startswith("https://openmed.ai/fhir/CodeSystem/")
    assert backend.identity.name == "openmed-synthetic-terminology"
    assert backend.identity.version == "2026.06"
    assert {system.system_id for system in backend.code_systems()} == {
        "synthetic-condition",
        "synthetic-observation",
        "synthetic-treatment",
    }


def test_candidate_generation_expands_abbreviations_and_ngrams():
    variants = generate_query_variants(
        "AF pattern",
        abbreviation_expansions={"af": ("aster fever",)},
        max_ngram=2,
    )

    assert variants[0] == "af pattern"
    assert "aster fever pattern" in variants
    assert "aster fever" in variants
    assert "fever pattern" in variants


def test_ranker_returns_confidence_and_offset_provenance():
    normalizer = ConceptNormalizer(SyntheticTerminologyBackend())

    ranked = normalizer.normalize("AF", start=21, end=23)

    assert ranked[0].concept.code == "SYN-COND-001"
    assert ranked[0].confidence >= 0.8
    assert ranked[0].provenance.mention_start == 21
    assert ranked[0].provenance.mention_end == 23
    assert ranked[0].provenance.backend_version == "2026.06"
    assert ranked[0].provenance.query_variant_count >= 1
    assert "AF" not in ranked[0].provenance.matched_term_hash


def test_synthetic_gold_accuracy_and_cache_hit_rate_are_ci_gated():
    cache = ConceptNormalizationCache(max_entries=32)
    normalizer = ConceptNormalizer(SyntheticTerminologyBackend(), cache=cache)

    result = evaluate_normalization_gold(normalizer, SYNTHETIC_GOLD_SET)

    assert result.case_count == len(SYNTHETIC_GOLD_SET)
    assert result.top1_accuracy >= 0.80
    assert result.top5_accuracy >= 0.92
    assert result.cache_hit_rate >= 0.5


def test_cache_equivalence_and_repeated_mentions_skip_backend_calls():
    backend = CountingBackend()
    cache = ConceptNormalizationCache(max_entries=8)
    cached_normalizer = ConceptNormalizer(backend, cache=cache)
    uncached_normalizer = ConceptNormalizer(SyntheticTerminologyBackend())

    uncached = uncached_normalizer.normalize("Aster pyrexia", use_cache=False)
    first = cached_normalizer.normalize("Aster pyrexia")
    calls_after_first = backend.call_count
    second = cached_normalizer.normalize("Aster pyrexia")

    assert first == uncached
    assert second == first
    assert backend.call_count == calls_after_first
    assert cache.stats().hit_rate == 0.5
    assert cache.stats().size == 1


def test_concept_cache_is_bounded_and_evicts_least_recent_entry():
    backend = CountingBackend()
    cache = ConceptNormalizationCache(max_entries=2)
    normalizer = ConceptNormalizer(backend, cache=cache)

    normalizer.normalize("Aster pyrexia")
    normalizer.normalize("beryl cough")
    normalizer.normalize("corin rash")
    calls_after_eviction = backend.call_count
    normalizer.normalize("Aster pyrexia")

    assert cache.stats().size == 2
    assert backend.call_count > calls_after_eviction


def test_cache_key_includes_backend_identity_version():
    identity_v1 = BackendIdentity("synthetic-test", "v1")
    identity_v2 = BackendIdentity("synthetic-test", "v2")

    key_v1 = make_normalization_cache_key("Aster pyrexia", identity_v1)
    key_v2 = make_normalization_cache_key("Aster pyrexia", identity_v2)

    assert key_v1 != key_v2
    assert "Aster" not in key_v1
    assert "pyrexia" not in key_v1


def test_swapping_backends_does_not_return_cross_backend_stale_codes():
    base = SYNTHETIC_CONCEPTS[0]
    alternate = replace(base, code="SYN-COND-ALT", display="Aster fever alternate")
    cache = ConceptNormalizationCache(max_entries=8)
    normalizer_v1 = ConceptNormalizer(
        SyntheticTerminologyBackend(
            identity=BackendIdentity("synthetic-swap", "v1"),
            concepts=(base,),
        ),
        cache=cache,
    )
    normalizer_v2 = ConceptNormalizer(
        SyntheticTerminologyBackend(
            identity=BackendIdentity("synthetic-swap", "v2"),
            concepts=(alternate,),
        ),
        cache=cache,
    )

    first = normalizer_v1.normalize("Aster fever")
    second = normalizer_v2.normalize("Aster fever alternate")

    assert first[0].concept.code == "SYN-COND-001"
    assert second[0].concept.code == "SYN-COND-ALT"
    assert cache.stats().misses == 2


def test_backend_identity_requires_name_and_version():
    with pytest.raises(ValueError, match="version"):
        BackendIdentity("synthetic-test", "")


def test_custom_backend_can_supply_synthetic_concepts_only():
    custom = TerminologyConcept(
        system_id="synthetic-condition",
        system_uri="https://openmed.ai/fhir/CodeSystem/synthetic-condition",
        code="SYN-CUSTOM-001",
        display="Kilo balance cue",
        aliases=("kbc",),
        version="2026.06-synthetic",
    )
    normalizer = ConceptNormalizer(
        SyntheticTerminologyBackend(concepts=(custom,)),
        abbreviation_expansions={"kbc": "kilo balance cue"},
    )

    ranked = normalizer.normalize("KBC")

    assert ranked[0].concept.code == "SYN-CUSTOM-001"
