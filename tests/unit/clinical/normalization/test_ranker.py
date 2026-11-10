"""Tests for the context-aware sparse+dense candidate reranker.

All candidate data is synthetic and algorithmically generated; no real patient
data or licensed terminology is used. The reranker is pure Python and fully
offline, so the suite is byte-for-byte reproducible across runs.
"""

from __future__ import annotations

import socket

import pytest

from openmed.clinical.context import ClinicalAssertion, RerankContext
from openmed.clinical.grounding.types import Candidate
from openmed.clinical.normalization import (
    RankedCandidate,
    RankedCandidateCache,
    SourceContribution,
    rank_candidates,
)
from openmed.clinical.normalization.ranker import DEFAULT_ASSERTION_WEIGHT
from openmed.eval.suites.candidate_ranking import (
    build_candidate_ranking_gold,
    evaluate_candidate_ranking,
)


def _candidate(
    system: str,
    code: str,
    score: float,
    source: str,
    match_kind: str = "exact",
) -> Candidate:
    return Candidate(
        system=system,
        code=code,
        display=f"{system} {code}",
        score=score,
        source=source,
        matched_alias=code.lower(),
        match_kind=match_kind,
        vocab_version="v1",
    )


def test_offline_no_socket(monkeypatch):
    """The reranker performs no network I/O."""

    def _forbidden(*_args, **_kwargs):  # pragma: no cover - guard only
        raise AssertionError("reranking must not open a socket")

    monkeypatch.setattr(socket, "socket", _forbidden)
    result = rank_candidates("m", None, [_candidate("ICD10CM", "A", 1.0, "sparse")])
    assert result[0].concept_key == ("ICD10CM", "A")


def test_returns_ranked_candidates_with_fused_score_and_attribution():
    """Output items carry a fused score, per-source contributions, and sources."""

    sparse = [
        _candidate("ICD10CM", "D", 0.95, "sparse", "fuzzy"),
        _candidate("ICD10CM", "G", 0.85, "sparse", "fuzzy"),
    ]
    dense = [_candidate("ICD10CM", "G", 0.90, "dense", "dense")]

    ranked = rank_candidates("dx", None, sparse + dense)

    assert all(isinstance(item, RankedCandidate) for item in ranked)
    top = ranked[0]
    assert top.concept_key == ("ICD10CM", "G")
    assert top.sources == ("dense", "sparse")
    assert {c.source for c in top.contributions} == {"dense", "sparse"}
    assert all(isinstance(c, SourceContribution) for c in top.contributions)
    rrf_total = sum(c.rrf for c in top.contributions)
    # No section preference here, so only the uniform assertion term is added.
    assert top.fused_score == pytest.approx(rrf_total + DEFAULT_ASSERTION_WEIGHT)
    assert top.feature_map["rrf"] == pytest.approx(rrf_total)
    assert top.feature_map["section_match"] == 0.0


def test_fusion_promotes_dual_source_gold_over_lexical_distractor():
    """A concept returned by both channels outranks a sparse-only distractor."""

    sparse = [
        _candidate("RXNORM", "DISTRACT", 0.99, "sparse", "fuzzy"),
        _candidate("RXNORM", "GOLD", 0.80, "sparse", "fuzzy"),
    ]
    dense = [_candidate("RXNORM", "GOLD", 0.88, "dense", "dense")]

    baseline = rank_candidates("m", None, sparse)
    fused = rank_candidates("m", None, sparse + dense)

    assert baseline[0].concept_key == ("RXNORM", "DISTRACT")
    assert fused[0].concept_key == ("RXNORM", "GOLD")


def test_ranking_is_deterministic_across_runs():
    """Identical input yields a byte-identical ranked list across runs."""

    sparse = [
        _candidate("HPO", "H1", 0.9, "sparse", "fuzzy"),
        _candidate("HPO", "H2", 0.7, "sparse", "fuzzy"),
    ]
    dense = [
        _candidate("HPO", "H2", 0.95, "dense", "dense"),
        _candidate("HPO", "H1", 0.6, "dense", "dense"),
    ]
    first = rank_candidates("m", None, sparse + dense)
    second = rank_candidates("m", None, sparse + dense)

    assert first == second
    assert repr(first) == repr(second)


def test_dedup_keeps_strongest_candidate_per_concept_key():
    """A concept present in both sources is emitted once with both sources."""

    candidates = [
        _candidate("LOINC", "L1", 0.6, "sparse", "fuzzy"),
        _candidate("LOINC", "L1", 1.0, "dense", "dense"),
    ]
    ranked = rank_candidates("m", None, candidates)

    assert len(ranked) == 1
    assert ranked[0].candidate.score == pytest.approx(1.0)
    assert ranked[0].sources == ("dense", "sparse")


def test_graceful_degradation_without_dense_matches_sparse_baseline():
    """With dense removed, the order equals the sparse-only baseline order."""

    sparse = [
        _candidate("ICD10CM", "S1", 0.90, "sparse", "fuzzy"),
        _candidate("ICD10CM", "S2", 0.70, "sparse", "fuzzy"),
        _candidate("ICD10CM", "S3", 0.50, "sparse", "fuzzy"),
    ]
    baseline_order = [(c.system, c.code) for c in sparse]

    without_context = [rc.concept_key for rc in rank_candidates("m", None, sparse)]
    empty_context = RerankContext(section="assessment")
    with_context = [
        rc.concept_key for rc in rank_candidates("m", empty_context, sparse)
    ]

    assert without_context == baseline_order
    assert with_context == baseline_order


def test_section_context_resolves_same_surface_collision():
    """A shared surface resolves to the section-preferred concept."""

    first = ("ICD10CM", "MS-NEURO")
    second = ("ICD10CM", "MS-CARDIO")
    sparse = [
        _candidate(*first, 1.0, "sparse", "exact"),
        _candidate(*second, 1.0, "sparse", "exact"),
    ]
    dense = [
        _candidate(*first, 0.9, "dense", "dense"),
        _candidate(*second, 0.9, "dense", "dense"),
    ]
    candidates = sparse + dense

    # Without a section signal the deterministic tie-break always wins.
    neutral = rank_candidates("ms", None, candidates)
    assert neutral[0].concept_key == first

    neuro_ctx = RerankContext(
        section="assessment", preferred_concepts=frozenset({first})
    )
    cardio_ctx = RerankContext(
        section="assessment", preferred_concepts=frozenset({second})
    )
    assert rank_candidates("ms", neuro_ctx, candidates)[0].concept_key == first
    assert rank_candidates("ms", cardio_ctx, candidates)[0].concept_key == second


def test_assertion_feature_is_recorded_without_reordering():
    """The assertion axis is recorded per candidate but never reorders them."""

    candidates = [
        _candidate("MESH", "M1", 0.9, "sparse", "fuzzy"),
        _candidate("MESH", "M2", 0.7, "sparse", "fuzzy"),
    ]
    negated = RerankContext(
        assertion=ClinicalAssertion(
            temporality="recent", certainty="certain", negation="negated"
        )
    )
    present = rank_candidates("m", None, candidates)
    absent = rank_candidates("m", negated, candidates)

    assert [rc.concept_key for rc in present] == [rc.concept_key for rc in absent]
    assert absent[0].feature_map["assertion_present"] == 0.0
    assert present[0].feature_map["assertion_present"] == 1.0


def test_cache_reuses_ranked_list_per_mention_and_vocab_version():
    """A cached ranking is reused for the same mention and vocab version."""

    cache = RankedCandidateCache()
    candidates = [_candidate("ICD10CM", "A", 1.0, "sparse")]

    first = rank_candidates(
        "reflux", None, candidates, cache=cache, vocab_version="edition-1"
    )
    second = rank_candidates(
        "reflux", None, candidates, cache=cache, vocab_version="edition-1"
    )
    assert first is second
    assert cache.stats().hits == 1

    # A changed vocabulary version does not serve the stale ranking.
    rank_candidates("reflux", None, candidates, cache=cache, vocab_version="edition-2")
    assert cache.stats().misses == 2


def test_cache_key_distinguishes_context_and_parameters():
    """A shared cache must not serve a ranking computed under a different section
    or fusion parameters for the same surface + vocab version."""

    cache = RankedCandidateCache()
    first = ("ICD10CM", "A")
    second = ("ICD10CM", "B")
    candidates = [
        _candidate(*first, 1.0, "sparse", "fuzzy"),
        _candidate(*second, 1.0, "sparse", "fuzzy"),
        _candidate(*first, 1.0, "dense", "dense"),
        _candidate(*second, 1.0, "dense", "dense"),
    ]
    neuro = RerankContext(section="assessment", preferred_concepts=frozenset({first}))
    cardio = RerankContext(section="assessment", preferred_concepts=frozenset({second}))

    top_neuro = rank_candidates(
        "ms", neuro, candidates, cache=cache, vocab_version="v1"
    )[0].concept_key
    top_cardio = rank_candidates(
        "ms", cardio, candidates, cache=cache, vocab_version="v1"
    )[0].concept_key
    # The second section must resolve to its own preferred concept, not the
    # first section's cached ranking for the same surface.
    assert top_neuro == first
    assert top_cardio == second

    # A changed fusion parameter also misses rather than serving the stale list.
    misses_before = cache.stats().misses
    rank_candidates("ms", neuro, candidates, cache=cache, vocab_version="v1", rrf_k=1)
    assert cache.stats().misses == misses_before + 1


def test_synthetic_gold_top1_improves_and_top5_recall_holds():
    """Fusion lifts top-1 by >= 8 points and keeps top-5 recall >= 0.90."""

    report = evaluate_candidate_ranking()

    assert report["top1_improvement"] >= 0.08
    assert (
        report["fused_top1_accuracy"] - report["baseline_top1_accuracy"]
    ) * 100 >= 8.0
    assert report["fused_top5_recall"] >= 0.90


def test_synthetic_section_collision_resolution_rate():
    """The section-collision suite resolves >= 90% of cases correctly."""

    report = evaluate_candidate_ranking()

    assert report["collision_case_count"] > 0
    assert report["collision_resolution_rate"] >= 0.90


def test_gold_set_is_stable_and_nonempty():
    """The synthetic gold set is deterministic and non-empty."""

    first = build_candidate_ranking_gold()
    second = build_candidate_ranking_gold()
    assert first == second
    assert len(first) >= 20
