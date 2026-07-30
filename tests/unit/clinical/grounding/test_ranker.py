"""Tests for the context-aware candidate ranking stage.

All vocabulary content is synthetic and algorithmically generated; no real
patient data or licensed terminology is used. The dense channel uses the
dependency-free deterministic hashing encoder, so the suite is fully offline and
byte-for-byte reproducible.
"""

from __future__ import annotations

import hashlib
import socket
from pathlib import Path

from openmed.clinical.context import CERTAIN, NEGATED, ClinicalAssertion, RerankContext
from openmed.clinical.grounding import (
    CandidateRankingStage,
    HashingAliasEncoder,
    RankingConfig,
    TwoStageRetriever,
    rank_mention,
    retrieve_candidates,
)
from openmed.clinical.grounding.types import Candidate
from openmed.clinical.normalization.cache import RankedCandidateCache
from openmed.clinical.normalization.ranker import RankedCandidate

_REPO_ROOT = Path(__file__).resolve().parents[4]
_FIXTURE = _REPO_ROOT / "openmed/eval/golden/fixtures/grounding_vocab_synthetic.jsonl"
_SYSTEMS = ("icd10cm", "rxnorm", "loinc", "hpo", "mesh")


def _loader(tmp_path: Path, path: Path = _FIXTURE):
    from openmed.clinical.grounding.vocab import VocabLoader, VocabSource

    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    registry = {
        system: VocabSource(system=system, path=path, sha256=sha256)
        for system in _SYSTEMS
    }
    return VocabLoader(cache_dir=tmp_path / "cache", registry=registry)


# --------------------------------------------------------------------------- #
# Two-stage retrieval union
# --------------------------------------------------------------------------- #
def test_retrieve_unions_sparse_and_dense_sources(tmp_path):
    encoder = HashingAliasEncoder()
    candidates = retrieve_candidates(
        "type 2 diabetes",
        systems=["icd10cm"],
        loader=_loader(tmp_path),
        encoder=encoder,
    )

    assert candidates
    sources = {candidate.source for candidate in candidates}
    assert sources == {"sparse", "dense"}


def test_retrieve_without_encoder_is_sparse_only(tmp_path):
    candidates = retrieve_candidates(
        "type 2 diabetes",
        systems=["icd10cm"],
        loader=_loader(tmp_path),
    )

    assert candidates
    assert {candidate.source for candidate in candidates} == {"sparse"}


# --------------------------------------------------------------------------- #
# Stage returns the reranked order
# --------------------------------------------------------------------------- #
def test_stage_returns_reranked_candidates(tmp_path):
    stage = CandidateRankingStage(
        _loader(tmp_path),
        encoder=HashingAliasEncoder(),
    )
    ranked = stage.rank("type 2 diabetes", systems=["icd10cm"])

    assert ranked
    assert all(isinstance(item, RankedCandidate) for item in ranked)
    assert ranked[0].candidate.code == "E11.9"
    # Deterministically ordered: fused score descending.
    scores = [item.fused_score for item in ranked]
    assert scores == sorted(scores, reverse=True)


def test_stage_matches_direct_reranker_on_the_union(tmp_path):
    # The stage must not re-derive scoring: its output equals rank_candidates
    # applied to the same retrieved union.
    from openmed.clinical.normalization.ranker import rank_candidates

    loader = _loader(tmp_path)
    encoder = HashingAliasEncoder()
    stage = CandidateRankingStage(loader, encoder=encoder)
    ranked = stage.rank("type 2 diabetes", systems=["icd10cm"])

    union = TwoStageRetriever(loader, encoder=encoder).retrieve(
        "type 2 diabetes", ["icd10cm"], 10
    )
    expected = rank_candidates("type 2 diabetes", None, union)
    assert [item.concept_key for item in ranked] == [
        item.concept_key for item in expected
    ]


def test_stage_is_deterministic(tmp_path):
    stage = CandidateRankingStage(_loader(tmp_path), encoder=HashingAliasEncoder())
    first = stage.rank("type 2 diabetes", systems=["icd10cm"])
    second = stage.rank("type 2 diabetes", systems=["icd10cm"])
    assert [item.concept_key for item in first] == [item.concept_key for item in second]
    assert [item.fused_score for item in first] == [item.fused_score for item in second]


# --------------------------------------------------------------------------- #
# Graceful sparse-only degradation (no-op fallback)
# --------------------------------------------------------------------------- #
def test_absent_encoder_degrades_to_sparse_only(tmp_path):
    # No encoder configured, and no encoder_path in the config: the stage must be
    # a pure no-op fallback that reproduces the sparse-only baseline order.
    loader = _loader(tmp_path)
    stage = CandidateRankingStage(loader)  # no encoder
    assert stage.rerank_enabled is False

    ranked = stage.rank("type 2 diabetes", systems=["icd10cm"])
    assert ranked
    assert all(item.sources == ("sparse",) for item in ranked)

    baseline = rank_mention("type 2 diabetes", systems=["icd10cm"], loader=loader)
    assert [item.concept_key for item in ranked] == [
        item.concept_key for item in baseline
    ]


def test_missing_encoder_path_yields_sparse_only(tmp_path):
    # A configured but unavailable encoder path resolves to None (no download),
    # so the stage still degrades gracefully rather than raising.
    config = RankingConfig(encoder_path=str(tmp_path / "no-such-weights"))
    stage = CandidateRankingStage(_loader(tmp_path), config=config)
    assert stage.rerank_enabled is False
    ranked = stage.rank("type 2 diabetes", systems=["icd10cm"])
    assert ranked
    assert all(item.sources == ("sparse",) for item in ranked)


def test_rerank_off_ignores_dense_and_context(tmp_path):
    # rerank=False forces the sparse baseline even with an encoder and a context.
    loader = _loader(tmp_path)
    config = RankingConfig(rerank=False)
    stage = CandidateRankingStage(loader, encoder=HashingAliasEncoder(), config=config)
    assert stage.rerank_enabled is False

    context = RerankContext(
        section="assessment",
        preferred_concepts=frozenset({("ICD10CM", "E11.65")}),
    )
    ranked = stage.rank("type 2 diabetes", systems=["icd10cm"], context=context)
    assert all(item.sources == ("sparse",) for item in ranked)
    # Section preference is ignored when rerank is off.
    assert all(item.feature_map["section_match"] == 0.0 for item in ranked)


def test_absent_encoder_still_applies_context(tmp_path):
    # With no dense encoder (rerank_enabled False) but rerank left on, the section
    # context must still refine the sparse ranking: the dense channel degrades,
    # the context does not.
    loader = _loader(tmp_path)
    stage = CandidateRankingStage(loader)  # no encoder
    assert stage.rerank_enabled is False

    plain = stage.rank("type 2 diabetes", systems=["icd10cm"])
    assert plain
    preferred_key = plain[-1].concept_key  # a sparse candidate the section prefers

    context = RerankContext(
        section="assessment",
        preferred_concepts=frozenset({preferred_key}),
    )
    ranked = stage.rank("type 2 diabetes", systems=["icd10cm"], context=context)
    assert all(item.sources == ("sparse",) for item in ranked)
    by_key = {item.concept_key: item for item in ranked}
    # The section made the preferred sense salient even without a dense encoder.
    assert by_key[preferred_key].feature_map["section_match"] == 1.0


# --------------------------------------------------------------------------- #
# Context influences ranking: section-collision resolution
# --------------------------------------------------------------------------- #
def _collision_union(surface: str) -> list[Candidate]:
    """Two same-surface senses tied by the sparse channel (A wins the tie-break)."""

    return [
        Candidate(
            system="ICD10CM",
            code="A",
            display="sense A",
            score=1.0,
            source="sparse",
            matched_alias=surface,
            match_kind="exact",
            vocab_version="sha256:collision",
        ),
        Candidate(
            system="ICD10CM",
            code="B",
            display="sense B",
            score=1.0,
            source="sparse",
            matched_alias=surface,
            match_kind="exact",
            vocab_version="sha256:collision",
        ),
    ]


def test_section_context_resolves_same_surface_collision():
    from openmed.clinical.normalization.ranker import rank_candidates

    union = _collision_union("discharge")

    # Baseline (no context): the deterministic tie-break keeps "A" first.
    baseline = rank_candidates("discharge", None, union)
    assert baseline[0].candidate.code == "A"

    # With a section that makes "B" the salient sense, "B" wins.
    context = RerankContext(
        section="assessment",
        preferred_concepts=frozenset({("ICD10CM", "B")}),
    )
    resolved = rank_candidates("discharge", context, union)
    assert resolved[0].candidate.code == "B"
    assert resolved[0].feature_map["section_match"] == 1.0


def test_section_collision_suite_resolves_at_least_ninety_percent():
    # A synthetic section-collision suite: each surface has two senses tied by
    # the sparse channel; the section makes the gold sense salient. The stage
    # must resolve the section-correct sense in >= 90% of cases, versus a
    # sparse-only baseline that is wrong on every case.
    from openmed.clinical.normalization.ranker import rank_candidates

    case_count = 20
    baseline_correct = 0
    resolved_correct = 0
    for index in range(case_count):
        surface = f"surface{index:02d}"
        union = _collision_union(surface)
        gold = "B"  # the tie-break loser, salient only via the section
        context = RerankContext(
            section="assessment",
            preferred_concepts=frozenset({("ICD10CM", gold)}),
        )
        baseline = rank_candidates(surface, None, union)
        resolved = rank_candidates(surface, context, union)
        baseline_correct += int(baseline[0].candidate.code == gold)
        resolved_correct += int(resolved[0].candidate.code == gold)

    assert baseline_correct == 0
    resolution_rate = resolved_correct / case_count
    assert resolution_rate >= 0.90
    # Top-1 accuracy improvement is far beyond the >= 8 absolute point bar.
    improvement = resolution_rate - (baseline_correct / case_count)
    assert improvement >= 0.08


def test_negated_assertion_is_recorded_as_a_feature(tmp_path):
    stage = CandidateRankingStage(_loader(tmp_path))
    context = RerankContext(
        section="assessment",
        assertion=ClinicalAssertion(
            temporality="current", certainty=CERTAIN, negation=NEGATED
        ),
    )
    ranked = stage.rank("type 2 diabetes", systems=["icd10cm"], context=context)
    assert ranked
    assert all(item.feature_map["assertion_present"] == 0.0 for item in ranked)


# --------------------------------------------------------------------------- #
# Per-(mention, vocab-version) caching
# --------------------------------------------------------------------------- #
def test_stage_caches_per_mention_and_vocab_version(tmp_path):
    cache = RankedCandidateCache()
    stage = CandidateRankingStage(
        _loader(tmp_path),
        encoder=HashingAliasEncoder(),
        cache=cache,
    )
    first = stage.rank("type 2 diabetes", systems=["icd10cm"])
    second = stage.rank("type 2 diabetes", systems=["icd10cm"])
    # The second call is served from cache: identical object, one stored entry.
    assert second is first
    assert cache.stats().hits == 1


# --------------------------------------------------------------------------- #
# Offline / zero network egress
# --------------------------------------------------------------------------- #
def test_stage_runs_with_sockets_blocked(tmp_path, monkeypatch):
    def fail_socket(*args, **kwargs):
        raise AssertionError("network egress attempted during ranking")

    monkeypatch.setattr(socket.socket, "connect", fail_socket)
    monkeypatch.setattr(socket.socket, "connect_ex", fail_socket)
    monkeypatch.setattr(socket, "create_connection", fail_socket)

    stage = CandidateRankingStage(_loader(tmp_path), encoder=HashingAliasEncoder())
    ranked = stage.rank("type 2 diabetes", systems=["icd10cm"])
    assert ranked
    assert ranked[0].candidate.code == "E11.9"


# --------------------------------------------------------------------------- #
# Wheel ships no vocabulary content
# --------------------------------------------------------------------------- #
def test_ranking_stage_ships_no_vocabulary_content():
    import openmed.clinical.grounding as grounding

    package_root = Path(grounding.__file__).resolve().parent
    vocab_suffixes = {".tsv", ".rrf", ".obo", ".csv", ".jsonl", ".txt", ".xml"}
    bundled = [
        path
        for path in package_root.rglob("*")
        if path.suffix.lower() in vocab_suffixes
    ]
    assert bundled == []
