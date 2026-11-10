"""Synthetic entity->code benchmark for the sparse+dense candidate reranker.

This suite measures the context-aware reciprocal-rank-fusion reranker
(:func:`openmed.clinical.normalization.rank_candidates`) against the sparse-only
first-hit baseline it is meant to improve on. Every case supplies a merged
sparse and dense candidate list plus an entity->code gold; the suite reports
top-1 accuracy for both the baseline and the fused ranking, top-5 recall of the
fused ranking, and a same-surface section-collision resolution rate.

The corpus is fully synthetic and algorithmically generated: concept codes,
surfaces, and per-source candidate scores are derived from case indices, so the
suite is offline and byte-for-byte deterministic. It exercises the reranker's
scoring, not candidate generation or the index, which are owned by sibling
tasks and covered by their own suites.

Three case families are generated:

* **Disambiguation** cases model sparse's known weakness: a lexical distractor
  outranks the gold in the sparse list but is absent from the dense list, so
  reciprocal-rank fusion (which rewards a concept returned by both channels)
  promotes the gold to rank 1 while the sparse-only baseline stays wrong.
* **Easy** cases place the gold first in both channels, so the baseline is
  already correct and the reranker must not regress it.
* **Collision** cases share one surface across two sections, each preferring a
  different concept. The channels tie, so only the section context feature
  resolves the section-appropriate sense.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from openmed.clinical.context import RerankContext
from openmed.clinical.grounding.types import Candidate
from openmed.clinical.normalization.ranker import rank_candidates

CANDIDATE_RANKING = "candidate_ranking"

_SYSTEMS = ("ICD10CM", "RXNORM", "LOINC", "HPO", "MESH")
_VOCAB_VERSION = "sha256:synthetic-candidate-ranking-v1"

_DEFAULT_DISAMBIGUATION = 12
_DEFAULT_EASY = 6
_DEFAULT_COLLISION_SURFACES = 6


@dataclass(frozen=True)
class CandidateRankingCase:
    """One synthetic reranking case with its gold and per-source candidates."""

    mention: str
    gold_system: str
    gold_code: str
    section: str | None
    preferred_concepts: tuple[tuple[str, str], ...]
    sparse: tuple[Candidate, ...]
    dense: tuple[Candidate, ...]
    collision: bool

    @property
    def gold_key(self) -> tuple[str, str]:
        """Return the ``(system, code)`` gold identity for the case."""

        return (self.gold_system, self.gold_code)

    def context(self) -> RerankContext:
        """Return the rerank context payload for this case."""

        return RerankContext(
            section=self.section,
            preferred_concepts=frozenset(self.preferred_concepts),
        )


def _candidate(
    system: str, code: str, score: float, source: str, match_kind: str
) -> Candidate:
    return Candidate(
        system=system,
        code=code,
        display=f"{system.lower()} concept {code}",
        score=round(score, 6),
        source=source,
        matched_alias=code.lower(),
        match_kind=match_kind,
        vocab_version=_VOCAB_VERSION,
    )


def _disambiguation_cases(count: int) -> list[CandidateRankingCase]:
    cases: list[CandidateRankingCase] = []
    for index in range(count):
        system = _SYSTEMS[index % len(_SYSTEMS)]
        gold = f"G{index:03d}"
        distractor = f"D{index:03d}"
        # Sparse ranks a lexical-only distractor first; the gold trails it. The
        # distractor never appears in the dense list, so fusion favors the gold.
        sparse = (
            _candidate(system, distractor, 0.95, "sparse", "fuzzy"),
            _candidate(system, gold, 0.85, "sparse", "fuzzy"),
        )
        dense = (_candidate(system, gold, 0.90, "dense", "dense"),)
        cases.append(
            CandidateRankingCase(
                mention=f"dx{index:03d}",
                gold_system=system,
                gold_code=gold,
                section=None,
                preferred_concepts=(),
                sparse=sparse,
                dense=dense,
                collision=False,
            )
        )
    return cases


def _easy_cases(count: int) -> list[CandidateRankingCase]:
    cases: list[CandidateRankingCase] = []
    for index in range(count):
        system = _SYSTEMS[index % len(_SYSTEMS)]
        gold = f"E{index:03d}"
        noise = f"N{index:03d}"
        sparse = (
            _candidate(system, gold, 1.0, "sparse", "exact"),
            _candidate(system, noise, 0.40, "sparse", "fuzzy"),
        )
        dense = (
            _candidate(system, gold, 0.95, "dense", "dense"),
            _candidate(system, noise, 0.35, "dense", "dense"),
        )
        cases.append(
            CandidateRankingCase(
                mention=f"easy{index:03d}",
                gold_system=system,
                gold_code=gold,
                section=None,
                preferred_concepts=(),
                sparse=sparse,
                dense=dense,
                collision=False,
            )
        )
    return cases


def _collision_cases(surfaces: int) -> list[CandidateRankingCase]:
    cases: list[CandidateRankingCase] = []
    for index in range(surfaces):
        system = _SYSTEMS[index % len(_SYSTEMS)]
        # Two concepts share one surface; "A" sorts before "B" on the code
        # tie-break, so without a section signal the baseline always picks "A".
        first = f"A{index:03d}"
        second = f"B{index:03d}"
        sparse = (
            _candidate(system, first, 1.0, "sparse", "exact"),
            _candidate(system, second, 1.0, "sparse", "exact"),
        )
        dense = (
            _candidate(system, first, 0.90, "dense", "dense"),
            _candidate(system, second, 0.90, "dense", "dense"),
        )
        surface = f"amb{index:03d}"
        cases.append(
            CandidateRankingCase(
                mention=surface,
                gold_system=system,
                gold_code=first,
                section="assessment",
                preferred_concepts=((system, first),),
                sparse=sparse,
                dense=dense,
                collision=True,
            )
        )
        cases.append(
            CandidateRankingCase(
                mention=surface,
                gold_system=system,
                gold_code=second,
                section="past_medical_history",
                preferred_concepts=((system, second),),
                sparse=sparse,
                dense=dense,
                collision=True,
            )
        )
    return cases


def build_candidate_ranking_gold(
    *,
    disambiguation: int = _DEFAULT_DISAMBIGUATION,
    easy: int = _DEFAULT_EASY,
    collision_surfaces: int = _DEFAULT_COLLISION_SURFACES,
) -> tuple[CandidateRankingCase, ...]:
    """Return the deterministic synthetic entity->code reranking gold set."""

    cases = (
        _disambiguation_cases(disambiguation)
        + _easy_cases(easy)
        + _collision_cases(collision_surfaces)
    )
    return tuple(cases)


def candidate_ranking_metadata() -> dict[str, Any]:
    """Return provenance metadata for the candidate ranking suite."""

    return {
        "suite": CANDIDATE_RANKING,
        "source": "synthetic; algorithmically generated candidate lists",
        "redistribution": "safe; no DUA or production terminology",
        "systems": list(_SYSTEMS),
    }


def evaluate_candidate_ranking(
    cases: Sequence[CandidateRankingCase] | None = None,
    *,
    k: int = 5,
) -> dict[str, Any]:
    """Score baseline vs. fused reranking over the synthetic gold set.

    For each case the sparse-only baseline ranks the sparse candidates with no
    context; the fused ranking fuses the sparse and dense candidates with the
    case's section context. Reports top-1 accuracy for both, the absolute
    improvement, fused top-``k`` recall, and the collision resolution rate.
    """

    gold = tuple(cases) if cases is not None else build_candidate_ranking_gold()
    if not gold:
        raise ValueError("candidate ranking gold set must not be empty")

    baseline_top1 = 0
    fused_top1 = 0
    fused_topk = 0
    collision_total = 0
    collision_resolved = 0
    for case in gold:
        baseline = rank_candidates(case.mention, None, case.sparse)
        fused = rank_candidates(
            case.mention,
            case.context(),
            (*case.sparse, *case.dense),
            vocab_version=_VOCAB_VERSION,
        )
        if baseline and baseline[0].concept_key == case.gold_key:
            baseline_top1 += 1
        if fused and fused[0].concept_key == case.gold_key:
            fused_top1 += 1
        if any(ranked.concept_key == case.gold_key for ranked in fused[:k]):
            fused_topk += 1
        if case.collision:
            collision_total += 1
            if fused and fused[0].concept_key == case.gold_key:
                collision_resolved += 1

    case_count = len(gold)
    baseline_accuracy = baseline_top1 / case_count
    fused_accuracy = fused_top1 / case_count
    return {
        "suite": CANDIDATE_RANKING,
        "k": k,
        "case_count": case_count,
        "baseline_top1_accuracy": baseline_accuracy,
        "fused_top1_accuracy": fused_accuracy,
        "top1_improvement": fused_accuracy - baseline_accuracy,
        "fused_top5_recall": fused_topk / case_count,
        "collision_case_count": collision_total,
        "collision_resolution_rate": (
            collision_resolved / collision_total if collision_total else 0.0
        ),
        "metadata": candidate_ranking_metadata(),
    }


def run_candidate_ranking(
    *,
    cases: Sequence[CandidateRankingCase] | None = None,
    k: int = 5,
) -> dict[str, Any]:
    """Run the candidate ranking suite and return its report."""

    return evaluate_candidate_ranking(cases, k=k)
