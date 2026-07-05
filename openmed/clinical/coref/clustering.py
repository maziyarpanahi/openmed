"""Deterministic clinical mention coreference and entity linking."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from openmed.clinical.context import FAMILY_EXPERIENCER, PATIENT_EXPERIENCER

from .mentions import (
    DEFAULT_DOCUMENT_ID,
    CanonicalMention,
    SpanOffset,
    canonicalize_mentions,
    event_coreference_mentions,
)

COREFERENCE_ADVISORY = (
    "Clinical mention coreference and entity linking are deterministic "
    "assistive grouping aids for review and downstream organization, not a "
    "clinical decision."
)

COMPATIBILITY_SCORER_VERSION = "clinical-coref-compat-v1"

# Versioned feature weights for COMPATIBILITY_SCORER_VERSION. The features are
# intentionally transparent: lexical canonicalization carries most of the
# signal, semantic type and code agreement add conservative support, clinical
# context prevents unsafe merges, and distance is only a weak tie-breaker.
COMPATIBILITY_WEIGHTS: dict[str, float] = {
    "string_similarity": 0.42,
    "semantic_type": 0.18,
    "section_temporality": 0.14,
    "distance": 0.10,
    "code": 0.16,
}

DEFAULT_LINK_THRESHOLD = 0.72


@dataclass(frozen=True)
class PairCompatibility:
    """Feature-based mention-pair compatibility score and constraints."""

    score: float
    features: Mapping[str, float]
    version: str
    must_link: bool = False
    cannot_link: bool = False
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class EntityCluster:
    """A deterministic per-document entity cluster with provenance offsets."""

    entity_id: str
    document_id: str
    representative: str
    canonical_text: str
    semantic_type: str | None
    members: tuple[CanonicalMention, ...]
    member_offsets: tuple[SpanOffset, ...]
    advisory: str = COREFERENCE_ADVISORY


@dataclass(frozen=True)
class ResolvedCoreferenceCluster:
    """Privacy-safe cluster record for public document coreference output."""

    cluster_id: str
    document_id: str
    semantic_type: str | None
    member_offsets: tuple[SpanOffset, ...]
    member_hashes: tuple[str, ...]
    canonical_hash: str
    mention_count: int
    advisory: str = COREFERENCE_ADVISORY

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible cluster without raw mention text."""

        return {
            "cluster_id": self.cluster_id,
            "document_id": self.document_id,
            "semantic_type": self.semantic_type,
            "member_offsets": [list(offset) for offset in self.member_offsets],
            "member_hashes": list(self.member_hashes),
            "canonical_hash": self.canonical_hash,
            "mention_count": self.mention_count,
            "advisory": self.advisory,
        }


@dataclass(frozen=True)
class ResolvedCoreferenceResult:
    """Privacy-safe result returned by ``resolve_coreference``."""

    clusters: tuple[ResolvedCoreferenceCluster, ...]
    advisory: str = COREFERENCE_ADVISORY
    scorer_version: str = COMPATIBILITY_SCORER_VERSION

    def entity_ids_by_offset(self) -> dict[tuple[str, SpanOffset], str]:
        """Return ``(document_id, offset) -> cluster_id`` for all members."""

        return {
            (cluster.document_id, offset): cluster.cluster_id
            for cluster in self.clusters
            for offset in cluster.member_offsets
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible result without raw mention text."""

        return {
            "advisory": self.advisory,
            "scorer_version": self.scorer_version,
            "clusters": [cluster.to_dict() for cluster in self.clusters],
        }


@dataclass(frozen=True)
class CoreferenceResult:
    """Document-level coreference output returned by ``link_mentions``."""

    clusters: tuple[EntityCluster, ...]
    advisory: str = COREFERENCE_ADVISORY
    scorer_version: str = COMPATIBILITY_SCORER_VERSION

    def entity_ids_by_offset(self) -> dict[tuple[str, SpanOffset], str]:
        """Return ``(document_id, offset) -> entity_id`` for all members."""

        return {
            (cluster.document_id, member.offset): cluster.entity_id
            for cluster in self.clusters
            for member in cluster.members
        }


@dataclass(frozen=True)
class ClusteringMetric:
    """Precision/recall/F1 metric result for clustering evaluations."""

    precision: float
    recall: float
    f1: float


def link_mentions(
    mentions: Iterable[Any],
    *,
    document_text: str | None = None,
    abbreviation_expansions: Mapping[str, str] | None = None,
    canonical_aliases: Mapping[str, str] | None = None,
    threshold: float = DEFAULT_LINK_THRESHOLD,
) -> CoreferenceResult:
    """Link clinical mentions into deterministic per-document entity clusters.

    Args:
        mentions: Candidate mention mappings or objects with text and offsets.
        document_text: Optional source text used by canonicalization when
            mentions need to be located or validated.
        abbreviation_expansions: Optional expansion hook for local clinical
            abbreviations.
        canonical_aliases: Optional alias hook for local surface-form variants.
        threshold: Minimum pair compatibility score for an agglomerative link.

    Returns:
        A ``CoreferenceResult`` containing stable entity ids, cluster members,
        member offsets, representative mentions, and the advisory disclaimer.
    """

    canonical_mentions = canonicalize_mentions(
        mentions,
        document_text=document_text,
        abbreviation_expansions=abbreviation_expansions,
        canonical_aliases=canonical_aliases,
    )
    if not canonical_mentions:
        return CoreferenceResult(clusters=())
    if not 0 <= threshold <= 1:
        raise ValueError("coreference threshold must be between 0 and 1")

    parents = list(range(len(canonical_mentions)))
    pair_scores = _pair_scores(canonical_mentions)
    candidates = [
        (left, right, compatibility)
        for (left, right), compatibility in pair_scores.items()
        if compatibility.must_link or compatibility.score >= threshold
    ]
    candidates.sort(
        key=lambda item: (
            not item[2].must_link,
            -item[2].score,
            canonical_mentions[item[0]].stable_key,
            canonical_mentions[item[1]].stable_key,
        )
    )

    for left, right, compatibility in candidates:
        if compatibility.cannot_link:
            continue
        left_root = _find(parents, left)
        right_root = _find(parents, right)
        if left_root == right_root:
            continue
        if _would_violate_cannot_link(
            parents,
            left_root,
            right_root,
            pair_scores,
            len(canonical_mentions),
        ):
            continue
        _union(parents, left_root, right_root, canonical_mentions)

    clusters = _build_clusters(canonical_mentions, parents)
    return CoreferenceResult(clusters=clusters)


def resolve_coreference(
    mentions: Iterable[Any] | None = None,
    *,
    document_text: str | None = None,
    document_id: str = DEFAULT_DOCUMENT_ID,
    include_anaphora: bool = True,
    abbreviation_expansions: Mapping[str, str] | None = None,
    canonical_aliases: Mapping[str, str] | None = None,
    threshold: float = DEFAULT_LINK_THRESHOLD,
) -> ResolvedCoreferenceResult:
    """Resolve event coreference into privacy-safe document clusters.

    When ``document_text`` is supplied, typed PROBLEM/TEST/TREATMENT-like spans
    are augmented with deterministic definite-NP and pronoun candidates before
    clustering. The returned clusters contain only ids, offsets, and hashes.
    """

    source_mentions: Iterable[Any]
    validation_text: str | None
    if document_text is not None:
        source_mentions = event_coreference_mentions(
            document_text,
            tuple(mentions or ()),
            document_id=document_id,
            include_anaphora=include_anaphora,
        )
        validation_text = document_text
    else:
        source_mentions = tuple(mentions or ())
        validation_text = None

    linked = link_mentions(
        source_mentions,
        document_text=validation_text,
        abbreviation_expansions=abbreviation_expansions,
        canonical_aliases=canonical_aliases,
        threshold=threshold,
    )
    return ResolvedCoreferenceResult(
        clusters=tuple(_sanitize_cluster(cluster) for cluster in linked.clusters)
    )


def score_mention_pair(
    left: CanonicalMention,
    right: CanonicalMention,
) -> PairCompatibility:
    """Score whether two canonical mentions refer to the same entity."""

    reasons: list[str] = []
    cannot_link = _cannot_link(left, right, reasons)
    string_similarity = _string_similarity(left.canonical_text, right.canonical_text)
    semantic_type = _semantic_type_score(left, right)
    section_temporality = _section_temporality_score(left, right)
    distance = _distance_score(left, right)
    code = _code_score(left, right)
    features = {
        "string_similarity": string_similarity,
        "semantic_type": semantic_type,
        "section_temporality": section_temporality,
        "distance": distance,
        "code": code,
    }
    raw_score = sum(
        COMPATIBILITY_WEIGHTS[name] * value for name, value in features.items()
    )
    must_link = not cannot_link and (
        _same_code(left, right)
        or (
            left.canonical_text == right.canonical_text
            and _semantic_types_compatible(left, right)
        )
    )
    if must_link:
        reasons.append("canonical identity")
    score = 0.0 if cannot_link else min(1.0, max(raw_score, 0.95 if must_link else 0))
    return PairCompatibility(
        score=round(score, 6),
        features=features,
        version=COMPATIBILITY_SCORER_VERSION,
        must_link=must_link,
        cannot_link=cannot_link,
        reasons=tuple(reasons),
    )


def bcubed_precision_recall_f1(
    predicted: Mapping[Any, str],
    gold: Mapping[Any, str],
) -> ClusteringMetric:
    """Compute B-cubed precision/recall/F1 for clustering labels."""

    if set(predicted) != set(gold):
        raise ValueError("predicted and gold labels must contain the same items")
    if not gold:
        return ClusteringMetric(precision=1.0, recall=1.0, f1=1.0)

    items = tuple(gold)
    precision_sum = 0.0
    recall_sum = 0.0
    for item in items:
        predicted_cluster = {
            other for other in items if predicted[other] == predicted[item]
        }
        gold_cluster = {other for other in items if gold[other] == gold[item]}
        overlap = predicted_cluster & gold_cluster
        precision_sum += len(overlap) / len(predicted_cluster)
        recall_sum += len(overlap) / len(gold_cluster)

    precision = precision_sum / len(items)
    recall = recall_sum / len(items)
    f1 = (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
    return ClusteringMetric(precision=precision, recall=recall, f1=f1)


def _pair_scores(
    mentions: tuple[CanonicalMention, ...],
) -> dict[tuple[int, int], PairCompatibility]:
    return {
        (left, right): score_mention_pair(mentions[left], mentions[right])
        for left in range(len(mentions))
        for right in range(left + 1, len(mentions))
    }


def _cannot_link(
    left: CanonicalMention,
    right: CanonicalMention,
    reasons: list[str],
) -> bool:
    cannot = False
    if left.document_id != right.document_id:
        reasons.append("different documents")
        cannot = True
    if {
        left.experiencer,
        right.experiencer,
    } == {PATIENT_EXPERIENCER, FAMILY_EXPERIENCER}:
        reasons.append("patient/family experiencer boundary")
        cannot = True
    if _conflicting_codes(left, right):
        reasons.append("conflicting coded identity")
        cannot = True
    if left.negation != right.negation:
        reasons.append("opposite assertion polarity")
        cannot = True
    return cannot


def _same_code(left: CanonicalMention, right: CanonicalMention) -> bool:
    return bool(
        left.system
        and right.system
        and left.code
        and right.code
        and left.system == right.system
        and left.code == right.code
    )


def _conflicting_codes(left: CanonicalMention, right: CanonicalMention) -> bool:
    return bool(
        left.system
        and right.system
        and left.code
        and right.code
        and left.system == right.system
        and left.code != right.code
    )


def _semantic_types_compatible(
    left: CanonicalMention,
    right: CanonicalMention,
) -> bool:
    return (
        not left.semantic_type
        or not right.semantic_type
        or (left.semantic_type == right.semantic_type)
    )


def _string_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    token_score = (
        len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
        if left_tokens and right_tokens
        else 0.0
    )
    sequence_score = SequenceMatcher(None, left, right).ratio()
    return max(token_score, sequence_score)


def _semantic_type_score(left: CanonicalMention, right: CanonicalMention) -> float:
    if left.semantic_type and right.semantic_type:
        return 1.0 if left.semantic_type == right.semantic_type else 0.0
    return 0.5


def _section_temporality_score(
    left: CanonicalMention,
    right: CanonicalMention,
) -> float:
    if {
        left.experiencer,
        right.experiencer,
    } == {PATIENT_EXPERIENCER, FAMILY_EXPERIENCER}:
        return 0.0

    score = 0.75
    if left.temporality == right.temporality:
        score += 0.15
    elif "hypothetical" in {left.temporality, right.temporality}:
        score -= 0.2
    else:
        score += 0.05

    if left.canonical_section and left.canonical_section == right.canonical_section:
        score += 0.1
    return max(0.0, min(1.0, score))


def _distance_score(left: CanonicalMention, right: CanonicalMention) -> float:
    if left.document_id != right.document_id:
        return 0.0
    gap = max(0, max(left.start, right.start) - min(left.end, right.end))
    if gap <= 80:
        return 1.0
    if gap <= 250:
        return 0.8
    if gap <= 750:
        return 0.55
    return 0.3


def _code_score(left: CanonicalMention, right: CanonicalMention) -> float:
    if _same_code(left, right):
        return 1.0
    if _conflicting_codes(left, right):
        return 0.0
    if left.code or right.code:
        return 0.4
    return 0.5


def _find(parents: list[int], item: int) -> int:
    while parents[item] != item:
        parents[item] = parents[parents[item]]
        item = parents[item]
    return item


def _union(
    parents: list[int],
    left_root: int,
    right_root: int,
    mentions: tuple[CanonicalMention, ...],
) -> None:
    left_key = _cluster_root_key(parents, left_root, mentions)
    right_key = _cluster_root_key(parents, right_root, mentions)
    keep, replace = (
        (left_root, right_root) if left_key <= right_key else (right_root, left_root)
    )
    parents[replace] = keep


def _cluster_root_key(
    parents: list[int],
    root: int,
    mentions: tuple[CanonicalMention, ...],
) -> tuple[str, int, int, str, str]:
    return min(
        mentions[index].stable_key
        for index in range(len(mentions))
        if _find(parents, index) == root
    )


def _would_violate_cannot_link(
    parents: list[int],
    left_root: int,
    right_root: int,
    pair_scores: Mapping[tuple[int, int], PairCompatibility],
    mention_count: int,
) -> bool:
    left_members = [
        index for index in range(mention_count) if _find(parents, index) == left_root
    ]
    right_members = [
        index for index in range(mention_count) if _find(parents, index) == right_root
    ]
    for left in left_members:
        for right in right_members:
            key = (left, right) if left < right else (right, left)
            if pair_scores[key].cannot_link:
                return True
    return False


def _build_clusters(
    mentions: tuple[CanonicalMention, ...],
    parents: list[int],
) -> tuple[EntityCluster, ...]:
    grouped: dict[int, list[CanonicalMention]] = {}
    for index, mention in enumerate(mentions):
        grouped.setdefault(_find(parents, index), []).append(mention)

    clusters = []
    for members in grouped.values():
        sorted_members = tuple(sorted(members, key=lambda mention: mention.stable_key))
        representative = _representative(sorted_members)
        clusters.append(
            EntityCluster(
                entity_id=_entity_id(sorted_members),
                document_id=sorted_members[0].document_id,
                representative=representative,
                canonical_text=representative.casefold(),
                semantic_type=_cluster_semantic_type(sorted_members),
                members=sorted_members,
                member_offsets=tuple(member.offset for member in sorted_members),
            )
        )
    return tuple(
        sorted(
            clusters,
            key=lambda cluster: (
                cluster.document_id,
                cluster.member_offsets[0],
                cluster.canonical_text,
            ),
        )
    )


def _representative(members: tuple[CanonicalMention, ...]) -> str:
    canonical_counts: dict[str, int] = {}
    for member in members:
        canonical_counts[member.canonical_text] = (
            canonical_counts.get(member.canonical_text, 0) + 1
        )
    return min(
        canonical_counts,
        key=lambda text: (-canonical_counts[text], len(text), text),
    )


def _cluster_semantic_type(members: tuple[CanonicalMention, ...]) -> str | None:
    counts: dict[str, int] = {}
    for member in members:
        if member.semantic_type:
            counts[member.semantic_type] = counts.get(member.semantic_type, 0) + 1
    if not counts:
        return None
    return min(counts, key=lambda value: (-counts[value], value))


def _entity_id(members: tuple[CanonicalMention, ...]) -> str:
    digest = hashlib.sha256()
    document_id = members[0].document_id
    digest.update(document_id.encode("utf-8"))
    for member in members:
        digest.update(
            f"|{member.start}:{member.end}:{member.canonical_text}:"
            f"{member.semantic_type or ''}".encode("utf-8")
        )
    return f"{document_id}:entity:{digest.hexdigest()[:12]}"


def _sanitize_cluster(cluster: EntityCluster) -> ResolvedCoreferenceCluster:
    return ResolvedCoreferenceCluster(
        cluster_id=cluster.entity_id,
        document_id=cluster.document_id,
        semantic_type=cluster.semantic_type,
        member_offsets=cluster.member_offsets,
        member_hashes=tuple(_mention_hash(member) for member in cluster.members),
        canonical_hash=_hash_payload(
            "canonical",
            cluster.document_id,
            cluster.canonical_text,
            cluster.semantic_type or "",
        ),
        mention_count=len(cluster.member_offsets),
    )


def _mention_hash(member: CanonicalMention) -> str:
    return _hash_payload(
        "mention",
        member.document_id,
        str(member.start),
        str(member.end),
        member.canonical_text,
        member.semantic_type or "",
        member.negation,
    )


def _hash_payload(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(b"\0")
        digest.update(part.encode("utf-8"))
    return digest.hexdigest()[:16]


__all__ = [
    "COREFERENCE_ADVISORY",
    "COMPATIBILITY_SCORER_VERSION",
    "COMPATIBILITY_WEIGHTS",
    "DEFAULT_LINK_THRESHOLD",
    "ClusteringMetric",
    "CoreferenceResult",
    "EntityCluster",
    "PairCompatibility",
    "ResolvedCoreferenceCluster",
    "ResolvedCoreferenceResult",
    "bcubed_precision_recall_f1",
    "link_mentions",
    "resolve_coreference",
    "score_mention_pair",
]
