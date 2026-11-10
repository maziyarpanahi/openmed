"""Split composite mentions, re-link children, and record safe provenance."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from openmed.clinical.normalization.composite import (
    CompositeMention,
    CompositeNormalization,
    normalize_composite,
)
from openmed.core.audit import hash_text

from .provenance import GROUNDING_ASSIST_ONLY_ADVISORY
from .types import Candidate, GroundedSpan

if TYPE_CHECKING:
    from openmed.clinical.context import ClinicalAssertion

__all__ = [
    "COMPOSITE_GROUNDING_DECISIONS",
    "CompositeChildProvenance",
    "CompositeDecompositionProvenance",
    "CompositeGroundingResult",
    "PostCoordinationRequest",
    "decompose_and_relink",
]

COMPOSITE_GROUNDING_DECISIONS = frozenset(
    {"unchanged", "precoordinated", "multiple", "postcoordination"}
)

CandidateLinker = Callable[[str], Sequence[Candidate]]


@dataclass(frozen=True)
class CompositeChildProvenance:
    """PHI-safe provenance for one proposed or emitted child mention."""

    start: int
    end: int
    byte_start: int
    byte_end: int
    text_hash: str
    linked: bool
    codes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the child record without storing its raw surface."""

        return {
            "start": self.start,
            "end": self.end,
            "byte_start": self.byte_start,
            "byte_end": self.byte_end,
            "text_hash": self.text_hash,
            "linked": self.linked,
            "codes": list(self.codes),
        }


@dataclass(frozen=True)
class CompositeDecompositionProvenance:
    """Trace one parent mention to proposed or emitted child coordinates.

    Only offsets, hashes, closed-vocabulary decisions, and terminology codes are
    retained. Parent and child surfaces are intentionally excluded.
    """

    parent_start: int
    parent_end: int
    parent_byte_start: int
    parent_byte_end: int
    parent_text_hash: str
    strategy: str
    decision: str
    children: tuple[CompositeChildProvenance, ...]
    blocked_reason: str | None = None

    def __post_init__(self) -> None:
        if self.decision not in COMPOSITE_GROUNDING_DECISIONS:
            raise ValueError(f"unknown composite decision {self.decision!r}")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready record with no raw mention text."""

        return {
            "parent_start": self.parent_start,
            "parent_end": self.parent_end,
            "parent_byte_start": self.parent_byte_start,
            "parent_byte_end": self.parent_byte_end,
            "parent_text_hash": self.parent_text_hash,
            "strategy": self.strategy,
            "decision": self.decision,
            "blocked_reason": self.blocked_reason,
            "children": [child.to_dict() for child in self.children],
            "advisory": GROUNDING_ASSIST_ONLY_ADVISORY,
        }


@dataclass(frozen=True)
class PostCoordinationRequest:
    """Deferred input for the user-key-gated post-coordination stage.

    The request preserves the parent and proposed child mentions for that
    stage, along with any children that could already be linked and a PHI-safe
    audit record.
    """

    parent: CompositeMention
    children: tuple[CompositeMention, ...]
    linked_children: tuple[GroundedSpan, ...]
    reason: str
    provenance: CompositeDecompositionProvenance


@dataclass(frozen=True)
class CompositeGroundingResult:
    """Grounding decision for one normalized composite mention."""

    decision: str
    normalization: CompositeNormalization
    spans: tuple[GroundedSpan, ...]
    provenance: CompositeDecompositionProvenance
    postcoordination: PostCoordinationRequest | None = None

    def __post_init__(self) -> None:
        if self.decision not in COMPOSITE_GROUNDING_DECISIONS:
            raise ValueError(f"unknown composite decision {self.decision!r}")
        if not self.spans:
            raise ValueError("composite grounding must retain an output span")


def decompose_and_relink(
    mention: str,
    *,
    linker: CandidateLinker,
    start: int = 0,
    byte_start: int | None = None,
    atomic_terms: Iterable[str] | None = None,
    canonical_label: str | None = None,
    assertion: "ClinicalAssertion | None" = None,
    source_language: str = "en",
    metadata: Mapping[str, Any] | None = None,
) -> CompositeGroundingResult:
    """Normalize one mention and re-link accepted children through ``linker``.

    An exact whole-span candidate wins as a pre-coordinated concept. Otherwise,
    a fully linkable decomposition emits one grounded span per child. If a rule
    proposes a split but any child is uncodable, the parent is emitted as an
    abstention and a :class:`PostCoordinationRequest` preserves the proposal for
    a later caller-owned builder. Nothing is dropped.

    Args:
        mention: Exact source mention surface.
        linker: Existing candidate generation/ranking stack exposed as a local
            callable returning ordered candidates for one surface.
        start: Character-coordinate base in the source document.
        byte_start: UTF-8 byte-coordinate base; defaults to ``start``.
        atomic_terms: Optional additional atomic stop-list entries.
        canonical_label: Optional canonical clinical label inherited by children.
        assertion: Optional assertion context inherited by children.
        source_language: Normalized source language inherited by children.
        metadata: Optional structured caller context inherited by children.

    Returns:
        The decision, output spans, safe provenance, and optional deferred
        post-coordination request.
    """

    linked_by_text: dict[str, tuple[Candidate, ...]] = {}

    def linked(surface: str) -> tuple[Candidate, ...]:
        cached = linked_by_text.get(surface)
        if cached is None:
            candidates = tuple(linker(surface))
            if any(not isinstance(candidate, Candidate) for candidate in candidates):
                raise TypeError("linker must return Candidate objects")
            cached = _one_per_system(candidates)
            linked_by_text[surface] = cached
        return cached

    normalization = normalize_composite(
        mention,
        start=start,
        byte_start=byte_start,
        atomic_terms=atomic_terms,
        is_linkable=lambda surface: bool(linked(surface)),
    )
    parent_candidates = linked(normalization.parent.text)
    proposal = (
        normalization.children
        if normalization.was_split
        else normalization.proposed_children
    )

    if proposal and _has_exact_candidate(parent_candidates):
        decision = "precoordinated"
        output_mentions = (normalization.parent,)
        output_candidates = (parent_candidates,)
    elif normalization.was_split:
        decision = "multiple"
        output_mentions = normalization.children
        output_candidates = tuple(linked(child.text) for child in output_mentions)
    elif normalization.needs_postcoordination:
        decision = "postcoordination"
        output_mentions = (normalization.parent,)
        # A fuzzy whole-span hit must not coerce a partially uncodable composite
        # into an unrelated code. The parent is retained as an explicit abstention.
        output_candidates = ((),)
    else:
        decision = "unchanged"
        output_mentions = (normalization.parent,)
        output_candidates = (parent_candidates,)

    provenance_mentions = proposal or output_mentions
    provenance = _decomposition_provenance(
        normalization,
        decision=decision,
        mentions=provenance_mentions,
        linked=linked,
    )
    provenance_payload = (
        {"composite_decomposition": provenance.to_dict()} if proposal else {}
    )
    inherited_metadata = dict(metadata or {})
    spans = tuple(
        _grounded_span(
            child,
            candidates,
            provenance=provenance_payload,
            canonical_label=canonical_label,
            assertion=assertion,
            source_language=source_language,
            metadata=inherited_metadata,
        )
        for child, candidates in zip(output_mentions, output_candidates)
    )

    postcoordination = None
    if decision == "postcoordination":
        linked_children = tuple(
            _grounded_span(
                child,
                linked(child.text),
                provenance=provenance_payload,
                canonical_label=canonical_label,
                assertion=assertion,
                source_language=source_language,
                metadata=inherited_metadata,
            )
            for child in normalization.proposed_children
        )
        postcoordination = PostCoordinationRequest(
            parent=normalization.parent,
            children=normalization.proposed_children,
            linked_children=linked_children,
            reason=normalization.blocked_reason or "uncodable_composite",
            provenance=provenance,
        )

    return CompositeGroundingResult(
        decision=decision,
        normalization=normalization,
        spans=spans,
        provenance=provenance,
        postcoordination=postcoordination,
    )


def _decomposition_provenance(
    normalization: CompositeNormalization,
    *,
    decision: str,
    mentions: Sequence[CompositeMention],
    linked: Callable[[str], tuple[Candidate, ...]],
) -> CompositeDecompositionProvenance:
    parent = normalization.parent
    children = tuple(
        CompositeChildProvenance(
            start=child.start,
            end=child.end,
            byte_start=child.byte_start,
            byte_end=child.byte_end,
            text_hash=hash_text(child.text),
            linked=bool(linked(child.text)),
            codes=tuple(
                f"{candidate.system}:{candidate.code}"
                for candidate in linked(child.text)
            ),
        )
        for child in mentions
    )
    return CompositeDecompositionProvenance(
        parent_start=parent.start,
        parent_end=parent.end,
        parent_byte_start=parent.byte_start,
        parent_byte_end=parent.byte_end,
        parent_text_hash=hash_text(parent.text),
        strategy=normalization.strategy,
        decision=decision,
        children=children,
        blocked_reason=normalization.blocked_reason,
    )


def _grounded_span(
    mention: CompositeMention,
    candidates: Sequence[Candidate],
    *,
    provenance: Mapping[str, Any],
    canonical_label: str | None,
    assertion: "ClinicalAssertion | None",
    source_language: str,
    metadata: Mapping[str, Any],
) -> GroundedSpan:
    return GroundedSpan(
        text=mention.text,
        start=mention.start,
        end=mention.end,
        candidates=tuple(candidates),
        provenance=provenance,
        canonical_label=canonical_label,
        assertion=assertion,
        source_language=source_language,
        metadata=metadata,
    )


def _one_per_system(candidates: Sequence[Candidate]) -> tuple[Candidate, ...]:
    selected: list[Candidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        system = candidate.system.casefold()
        if system in seen:
            continue
        selected.append(candidate)
        seen.add(system)
    return tuple(selected)


def _has_exact_candidate(candidates: Sequence[Candidate]) -> bool:
    return any(candidate.match_kind == "exact" for candidate in candidates)
