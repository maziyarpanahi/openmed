"""Rules-first decomposition of composite clinical concept mentions.

The normalizer deliberately uses a small, auditable rule set rather than a
syntactic parser.  It recognizes coordination, complication phrases, and a
small modifier/head pattern while retaining both character and UTF-8 byte
offsets into the original mention.  Callers may supply a linkability predicate;
when any proposed child is not linkable the split is withheld and exposed as a
post-coordination proposal instead of silently dropping part of the mention.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass

__all__ = [
    "COMPOSITE_SPLIT_STRATEGIES",
    "KNOWN_ATOMIC_COMPOSITES",
    "CompositeMention",
    "CompositeNormalization",
    "normalize_composite",
]

COMPOSITE_SPLIT_STRATEGIES = frozenset(
    {"atomic", "unsplit", "coordination", "complication", "modifier_head"}
)

# These are lexical atoms even though their surfaces contain a separator that
# would otherwise look compositional to the rules below.  The set is intentionally
# small and caller-extensible; it protects common false-positive shapes without
# pretending to be a terminology bundle.
KNOWN_ATOMIC_COMPOSITES: frozenset[str] = frozenset(
    {
        "attention deficit and hyperactivity disorder",
        "bed and breakfast sign",
        "ear, nose and throat disorder",
        "hand, foot and mouth disease",
        "head and neck cancer",
        "heart and lung transplant status",
        "migraine with aura",
        "mother and baby unit",
        "pain with psychological factors",
        "salt and pepper retinopathy",
        "signs and symptoms",
        "sickle cell disease with crisis",
    }
)

_COMPLICATION_RE = re.compile(
    r"\s+(?:with|complicated\s+by|associated\s+with)\s+",
    flags=re.IGNORECASE,
)
_COORDINATION_RE = re.compile(
    r"\s*(?:,|;|&|\band\b|\bor\b)\s*",
    flags=re.IGNORECASE,
)
_MODIFIER_HEAD_RE = re.compile(
    r"\s+(?:acute\s+exacerbation|exacerbation|flare|complication)$",
    flags=re.IGNORECASE,
)

LinkabilityCheck = Callable[[str], bool]


@dataclass(frozen=True)
class CompositeMention:
    """One contiguous sub-mention with source-relative coordinates.

    Args:
        text: Exact contiguous text copied from the parent mention.
        start: Inclusive character offset in the caller's source coordinates.
        end: Exclusive character offset in the caller's source coordinates.
        byte_start: Inclusive UTF-8 byte offset in the caller's byte coordinates.
        byte_end: Exclusive UTF-8 byte offset in the caller's byte coordinates.
    """

    text: str
    start: int
    end: int
    byte_start: int
    byte_end: int

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("composite mention text must be non-empty")
        for name, value in (
            ("start", self.start),
            ("end", self.end),
            ("byte_start", self.byte_start),
            ("byte_end", self.byte_end),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.end < self.start or self.byte_end < self.byte_start:
            raise ValueError("composite mention offsets must be ordered")
        if self.end - self.start != len(self.text):
            raise ValueError("character offsets must match composite mention text")
        if self.byte_end - self.byte_start != len(self.text.encode("utf-8")):
            raise ValueError("byte offsets must match UTF-8 composite mention text")


@dataclass(frozen=True)
class CompositeNormalization:
    """Result of applying the rules-first composite normalizer.

    ``children`` is the accepted decomposition.  It contains only ``parent``
    when no rule applies, an atomic stop-list entry matches, or a linkability
    check blocks a proposed split.  ``proposed_children`` retains the latter
    proposal so a downstream stage can route it to post-coordination.
    """

    parent: CompositeMention
    children: tuple[CompositeMention, ...]
    strategy: str
    proposed_children: tuple[CompositeMention, ...] = ()
    blocked_reason: str | None = None

    def __post_init__(self) -> None:
        if self.strategy not in COMPOSITE_SPLIT_STRATEGIES:
            raise ValueError(f"unknown composite split strategy {self.strategy!r}")
        if not self.children:
            raise ValueError("composite normalization must retain a child span")

    @property
    def was_split(self) -> bool:
        """Return whether the accepted output contains multiple children."""

        return len(self.children) > 1

    @property
    def needs_postcoordination(self) -> bool:
        """Return whether an unlinked proposal needs post-coordination."""

        return bool(self.proposed_children and self.blocked_reason)


def normalize_composite(
    mention: str,
    *,
    start: int = 0,
    byte_start: int | None = None,
    atomic_terms: Iterable[str] | None = None,
    is_linkable: LinkabilityCheck | None = None,
) -> CompositeNormalization:
    """Split one clinical mention into contiguous, offset-bearing children.

    Rules run in conservative order: known lexical atoms, complication phrases,
    modifier/head suffixes, then coordination.  A supplied ``is_linkable``
    predicate is evaluated for every proposed child before the split is
    accepted.  If any child cannot be linked, the function returns the intact
    parent in ``children`` and keeps the proposal in ``proposed_children`` for a
    post-coordination builder.

    Args:
        mention: Exact source surface to normalize.
        start: Character-coordinate base of ``mention`` in its source document.
        byte_start: UTF-8 byte-coordinate base. Defaults to ``start`` for
            backward-compatible ASCII/source-relative use.
        atomic_terms: Additional known atomic multi-word surfaces.
        is_linkable: Optional offline predicate used to validate every child.

    Returns:
        A deterministic normalization result with exact character and byte
        offsets. No network or terminology lookup occurs unless the caller's
        predicate performs one.

    Raises:
        TypeError: If ``mention`` is not a string.
        ValueError: If the mention is empty or an offset base is invalid.
    """

    if not isinstance(mention, str):
        raise TypeError("mention must be a string")
    if not mention.strip():
        raise ValueError("mention must contain non-whitespace text")
    if type(start) is not int or start < 0:
        raise ValueError("start must be a non-negative integer")
    resolved_byte_start = start if byte_start is None else byte_start
    if type(resolved_byte_start) is not int or resolved_byte_start < 0:
        raise ValueError("byte_start must be a non-negative integer")

    local_start, local_end = _trim_range(mention, 0, len(mention))
    parent = _span(
        mention,
        0,
        len(mention),
        char_base=start,
        byte_base=resolved_byte_start,
    )
    known_atomic = set(KNOWN_ATOMIC_COMPOSITES)
    if atomic_terms is not None:
        known_atomic.update(_normalize_atomic(term) for term in atomic_terms)
    if _normalize_atomic(parent.text) in known_atomic:
        return CompositeNormalization(
            parent=parent,
            children=(parent,),
            strategy="atomic",
        )

    proposal, strategy = _propose_split(
        mention,
        local_start,
        local_end,
        char_base=start,
        byte_base=resolved_byte_start,
    )
    if len(proposal) < 2:
        return CompositeNormalization(
            parent=parent,
            children=(parent,),
            strategy="unsplit",
        )

    if is_linkable is not None and not all(
        bool(is_linkable(child.text)) for child in proposal
    ):
        return CompositeNormalization(
            parent=parent,
            children=(parent,),
            strategy=strategy,
            proposed_children=proposal,
            blocked_reason="unlinked_child",
        )
    return CompositeNormalization(
        parent=parent,
        children=proposal,
        strategy=strategy,
    )


def _propose_split(
    mention: str,
    local_start: int,
    local_end: int,
    *,
    char_base: int,
    byte_base: int,
) -> tuple[tuple[CompositeMention, ...], str]:
    complication = _split_matches(
        mention,
        local_start,
        local_end,
        _COMPLICATION_RE,
        char_base=char_base,
        byte_base=byte_base,
    )
    if len(complication) >= 2:
        return complication, "complication"

    modifier = _MODIFIER_HEAD_RE.search(mention, local_start, local_end)
    if modifier is not None and modifier.start() > local_start:
        spans = _spans_from_ranges(
            mention,
            ((local_start, modifier.start()), (modifier.start(), local_end)),
            char_base=char_base,
            byte_base=byte_base,
        )
        if len(spans) == 2:
            return spans, "modifier_head"

    coordination = _split_matches(
        mention,
        local_start,
        local_end,
        _COORDINATION_RE,
        char_base=char_base,
        byte_base=byte_base,
    )
    if len(coordination) >= 2:
        return coordination, "coordination"
    return (), "unsplit"


def _split_matches(
    mention: str,
    local_start: int,
    local_end: int,
    pattern: re.Pattern[str],
    *,
    char_base: int,
    byte_base: int,
) -> tuple[CompositeMention, ...]:
    ranges: list[tuple[int, int]] = []
    cursor = local_start
    for match in pattern.finditer(mention, local_start, local_end):
        if match.start() == match.end():
            continue
        ranges.append((cursor, match.start()))
        cursor = match.end()
    if not ranges:
        return ()
    ranges.append((cursor, local_end))
    return _spans_from_ranges(
        mention,
        ranges,
        char_base=char_base,
        byte_base=byte_base,
    )


def _spans_from_ranges(
    mention: str,
    ranges: Iterable[tuple[int, int]],
    *,
    char_base: int,
    byte_base: int,
) -> tuple[CompositeMention, ...]:
    spans: list[CompositeMention] = []
    for range_start, range_end in ranges:
        trimmed_start, trimmed_end = _trim_range(mention, range_start, range_end)
        if trimmed_start >= trimmed_end:
            return ()
        spans.append(
            _span(
                mention,
                trimmed_start,
                trimmed_end,
                char_base=char_base,
                byte_base=byte_base,
            )
        )
    return tuple(spans)


def _span(
    mention: str,
    local_start: int,
    local_end: int,
    *,
    char_base: int,
    byte_base: int,
) -> CompositeMention:
    return CompositeMention(
        text=mention[local_start:local_end],
        start=char_base + local_start,
        end=char_base + local_end,
        byte_start=byte_base + len(mention[:local_start].encode("utf-8")),
        byte_end=byte_base + len(mention[:local_end].encode("utf-8")),
    )


def _trim_range(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _normalize_atomic(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"\s+", " ", text).strip()
