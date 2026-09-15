"""Deterministic, privacy-safe resolution of terminology candidates.

Terminology adapters are allowed to disagree.  This module makes the policy
used to reconcile those disagreements explicit and reproducible without
consulting a remote catalog.  It accepts the grounding :class:`Candidate`
shape, small mapping records, and the local :class:`TerminologyCandidate`
record defined here.

Only terminology identifiers and decision metadata enter the serialized
result.  Candidate displays, aliases, arbitrary metadata, and any query
surface are deliberately excluded from provenance.  The resolver does not
accept a query string and does not log candidate values, so callers can keep
source text outside the conflict-resolution record.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openmed.clinical.grounding.types import Candidate

__all__ = [
    "CONFLICT_RESOLUTION_SCHEMA_VERSION",
    "CandidateProvenance",
    "DISCARD_CATEGORIES",
    "TERMINOLOGY_CONFLICT_ADVISORY",
    "ConflictResolution",
    "ConflictResolutionPolicy",
    "DiscardedCandidate",
    "TerminologyCandidate",
    "TerminologyCandidateProvenance",
    "TerminologyConflictResolver",
    "resolve_conflicts",
    "resolve_terminology_conflicts",
]


CONFLICT_RESOLUTION_SCHEMA_VERSION = 1
TERMINOLOGY_CONFLICT_ADVISORY = (
    "Terminology candidates are machine-suggested records for human review; "
    "resolution is deterministic and does not constitute a clinical decision."
)

# The order is also the order used when serializing empty discarded buckets.
# Keeping every bucket present makes downstream reports schema-stable.
DISCARD_CATEGORIES: tuple[str, ...] = (
    "duplicate",
    "lower_source_priority",
    "older_version",
    "less_exact",
    "lower_score",
    "stable_tiebreak",
)

_SELECTION_RULE = "source_priority>version>exactness>score>stable_identity"
_VERSION_TOKEN_RE = re.compile(r"\d+|[a-z]+", re.IGNORECASE)
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_POST_RELEASE_LABELS = frozenset({"post", "rev", "r"})

_DEFAULT_EXACTNESS_PRIORITY: dict[str, float] = {
    "unknown": 0.0,
    "fuzzy": 1.0,
    "partial": 1.0,
    "synonym": 2.0,
    "alias": 2.0,
    "normalized": 2.0,
    "exact": 3.0,
}


@dataclass(frozen=True)
class TerminologyCandidate:
    """One candidate returned by a terminology source.

    ``system`` and ``code`` identify the coded concept.  ``source`` and
    ``version`` identify the local source snapshot.  ``exactness`` may be a
    label such as ``"exact"`` or ``"fuzzy"``; the legacy ``exact`` and
    ``match_kind`` fields are accepted for adapters that use those names.

    ``display``, ``synonym``, ``matched_alias``, and ``metadata`` are retained
    for the caller's in-memory use but are not included in :meth:`to_dict` or
    any resolver provenance.  They are hidden from the default representation
    as an additional guard against accidental logging.
    """

    # All fields have defaults so callers can use keyword records regardless of
    # whether their source calls the first identity field ``code`` or ``id``.
    system: str = ""
    code: str = ""
    display: str = field(default="", repr=False)
    score: float = 0.0
    source: str = ""
    version: str = ""
    exactness: str | int | float | bool | None = None
    exact: bool | None = None
    match_kind: str | None = field(default=None, repr=False)
    synonym: str | None = field(default=None, repr=False)
    matched_alias: str | None = field(default=None, repr=False)
    vocab_version: str | None = field(default=None, repr=False)
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        system = _clean_text(self.system, "candidate system", optional=True)
        code = _clean_text(self.code, "candidate code")
        display = _clean_text(self.display, "candidate display", optional=True)
        source = _clean_text(self.source, "candidate source", optional=True)
        version = _clean_text(self.version, "candidate version", optional=True)
        vocab_version = _clean_optional_text(self.vocab_version, "candidate version")
        if not version and vocab_version:
            version = vocab_version

        synonym = _clean_optional_text(self.synonym, "candidate synonym")
        matched_alias = _clean_optional_text(
            self.matched_alias, "candidate matched alias"
        )
        if self.exact is not None and not isinstance(self.exact, bool):
            raise TypeError("candidate exact must be a boolean when provided")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("candidate metadata must be a mapping")

        exactness_input: object = self.exactness
        if exactness_input is None and self.match_kind is not None:
            exactness_input = self.match_kind
        if exactness_input is None and self.exact is not None:
            exactness_input = self.exact
        exactness = _canonical_exactness(exactness_input)
        score = _safe_score(self.score, "candidate score")

        object.__setattr__(self, "system", system)
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "display", display)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "vocab_version", vocab_version)
        object.__setattr__(self, "synonym", synonym)
        object.__setattr__(self, "matched_alias", matched_alias)
        object.__setattr__(self, "exactness", exactness)
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "score", score)

    @property
    def concept_key(self) -> tuple[str, str]:
        """Return the stable system/code identity used for deduplication."""

        return (self.system.casefold(), self.code.casefold())

    def to_dict(self) -> dict[str, object]:
        """Return a provenance-safe candidate summary.

        The summary intentionally omits all surface-like fields.  Use the
        in-memory attributes when a caller needs a display or alias locally.
        """

        return {
            "system": self.system,
            "code": self.code,
            "source": self.source,
            "version": self.version,
            "exactness": self.exactness,
            "score": self.score,
        }


@dataclass(frozen=True)
class TerminologyCandidateProvenance:
    """Safe provenance for one selected or discarded candidate."""

    system: str
    code: str
    source: str
    version: str
    exactness: str
    score: float
    source_priority: float
    exactness_priority: float
    candidate_id: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready provenance without candidate surface text."""

        return {
            "system": self.system,
            "code": self.code,
            "source": self.source,
            "version": self.version,
            "exactness": self.exactness,
            "score": self.score,
            "source_priority": self.source_priority,
            "exactness_priority": self.exactness_priority,
            "candidate_id": self.candidate_id,
        }


# A shorter alias is convenient for clients that already use provenance as a
# generic concept, while the long name remains explicit in generated docs.
CandidateProvenance = TerminologyCandidateProvenance


@dataclass(frozen=True)
class DiscardedCandidate:
    """A candidate rejected for one explicit deterministic reason."""

    category: str
    provenance: TerminologyCandidateProvenance
    candidate: TerminologyCandidate = field(compare=False, repr=False)

    def to_dict(self) -> dict[str, object]:
        """Return a safe discarded-candidate report record."""

        return {
            "category": self.category,
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True)
class ConflictResolutionPolicy:
    """The explicit policy used to order candidates."""

    source_priority: tuple[tuple[str, float], ...] = ()
    version_priority: tuple[tuple[str, float], ...] = ()
    exactness_priority: tuple[tuple[str, float], ...] = tuple(
        sorted(_DEFAULT_EXACTNESS_PRIORITY.items())
    )
    version_rule: str = "newest"

    def to_dict(self) -> dict[str, object]:
        """Return the policy in a stable JSON-ready form."""

        return {
            "source_priority": dict(self.source_priority),
            "version_priority": dict(self.version_priority),
            "exactness_priority": dict(self.exactness_priority),
            "version_rule": self.version_rule,
            "selection_rule": _SELECTION_RULE,
        }


@dataclass(frozen=True)
class ConflictResolution:
    """Result of resolving one set of terminology candidates.

    ``discarded`` contains every known category, including empty tuples, so a
    report consumer does not need to infer the schema from a particular run.
    The original candidate is available on each in-memory
    :class:`DiscardedCandidate`; serialized records contain only safe
    provenance.
    """

    selected: TerminologyCandidate | None
    selected_provenance: TerminologyCandidateProvenance | None
    discarded: Mapping[str, tuple[DiscardedCandidate, ...]]
    candidate_count: int
    policy: ConflictResolutionPolicy
    abstained: bool = False

    def __post_init__(self) -> None:
        if self.candidate_count < 0:
            raise ValueError("candidate count must be non-negative")
        if set(self.discarded) - set(DISCARD_CATEGORIES):
            raise ValueError("discarded candidate category is not supported")
        normalized = {
            category: tuple(self.discarded.get(category, ()))
            for category in DISCARD_CATEGORIES
        }
        object.__setattr__(self, "discarded", normalized)
        object.__setattr__(
            self,
            "abstained",
            bool(self.abstained or self.selected is None),
        )

    @property
    def provenance(self) -> TerminologyCandidateProvenance | None:
        """Return the selected candidate's safe provenance."""

        return self.selected_provenance

    @property
    def selected_candidate(self) -> TerminologyCandidate | None:
        """Alias for :attr:`selected` used by pipeline integrations."""

        return self.selected

    @property
    def discarded_candidates(self) -> tuple[DiscardedCandidate, ...]:
        """Return discarded records in stable category/id order."""

        return tuple(
            record
            for category in DISCARD_CATEGORIES
            for record in self.discarded[category]
        )

    @property
    def discarded_categories(self) -> tuple[str, ...]:
        """Return the categories that discarded at least one candidate."""

        return tuple(
            category for category in DISCARD_CATEGORIES if self.discarded[category]
        )

    @property
    def discarded_by_category(self) -> Mapping[str, tuple[DiscardedCandidate, ...]]:
        """Return the category-indexed discarded records."""

        return self.discarded

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic, provenance-safe report payload."""

        selected = (
            self.selected_provenance.to_dict()
            if self.selected_provenance is not None
            else None
        )
        return {
            "schema_version": CONFLICT_RESOLUTION_SCHEMA_VERSION,
            "candidate_count": self.candidate_count,
            "abstained": self.abstained,
            "selected": selected,
            "selected_provenance": selected,
            "discarded_categories": list(self.discarded_categories),
            "discarded": {
                category: [record.to_dict() for record in self.discarded[category]]
                for category in DISCARD_CATEGORIES
            },
            "policy": self.policy.to_dict(),
            "advisory": TERMINOLOGY_CONFLICT_ADVISORY,
        }


@dataclass(frozen=True)
class _ScoredCandidate:
    candidate: TerminologyCandidate
    source_priority: float
    version_rank: tuple[Any, ...]
    exactness_priority: float
    candidate_id: str

    @property
    def provenance(self) -> TerminologyCandidateProvenance:
        """Build the safe provenance view for this scored candidate."""

        return TerminologyCandidateProvenance(
            system=self.candidate.system,
            code=self.candidate.code,
            source=self.candidate.source,
            version=self.candidate.version,
            exactness=self.candidate.exactness,
            score=self.candidate.score,
            source_priority=self.source_priority,
            exactness_priority=self.exactness_priority,
            candidate_id=self.candidate_id,
        )


class TerminologyConflictResolver:
    """Resolve candidates with a local, explicit, deterministic policy.

    Candidates are ordered by the following rules, in order:

    1. higher configured ``source_priority``;
    2. newer ``version`` (or higher configured ``version_priority``);
    3. higher configured ``exactness_priority``;
    4. higher candidate ``score``;
    5. lexicographically smallest stable candidate identity.

    The first four rules are applied only when the earlier rule ties.  Source
    priority is intentionally explicit: an unknown source receives ``0`` and
    cannot silently outrank a configured source.  No rule performs I/O.
    """

    def __init__(
        self,
        source_priority: Mapping[str, int | float] | Sequence[str] | None = None,
        *,
        version_priority: Mapping[str, int | float] | Sequence[str] | None = None,
        exactness_priority: Mapping[str, int | float] | None = None,
    ) -> None:
        self._source_priority = _normalize_priority(source_priority, "source")
        self._version_priority = _normalize_priority(version_priority, "version")
        self._exactness_priority = _normalize_exactness_priority(exactness_priority)
        self._policy = ConflictResolutionPolicy(
            source_priority=tuple(sorted(self._source_priority.items())),
            version_priority=tuple(sorted(self._version_priority.items())),
            exactness_priority=tuple(sorted(self._exactness_priority.items())),
            version_rule="configured" if self._version_priority else "newest",
        )

    @property
    def policy(self) -> ConflictResolutionPolicy:
        """Return the immutable policy used by this resolver."""

        return self._policy

    def resolve(self, candidates: Iterable[object]) -> ConflictResolution:
        """Resolve one candidate set and retain every discarded category.

        The input is consumed once and may contain :class:`Candidate` objects,
        :class:`TerminologyCandidate` objects, or mappings with equivalent
        fields.  Candidate ordering does not affect the result.
        """

        if isinstance(candidates, (str, bytes, bytearray)):
            raise TypeError("candidates must be an iterable of candidate records")
        try:
            iterator = iter(candidates)
        except TypeError:
            raise TypeError(
                "candidates must be an iterable of candidate records"
            ) from None
        normalized = tuple(_coerce_candidate(item) for item in iterator)

        if not normalized:
            return ConflictResolution(
                selected=None,
                selected_provenance=None,
                discarded={category: () for category in DISCARD_CATEGORIES},
                candidate_count=0,
                policy=self._policy,
                abstained=True,
            )

        scored = tuple(self._score(candidate) for candidate in normalized)
        grouped: dict[tuple[str, str], list[_ScoredCandidate]] = defaultdict(list)
        for item in scored:
            grouped[item.candidate.concept_key].append(item)

        representatives: list[_ScoredCandidate] = []
        discarded: dict[str, list[DiscardedCandidate]] = {
            category: [] for category in DISCARD_CATEGORIES
        }
        for key in sorted(grouped):
            group = grouped[key]
            representative = _choose_best(group)
            representatives.append(representative)
            for item in group:
                if item is representative:
                    continue
                discarded["duplicate"].append(
                    DiscardedCandidate(
                        category="duplicate",
                        provenance=item.provenance,
                        candidate=item.candidate,
                    )
                )

        selected = _choose_best(representatives)
        for item in representatives:
            if item is selected:
                continue
            category = _discard_category(selected, item)
            discarded[category].append(
                DiscardedCandidate(
                    category=category,
                    provenance=item.provenance,
                    candidate=item.candidate,
                )
            )

        frozen_discarded = {
            category: tuple(
                sorted(
                    records,
                    key=lambda record: record.provenance.candidate_id,
                )
            )
            for category, records in discarded.items()
        }
        return ConflictResolution(
            selected=selected.candidate,
            selected_provenance=selected.provenance,
            discarded=frozen_discarded,
            candidate_count=len(normalized),
            policy=self._policy,
        )

    def resolve_candidates(self, candidates: Iterable[object]) -> ConflictResolution:
        """Alias for :meth:`resolve` for pipeline adapters."""

        return self.resolve(candidates)

    def _score(self, candidate: TerminologyCandidate) -> _ScoredCandidate:
        source = candidate.source.casefold()
        source_priority = self._source_priority.get(
            source,
            _unknown_priority(self._source_priority),
        )
        version_rank = _version_rank(candidate.version, self._version_priority)
        exactness_priority = self._exactness_priority.get(
            candidate.exactness,
            _unknown_priority(self._exactness_priority),
        )
        return _ScoredCandidate(
            candidate=candidate,
            source_priority=source_priority,
            version_rank=version_rank,
            exactness_priority=exactness_priority,
            candidate_id=_candidate_id(candidate),
        )


def resolve_terminology_conflicts(
    candidates: Iterable[object],
    source_priority: Mapping[str, int | float] | Sequence[str] | None = None,
    *,
    version_priority: Mapping[str, int | float] | Sequence[str] | None = None,
    exactness_priority: Mapping[str, int | float] | None = None,
) -> ConflictResolution:
    """Resolve terminology candidates with a one-shot local resolver."""

    return TerminologyConflictResolver(
        source_priority,
        version_priority=version_priority,
        exactness_priority=exactness_priority,
    ).resolve(candidates)


resolve_conflicts = resolve_terminology_conflicts


def _coerce_candidate(raw: object) -> TerminologyCandidate:
    if isinstance(raw, TerminologyCandidate):
        return raw
    if isinstance(raw, Mapping):
        nested = raw.get("concept")
        nested_mapping = nested if isinstance(nested, Mapping) else {}
        provenance = raw.get("provenance")
        provenance_mapping = provenance if isinstance(provenance, Mapping) else {}

        def value(*keys: str, default: object = None) -> object:
            for key in keys:
                if key in raw and raw[key] is not None:
                    return raw[key]
                if key in nested_mapping and nested_mapping[key] is not None:
                    return nested_mapping[key]
                if key in provenance_mapping and provenance_mapping[key] is not None:
                    return provenance_mapping[key]
            return default

        return TerminologyCandidate(
            system=value("system", "system_uri", "vocabulary", default=""),
            code=value("code", "concept_id", "id", default=""),
            display=value("display", "label", "term", "preferred_term", default=""),
            score=value("score", "confidence", default=0.0),
            source=value("source", "source_name", "provider", "origin", default=""),
            version=value(
                "version",
                "vocab_version",
                "release_version",
                "terminology_version",
                default="",
            ),
            exactness=value("exactness", "match_kind", "match", default=None),
            exact=value("exact", "is_exact", default=None),
            synonym=value("synonym", "alias", default=None),
            matched_alias=value("matched_alias", "matched_term", default=None),
            metadata=value("metadata", default={}) or {},
        )

    concept = getattr(raw, "concept", None)
    provenance = getattr(raw, "provenance", None)

    def attr(*names: str, default: object = None) -> object:
        for name in names:
            value = getattr(raw, name, None)
            if value is not None:
                return value
            if concept is not None:
                value = getattr(concept, name, None)
                if value is not None:
                    return value
            if provenance is not None:
                value = getattr(provenance, name, None)
                if value is not None:
                    return value
        return default

    code = attr("code", "concept_id", "id", default=None)
    if code is None:
        raise TypeError("candidate records must provide a code")
    return TerminologyCandidate(
        system=attr("system", "system_uri", "vocabulary", default=""),
        code=code,
        display=attr("display", "label", "term", default=""),
        score=attr("score", "confidence", default=0.0),
        source=attr("source", "source_name", "provider", "origin", default=""),
        version=attr(
            "version",
            "vocab_version",
            "release_version",
            "backend_version",
            "terminology_version",
            default="",
        ),
        exactness=attr("exactness", "match_kind", "match", default=None),
        exact=attr("exact", "is_exact", default=None),
        synonym=attr("synonym", "alias", default=None),
        matched_alias=attr("matched_alias", "matched_term", default=None),
        metadata=attr("metadata", default={}) or {},
    )


def _choose_best(items: Sequence[_ScoredCandidate]) -> _ScoredCandidate:
    if not items:
        raise ValueError("cannot choose from an empty candidate group")

    pool = list(items)
    best_source = max(item.source_priority for item in pool)
    pool = [item for item in pool if item.source_priority == best_source]
    best_version = max(item.version_rank for item in pool)
    pool = [item for item in pool if item.version_rank == best_version]
    best_exactness = max(item.exactness_priority for item in pool)
    pool = [item for item in pool if item.exactness_priority == best_exactness]
    best_score = max(item.candidate.score for item in pool)
    pool = [item for item in pool if item.candidate.score == best_score]
    return min(pool, key=lambda item: item.candidate_id)


def _discard_category(
    selected: _ScoredCandidate,
    discarded: _ScoredCandidate,
) -> str:
    if selected.source_priority != discarded.source_priority:
        return "lower_source_priority"
    if selected.version_rank != discarded.version_rank:
        return "older_version"
    if selected.exactness_priority != discarded.exactness_priority:
        return "less_exact"
    if selected.candidate.score != discarded.candidate.score:
        return "lower_score"
    return "stable_tiebreak"


def _candidate_id(candidate: TerminologyCandidate) -> str:
    payload = "\x1f".join(
        (
            candidate.system.casefold(),
            candidate.code.casefold(),
            candidate.source.casefold(),
            candidate.version.casefold(),
            candidate.exactness,
            repr(candidate.score),
        )
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _version_rank(
    version: str,
    configured: Mapping[str, float],
) -> tuple[Any, ...]:
    normalized = version.casefold()
    if configured:
        if normalized in configured:
            return (2, configured[normalized])
        return (1, _natural_version_key(version))
    return _natural_version_key(version)


def _natural_version_key(version: str) -> tuple[Any, ...]:
    normalized = version.casefold().strip()
    if not normalized:
        return (0, (), 0, ())
    if normalized.startswith("v") and len(normalized) > 1 and normalized[1].isdigit():
        normalized = normalized[1:]
    tokens = _VERSION_TOKEN_RE.findall(normalized)
    if not tokens:
        return (1, (), 1, ((1, normalized),))

    core: list[tuple[int, int | str]] = []
    suffix: list[tuple[int, int | str]] = []
    in_suffix = False
    for token in tokens:
        if not token.isdigit():
            in_suffix = True
        target = suffix if in_suffix else core
        target.append((0, int(token)) if token.isdigit() else (1, token))

    if not suffix:
        release_stage = 2
    elif str(suffix[0][1]) in _POST_RELEASE_LABELS:
        release_stage = 3
    else:
        release_stage = 1
    return (
        1,
        tuple(core),
        release_stage,
        tuple(suffix),
    )


def _canonical_exactness(value: object) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "exact" if value else "fuzzy"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ValueError("candidate exactness must be finite")
        return f"custom:{float(value):g}"
    if not isinstance(value, str):
        raise TypeError("candidate exactness must be text, numeric, or boolean")
    normalized = value.strip().casefold().replace(" ", "_").replace("-", "_")
    if not normalized:
        return "unknown"
    aliases = {
        "exact_match": "exact",
        "preferred": "exact",
        "preferred_term": "exact",
        "alias_match": "alias",
        "normalized_alias": "normalized",
        "partial_match": "partial",
        "fuzzy_match": "fuzzy",
        "not_exact": "fuzzy",
    }
    return aliases.get(normalized, normalized)


def _normalize_exactness_priority(
    values: Mapping[str, int | float] | None,
) -> dict[str, float]:
    normalized = dict(_DEFAULT_EXACTNESS_PRIORITY)
    if values is not None:
        normalized.update(_normalize_priority(values, "exactness"))
    return normalized


def _normalize_priority(
    values: Mapping[str, int | float] | Sequence[str] | None,
    label: str,
) -> dict[str, float]:
    if values is None:
        return {}
    if isinstance(values, Mapping):
        result: dict[str, float] = {}
        for key, rank in values.items():
            normalized_key = _clean_text(key, f"{label} priority key").casefold()
            result[normalized_key] = _safe_priority(rank, label)
        return result
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{label} priority must be a mapping or sequence")
    items = tuple(values)
    result = {}
    for index, key in enumerate(items):
        normalized_key = _clean_text(key, f"{label} priority key").casefold()
        result[normalized_key] = float(len(items) - index)
    return result


def _unknown_priority(values: Mapping[str, float]) -> float:
    return min(values.values(), default=0.0) - 1.0


def _safe_priority(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} priority values must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{label} priority values must be finite")
    return converted


def _safe_score(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{label} must be finite")
    return converted


def _clean_text(value: object, label: str, *, optional: bool = False) -> str:
    if value is None and optional:
        return ""
    if not isinstance(value, str):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = str(value)
        else:
            raise TypeError(f"{label} must be text")
    if _CONTROL_RE.search(value):
        raise ValueError(f"{label} must not contain control characters")
    cleaned = value.strip()
    if not cleaned and not optional:
        raise ValueError(f"{label} must be non-empty")
    return cleaned


def _clean_optional_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _clean_text(value, label, optional=True) or None
