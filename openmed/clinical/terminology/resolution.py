"""Deterministic, offline terminology resolution with reviewable outcomes.

The resolver applies a strict cascade: source coding, preferred lexical match,
alias match, caller-supplied relationship rules, then an optional local
semantic provider.  It never serializes the source surface.  Ambiguous,
unmapped, and rejected outcomes remain visible states and can be persisted in
the review queue implemented by :mod:`openmed.clinical.terminology.store`.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from openmed.clinical.grounding.vocab import (
    VocabConcept,
    VocabularyIndex,
    normalize_alias,
)
from openmed.clinical.journey_contracts import canonical_digest, canonical_json
from openmed.structured.store import StoreResult, StoreState

TERMINOLOGY_RESOLUTION_SCHEMA_VERSION = "1.0.0"
TERMINOLOGY_RESOLUTION_COMPATIBILITY_POLICY = "same_major"

MAPPING_STATES = frozenset({"mapped", "ambiguous", "unmapped", "rejected"})
MAPPING_RULES = frozenset(
    {"explicit_code", "normalized_lexical", "alias", "hierarchy", "semantic"}
)
MAPPING_RELATIONSHIPS = frozenset(
    {"equivalent", "narrower", "broader", "related", "source_code"}
)

_CONTROLLED_RE = re.compile(r"^[a-z][a-z0-9_.:/-]{0,127}$")
_DIGEST_RE = re.compile(r"^(?:sha256|hmac-sha256):[0-9a-f]{64}$")
_SCHEMA_VERSION_RE = re.compile(r"^[1-9][0-9]*\.[0-9]+\.[0-9]+$")
_RULE_PRECEDENCE = {
    "explicit_code": 0,
    "normalized_lexical": 1,
    "alias": 2,
    "hierarchy": 3,
    "semantic": 4,
}


class TerminologyResolutionError(ValueError):
    """Raised when terminology-resolution inputs violate the contract."""


@dataclass(frozen=True, slots=True)
class TerminologyQuery:
    """One source value to resolve without making it serializable.

    Args:
        source_value: Optional lexical surface. It is hidden from ``repr`` and
            never appears in a public result.
        source_system: Optional source coding system.
        source_code: Optional source code. System and code must be supplied
            together.
        language: BCP-47-like language key used by the vocabulary index.
    """

    source_value: str | None = field(default=None, repr=False, compare=False)
    source_system: str | None = None
    source_code: str | None = None
    language: str = "en"

    def __post_init__(self) -> None:
        value = (
            self.source_value.strip() if isinstance(self.source_value, str) else None
        )
        system = (
            self.source_system.strip() if isinstance(self.source_system, str) else None
        )
        code = self.source_code.strip() if isinstance(self.source_code, str) else None
        language = self.language.strip().lower().replace("_", "-")
        if not value and not (system and code):
            raise TerminologyResolutionError(
                "a lexical value or complete source coding is required"
            )
        if bool(system) != bool(code):
            raise TerminologyResolutionError(
                "source_system and source_code must be supplied together"
            )
        if system and _CONTROLLED_RE.fullmatch(system.lower()) is None:
            raise TerminologyResolutionError(
                "source_system must be a controlled identifier"
            )
        if not language or len(language) > 35:
            raise TerminologyResolutionError("language must be a short language tag")
        object.__setattr__(self, "source_value", value)
        object.__setattr__(self, "source_system", system.lower() if system else None)
        object.__setattr__(self, "source_code", code)
        object.__setattr__(self, "language", language)


@dataclass(frozen=True, slots=True)
class TerminologySnapshot:
    """One immutable local vocabulary snapshot."""

    vocabulary: str
    version: str
    index: VocabularyIndex = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        vocabulary = self.vocabulary.strip().lower()
        version = self.version.strip()
        if _CONTROLLED_RE.fullmatch(vocabulary) is None:
            raise TerminologyResolutionError(
                "vocabulary must be a controlled identifier"
            )
        if not version or len(version) > 128:
            raise TerminologyResolutionError("snapshot version must not be blank")
        if not isinstance(self.index, VocabularyIndex):
            raise TypeError("snapshot index must be a VocabularyIndex")
        object.__setattr__(self, "vocabulary", vocabulary)
        object.__setattr__(self, "version", version)

    @property
    def digest(self) -> str:
        """Return an identity digest over version and complete index content."""

        return canonical_digest(
            {
                "content_hash": self.index.content_hash,
                "system": self.index.system,
                "version": self.version,
                "vocabulary": self.vocabulary,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Return source-free snapshot provenance."""

        return {
            "content_hash": self.index.content_hash,
            "digest": self.digest,
            "system": self.index.system,
            "version": self.version,
            "vocabulary": self.vocabulary,
        }


@dataclass(frozen=True, slots=True)
class TerminologyRelationshipRule:
    """Caller-supplied code crosswalk or hierarchy relationship.

    Exactly one source selector is required.  A ``source_term`` is retained
    only in memory and omitted from every serialized representation.
    """

    target_code: str
    relationship: str
    source_system: str | None = None
    source_code: str | None = None
    source_term: str | None = field(default=None, repr=False, compare=False)
    confidence: float = 1.0
    mapping_rule: str = "hierarchy"

    def __post_init__(self) -> None:
        source_term = self.source_term.strip() if self.source_term else None
        source_system = (
            self.source_system.strip().lower() if self.source_system else None
        )
        source_code = self.source_code.strip() if self.source_code else None
        coded = bool(source_system and source_code)
        if bool(source_system) != bool(source_code) or coded == bool(source_term):
            raise TerminologyResolutionError(
                "relationship rule needs exactly one complete source selector"
            )
        if self.mapping_rule not in {"explicit_code", "hierarchy"}:
            raise TerminologyResolutionError(
                "relationship rule must be explicit_code or hierarchy"
            )
        if self.relationship not in MAPPING_RELATIONSHIPS:
            raise TerminologyResolutionError("unsupported mapping relationship")
        if not self.target_code.strip():
            raise TerminologyResolutionError("target_code must not be blank")
        confidence = _confidence(self.confidence, "relationship confidence")
        object.__setattr__(self, "source_term", source_term)
        object.__setattr__(self, "source_system", source_system)
        object.__setattr__(self, "source_code", source_code)
        object.__setattr__(self, "target_code", self.target_code.strip())
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True, slots=True)
class SemanticCandidate:
    """One coded candidate returned by an explicitly local semantic provider."""

    code: str
    confidence: float
    relationship: str = "related"

    def __post_init__(self) -> None:
        if not self.code.strip():
            raise TerminologyResolutionError("semantic candidate code is required")
        if self.relationship not in MAPPING_RELATIONSHIPS:
            raise TerminologyResolutionError("unsupported semantic relationship")
        object.__setattr__(self, "code", self.code.strip())
        object.__setattr__(
            self, "confidence", _confidence(self.confidence, "semantic confidence")
        )


@runtime_checkable
class LocalSemanticCandidateProvider(Protocol):
    """Protocol for opt-in semantic retrieval that cannot imply cloud access."""

    local_only: bool
    provider_id: str
    version: str

    def candidates(
        self,
        source_value: str,
        snapshot: TerminologySnapshot,
        *,
        language: str,
        limit: int,
    ) -> Sequence[SemanticCandidate]:
        """Return local candidates for one source surface."""


@dataclass(frozen=True, slots=True)
class TerminologyResolutionPolicy:
    """Versioned policy controlling optional semantic resolution."""

    semantic_enabled: bool = False
    semantic_min_confidence: float = 0.75
    semantic_ambiguity_margin: float = 0.05
    semantic_limit: int = 8
    policy_id: str = "openmed.terminology.cascade"
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        if _CONTROLLED_RE.fullmatch(self.policy_id) is None:
            raise TerminologyResolutionError("policy_id must be controlled")
        if _SCHEMA_VERSION_RE.fullmatch(self.version) is None:
            raise TerminologyResolutionError("policy version must be semantic")
        _confidence(self.semantic_min_confidence, "semantic_min_confidence")
        _confidence(self.semantic_ambiguity_margin, "semantic_ambiguity_margin")
        if type(self.semantic_limit) is not int or not 1 <= self.semantic_limit <= 100:
            raise TerminologyResolutionError("semantic_limit must be between 1 and 100")

    @property
    def digest(self) -> str:
        """Return a stable digest of public policy controls."""

        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic policy metadata."""

        return {
            "policy_id": self.policy_id,
            "semantic_ambiguity_margin": self.semantic_ambiguity_margin,
            "semantic_enabled": self.semantic_enabled,
            "semantic_limit": self.semantic_limit,
            "semantic_min_confidence": self.semantic_min_confidence,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class TerminologyMappingCandidate:
    """One ranked coded candidate with complete mapping provenance."""

    candidate_id: str
    system: str
    code: str
    rank: int
    confidence: float
    mapping_rule: str
    relationship: str
    vocabulary: str
    vocabulary_version: str
    snapshot_digest: str

    def __post_init__(self) -> None:
        if _DIGEST_RE.fullmatch(self.candidate_id) is None:
            raise TerminologyResolutionError("candidate_id must be a digest")
        if self.mapping_rule not in MAPPING_RULES:
            raise TerminologyResolutionError("unsupported mapping rule")
        if self.relationship not in MAPPING_RELATIONSHIPS:
            raise TerminologyResolutionError("unsupported mapping relationship")
        if type(self.rank) is not int or self.rank < 1:
            raise TerminologyResolutionError("candidate rank must be positive")
        _confidence(self.confidence, "candidate confidence")
        if _DIGEST_RE.fullmatch(self.snapshot_digest) is None:
            raise TerminologyResolutionError("snapshot_digest must be a digest")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible coded candidate."""

        return {
            "candidate_id": self.candidate_id,
            "code": self.code,
            "confidence": self.confidence,
            "mapping_rule": self.mapping_rule,
            "rank": self.rank,
            "relationship": self.relationship,
            "snapshot_digest": self.snapshot_digest,
            "system": self.system,
            "vocabulary": self.vocabulary,
            "vocabulary_version": self.vocabulary_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TerminologyMappingCandidate":
        """Parse a serialized mapping candidate."""

        try:
            return cls(
                candidate_id=str(payload["candidate_id"]),
                system=str(payload["system"]),
                code=str(payload["code"]),
                rank=int(payload["rank"]),
                confidence=float(payload["confidence"]),
                mapping_rule=str(payload["mapping_rule"]),
                relationship=str(payload["relationship"]),
                vocabulary=str(payload["vocabulary"]),
                vocabulary_version=str(payload["vocabulary_version"]),
                snapshot_digest=str(payload["snapshot_digest"]),
            )
        except (KeyError, TypeError, ValueError):
            raise TerminologyResolutionError(
                "mapping candidate is missing or has invalid fields"
            ) from None


@dataclass(frozen=True, slots=True)
class TerminologyMappingResult:
    """Public resolution record with no source surface."""

    mapping_id: str
    source_digest: str
    state: str
    snapshot: TerminologySnapshot = field(repr=False, compare=False)
    candidates: tuple[TerminologyMappingCandidate, ...]
    selected_candidate_id: str | None
    policy_id: str
    policy_version: str
    policy_digest: str
    semantic_enabled: bool
    reason_code: str
    source_system: str | None = None
    source_code: str | None = None
    schema_version: str = TERMINOLOGY_RESOLUTION_SCHEMA_VERSION
    compatibility_policy: str = TERMINOLOGY_RESOLUTION_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        if _DIGEST_RE.fullmatch(self.mapping_id) is None:
            raise TerminologyResolutionError("mapping_id must be a digest")
        if _DIGEST_RE.fullmatch(self.source_digest) is None:
            raise TerminologyResolutionError("source_digest must be a digest")
        if self.state not in MAPPING_STATES:
            raise TerminologyResolutionError("unsupported mapping state")
        if _CONTROLLED_RE.fullmatch(self.reason_code) is None:
            raise TerminologyResolutionError("reason_code must be controlled")
        if self.schema_version != TERMINOLOGY_RESOLUTION_SCHEMA_VERSION:
            raise TerminologyResolutionError("unsupported mapping schema version")
        if self.compatibility_policy != TERMINOLOGY_RESOLUTION_COMPATIBILITY_POLICY:
            raise TerminologyResolutionError("unsupported compatibility policy")
        candidate_ids = {candidate.candidate_id for candidate in self.candidates}
        if self.selected_candidate_id is not None:
            if self.selected_candidate_id not in candidate_ids:
                raise TerminologyResolutionError("selected candidate is not ranked")
            if self.state != "mapped":
                raise TerminologyResolutionError(
                    "only mapped results may select a candidate"
                )
        if self.state == "mapped" and self.selected_candidate_id is None:
            raise TerminologyResolutionError("mapped result requires a candidate")
        if tuple(candidate.rank for candidate in self.candidates) != tuple(
            range(1, len(self.candidates) + 1)
        ):
            raise TerminologyResolutionError("candidate ranks must be contiguous")

    @property
    def review_required(self) -> bool:
        """Return whether this result belongs in the visible review queue."""

        return self.state != "mapped"

    @property
    def selected_candidate(self) -> TerminologyMappingCandidate | None:
        """Return the selected candidate without trusting a positional index."""

        return next(
            (
                candidate
                for candidate in self.candidates
                if candidate.candidate_id == self.selected_candidate_id
            ),
            None,
        )

    def reject(
        self, reason_code: str = "review_rejected"
    ) -> "TerminologyMappingResult":
        """Return an appendable rejected decision without changing its identity."""

        if _CONTROLLED_RE.fullmatch(reason_code) is None:
            raise TerminologyResolutionError("reason_code must be controlled")
        identity = canonical_digest(
            {
                "mapping_id": self.mapping_id,
                "reason_code": reason_code,
                "state": "rejected",
            }
        )
        return replace(
            self,
            mapping_id=identity,
            state="rejected",
            selected_candidate_id=None,
            reason_code=reason_code,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic public data without the source surface."""

        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "compatibility_policy": self.compatibility_policy,
            "mapping_id": self.mapping_id,
            "policy": {
                "digest": self.policy_digest,
                "id": self.policy_id,
                "semantic_enabled": self.semantic_enabled,
                "version": self.policy_version,
            },
            "reason_code": self.reason_code,
            "review_required": self.review_required,
            "schema_version": self.schema_version,
            "selected_candidate_id": self.selected_candidate_id,
            "snapshot": self.snapshot.to_dict(),
            "source_code": self.source_code,
            "source_digest": self.source_digest,
            "source_system": self.source_system,
            "state": self.state,
        }

    def to_json(self) -> str:
        """Return canonical JSON."""

        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        snapshot: TerminologySnapshot,
    ) -> "TerminologyMappingResult":
        """Parse a record while requiring its locally loaded snapshot."""

        try:
            snapshot_payload = payload["snapshot"]
            policy = payload["policy"]
            if not isinstance(snapshot_payload, Mapping) or not isinstance(
                policy, Mapping
            ):
                raise TypeError
            if str(snapshot_payload["digest"]) != snapshot.digest:
                raise TerminologyResolutionError("mapping snapshot digest mismatch")
            return cls(
                mapping_id=str(payload["mapping_id"]),
                source_digest=str(payload["source_digest"]),
                state=str(payload["state"]),
                snapshot=snapshot,
                candidates=tuple(
                    TerminologyMappingCandidate.from_dict(item)
                    for item in payload["candidates"]
                ),
                selected_candidate_id=payload.get("selected_candidate_id"),
                policy_id=str(policy["id"]),
                policy_version=str(policy["version"]),
                policy_digest=str(policy["digest"]),
                semantic_enabled=bool(policy["semantic_enabled"]),
                reason_code=str(payload["reason_code"]),
                source_system=payload.get("source_system"),
                source_code=payload.get("source_code"),
                schema_version=str(payload["schema_version"]),
                compatibility_policy=str(payload["compatibility_policy"]),
            )
        except TerminologyResolutionError:
            raise
        except (KeyError, TypeError, ValueError):
            raise TerminologyResolutionError(
                "invalid terminology mapping result"
            ) from None


@dataclass(frozen=True, slots=True)
class TerminologyCoverageSummary:
    """Aggregate mapping coverage suitable for evaluation and release gates."""

    total: int
    mapped: int
    ambiguous: int
    unmapped: int
    rejected: int
    schema_version: str = TERMINOLOGY_RESOLUTION_SCHEMA_VERSION
    compatibility_policy: str = TERMINOLOGY_RESOLUTION_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        counts = (self.total, self.mapped, self.ambiguous, self.unmapped, self.rejected)
        if any(type(value) is not int or value < 0 for value in counts):
            raise TerminologyResolutionError("coverage counts must be non-negative")
        if sum(counts[1:]) != self.total:
            raise TerminologyResolutionError("coverage state counts must equal total")

    @classmethod
    def from_results(
        cls, results: Iterable[TerminologyMappingResult]
    ) -> "TerminologyCoverageSummary":
        """Aggregate explicit states without retaining any record identifiers."""

        counts = {state: 0 for state in MAPPING_STATES}
        total = 0
        for result in results:
            counts[result.state] += 1
            total += 1
        return cls(
            total=total,
            mapped=counts["mapped"],
            ambiguous=counts["ambiguous"],
            unmapped=counts["unmapped"],
            rejected=counts["rejected"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Return counts and rates only; no source or candidate values."""

        denominator = self.total or 1
        return {
            "ambiguous": self.ambiguous,
            "ambiguous_rate": self.ambiguous / denominator if self.total else 0.0,
            "compatibility_policy": self.compatibility_policy,
            "mapped": self.mapped,
            "mapped_rate": self.mapped / denominator if self.total else 0.0,
            "rejected": self.rejected,
            "rejected_rate": self.rejected / denominator if self.total else 0.0,
            "schema_version": self.schema_version,
            "total": self.total,
            "unmapped": self.unmapped,
            "unmapped_rate": self.unmapped / denominator if self.total else 0.0,
        }


class TerminologyResolver:
    """Resolve source coding and text through a strict local cascade."""

    def __init__(
        self,
        snapshot: TerminologySnapshot,
        *,
        hmac_secret: str | bytes,
        relationships: Iterable[TerminologyRelationshipRule] = (),
        semantic_provider: LocalSemanticCandidateProvider | None = None,
        policy: TerminologyResolutionPolicy | None = None,
    ) -> None:
        self.snapshot = snapshot
        self.policy = policy or TerminologyResolutionPolicy()
        self._secret = _secret_bytes(hmac_secret)
        self._relationships = tuple(relationships)
        self._semantic_provider = semantic_provider
        if semantic_provider is not None:
            if not isinstance(semantic_provider, LocalSemanticCandidateProvider):
                raise TerminologyResolutionError(
                    "semantic provider does not implement the local protocol"
                )
            if semantic_provider.local_only is not True:
                raise TerminologyResolutionError(
                    "semantic provider must explicitly declare local_only=True"
                )
            if _CONTROLLED_RE.fullmatch(semantic_provider.provider_id) is None:
                raise TerminologyResolutionError("semantic provider_id is invalid")
            if _SCHEMA_VERSION_RE.fullmatch(semantic_provider.version) is None:
                raise TerminologyResolutionError("semantic provider version is invalid")
        self._concepts_by_code = _concepts_by_code(snapshot.index.concepts)

    def resolve(self, query: TerminologyQuery) -> StoreResult[TerminologyMappingResult]:
        """Resolve one query and preserve every non-success as a typed outcome."""

        source_digest = _source_digest(query, self._secret)
        stage: list[_CandidateSeed] = []

        if query.source_system and query.source_code:
            stage = self._coded_candidates(query)
        if not stage and query.source_value:
            stage = self._lexical_candidates(query)
        if not stage:
            stage = self._relationship_candidates(query, mapping_rule="hierarchy")
        if not stage and self.policy.semantic_enabled:
            semantic = self._semantic_candidates(query)
            if isinstance(semantic, StoreResult):
                return semantic
            stage = semantic

        candidates = self._rank(stage)
        state, selected_id, reason = self._decision(candidates)
        result = self._result(
            query=query,
            source_digest=source_digest,
            state=state,
            candidates=candidates,
            selected_candidate_id=selected_id,
            reason_code=reason,
        )
        if state == "mapped":
            return StoreResult.success(result)
        if state == "ambiguous":
            return StoreResult.outcome(
                StoreState.CONFLICT, "terminology_ambiguous", value=result
            )
        return StoreResult.outcome(
            StoreState.UNKNOWN, "terminology_unmapped", value=result
        )

    def _coded_candidates(self, query: TerminologyQuery) -> list["_CandidateSeed"]:
        assert query.source_system is not None and query.source_code is not None
        if query.source_system == self.snapshot.index.system:
            return [
                _CandidateSeed(
                    concept=concept,
                    confidence=1.0,
                    mapping_rule="explicit_code",
                    relationship="source_code",
                )
                for concept in self._concepts_by_code.get(query.source_code, ())
            ]
        return self._relationship_candidates(query, mapping_rule="explicit_code")

    def _lexical_candidates(self, query: TerminologyQuery) -> list["_CandidateSeed"]:
        assert query.source_value is not None
        normalized = normalize_alias(query.source_value)
        matches = self.snapshot.index.lookup_all(
            query.source_value, language=query.language
        )
        preferred = [
            concept
            for concept in matches
            if normalize_alias(concept.preferred_term) == normalized
        ]
        concepts = preferred or list(matches)
        mapping_rule = "normalized_lexical" if preferred else "alias"
        confidence = 1.0 if preferred else 0.98
        return [
            _CandidateSeed(
                concept=concept,
                confidence=confidence,
                mapping_rule=mapping_rule,
                relationship="equivalent",
            )
            for concept in concepts
        ]

    def _relationship_candidates(
        self,
        query: TerminologyQuery,
        *,
        mapping_rule: str,
    ) -> list["_CandidateSeed"]:
        matches: list[_CandidateSeed] = []
        normalized = normalize_alias(query.source_value) if query.source_value else None
        for rule in self._relationships:
            if rule.mapping_rule != mapping_rule:
                continue
            coded_match = bool(
                query.source_system
                and query.source_code
                and rule.source_system == query.source_system
                and rule.source_code == query.source_code
            )
            term_match = bool(
                normalized
                and rule.source_term
                and normalize_alias(rule.source_term) == normalized
            )
            if not (coded_match or term_match):
                continue
            for concept in self._concepts_by_code.get(rule.target_code, ()):
                matches.append(
                    _CandidateSeed(
                        concept=concept,
                        confidence=rule.confidence,
                        mapping_rule=rule.mapping_rule,
                        relationship=rule.relationship,
                    )
                )
        return matches

    def _semantic_candidates(
        self, query: TerminologyQuery
    ) -> list["_CandidateSeed"] | StoreResult[TerminologyMappingResult]:
        if self._semantic_provider is None:
            result = self._result(
                query=query,
                source_digest=_source_digest(query, self._secret),
                state="unmapped",
                candidates=(),
                selected_candidate_id=None,
                reason_code="semantic_provider_unavailable",
            )
            return StoreResult.outcome(
                StoreState.UNSUPPORTED,
                "semantic_provider_unavailable",
                value=result,
            )
        if query.source_value is None:
            return []
        try:
            supplied = self._semantic_provider.candidates(
                query.source_value,
                self.snapshot,
                language=query.language,
                limit=self.policy.semantic_limit,
            )
            if not isinstance(supplied, Sequence):
                raise TypeError("semantic provider returned invalid candidates")
            supplied = supplied[: self.policy.semantic_limit]
            if not all(isinstance(item, SemanticCandidate) for item in supplied):
                raise TypeError("semantic provider returned invalid candidates")
        except Exception:
            result = self._result(
                query=query,
                source_digest=_source_digest(query, self._secret),
                state="unmapped",
                candidates=(),
                selected_candidate_id=None,
                reason_code="semantic_provider_failed",
            )
            return StoreResult.outcome(
                StoreState.FAILURE, "semantic_provider_failed", value=result
            )
        seeds: list[_CandidateSeed] = []
        for candidate in supplied[: self.policy.semantic_limit]:
            if candidate.confidence < self.policy.semantic_min_confidence:
                continue
            for concept in self._concepts_by_code.get(candidate.code, ()):
                seeds.append(
                    _CandidateSeed(
                        concept=concept,
                        confidence=candidate.confidence,
                        mapping_rule="semantic",
                        relationship=candidate.relationship,
                    )
                )
        return seeds

    def _rank(
        self, seeds: Iterable["_CandidateSeed"]
    ) -> tuple[TerminologyMappingCandidate, ...]:
        deduplicated: dict[tuple[str, str], _CandidateSeed] = {}
        for seed in seeds:
            key = (seed.concept.system, seed.concept.code)
            previous = deduplicated.get(key)
            if previous is None or _seed_sort_key(seed) < _seed_sort_key(previous):
                deduplicated[key] = seed
        ordered = sorted(deduplicated.values(), key=_seed_sort_key)
        return tuple(
            TerminologyMappingCandidate(
                candidate_id=canonical_digest(
                    {
                        "code": seed.concept.code,
                        "mapping_rule": seed.mapping_rule,
                        "relationship": seed.relationship,
                        "snapshot_digest": self.snapshot.digest,
                        "system": seed.concept.system,
                    }
                ),
                system=seed.concept.system,
                code=seed.concept.code,
                rank=index,
                confidence=seed.confidence,
                mapping_rule=seed.mapping_rule,
                relationship=seed.relationship,
                vocabulary=self.snapshot.vocabulary,
                vocabulary_version=self.snapshot.version,
                snapshot_digest=self.snapshot.digest,
            )
            for index, seed in enumerate(ordered, start=1)
        )

    def _decision(
        self, candidates: tuple[TerminologyMappingCandidate, ...]
    ) -> tuple[str, str | None, str]:
        if not candidates:
            return "unmapped", None, "no_candidate"
        if len(candidates) == 1:
            return "mapped", candidates[0].candidate_id, "unique_candidate"
        first, second = candidates[:2]
        if (
            first.mapping_rule == "semantic"
            and first.confidence - second.confidence
            >= self.policy.semantic_ambiguity_margin
        ):
            return "mapped", first.candidate_id, "semantic_margin"
        return "ambiguous", None, "candidate_tie"

    def _result(
        self,
        *,
        query: TerminologyQuery,
        source_digest: str,
        state: str,
        candidates: tuple[TerminologyMappingCandidate, ...],
        selected_candidate_id: str | None,
        reason_code: str,
    ) -> TerminologyMappingResult:
        identity = canonical_digest(
            {
                "candidates": [candidate.to_dict() for candidate in candidates],
                "policy_digest": self.policy.digest,
                "reason_code": reason_code,
                "selected_candidate_id": selected_candidate_id,
                "snapshot_digest": self.snapshot.digest,
                "source_digest": source_digest,
                "state": state,
            }
        )
        return TerminologyMappingResult(
            mapping_id=identity,
            source_digest=source_digest,
            state=state,
            snapshot=self.snapshot,
            candidates=candidates,
            selected_candidate_id=selected_candidate_id,
            policy_id=self.policy.policy_id,
            policy_version=self.policy.version,
            policy_digest=self.policy.digest,
            semantic_enabled=self.policy.semantic_enabled,
            reason_code=reason_code,
            source_system=query.source_system,
            source_code=query.source_code,
        )


@dataclass(frozen=True, slots=True)
class _CandidateSeed:
    concept: VocabConcept = field(repr=False)
    confidence: float
    mapping_rule: str
    relationship: str


def load_terminology_mapping_schema(
    name: str = "terminology_mapping_result",
) -> dict[str, Any]:
    """Load a bundled terminology mapping or coverage JSON Schema."""

    supported = {
        "terminology_mapping_result",
        "terminology_review_item",
        "terminology_coverage_summary",
    }
    if name not in supported:
        raise KeyError(f"unknown terminology mapping schema: {name!r}")
    path = (
        Path(__file__).resolve().parents[2]
        / "core"
        / "schemas"
        / "json"
        / f"{name}.schema.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _confidence(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TerminologyResolutionError(f"{name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise TerminologyResolutionError(f"{name} must be between 0 and 1")
    return normalized


def _secret_bytes(secret: str | bytes) -> bytes:
    if isinstance(secret, str):
        value = secret.encode("utf-8")
    elif isinstance(secret, bytes):
        value = secret
    else:
        raise TypeError("hmac_secret must be text or bytes")
    if len(value) < 16:
        raise TerminologyResolutionError("hmac_secret must contain at least 16 bytes")
    return value


def _source_digest(query: TerminologyQuery, secret: bytes) -> str:
    value = canonical_json(
        {
            "language": query.language,
            "source_code": query.source_code,
            "source_system": query.source_system,
            "source_value": query.source_value,
        }
    ).encode("utf-8")
    return "hmac-sha256:" + hmac.new(secret, value, hashlib.sha256).hexdigest()


def _concepts_by_code(
    concepts: Iterable[VocabConcept],
) -> dict[str, tuple[VocabConcept, ...]]:
    grouped: dict[str, list[VocabConcept]] = {}
    for concept in concepts:
        grouped.setdefault(concept.code, []).append(concept)
    return {
        code: tuple(sorted(values, key=lambda item: (item.system, item.code)))
        for code, values in grouped.items()
    }


def _seed_sort_key(seed: _CandidateSeed) -> tuple[int, float, str, str, str]:
    return (
        _RULE_PRECEDENCE[seed.mapping_rule],
        -seed.confidence,
        seed.concept.system,
        seed.concept.code,
        seed.relationship,
    )


__all__ = [
    "LocalSemanticCandidateProvider",
    "MAPPING_RELATIONSHIPS",
    "MAPPING_RULES",
    "MAPPING_STATES",
    "SemanticCandidate",
    "TERMINOLOGY_RESOLUTION_COMPATIBILITY_POLICY",
    "TERMINOLOGY_RESOLUTION_SCHEMA_VERSION",
    "TerminologyCoverageSummary",
    "TerminologyMappingCandidate",
    "TerminologyMappingResult",
    "TerminologyQuery",
    "TerminologyRelationshipRule",
    "TerminologyResolutionError",
    "TerminologyResolutionPolicy",
    "TerminologyResolver",
    "TerminologySnapshot",
    "load_terminology_mapping_schema",
]
