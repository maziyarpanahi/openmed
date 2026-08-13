"""Pure-offline multilingual grounding to international clinical codes.

The router searches only the alias spaces matching the source locale.  A local
MLX SapBERT-class encoder can add cross-lingual dense candidates; when it is
absent, deterministic exact and normalized-string matching remain available.
No translation service, model download, or other network path exists here.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from .crosswalk import CrosswalkEntry, CrosswalkResource, load_default_crosswalks
from .embeddings import AliasEncoder, load_encoder
from .matcher import LexicalMatcher
from .restricted import UserKeyVocabularyLoader
from .types import Candidate
from .vocab import RestrictedVocabularyError, normalize_alias, normalize_language

__all__ = [
    "MultilingualGrounder",
    "MultilingualGroundingResult",
    "ground_multilingual",
]

_DEFAULT_TOP_K = 5
_DEFAULT_STRING_SCORE = 0.72
_DEFAULT_DENSE_SCORE = 0.50


@dataclass(frozen=True, repr=False)
class MultilingualGroundingResult:
    """One source-language mention and its ranked international concepts.

    The first candidate is the selected international concept.  Up to ``top_k``
    candidates remain available for Acc@k evaluation and human review.
    ``provenance`` contains resource and encoder versions but never duplicates
    the raw source surface.
    """

    surface: str
    locale: str
    source_language: str
    candidates: tuple[Candidate, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.surface, str) or not self.surface.strip():
            raise ValueError("multilingual grounding surface must not be blank")
        if not isinstance(self.locale, str) or not self.locale:
            raise ValueError("multilingual grounding locale must not be blank")
        candidates = tuple(self.candidates)
        if any(not isinstance(candidate, Candidate) for candidate in candidates):
            raise TypeError("multilingual candidates must be Candidate objects")
        if not isinstance(self.provenance, Mapping):
            raise TypeError("multilingual provenance must be a mapping")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "provenance", dict(self.provenance))

    def __repr__(self) -> str:
        """Return a source-text-free representation safe for diagnostics."""

        return (
            "MultilingualGroundingResult("
            f"locale={self.locale!r}, source_language={self.source_language!r}, "
            f"candidate_count={len(self.candidates)})"
        )

    @property
    def code(self) -> str | None:
        """Return the selected international code, if one was found."""

        return self.candidates[0].code if self.candidates else None

    @property
    def international_code(self) -> str | None:
        """Alias for :attr:`code` emphasizing the crosswalk destination."""

        return self.code

    @property
    def system(self) -> str | None:
        """Return the selected international vocabulary system."""

        return self.candidates[0].system if self.candidates else None

    @property
    def display(self) -> str | None:
        """Return the selected international preferred term."""

        return self.candidates[0].display if self.candidates else None

    @property
    def score(self) -> float:
        """Return the selected cross-lingual match score, or zero."""

        return self.candidates[0].score if self.candidates else 0.0

    @property
    def cui(self) -> str | None:
        """Return a UMLS CUI when the gated local path produced one."""

        for candidate in self.candidates:
            if candidate.system == "UMLS":
                return candidate.code
        return None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result including the requested source surface."""

        return {
            "surface": self.surface,
            "locale": self.locale,
            "source_language": self.source_language,
            "international_code": self.international_code,
            "system": self.system,
            "display": self.display,
            "cross_lingual_match_score": self.score,
            "cui": self.cui,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class _CandidateOffer:
    candidate: Candidate
    resource: CrosswalkResource | None


class MultilingualGrounder:
    """Locale-aware router over local crosswalk, encoder, and gated UMLS data.

    Args:
        resources: Free local crosswalks. ``None`` loads the bundled CC0 starter
            tables; an empty sequence disables free mappings.
        encoder: Optional already-loaded cross-lingual alias encoder.
        encoder_path: Optional local MLX SapBERT-class weights path. It is
            ignored when ``encoder`` is supplied and never downloaded.
        restricted_loaders: Optional caller-owned local UMLS loader. Constructing
            that loader requires an explicit user key; its data is never bundled.
        string_score_cutoff: Minimum normalized-string fallback score.
        dense_score_cutoff: Minimum encoder cosine score.
    """

    def __init__(
        self,
        resources: Sequence[CrosswalkResource] | None = None,
        *,
        encoder: AliasEncoder | None = None,
        encoder_path: str | None = None,
        restricted_loaders: Mapping[str, UserKeyVocabularyLoader] | None = None,
        string_score_cutoff: float = _DEFAULT_STRING_SCORE,
        dense_score_cutoff: float = _DEFAULT_DENSE_SCORE,
    ) -> None:
        self._resources = (
            load_default_crosswalks() if resources is None else tuple(resources)
        )
        if any(
            not isinstance(resource, CrosswalkResource) for resource in self._resources
        ):
            raise TypeError("resources must contain CrosswalkResource objects")
        self._encoder = encoder if encoder is not None else load_encoder(encoder_path)
        self._string_score_cutoff = _score_cutoff(
            string_score_cutoff, "string_score_cutoff"
        )
        self._dense_score_cutoff = _score_cutoff(
            dense_score_cutoff, "dense_score_cutoff"
        )
        self._umls_loader = _resolve_umls_loader(restricted_loaders)
        self._umls_matcher = (
            LexicalMatcher(
                self._umls_loader.load(),
                system_uri=self._umls_loader.system_uri,
            )
            if self._umls_loader is not None
            else None
        )

    @property
    def resources(self) -> tuple[CrosswalkResource, ...]:
        """Return the configured local crosswalk resources."""

        return self._resources

    @property
    def encoder_enabled(self) -> bool:
        """Return whether local cross-lingual dense matching is active."""

        return self._encoder is not None

    def ground(
        self,
        mention: str,
        locale: str,
        *,
        top_k: int = _DEFAULT_TOP_K,
    ) -> MultilingualGroundingResult:
        """Ground one source-locale mention without translation or network I/O."""

        if not isinstance(mention, str) or not mention.strip():
            raise ValueError("mention must be non-empty text")
        resolved_locale = _normalize_locale(locale)
        source_language = normalize_language(resolved_locale)
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        routed = tuple(
            (resource, entry)
            for resource in self._resources
            for entry in resource.entries_for_locale(resolved_locale)
        )
        best: dict[tuple[str, str], _CandidateOffer] = {}
        self._offer_string_candidates(
            best,
            mention,
            source_language=source_language,
            routed=routed,
        )
        if self._encoder is not None and routed:
            self._offer_dense_candidates(
                best,
                mention,
                source_language=source_language,
                routed=routed,
            )
        self._offer_umls_candidates(best, mention, source_language=source_language)

        offers = sorted(
            best.values(),
            key=lambda offer: (
                -offer.candidate.score,
                offer.candidate.system,
                offer.candidate.code,
            ),
        )[:top_k]
        primary = offers[0] if offers else None
        provenance = {
            "source_language": source_language,
            "source_locale": resolved_locale,
            "mapping_resource_version": (
                primary.resource.resource_version
                if primary is not None and primary.resource is not None
                else None
            ),
            "mapping_resources": [
                {
                    "name": resource.name,
                    "version": resource.version,
                    "content_hash": resource.content_hash,
                    "license": resource.license_id,
                }
                for resource in self._resources
                if resource.entries_for_locale(resolved_locale)
            ],
            "encoder_id": (
                self._encoder.encoder_id if self._encoder is not None else None
            ),
            "match_method": (
                primary.candidate.match_kind if primary is not None else None
            ),
            "offline": True,
        }
        return MultilingualGroundingResult(
            surface=mention,
            locale=resolved_locale,
            source_language=source_language,
            candidates=tuple(offer.candidate for offer in offers),
            provenance=provenance,
        )

    def _offer_string_candidates(
        self,
        best: dict[tuple[str, str], _CandidateOffer],
        mention: str,
        *,
        source_language: str,
        routed: Sequence[tuple[CrosswalkResource, CrosswalkEntry]],
    ) -> None:
        query = normalize_alias(mention)
        for resource, entry in routed:
            strongest_score = 0.0
            strongest_alias: str | None = None
            for surface in entry.surfaces:
                alias = normalize_alias(surface)
                score = (
                    1.0
                    if query == alias
                    else SequenceMatcher(None, query, alias).ratio()
                )
                if score > strongest_score:
                    strongest_score = score
                    strongest_alias = surface
            if strongest_alias is None or strongest_score < self._string_score_cutoff:
                continue
            exact = strongest_score == 1.0
            candidate = Candidate(
                system=entry.target_system,
                code=entry.target_code,
                display=entry.target_display,
                score=round(strongest_score, 6),
                source_language=source_language,
                source="crosswalk-string",
                matched_alias=strongest_alias,
                match_kind="exact-crosswalk" if exact else "string-crosswalk",
                vocab_version=resource.resource_version,
            )
            _offer(best, _CandidateOffer(candidate, resource))

    def _offer_dense_candidates(
        self,
        best: dict[tuple[str, str], _CandidateOffer],
        mention: str,
        *,
        source_language: str,
        routed: Sequence[tuple[CrosswalkResource, CrosswalkEntry]],
    ) -> None:
        assert self._encoder is not None
        alias_rows: list[tuple[CrosswalkResource, CrosswalkEntry, str]] = []
        for resource, entry in routed:
            for alias in (*entry.aliases, entry.target_display):
                alias_rows.append((resource, entry, alias))
        texts = (mention, *(alias for _, _, alias in alias_rows))
        vectors = self._encoder.encode(texts)
        if len(vectors) != len(texts):
            raise ValueError("cross-lingual encoder returned the wrong vector count")
        query_vector = _validated_vector(
            vectors[0], expected_dimension=self._encoder.dimension
        )
        for (resource, entry, alias), raw_vector in zip(alias_rows, vectors[1:]):
            vector = _validated_vector(
                raw_vector, expected_dimension=self._encoder.dimension
            )
            score = max(0.0, _cosine(query_vector, vector))
            if score < self._dense_score_cutoff:
                continue
            candidate = Candidate(
                system=entry.target_system,
                code=entry.target_code,
                display=entry.target_display,
                score=round(score, 6),
                source="cross-lingual-dense",
                source_language=source_language,
                matched_alias=alias,
                match_kind="dense-cross-lingual",
                vocab_version=resource.resource_version,
            )
            _offer(best, _CandidateOffer(candidate, resource))

    def _offer_umls_candidates(
        self,
        best: dict[tuple[str, str], _CandidateOffer],
        mention: str,
        *,
        source_language: str,
    ) -> None:
        if self._umls_matcher is None or self._umls_loader is None:
            return
        for match in self._umls_matcher.lookup(mention):
            candidate = Candidate(
                system="UMLS",
                code=match.code,
                display=match.display,
                score=match.score,
                source_language=source_language,
                source="user-key-local",
                matched_alias=match.matched_term,
                match_kind=match.match_type,
                vocab_version=self._umls_loader.content_hash,
            )
            _offer(best, _CandidateOffer(candidate, None))


def ground_multilingual(
    mention: str,
    locale: str,
    *,
    resources: Sequence[CrosswalkResource] | None = None,
    encoder: AliasEncoder | None = None,
    encoder_path: str | None = None,
    restricted_loaders: Mapping[str, UserKeyVocabularyLoader] | None = None,
    top_k: int = _DEFAULT_TOP_K,
) -> MultilingualGroundingResult:
    """Ground one multilingual mention through an entirely local router.

    ``None`` resources use the bundled free starter mappings.  Pass resources
    returned by :func:`openmed.clinical.grounding.load_crosswalk` for larger
    caller-controlled tables.  Restricted UMLS aliases activate only through a
    key-gated :class:`UserKeyVocabularyLoader`.
    """

    return MultilingualGrounder(
        resources,
        encoder=encoder,
        encoder_path=encoder_path,
        restricted_loaders=restricted_loaders,
    ).ground(mention, locale, top_k=top_k)


def _offer(
    best: dict[tuple[str, str], _CandidateOffer], offer: _CandidateOffer
) -> None:
    key = (offer.candidate.system, offer.candidate.code)
    current = best.get(key)
    if current is None or offer.candidate.score > current.candidate.score:
        best[key] = offer


def _resolve_umls_loader(
    loaders: Mapping[str, UserKeyVocabularyLoader] | None,
) -> UserKeyVocabularyLoader | None:
    if loaders is None:
        return None
    normalized = {key.strip().casefold(): value for key, value in loaders.items()}
    loader = normalized.get("umls")
    if loader is None:
        return None
    if not isinstance(loader, UserKeyVocabularyLoader) or loader.system != "umls":
        raise RestrictedVocabularyError(
            "multilingual UMLS grounding requires a matching UserKeyVocabularyLoader"
        )
    return loader


def _normalize_locale(locale: str) -> str:
    if not isinstance(locale, str) or not locale.strip():
        raise ValueError("locale must be non-empty text")
    folded = unicodedata.normalize("NFKC", locale).strip().replace("_", "-")
    folded = re.sub(r"[^A-Za-z0-9-]+", "", folded)
    parts = [part for part in folded.split("-") if part]
    if not parts:
        raise ValueError("locale must contain a language tag")
    language = normalize_language(parts[0])
    return "-".join((language, *(part.upper() for part in parts[1:])))


def _score_cutoff(value: float, name: str) -> float:
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{name} must be finite and between 0.0 and 1.0")
    return score


def _validated_vector(
    vector: Sequence[float], *, expected_dimension: int
) -> tuple[float, ...]:
    values = tuple(float(value) for value in vector)
    if len(values) != expected_dimension:
        raise ValueError(
            f"cross-lingual encoder returned dimension {len(values)}, "
            f"expected {expected_dimension}"
        )
    if any(not math.isfinite(value) for value in values):
        raise ValueError("cross-lingual encoder returned a non-finite vector")
    return values


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
