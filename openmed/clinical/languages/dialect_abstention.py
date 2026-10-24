"""Dialect-aware fallback and abstention for validated language routes.

Deterministic routing can send an unvalidated dialect to a nearby language
pack, and the model behind that pack may then return confident but unsafe
extraction or de-identification results. This module estimates how far an
input sits from the language varieties a route was validated on and applies a
declared policy that accepts the route, requires a fallback or an ensemble, or
abstains.

Distances use privacy-safe features only: script shares, character-class
shares, a keyed and hashed character n-gram histogram, the declared locale, the
declared domain and the deterministic language route. Input text is never
stored, logged or echoed. Input histograms exist only in memory and are never
serialized; validated references are aggregates over several validation
documents. Everything is stdlib-only, deterministic and offline.

Distances are routing heuristics. They are not a measurement of extraction or
de-identification accuracy, and an accepted route is not a clinical
qualification.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Collection, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import islice
from types import MappingProxyType
from typing import Any, Final

from ...core.language_router import LanguageRouter
from ...core.locale_tag import normalize_locale_tag
from ...core.script_detect import UNKNOWN_SCRIPT, detect_script

__all__ = [
    "CHAR_CLASSES",
    "DIALECT_ABSTENTION_ADVISORY",
    "DIALECT_ACTIONS",
    "DIALECT_LIMITATIONS",
    "MIN_REFERENCE_DOCUMENTS",
    "NGRAM_BUCKETS",
    "SCHEMA_VERSION",
    "DialectDecision",
    "DialectFeatureProfile",
    "DialectPolicy",
    "DialectRouteError",
    "ValidatedVariety",
    "VarietyDistance",
    "apply_dialect_policy",
    "assess_dialect",
    "build_validated_variety",
    "enforce_dialect_decision",
    "estimate_variety_distance",
    "extract_dialect_features",
]

SCHEMA_VERSION: Final = 1

DIALECT_ABSTENTION_ADVISORY: Final = (
    "Dialect distance is a deterministic routing heuristic, not a measurement "
    "of extraction or de-identification accuracy. An accepted route is not a "
    "clinical qualification, and abstained inputs need human review."
)

#: Number of hashed character n-gram buckets in every histogram.
NGRAM_BUCKETS: Final = 512

#: Fixed order of the character classes used for domain-shape features.
CHAR_CLASSES: Final = (
    "letter",
    "mark",
    "number",
    "space",
    "line_break",
    "punctuation",
    "symbol",
    "other",
)

#: Routing actions a policy can require, ordered from least to most cautious.
DIALECT_ACTIONS: Final = ("accept", "fallback", "ensemble", "abstain")

#: Minimum number of validation documents aggregated into one reference.
MIN_REFERENCE_DOCUMENTS: Final = 3

#: Controlled limitation codes surfaced in decisions, with fixed descriptions.
DIALECT_LIMITATIONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "declared_locale_conflict": (
            "The declared locale's language differs from a decisive "
            "deterministic route."
        ),
        "dialect_out_of_range": (
            "The input is beyond the abstention distance of every validated variety."
        ),
        "dialect_shift": (
            "The input is outside the accept distance of the nearest validated variety."
        ),
        "domain_out_of_range": (
            "The input domain is beyond the abstention distance of the nearest "
            "validated variety."
        ),
        "domain_shift": (
            "The input domain is outside the accept distance of the nearest "
            "validated variety."
        ),
        "domain_undeclared": "No document domain was declared.",
        "domain_unvalidated": (
            "The declared domain is not among the nearest variety's validated domains."
        ),
        "ensemble_insufficient_members": (
            "Fewer than two validated varieties are close enough for an ensemble."
        ),
        "ensemble_route": "The policy requires an ensemble of validated varieties.",
        "fallback_out_of_range": (
            "The declared fallback variety is beyond the abstention distance."
        ),
        "fallback_route": "The policy requires the declared fallback variety.",
        "fallback_unavailable": (
            "The declared fallback variety is not validated for this task."
        ),
        "insufficient_evidence": (
            "The input has too few letters for a stable distance estimate."
        ),
        "language_unconfirmed": (
            "Deterministic routing could not confirm the input language."
        ),
        "locale_undeclared": "No locale was declared for the input.",
        "mixed_language": "The input routes to more than one language.",
        "no_validated_variety": "No validated variety covers this task.",
        "unsupported_script": (
            "The input contains letters in a script with no registered pack."
        ),
        "variety_pack_unregistered": (
            "At least one validated variety names an unregistered language pack."
        ),
    }
)

MAX_TEXT_CHARS: Final = 1_000_000
MAX_REFERENCE_DOCUMENTS: Final = 10_000
MAX_VARIETIES: Final = 256
_MIN_REFERENCE_LETTERS: Final = 120
_MAX_TOKENS: Final = 64
_PRECISION: Final = 6
_SUM_TOLERANCE: Final = 1e-3
_HASH_PERSON: Final = b"omdialect.v1"
_DECISIVE_ROUTING_CONFIDENCE: Final = 0.9

# Dialect distance blends the locale, script and orthographic evidence; domain
# distance blends the declared domain with the document's character shape.
_LOCALE_WEIGHT: Final = 0.3
_SCRIPT_WEIGHT: Final = 0.2
_NGRAM_WEIGHT: Final = 0.5
_DECLARED_DOMAIN_WEIGHT: Final = 0.5
_SHAPE_WEIGHT: Final = 0.5

_MULTILINGUAL_LANGUAGES: Final = frozenset({"mul", "und"})
_LINE_BREAKS: Final = frozenset("\n\r\x0b\x0c\x85\u2028\u2029")
_TOKEN: Final = re.compile(r"[a-z][a-z0-9_]{0,63}")
_VARIETY_ID: Final = re.compile(r"[a-z][a-z0-9_.-]{0,63}")
_PACK_CODE: Final = re.compile(r"[a-z]{2}")
_SCRIPT_NAME: Final = re.compile(r"[A-Z][A-Za-z/]{1,31}")
_DIGEST: Final = re.compile(r"[0-9a-f]{64}")
_BAND_RANK: Final = {"accept": 0, "intermediate": 1, "abstain": 2}
_REVIEW_LIMITATIONS: Final = frozenset(
    {
        "declared_locale_conflict",
        "domain_unvalidated",
        "insufficient_evidence",
        "mixed_language",
        "unsupported_script",
    }
)


class DialectRouteError(ValueError):
    """Value-free failure raised when execution does not honor a decision.

    Attributes:
        code: Stable reason code, ``"dialect_abstained"`` or
            ``"dialect_route_mismatch"``.
    """

    def __init__(self, code: str) -> None:
        """Create an error that carries only a controlled reason code."""

        self.code = code
        super().__init__(code)


# ---------------------------------------------------------------------------
# Validation helpers


def _round(value: float) -> float:
    return round(float(value), _PRECISION)


def _token(value: object, pattern: re.Pattern[str], message: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(message)
    return value


def _tokens(value: object, message: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ValueError(message)
    items = tuple(islice(value, _MAX_TOKENS + 1))
    if not items or len(items) > _MAX_TOKENS:
        raise ValueError(message)
    for item in items:
        _token(item, _TOKEN, message)
    if len(set(items)) != len(items):
        raise ValueError(message)
    return tuple(sorted(items))


def _canonical_locale(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("locale must be a BCP 47 language tag")
    # ``normalize_locale_tag`` raises value-free ``LocaleTagError`` codes.
    return normalize_locale_tag(value.strip().replace("_", "-"))


def _probability(value: object, message: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(message)
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(message)
    return number


def _check_distribution(values: Sequence[float], message: str) -> None:
    if values and abs(sum(values) - 1.0) > _SUM_TOLERANCE:
        raise ValueError(message)


def _freeze_script_shares(value: object) -> tuple[tuple[str, float], ...]:
    message = "script shares must map script names to probabilities"
    items = value.items() if isinstance(value, Mapping) else value
    if isinstance(items, (str, bytes)) or not isinstance(items, Iterable):
        raise ValueError(message)
    entries = tuple(islice(items, _MAX_TOKENS + 1))
    if len(entries) > _MAX_TOKENS:
        raise ValueError(message)
    pairs: dict[str, float] = {}
    for entry in entries:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValueError(message)
        script = _token(entry[0], _SCRIPT_NAME, message)
        if script in pairs:
            raise ValueError(message)
        pairs[script] = _round(_probability(entry[1], message))
    pairs = {script: share for script, share in pairs.items() if share > 0.0}
    if not pairs:
        raise ValueError(message)
    _check_distribution(tuple(pairs.values()), message)
    return tuple(sorted(pairs.items()))


def _freeze_class_shares(value: object) -> tuple[float, ...]:
    message = "character-class shares must follow CHAR_CLASSES"
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ValueError(message)
    shares = tuple(islice(value, len(CHAR_CLASSES) + 1))
    if len(shares) != len(CHAR_CLASSES):
        raise ValueError(message)
    frozen = tuple(_round(_probability(share, message)) for share in shares)
    _check_distribution(frozen, message)
    return frozen


def _freeze_histogram(value: object) -> tuple[tuple[int, float], ...]:
    message = "n-gram histograms must map buckets to probabilities"
    items = value.items() if isinstance(value, Mapping) else value
    if isinstance(items, (str, bytes)) or not isinstance(items, Iterable):
        raise ValueError(message)
    buckets: dict[int, float] = {}
    # A histogram has at most one entry per bucket, so a longer input always
    # contains a duplicate and fails without being consumed further.
    for entry in islice(items, NGRAM_BUCKETS + 1):
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValueError(message)
        bucket, share = entry
        if (
            isinstance(bucket, bool)
            or not isinstance(bucket, int)
            or not 0 <= bucket < NGRAM_BUCKETS
            or bucket in buckets
        ):
            raise ValueError(message)
        buckets[bucket] = _round(_probability(share, message))
    buckets = {bucket: share for bucket, share in buckets.items() if share > 0.0}
    if not buckets:
        raise ValueError(message)
    _check_distribution(tuple(buckets.values()), message)
    return tuple(sorted(buckets.items()))


# ---------------------------------------------------------------------------
# Privacy-safe feature extraction


def _require_text(text: object) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError("text exceeds the dialect assessment bound")
    return text


def _char_class_index(char: str) -> int:
    if char in _LINE_BREAKS:
        return 4
    category = unicodedata.category(char)
    major = category[0]
    if major == "L":
        return 0
    if major == "M":
        return 1
    if major == "N":
        return 2
    if category == "Zs" or char == "\t":
        return 3
    if major == "P":
        return 5
    if major == "S":
        return 6
    return 7


@dataclass(slots=True)
class _Counts:
    """Transient per-call counters; never returned or retained."""

    chars: int = 0
    letters: int = 0
    scripts: dict[str, int] = field(default_factory=dict)
    classes: list[int] = field(default_factory=lambda: [0] * len(CHAR_CLASSES))
    ngrams: dict[int, int] = field(default_factory=dict)

    def add(self, text: str) -> None:
        script_of: dict[str, str] = {}
        bucket_of: dict[str, int] = {}
        token: list[str] = []
        for char in unicodedata.normalize("NFKC", text):
            self.chars += 1
            index = _char_class_index(char)
            self.classes[index] += 1
            if index in (0, 1):
                token.append(char)
                if index == 0:
                    self.letters += 1
                    script = script_of.get(char)
                    if script is None:
                        script = script_of[char] = detect_script(char)
                    self.scripts[script] = self.scripts.get(script, 0) + 1
                continue
            if token:
                self._add_token("".join(token), bucket_of)
                token.clear()
        if token:
            self._add_token("".join(token), bucket_of)

    def _add_token(self, token: str, bucket_of: dict[str, int]) -> None:
        # Digits and punctuation end a token, so identifiers and numbers never
        # contribute n-grams. Each gram is hashed into a small keyed bucket
        # space; the grams themselves are discarded with this call.
        padded = f" {token.casefold()} "
        for size in (2, 3):
            for start in range(len(padded) - size + 1):
                gram = padded[start : start + size]
                bucket = bucket_of.get(gram)
                if bucket is None:
                    digest = hashlib.blake2b(
                        gram.encode("utf-8"), digest_size=8, person=_HASH_PERSON
                    ).digest()
                    bucket = int.from_bytes(digest, "big") % NGRAM_BUCKETS
                    bucket_of[gram] = bucket
                self.ngrams[bucket] = self.ngrams.get(bucket, 0) + 1

    def script_shares(self) -> tuple[tuple[str, float], ...]:
        if not self.letters:
            return ()
        return tuple(
            sorted(
                (script, _round(count / self.letters))
                for script, count in self.scripts.items()
            )
        )

    def class_shares(self) -> tuple[float, ...]:
        if not self.chars:
            return tuple(0.0 for _ in CHAR_CLASSES)
        return tuple(_round(count / self.chars) for count in self.classes)

    def histogram(self) -> tuple[tuple[int, float], ...]:
        total = sum(self.ngrams.values())
        if not total:
            return ()
        return tuple(
            (bucket, _round(count / total))
            for bucket, count in sorted(self.ngrams.items())
        )


@dataclass(frozen=True, slots=True)
class DialectFeatureProfile:
    """Privacy-safe features of one input; the input text is not retained.

    Attributes:
        char_count: Number of code points after NFKC normalization.
        letter_count: Number of letters after NFKC normalization.
        script_shares: Sorted ``(script, share)`` pairs over letters.
        char_class_shares: Shares aligned with :data:`CHAR_CLASSES`.
        ngram_histogram: Sorted ``(bucket, share)`` pairs of keyed, hashed
            letter bigrams and trigrams. It is used in memory only and is
            excluded from :meth:`summary`.
        declared_locale: Canonical BCP 47 tag declared by the caller.
        declared_domain: Controlled domain code declared by the caller.
        routed_language: Dominant language of the deterministic route.
        routing_confidence: Letter-weighted route confidence; a routing hint,
            not an accuracy estimate.
        routing_decisive: Whether the route met the decisive confidence floor.
        mixed_language: Whether letter-bearing runs route to several languages.
        unsupported_script: Whether letters fall in a script without a pack.
    """

    char_count: int
    letter_count: int
    script_shares: tuple[tuple[str, float], ...]
    char_class_shares: tuple[float, ...]
    ngram_histogram: tuple[tuple[int, float], ...] = field(repr=False)
    declared_locale: str | None = None
    declared_domain: str | None = None
    routed_language: str = "und"
    routing_confidence: float = 0.0
    routing_decisive: bool = False
    mixed_language: bool = False
    unsupported_script: bool = False

    @property
    def observed_locale(self) -> str | None:
        """Return the declared locale, else a decisively routed language."""

        if self.declared_locale is not None:
            return self.declared_locale
        if self.routing_decisive and self.letter_count:
            return self.routed_language
        return None

    def summary(self) -> dict[str, Any]:
        """Return counts, shares and codes only, without the n-gram histogram."""

        return {
            "char_count": self.char_count,
            "letter_count": self.letter_count,
            "script_shares": dict(self.script_shares),
            "char_class_shares": dict(zip(CHAR_CLASSES, self.char_class_shares)),
            "declared_locale": self.declared_locale,
            "declared_domain": self.declared_domain,
            "routed_language": self.routed_language,
            "routing_confidence": self.routing_confidence,
            "routing_decisive": self.routing_decisive,
            "mixed_language": self.mixed_language,
            "unsupported_script": self.unsupported_script,
        }


def _default_router() -> LanguageRouter:
    # The optional CLD2 backend depends on the environment, so the default
    # assessment uses the stdlib-only route to stay reproducible everywhere.
    return LanguageRouter(use_optional_lid=False)


def _require_router(router: object) -> LanguageRouter:
    if router is None:
        return _default_router()
    if not isinstance(router, LanguageRouter):
        raise TypeError("router must be a LanguageRouter")
    return router


def extract_dialect_features(
    text: str,
    *,
    declared_locale: str | None = None,
    declared_domain: str | None = None,
    router: LanguageRouter | None = None,
) -> DialectFeatureProfile:
    """Measure privacy-safe dialect and domain features of ``text``.

    Args:
        text: Input text. It is read once and never stored or echoed.
        declared_locale: Optional BCP 47 tag such as ``"de-CH"``. Underscore
            separators are accepted. Validation is structural and offline.
        declared_domain: Optional controlled domain code such as
            ``"discharge_summary"``.
        router: Deterministic language router. The default disables the
            optional CLD2 backend so results do not depend on the environment.

    Returns:
        A :class:`DialectFeatureProfile` holding counts, shares and codes.

    Raises:
        TypeError: If ``text`` is not a string or ``router`` is not a router.
        ValueError: If an argument is malformed or ``text`` exceeds the bound.
            Messages never contain input values.
    """

    text = _require_text(text)
    locale = None if declared_locale is None else _canonical_locale(declared_locale)
    domain = (
        None
        if declared_domain is None
        else _token(declared_domain, _TOKEN, "declared domain must be a code")
    )
    resolved_router = _require_router(router)

    counts = _Counts()
    counts.add(text)

    lexical_length: dict[str, int] = {}
    weighted_confidence = 0.0
    # Letters outside every known script block attach to a neighbouring run in
    # the router, so the per-letter script counts are checked as well.
    unsupported = UNKNOWN_SCRIPT in counts.scripts
    for run in resolved_router.route_runs(text):
        segment = text[run.start : run.end]
        if not any(char.isalpha() for char in segment):
            continue
        length = run.end - run.start
        lexical_length[run.language] = lexical_length.get(run.language, 0) + length
        weighted_confidence += run.confidence * length
        unsupported = unsupported or run.reason == "unsupported_script"

    if lexical_length:
        routed_language = min(
            lexical_length, key=lambda code: (-lexical_length[code], code)
        )
        confidence = _round(weighted_confidence / sum(lexical_length.values()))
    else:
        routed_language, confidence = "und", 0.0

    return DialectFeatureProfile(
        char_count=counts.chars,
        letter_count=counts.letters,
        script_shares=counts.script_shares(),
        char_class_shares=counts.class_shares(),
        ngram_histogram=counts.histogram(),
        declared_locale=locale,
        declared_domain=domain,
        routed_language=routed_language,
        routing_confidence=confidence,
        routing_decisive=(
            bool(lexical_length)
            and not unsupported
            and confidence >= _DECISIVE_ROUTING_CONFIDENCE
        ),
        mixed_language=len(lexical_length) > 1,
        unsupported_script=unsupported,
    )


# ---------------------------------------------------------------------------
# Validated evidence


@dataclass(frozen=True, slots=True)
class ValidatedVariety:
    """Aggregate, text-free reference for one validated language variety.

    A variety records the locale, tasks and domains a route was validated on,
    plus aggregate feature statistics from its validation documents. It holds
    no text, n-grams or identifiers; ``evidence_digest`` references the
    validation evidence itself. References derived from restricted corpora
    remain derived artifacts of those corpora and must stay where the source
    data use agreement permits.

    Attributes:
        variety_id: Controlled identifier, for example ``"de-de.clinical"``.
        locale: Canonical BCP 47 locale the route was validated on.
        tasks: Validated task codes, for example ``("deidentification",)``.
        domains: Validated document-domain codes.
        evidence_digest: SHA-256 hex digest of the validation evidence.
        document_count: Number of validation documents aggregated.
        script_shares: Sorted ``(script, share)`` pairs over letters.
        char_class_shares: Shares aligned with :data:`CHAR_CLASSES`.
        ngram_centroid: Sorted ``(bucket, share)`` hashed n-gram centroid.
        language_pack: Optional language-pack code the route executes with.
    """

    variety_id: str
    locale: str
    tasks: tuple[str, ...]
    domains: tuple[str, ...]
    evidence_digest: str
    document_count: int
    script_shares: tuple[tuple[str, float], ...]
    char_class_shares: tuple[float, ...]
    ngram_centroid: tuple[tuple[int, float], ...] = field(repr=False)
    language_pack: str | None = None

    def __post_init__(self) -> None:
        """Validate and freeze the declaration with value-free errors."""

        _token(self.variety_id, _VARIETY_ID, "invalid variety identifier")
        object.__setattr__(self, "locale", _canonical_locale(self.locale))
        object.__setattr__(
            self, "tasks", _tokens(self.tasks, "invalid variety task codes")
        )
        object.__setattr__(
            self, "domains", _tokens(self.domains, "invalid variety domain codes")
        )
        _token(self.evidence_digest, _DIGEST, "invalid variety evidence digest")
        if (
            isinstance(self.document_count, bool)
            or not isinstance(self.document_count, int)
            or self.document_count < MIN_REFERENCE_DOCUMENTS
        ):
            raise ValueError("variety needs more validation documents")
        object.__setattr__(
            self, "script_shares", _freeze_script_shares(self.script_shares)
        )
        object.__setattr__(
            self, "char_class_shares", _freeze_class_shares(self.char_class_shares)
        )
        object.__setattr__(
            self, "ngram_centroid", _freeze_histogram(self.ngram_centroid)
        )
        if self.language_pack is not None:
            _token(self.language_pack, _PACK_CODE, "invalid variety language pack")

    def to_dict(self) -> dict[str, Any]:
        """Export the aggregate reference as JSON-compatible metadata."""

        return {
            "schema_version": SCHEMA_VERSION,
            "variety_id": self.variety_id,
            "locale": self.locale,
            "language_pack": self.language_pack,
            "tasks": list(self.tasks),
            "domains": list(self.domains),
            "evidence_digest": self.evidence_digest,
            "document_count": self.document_count,
            "script_shares": dict(self.script_shares),
            "char_class_shares": dict(zip(CHAR_CLASSES, self.char_class_shares)),
            "ngram_centroid": [list(item) for item in self.ngram_centroid],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ValidatedVariety:
        """Rebuild a reference exported by :meth:`to_dict`.

        Raises:
            ValueError: If the payload is malformed. Messages are value-free.
        """

        if not isinstance(payload, Mapping):
            raise ValueError("variety payload must be a mapping")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported variety schema version")
        class_shares = payload.get("char_class_shares")
        if not isinstance(class_shares, Mapping) or set(class_shares) != set(
            CHAR_CLASSES
        ):
            raise ValueError("character-class shares must follow CHAR_CLASSES")
        centroid = payload.get("ngram_centroid")
        if isinstance(centroid, (str, bytes)) or not isinstance(centroid, Sequence):
            raise ValueError("n-gram histograms must map buckets to probabilities")
        return cls(
            variety_id=payload.get("variety_id"),
            locale=payload.get("locale"),
            tasks=payload.get("tasks"),
            domains=payload.get("domains"),
            evidence_digest=payload.get("evidence_digest"),
            document_count=payload.get("document_count"),
            script_shares=payload.get("script_shares"),
            char_class_shares=tuple(class_shares[name] for name in CHAR_CLASSES),
            ngram_centroid=tuple(
                tuple(item) if isinstance(item, (list, tuple)) else item
                for item in islice(centroid, NGRAM_BUCKETS + 1)
            ),
            language_pack=payload.get("language_pack"),
        )


def build_validated_variety(
    variety_id: str,
    texts: Iterable[str],
    *,
    locale: str,
    tasks: Iterable[str],
    domains: Iterable[str],
    evidence_digest: str,
    language_pack: str | None = None,
) -> ValidatedVariety:
    """Aggregate validation documents into a text-free variety reference.

    Only pooled counts survive: script shares, character-class shares and the
    hashed n-gram centroid. Use synthetic or governed evaluation documents;
    the documents are read once and never stored.

    Args:
        variety_id: Controlled identifier for the validated variety.
        texts: At least :data:`MIN_REFERENCE_DOCUMENTS` validation documents.
        locale: BCP 47 locale the route was validated on.
        tasks: Validated task codes.
        domains: Validated document-domain codes.
        evidence_digest: SHA-256 hex digest of the validation evidence.
        language_pack: Optional language-pack code the route executes with.

    Returns:
        The aggregate :class:`ValidatedVariety`.

    Raises:
        TypeError: If a document is not a string.
        ValueError: If there are too few documents or letters, or a
            declaration is malformed. Messages never contain document text.
    """

    if isinstance(texts, (str, bytes)) or not isinstance(texts, Iterable):
        raise ValueError("texts must be a collection of documents")
    counts = _Counts()
    documents = 0
    for text in islice(texts, MAX_REFERENCE_DOCUMENTS + 1):
        documents += 1
        if documents > MAX_REFERENCE_DOCUMENTS:
            raise ValueError("too many validation documents")
        counts.add(_require_text(text))
    if documents < MIN_REFERENCE_DOCUMENTS:
        raise ValueError("variety needs more validation documents")
    if counts.letters < _MIN_REFERENCE_LETTERS:
        raise ValueError("validation documents contain too few letters")
    return ValidatedVariety(
        variety_id=variety_id,
        locale=locale,
        tasks=tasks,
        domains=domains,
        evidence_digest=evidence_digest,
        document_count=documents,
        script_shares=counts.script_shares(),
        char_class_shares=counts.class_shares(),
        ngram_centroid=counts.histogram(),
        language_pack=language_pack,
    )


# ---------------------------------------------------------------------------
# Distances


def _locale_parts(tag: str) -> tuple[str, str | None, str | None, tuple[str, ...]]:
    parts = tag.split("-")
    script = region = None
    index = 1
    if index < len(parts) and len(parts[index]) == 4 and parts[index].isalpha():
        script = parts[index]
        index += 1
    if index < len(parts) and (
        (len(parts[index]) == 2 and parts[index].isalpha())
        or (len(parts[index]) == 3 and parts[index].isdigit())
    ):
        region = parts[index]
        index += 1
    return parts[0], script, region, tuple(parts[index:])


def _locale_distance(observed: str | None, reference: str) -> float:
    ref_language, ref_script, ref_region, ref_variants = _locale_parts(reference)
    if observed is None or ref_language in _MULTILINGUAL_LANGUAGES:
        return 0.5
    language, script, region, variants = _locale_parts(observed)
    if language in _MULTILINGUAL_LANGUAGES:
        return 0.5
    if language != ref_language:
        return 1.0
    if script is not None and ref_script is not None and script != ref_script:
        return 0.75
    if region is None and ref_region is None:
        return 0.0 if variants == ref_variants else 0.25
    if region is None or ref_region is None:
        return 0.25
    if region != ref_region:
        return 0.5
    return 0.0 if variants == ref_variants else 0.25


def _total_variation(left: Mapping[Any, float], right: Mapping[Any, float]) -> float:
    keys = sorted(set(left) | set(right), key=str)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


def _cosine_distance(left: Mapping[int, float], right: Mapping[int, float]) -> float:
    if not left or not right:
        return 1.0
    dot = sum(
        share * right[bucket]
        for bucket, share in sorted(left.items())
        if bucket in right
    )
    norm = math.sqrt(sum(share * share for share in left.values())) * math.sqrt(
        sum(share * share for share in right.values())
    )
    return min(1.0, max(0.0, 1.0 - dot / norm)) if norm else 1.0


@dataclass(frozen=True, slots=True)
class VarietyDistance:
    """Distance components between one input profile and one variety.

    Every component lies in ``[0, 1]``. ``dialect_distance`` blends locale,
    script and hashed n-gram distances; ``domain_distance`` blends the
    declared-domain match with the character-class shape distance.
    """

    variety_id: str
    dialect_distance: float
    domain_distance: float
    locale_distance: float
    script_distance: float
    ngram_distance: float
    shape_distance: float
    declared_domain_distance: float

    def to_dict(self) -> dict[str, Any]:
        """Return the distance components as JSON-compatible metadata."""

        return {
            "variety_id": self.variety_id,
            "dialect_distance": self.dialect_distance,
            "domain_distance": self.domain_distance,
            "locale_distance": self.locale_distance,
            "script_distance": self.script_distance,
            "ngram_distance": self.ngram_distance,
            "shape_distance": self.shape_distance,
            "declared_domain_distance": self.declared_domain_distance,
        }


def estimate_variety_distance(
    profile: DialectFeatureProfile,
    variety: ValidatedVariety,
) -> VarietyDistance:
    """Estimate dialect and domain distance from one validated variety.

    Args:
        profile: Input features from :func:`extract_dialect_features`.
        variety: Validated reference to compare against.

    Returns:
        The deterministic, rounded :class:`VarietyDistance`.
    """

    if not isinstance(profile, DialectFeatureProfile):
        raise TypeError("profile must be a DialectFeatureProfile")
    if not isinstance(variety, ValidatedVariety):
        raise TypeError("variety must be a ValidatedVariety")

    locale = _locale_distance(profile.observed_locale, variety.locale)
    script = (
        _total_variation(dict(profile.script_shares), dict(variety.script_shares))
        if profile.script_shares
        else 1.0
    )
    ngram = _cosine_distance(
        dict(profile.ngram_histogram), dict(variety.ngram_centroid)
    )
    shape = _total_variation(
        dict(zip(CHAR_CLASSES, profile.char_class_shares)),
        dict(zip(CHAR_CLASSES, variety.char_class_shares)),
    )
    if profile.declared_domain is None:
        declared_domain = 0.5
    else:
        declared_domain = 0.0 if profile.declared_domain in variety.domains else 1.0
    dialect = _LOCALE_WEIGHT * locale + _SCRIPT_WEIGHT * script + _NGRAM_WEIGHT * ngram
    domain = _DECLARED_DOMAIN_WEIGHT * declared_domain + _SHAPE_WEIGHT * shape
    return VarietyDistance(
        variety_id=variety.variety_id,
        dialect_distance=_round(min(dialect, 1.0)),
        domain_distance=_round(min(domain, 1.0)),
        locale_distance=_round(locale),
        script_distance=_round(min(script, 1.0)),
        ngram_distance=_round(ngram),
        shape_distance=_round(min(shape, 1.0)),
        declared_domain_distance=_round(declared_domain),
    )


# ---------------------------------------------------------------------------
# Declared policy and decisions


@dataclass(frozen=True, slots=True)
class DialectPolicy:
    """Declared thresholds and actions for dialect and domain distance.

    A distance at or below an ``accept_*`` threshold accepts the nearest
    validated variety. A distance above an ``abstain_*`` threshold abstains.
    In between, ``intermediate_action`` decides whether the input requires the
    declared fallback variety, an ensemble of the nearest validated varieties,
    or abstention. The stricter of the dialect and domain bands applies. Inputs
    with fewer than ``min_letters`` letters use
    ``insufficient_evidence_action``. The defaults abstain, so a caller must
    declare a fallback or an ensemble explicitly.

    Attributes:
        policy_id: Controlled policy identifier surfaced in every decision.
        accept_dialect_distance: Highest dialect distance that is accepted.
        abstain_dialect_distance: Dialect distance above which to abstain.
        accept_domain_distance: Highest domain distance that is accepted.
        abstain_domain_distance: Domain distance above which to abstain.
        intermediate_action: ``"fallback"``, ``"ensemble"`` or ``"abstain"``.
        insufficient_evidence_action: ``"fallback"`` or ``"abstain"``.
        fallback_variety: Variety identifier required by fallback actions.
        ensemble_size: Maximum ensemble members, between 2 and 8.
        min_letters: Letters needed for a stable distance estimate.
    """

    policy_id: str = "default"
    accept_dialect_distance: float = 0.2
    abstain_dialect_distance: float = 0.4
    accept_domain_distance: float = 0.35
    abstain_domain_distance: float = 0.75
    intermediate_action: str = "abstain"
    insufficient_evidence_action: str = "abstain"
    fallback_variety: str | None = None
    ensemble_size: int = 2
    min_letters: int = 40

    def __post_init__(self) -> None:
        """Validate the declaration with value-free errors."""

        _token(self.policy_id, _VARIETY_ID, "invalid dialect policy identifier")
        for name in (
            "accept_dialect_distance",
            "abstain_dialect_distance",
            "accept_domain_distance",
            "abstain_domain_distance",
        ):
            object.__setattr__(
                self,
                name,
                _probability(
                    getattr(self, name), "policy thresholds must be in [0, 1]"
                ),
            )
        if (
            self.accept_dialect_distance > self.abstain_dialect_distance
            or self.accept_domain_distance > self.abstain_domain_distance
        ):
            raise ValueError("accept thresholds must not exceed abstain thresholds")
        if self.intermediate_action not in DIALECT_ACTIONS[1:]:
            raise ValueError("invalid intermediate dialect action")
        if self.insufficient_evidence_action not in ("fallback", "abstain"):
            raise ValueError("invalid insufficient-evidence dialect action")
        uses_fallback = "fallback" in (
            self.intermediate_action,
            self.insufficient_evidence_action,
        )
        if self.fallback_variety is None:
            if uses_fallback:
                raise ValueError("fallback actions require a fallback variety")
        else:
            _token(self.fallback_variety, _VARIETY_ID, "invalid fallback variety")
        for name, low, high in (("ensemble_size", 2, 8), ("min_letters", 1, 10_000)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("policy bounds must be integers")
            if not low <= value <= high:
                raise ValueError("policy bounds are out of range")

    def to_dict(self) -> dict[str, Any]:
        """Return the declared policy as JSON-compatible metadata."""

        return {
            "policy_id": self.policy_id,
            "accept_dialect_distance": self.accept_dialect_distance,
            "abstain_dialect_distance": self.abstain_dialect_distance,
            "accept_domain_distance": self.accept_domain_distance,
            "abstain_domain_distance": self.abstain_domain_distance,
            "intermediate_action": self.intermediate_action,
            "insufficient_evidence_action": self.insufficient_evidence_action,
            "fallback_variety": self.fallback_variety,
            "ensemble_size": self.ensemble_size,
            "min_letters": self.min_letters,
        }

    @property
    def digest(self) -> str:
        """Return the SHA-256 digest of the canonical policy declaration."""

        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _band(self, distance: float, accept: float, abstain: float) -> str:
        if distance <= accept:
            return "accept"
        if distance > abstain:
            return "abstain"
        return "intermediate"

    def dialect_band(self, distance: VarietyDistance) -> str:
        """Return the accept, intermediate or abstain band for dialect."""

        return self._band(
            distance.dialect_distance,
            self.accept_dialect_distance,
            self.abstain_dialect_distance,
        )

    def domain_band(self, distance: VarietyDistance) -> str:
        """Return the accept, intermediate or abstain band for domain."""

        return self._band(
            distance.domain_distance,
            self.accept_domain_distance,
            self.abstain_domain_distance,
        )

    def band(self, distance: VarietyDistance) -> str:
        """Return the stricter of the dialect and domain bands."""

        return max(
            (self.dialect_band(distance), self.domain_band(distance)),
            key=_BAND_RANK.__getitem__,
        )


@dataclass(frozen=True, slots=True)
class DialectDecision:
    """Routing decision that carries its limitations into downstream results.

    Attributes:
        action: One of :data:`DIALECT_ACTIONS`.
        selected: Variety identifiers that must run; empty when abstaining.
        task: Task code the decision covers.
        nearest_variety: Nearest validated variety, if any qualified.
        dialect_distance: Dialect distance of the nearest variety.
        domain_distance: Domain distance of the nearest variety.
        candidates: Distances to every eligible variety, nearest first.
        limitations: Sorted codes from :data:`DIALECT_LIMITATIONS`.
        profile: Count, share and code summary of the input features.
        policy_id: Identifier of the declared policy.
        policy_digest: SHA-256 digest of the declared policy.
        disclaimer: Fixed advisory text.
    """

    action: str
    selected: tuple[str, ...]
    task: str
    nearest_variety: str | None
    dialect_distance: float | None
    domain_distance: float | None
    candidates: tuple[VarietyDistance, ...]
    limitations: tuple[str, ...]
    profile: Mapping[str, Any] = field(repr=False, hash=False)
    policy_id: str = "default"
    policy_digest: str = ""
    disclaimer: str = DIALECT_ABSTENTION_ADVISORY

    @property
    def abstained(self) -> bool:
        """Return whether automated output must be withheld."""

        return self.action == "abstain"

    @property
    def requires_review(self) -> bool:
        """Return whether a human should review the routed output."""

        return self.action != "accept" or bool(
            _REVIEW_LIMITATIONS.intersection(self.limitations)
        )

    def to_dict(self) -> dict[str, Any]:
        """Return codes, distances and counts; never input text."""

        return {
            "schema_version": SCHEMA_VERSION,
            "action": self.action,
            "selected": list(self.selected),
            "task": self.task,
            "abstained": self.abstained,
            "requires_review": self.requires_review,
            "nearest_variety": self.nearest_variety,
            "dialect_distance": self.dialect_distance,
            "domain_distance": self.domain_distance,
            "limitations": list(self.limitations),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "profile": deepcopy(dict(self.profile)),
            "policy_id": self.policy_id,
            "policy_digest": self.policy_digest,
            "disclaimer": self.disclaimer,
        }

    def to_json(self) -> str:
        """Serialize the decision as deterministic JSON."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def to_metadata(self) -> dict[str, Any]:
        """Return a ``metadata`` entry that surfaces the decision in results."""

        return {"dialect_assessment": self.to_dict()}


def _validate_varieties(varieties: object) -> tuple[ValidatedVariety, ...]:
    if isinstance(varieties, (str, bytes, Mapping)) or not isinstance(
        varieties, Iterable
    ):
        raise ValueError("varieties must be a collection of ValidatedVariety")
    items = tuple(islice(varieties, MAX_VARIETIES + 1))
    if len(items) > MAX_VARIETIES or any(
        not isinstance(item, ValidatedVariety) for item in items
    ):
        raise ValueError("varieties must be a collection of ValidatedVariety")
    if len({item.variety_id for item in items}) != len(items):
        raise ValueError("duplicate validated variety identifier")
    return items


def apply_dialect_policy(
    profile: DialectFeatureProfile,
    varieties: Iterable[ValidatedVariety],
    *,
    task: str,
    policy: DialectPolicy | None = None,
    registered_packs: Collection[str] | None = None,
) -> DialectDecision:
    """Apply a declared policy to precomputed input features.

    Args:
        profile: Input features from :func:`extract_dialect_features`.
        varieties: Validated varieties to compare against.
        task: Controlled task code, for example ``"deidentification"``.
        policy: Declared policy. The default abstains outside the accept band.
        registered_packs: Optional language-pack codes that can execute. A
            variety naming another pack is not eligible.

    Returns:
        A :class:`DialectDecision` that names the varieties to run, or
        abstains, and lists every limitation.
    """

    if not isinstance(profile, DialectFeatureProfile):
        raise TypeError("profile must be a DialectFeatureProfile")
    task = _token(task, _TOKEN, "task must be a code")
    policy = DialectPolicy() if policy is None else policy
    if not isinstance(policy, DialectPolicy):
        raise TypeError("policy must be a DialectPolicy")
    items = _validate_varieties(varieties)
    if isinstance(registered_packs, (str, bytes)):
        raise TypeError("registered_packs must be a collection of pack codes")
    packs = None if registered_packs is None else frozenset(registered_packs)

    limitations: set[str] = set()
    if profile.declared_locale is None:
        limitations.add("locale_undeclared")
    if not profile.routing_decisive:
        limitations.add("language_unconfirmed")
    elif profile.declared_locale is not None and (
        _locale_parts(profile.declared_locale)[0] != profile.routed_language
    ):
        limitations.add("declared_locale_conflict")
    if profile.mixed_language:
        limitations.add("mixed_language")
    if profile.unsupported_script:
        limitations.add("unsupported_script")
    if profile.declared_domain is None:
        limitations.add("domain_undeclared")

    eligible = []
    for variety in items:
        if task not in variety.tasks:
            continue
        if (
            packs is not None
            and variety.language_pack is not None
            and variety.language_pack not in packs
        ):
            limitations.add("variety_pack_unregistered")
            continue
        eligible.append(variety)

    distances = sorted(
        (estimate_variety_distance(profile, variety) for variety in eligible),
        key=lambda item: (
            _BAND_RANK[policy.band(item)],
            item.dialect_distance,
            item.domain_distance,
            item.variety_id,
        ),
    )
    by_id = {item.variety_id: item for item in distances}
    nearest = distances[0] if distances else None
    action = "abstain"
    selected: tuple[str, ...] = ()

    def fallback(check_distance: bool) -> tuple[str, tuple[str, ...]]:
        target = by_id.get(policy.fallback_variety or "")
        if target is None:
            limitations.add("fallback_unavailable")
            return "abstain", ()
        if check_distance and policy.band(target) == "abstain":
            limitations.add("fallback_out_of_range")
            return "abstain", ()
        limitations.add("fallback_route")
        return "fallback", (target.variety_id,)

    if nearest is None:
        limitations.add("no_validated_variety")
    elif profile.letter_count < policy.min_letters:
        limitations.add("insufficient_evidence")
        if policy.insufficient_evidence_action == "fallback":
            action, selected = fallback(check_distance=False)
    else:
        dialect_band = policy.dialect_band(nearest)
        domain_band = policy.domain_band(nearest)
        if dialect_band == "intermediate":
            limitations.add("dialect_shift")
        elif dialect_band == "abstain":
            limitations.add("dialect_out_of_range")
        if domain_band == "intermediate":
            limitations.add("domain_shift")
        elif domain_band == "abstain":
            limitations.add("domain_out_of_range")
        if nearest.declared_domain_distance == 1.0:
            limitations.add("domain_unvalidated")

        band = policy.band(nearest)
        if band == "accept":
            action, selected = "accept", (nearest.variety_id,)
        elif band == "intermediate":
            if policy.intermediate_action == "fallback":
                action, selected = fallback(check_distance=True)
            elif policy.intermediate_action == "ensemble":
                members = tuple(
                    item.variety_id
                    for item in distances
                    if policy.band(item) != "abstain"
                )[: policy.ensemble_size]
                if len(members) < 2:
                    limitations.add("ensemble_insufficient_members")
                else:
                    limitations.add("ensemble_route")
                    action, selected = "ensemble", members

    return DialectDecision(
        action=action,
        selected=selected,
        task=task,
        nearest_variety=None if nearest is None else nearest.variety_id,
        dialect_distance=None if nearest is None else nearest.dialect_distance,
        domain_distance=None if nearest is None else nearest.domain_distance,
        candidates=tuple(distances),
        limitations=tuple(sorted(limitations)),
        profile=MappingProxyType(profile.summary()),
        policy_id=policy.policy_id,
        policy_digest=policy.digest,
    )


def assess_dialect(
    text: str,
    varieties: Iterable[ValidatedVariety],
    *,
    task: str,
    policy: DialectPolicy | None = None,
    declared_locale: str | None = None,
    declared_domain: str | None = None,
    router: LanguageRouter | None = None,
) -> DialectDecision:
    """Route ``text`` against validated varieties under a declared policy.

    Example:
        >>> decision = assess_dialect(
        ...     note, varieties, task="deidentification",
        ...     declared_locale="de-CH", declared_domain="clinical_note",
        ...     policy=DialectPolicy(intermediate_action="ensemble"),
        ... )  # doctest: +SKIP
        >>> decision.action, decision.limitations  # doctest: +SKIP
        ('ensemble', ('dialect_shift', 'ensemble_route'))

    Args:
        text: Input text. It is never stored, logged or echoed.
        varieties: Validated varieties to compare against.
        task: Controlled task code, for example ``"deidentification"``.
        policy: Declared policy. The default abstains outside the accept band.
        declared_locale: Optional BCP 47 locale declared by the caller.
        declared_domain: Optional controlled document-domain code.
        router: Deterministic language router. Its registered packs decide
            which varieties can execute.

    Returns:
        A :class:`DialectDecision` with value-free limitations.
    """

    resolved_router = _require_router(router)
    profile = extract_dialect_features(
        text,
        declared_locale=declared_locale,
        declared_domain=declared_domain,
        router=resolved_router,
    )
    return apply_dialect_policy(
        profile,
        varieties,
        task=task,
        policy=policy,
        registered_packs=tuple(pack.code for pack in resolved_router.packs),
    )


def enforce_dialect_decision(
    decision: DialectDecision,
    executed: Iterable[str],
) -> None:
    """Fail closed unless execution matches the decision exactly.

    Args:
        decision: Decision returned by :func:`assess_dialect`.
        executed: Variety identifiers whose models actually ran.

    Raises:
        DialectRouteError: ``"dialect_abstained"`` when the decision abstained,
            or ``"dialect_route_mismatch"`` when the executed set differs from
            the required set, for example a skipped ensemble member.
    """

    if not isinstance(decision, DialectDecision):
        raise TypeError("decision must be a DialectDecision")
    if isinstance(executed, (str, bytes)) or not isinstance(executed, Iterable):
        raise TypeError("executed must be a collection of variety identifiers")
    if decision.abstained:
        raise DialectRouteError("dialect_abstained")
    ran = tuple(islice(executed, MAX_VARIETIES + 1))
    if len(set(ran)) != len(ran) or set(ran) != set(decision.selected):
        raise DialectRouteError("dialect_route_mismatch")
