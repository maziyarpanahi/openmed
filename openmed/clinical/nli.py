"""Local-first natural-language-inference checks for clinical claims.

The public :func:`nli` entry point is deliberately backend-neutral.  The
default backend is a deterministic lexical heuristic suitable for offline
development and synthetic fixtures; a caller-supplied MLX head can implement
the same ``predict(premise, hypothesis)`` contract without changing callers.

The :func:`verify` helper is the small hook consumed by future summarization or
grounding stages when they expose ``verify=True``.  It evaluates every claim,
retains the original claim in the result, and adds an explicit
``contradicted`` flag.  Verification is assistive review metadata, not a
clinical decision.

MedNLI is not bundled.  It is DUA-gated and eval-only; the BigBio mirror is
represented by the repository's gated stub and must be supplied separately by
an authorized evaluator.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Literal, Protocol, TypedDict, runtime_checkable

NLI_LABELS = ("entailment", "contradiction", "neutral")
NliLabel = Literal["entailment", "contradiction", "neutral"]

NLI_ADVISORY = (
    "Clinical NLI verification is assistive grounding evidence for human "
    "review, not a diagnosis, treatment decision, or autonomous clinical "
    "judgment."
)
MEDNLI_DATA_POLICY = (
    "MedNLI is DUA-gated and eval-only. The BigBio mirror is a gated stub; "
    "no MedNLI data or model is bundled or downloaded by OpenMed."
)


class NLIResult(TypedDict):
    """The stable two-field result returned by :func:`nli`."""

    label: NliLabel
    score: float


class VerificationResult(TypedDict):
    """One claim result returned by :func:`verify`.

    ``claim`` and ``source`` retain the caller's original values so an
    audit/review layer can preserve its own span or provenance object.  The
    verifier never drops a claim merely because it is contradicted.
    """

    claim: Any
    source: Any
    label: NliLabel
    score: float
    contradicted: bool


@runtime_checkable
class NLIBackend(Protocol):
    """Contract implemented by a clinical NLI backend.

    A trained MLX head or another local model can implement this protocol.  A
    backend must return a mapping with a canonical ``label`` and a finite
    ``score`` in ``[0, 1]``; :func:`nli` validates and normalizes the result.
    """

    def predict(self, premise: str, hypothesis: str) -> Mapping[str, Any]:
        """Classify one premise/hypothesis pair."""


NLIBackendLike = NLIBackend | Callable[[str, str], Mapping[str, Any]]


class HeuristicNLIBackend:
    """Deterministic, dependency-free NLI backend for local operation.

    This backend is intentionally conservative.  It recognizes lexical
    containment, explicit negation, and a small set of common clinical
    opposites; unsupported inferences are returned as ``neutral``.
    """

    def predict(self, premise: str, hypothesis: str) -> NLIResult:
        """Return a deterministic three-way classification."""

        return _heuristic_prediction(premise, hypothesis)


HEURISTIC_NLI_BACKEND = HeuristicNLIBackend()
DEFAULT_NLI_BACKEND: NLIBackendLike = HEURISTIC_NLI_BACKEND


def get_default_backend() -> NLIBackendLike:
    """Return the process-wide backend used when none is passed to :func:`nli`."""

    return DEFAULT_NLI_BACKEND


def set_default_backend(backend: NLIBackendLike) -> None:
    """Replace the default backend used by :func:`nli`.

    Dependency injection through the ``backend=`` argument is preferred for
    request-scoped or concurrent applications.  This setter is provided for a
    process that installs one local model at startup.
    """

    _validate_backend(backend)
    global DEFAULT_NLI_BACKEND
    DEFAULT_NLI_BACKEND = backend


def nli(
    premise: str,
    hypothesis: str,
    *,
    backend: NLIBackendLike | None = None,
) -> NLIResult:
    """Classify a premise and hypothesis using a swappable NLI backend.

    Args:
        premise: Source span or other evidence text.
        hypothesis: Generated or grounded claim to check.
        backend: Optional backend implementing :class:`NLIBackend` or a
            two-argument callable.  The deterministic heuristic is used by
            default.

    Returns:
        A JSON-compatible mapping with exactly ``label`` and ``score`` keys.
        ``label`` is one of ``entailment``, ``contradiction``, or ``neutral``;
        ``score`` is a finite confidence in ``[0, 1]``.

    Raises:
        TypeError: If either text is not a string or the backend is invalid.
        ValueError: If either text is empty or the backend returns an invalid
            label or score.
    """

    premise = _required_text(premise, "premise")
    hypothesis = _required_text(hypothesis, "hypothesis")
    selected_backend = DEFAULT_NLI_BACKEND if backend is None else backend
    _validate_backend(selected_backend)
    raw_result = _call_backend(selected_backend, premise, hypothesis)
    return _normalize_result(raw_result)


def verify(
    claims: Iterable[Any] | Any,
    source: Any,
    *,
    backend: NLIBackendLike | None = None,
) -> list[VerificationResult]:
    """Verify claims against source text or source spans.

    A string source is reused for every claim.  A sequence of source spans is
    paired positionally when it has one item per claim, or its single item is
    reused for all claims.  Claim and source records may be strings, mappings,
    or objects exposing ``text``; mappings may use ``claim``/``hypothesis`` or
    ``source``/``evidence`` aliases.  A claim may also be a ``(source, claim)``
    pair, which is useful when a caller already has aligned spans.

    Each result preserves the claim and source values and contains the NLI
    label, score, and an explicit ``contradicted`` flag.  This is intentionally
    suitable for a caller's optional ``verify=True`` stage: contradicted
    claims remain visible for reviewer or audit handling.

    Args:
        claims: One claim or an iterable of claim records.
        source: Shared source text, one source span, or aligned source spans.
        backend: Optional backend forwarded to :func:`nli`.

    Returns:
        One verification mapping per input claim, in input order.

    Raises:
        TypeError: If claims, source, or a record cannot provide text.
        ValueError: If aligned source spans do not match the claims.
    """

    claim_items = _claim_items(claims)
    if not claim_items:
        return []

    source_items = _source_items(source)
    if not source_items:
        raise ValueError("source must contain at least one text span")
    aligned_sources = _align_sources(source_items, len(claim_items))

    results: list[VerificationResult] = []
    for index, (raw_claim, fallback_source) in enumerate(
        zip(claim_items, aligned_sources, strict=True)
    ):
        claim_value, claim_text, claim_source = _claim_parts(raw_claim)
        source_value = fallback_source if claim_source is None else claim_source
        source_text = _text_from_record(source_value, "source")
        result = nli(source_text, claim_text, backend=backend)
        results.append(
            {
                "claim": claim_value,
                "source": source_value,
                "label": result["label"],
                "score": result["score"],
                "contradicted": result["label"] == "contradiction",
            }
        )
    return results


def _call_backend(
    backend: NLIBackendLike,
    premise: str,
    hypothesis: str,
) -> Mapping[str, Any]:
    predictor = getattr(backend, "predict", None)
    if callable(predictor):
        result = predictor(premise, hypothesis)
    elif callable(backend):
        result = backend(premise, hypothesis)
    else:  # pragma: no cover - guarded by _validate_backend
        raise TypeError("NLI backend must implement predict or be callable")
    if not isinstance(result, Mapping):
        raise TypeError("NLI backend must return a mapping")
    return result


def _normalize_result(result: Mapping[str, Any]) -> NLIResult:
    label = result.get("label")
    if not isinstance(label, str):
        raise TypeError("NLI backend result label must be a string")
    normalized_label = label.strip().casefold()
    if normalized_label not in NLI_LABELS:
        allowed = ", ".join(NLI_LABELS)
        raise ValueError(f"NLI backend returned {label!r}; expected {allowed}")

    score = result.get("score")
    if isinstance(score, bool) or not isinstance(score, int | float):
        raise TypeError("NLI backend result score must be a number")
    normalized_score = float(score)
    if not math.isfinite(normalized_score) or not 0.0 <= normalized_score <= 1.0:
        raise ValueError("NLI backend result score must be finite and in [0, 1]")
    return {
        "label": normalized_label,  # type: ignore[typeddict-item]
        "score": normalized_score,
    }


def _validate_backend(backend: object) -> None:
    if not callable(getattr(backend, "predict", None)) and not callable(backend):
        raise TypeError("NLI backend must implement predict or be callable")


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


def _heuristic_prediction(premise: str, hypothesis: str) -> NLIResult:
    _required_text(premise, "premise")
    _required_text(hypothesis, "hypothesis")
    premise_normalized = _normalize_text(premise)
    hypothesis_normalized = _normalize_text(hypothesis)
    if premise_normalized == hypothesis_normalized:
        return {"label": "entailment", "score": 1.0}

    candidates = [
        _classify_pair(sentence, hypothesis)
        for sentence in _sentence_candidates(premise)
    ]
    contradictions = [
        result for result in candidates if result["label"] == "contradiction"
    ]
    if contradictions:
        return max(contradictions, key=lambda result: result["score"])
    entailments = [result for result in candidates if result["label"] == "entailment"]
    if entailments:
        return max(entailments, key=lambda result: result["score"])
    return {"label": "neutral", "score": 0.5}


def _classify_pair(premise: str, hypothesis: str) -> NLIResult:
    premise_tokens = _content_tokens(premise)
    hypothesis_tokens = _content_tokens(hypothesis)
    if not premise_tokens or not hypothesis_tokens:
        return {"label": "neutral", "score": 0.5}

    if _has_opposite_terms(premise_tokens, hypothesis_tokens):
        return {"label": "contradiction", "score": 0.95}

    shared = premise_tokens & hypothesis_tokens
    if not shared:
        return {"label": "neutral", "score": 0.5}

    premise_polarity = _polarity(premise)
    hypothesis_polarity = _polarity(hypothesis)
    if (
        premise_polarity != 0
        and hypothesis_polarity != 0
        and premise_polarity != hypothesis_polarity
    ):
        return {"label": "contradiction", "score": 0.96}

    hypothesis_coverage = len(shared) / len(hypothesis_tokens)
    if hypothesis_tokens <= premise_tokens:
        return {
            "label": "entailment",
            "score": min(0.99, 0.9 + 0.09 * hypothesis_coverage),
        }
    if hypothesis_coverage >= 0.8:
        return {"label": "entailment", "score": 0.86}
    return {"label": "neutral", "score": 0.5}


def _sentence_candidates(text: str) -> tuple[str, ...]:
    parts = re.split(r"(?<=[.!?;])\s+|\n+", text)
    candidates = tuple(part.strip() for part in parts if part.strip())
    return candidates or (text.strip(),)


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def _tokens(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(re.findall(r"[^\W_]+", normalized, flags=re.UNICODE))


_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "might",
        "of",
        "on",
        "or",
        "patient",
        "person",
        "should",
        "subject",
        "than",
        "that",
        "the",
        "their",
        "this",
        "those",
        "to",
        "under",
        "was",
        "were",
        "with",
        "would",
    }
)
_NEGATION_WORDS = frozenset(
    {
        "absent",
        "denied",
        "denies",
        "deny",
        "free",
        "lack",
        "lacks",
        "negative",
        "neither",
        "never",
        "no",
        "none",
        "nor",
        "not",
        "ruled",
        "without",
    }
)
_OPPOSITES = {
    "abnormal": "normal",
    "absent": "present",
    "decreased": "increased",
    "decrease": "increase",
    "declined": "improved",
    "declining": "improving",
    "discontinued": "continued",
    "discontinue": "continue",
    "dropped": "rose",
    "failed": "passed",
    "high": "low",
    "improved": "declined",
    "improving": "declining",
    "increased": "decreased",
    "increase": "decrease",
    "low": "high",
    "negative": "positive",
    "normal": "abnormal",
    "pass": "fail",
    "passed": "failed",
    "positive": "negative",
    "present": "absent",
    "rose": "dropped",
    "stable": "unstable",
    "stopped": "continued",
    "stop": "continue",
    "unstable": "stable",
    "worsened": "improved",
    "worsening": "improving",
}


def _content_tokens(text: str) -> frozenset[str]:
    result: set[str] = set()
    for token in _tokens(text):
        if token in _STOPWORDS or token in _NEGATION_WORDS:
            continue
        result.add(_singularize(token))
    return frozenset(result)


def _singularize(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _has_opposite_terms(
    premise_tokens: frozenset[str], hypothesis_tokens: frozenset[str]
) -> bool:
    return any(
        _singularize(_OPPOSITES.get(token, "")) in hypothesis_tokens
        for token in premise_tokens
        if token in _OPPOSITES
    )


_NEGATION_PATTERN = re.compile(
    r"\b(?:absent|den(?:y|ies|ied)|free\s+of|lack(?:s|ing)?|negative\s+for|"
    r"neither|never|no|none|nor|not|ruled\s+out|without)\b",
    flags=re.IGNORECASE,
)


def _polarity(text: str) -> int:
    return -1 if _NEGATION_PATTERN.search(text) else 1


def _claim_items(claims: Iterable[Any] | Any) -> tuple[Any, ...]:
    if _record_text(claims, ("claim", "hypothesis", "text", "content")):
        return (claims,)
    if isinstance(claims, (str, bytes)):
        return (claims,)
    try:
        return tuple(claims)
    except TypeError as exc:
        raise TypeError("claims must be a claim or iterable of claims") from exc


def _source_items(source: Any) -> tuple[Any, ...]:
    if _record_text(source, ("text", "source", "evidence", "content")):
        return (source,)
    if isinstance(source, (str, bytes)):
        return (source,)
    try:
        return tuple(source)
    except TypeError as exc:
        raise TypeError("source must be text, a span, or an iterable of spans") from exc


def _align_sources(source_items: tuple[Any, ...], claim_count: int) -> tuple[Any, ...]:
    if len(source_items) == 1:
        return source_items * claim_count
    if len(source_items) == claim_count:
        return source_items
    raise ValueError("source spans must contain one item or exactly one item per claim")


def _claim_parts(raw_claim: Any) -> tuple[Any, str, Any | None]:
    if _is_pair(raw_claim):
        local_source, claim = raw_claim
        return claim, _text_from_record(claim, "claim"), local_source
    claim_text = _text_from_record(raw_claim, "claim")
    local_source = _record_value(raw_claim, ("source", "source_span", "evidence"))
    claim_value = raw_claim
    if isinstance(raw_claim, Mapping):
        claim_value = raw_claim
    return claim_value, claim_text, local_source


def _is_pair(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
    )


def _text_from_record(value: Any, field_name: str) -> str:
    if isinstance(value, str):
        return _required_text(value, field_name)
    text = _record_value(
        value,
        (
            "text",
            field_name,
            "claim" if field_name == "claim" else "hypothesis",
            "source" if field_name == "source" else "evidence",
            "content",
            "surface",
            "value",
        ),
    )
    return _required_text(text, field_name)


def _record_text(value: Any, fields: Sequence[str]) -> bool:
    return _record_value(value, fields) is not None


def _record_value(value: Any, fields: Sequence[str]) -> Any | None:
    if isinstance(value, Mapping):
        for field in fields:
            if field in value and value[field] is not None:
                return value[field]
        return None
    if isinstance(value, (str, bytes)):
        return None
    for field in fields:
        candidate = getattr(value, field, None)
        if candidate is not None:
            return candidate
    return None


__all__ = [
    "DEFAULT_NLI_BACKEND",
    "HEURISTIC_NLI_BACKEND",
    "MEDNLI_DATA_POLICY",
    "NLI_ADVISORY",
    "NLI_LABELS",
    "NLIBackend",
    "NLIResult",
    "HeuristicNLIBackend",
    "VerificationResult",
    "get_default_backend",
    "nli",
    "set_default_backend",
    "verify",
]
