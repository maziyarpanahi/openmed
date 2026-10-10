"""Local-first natural-language-inference checks for clinical claims.

The public :func:`nli` entry point is backend-neutral. Its default selects a
released local sequence classifier and fails closed until one is registered.
The deterministic lexical heuristic remains an explicit development option.

The :func:`verify` helper evaluates every claim and returns value-free label,
score, backend, and review metadata. Verification is assistive review evidence,
not a clinical decision.

MedNLI is not bundled.  It is DUA-gated and eval-only; the BigBio mirror is
represented by the repository's gated stub and must be supplied separately by
an authorized evaluator.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypedDict, runtime_checkable

NLI_LABELS = ("entailment", "contradiction", "neutral", "abstention")
NliLabel = Literal["entailment", "contradiction", "neutral", "abstention"]

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
    """A value-free four-state result returned by :func:`nli`."""

    label: NliLabel
    score: float
    backend_id: str


class _RawNLIResult(TypedDict):
    label: NliLabel
    score: float


class VerificationResult(TypedDict):
    """One value-free claim result returned by :func:`verify`."""

    claim_index: int
    label: NliLabel
    score: float
    backend_id: str
    contradicted: bool
    review_required: bool


@dataclass(frozen=True)
class ClaimVerification:
    """Value-free NLI evidence attached to an opt-in public pipeline result.

    Offsets are half-open character ranges. Source offsets refer to the text
    passed to the verification stage; summary claim offsets refer to the
    returned summary. Grounded concept displays need no output-text offsets.
    """

    claim_index: int
    label: NliLabel
    score: float
    backend_id: str
    source_offset: tuple[int, int] | None
    claim_offset: tuple[int, int] | None
    source_digest: str | None
    claim_digest: str | None
    review_required: bool

    def __post_init__(self) -> None:
        valid = (
            type(self.claim_index) is int
            and self.claim_index >= 0
            and self.label in NLI_LABELS
            and type(self.score) in {int, float}
            and 0 <= self.score <= 1
            and math.isfinite(self.score)
            and isinstance(self.backend_id, str)
            and re.fullmatch(r"[a-z][a-z0-9-]{0,63}", self.backend_id) is not None
            and type(self.review_required) is bool
            and (self.label == "entailment" or self.review_required)
        )
        for offset in (self.source_offset, self.claim_offset):
            valid = valid and (
                offset is None
                or (
                    type(offset) is tuple
                    and len(offset) == 2
                    and all(type(value) is int for value in offset)
                    and 0 <= offset[0] < offset[1]
                )
            )
        for digest in (self.source_digest, self.claim_digest):
            valid = valid and (
                digest is None
                or (
                    isinstance(digest, str)
                    and re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is not None
                )
            )
        valid = valid and ((self.source_offset is None) == (self.source_digest is None))
        valid = valid and (self.claim_offset is None or self.claim_digest is not None)
        valid = valid and (
            self.label == "abstention"
            or (self.source_digest is not None and self.claim_digest is not None)
        )
        if not valid:
            raise ValueError("invalid claim verification metadata")
        object.__setattr__(self, "score", float(self.score))

    @property
    def contradicted(self) -> bool:
        """Return whether NLI contradicted the retained claim."""
        return self.label == "contradiction"

    def to_dict(self) -> dict[str, Any]:
        """Return labels, scores, offsets, digests and controlled review metadata."""
        return {
            "claim_index": self.claim_index,
            "label": self.label,
            "score": self.score,
            "backend_id": self.backend_id,
            "source_offset": list(self.source_offset)
            if self.source_offset is not None
            else None,
            "claim_offset": list(self.claim_offset)
            if self.claim_offset is not None
            else None,
            "source_digest": self.source_digest,
            "claim_digest": self.claim_digest,
            "contradicted": self.contradicted,
            "review_required": self.review_required,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ClaimVerification":
        """Reconstruct a closed, value-free verification record."""
        fields = {
            "claim_index",
            "label",
            "score",
            "backend_id",
            "source_offset",
            "claim_offset",
            "source_digest",
            "claim_digest",
            "contradicted",
            "review_required",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("invalid claim verification metadata")
        offsets = {}
        for name in ("source_offset", "claim_offset"):
            offset = value[name]
            if offset is not None and not isinstance(offset, (list, tuple)):
                raise ValueError("invalid claim verification metadata")
            offsets[name] = tuple(offset) if offset is not None else None
        result = cls(
            claim_index=value["claim_index"],
            label=value["label"],
            score=value["score"],
            backend_id=value["backend_id"],
            source_offset=offsets["source_offset"],
            claim_offset=offsets["claim_offset"],
            source_digest=value["source_digest"],
            claim_digest=value["claim_digest"],
            review_required=value["review_required"],
        )
        if (
            type(value["contradicted"]) is not bool
            or value["contradicted"] != result.contradicted
        ):
            raise ValueError("invalid claim verification metadata")
        return result


def _verification_digest(text: str) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            b"openmed-claim-verification-v1\x00" + text.encode("utf-8")
        ).hexdigest()
    )


def _verification_refusal(error: Exception, fallback: str) -> Exception:
    """Preserve typed local refusals without forwarding arbitrary provider text."""
    from .nli_backends import LocalNLIError, RemoteNLIBackendError

    if isinstance(error, RemoteNLIBackendError):
        return RemoteNLIBackendError("remote NLI backends are prohibited")
    safe_messages = {
        "NLI backend must be local or callable",
        "no released local NLI checkpoint is registered",
        "NLI model alias is not registered",
        "NLI checkpoint release metadata is incomplete",
        "NLI checkpoint calibration is invalid",
        "local NLI checkpoint is unavailable",
        "local NLI inference failed",
    }
    message = str(error) if type(error) is LocalNLIError else ""
    return LocalNLIError(message if message in safe_messages else fallback)


def _verify_claim_spans(
    claims: Sequence[str | None],
    source: str | None,
    source_offsets: Sequence[tuple[int, int] | None],
    *,
    option: object,
    claim_offsets: Sequence[tuple[int, int] | None] | None = None,
    review_flags: Sequence[bool] | None = None,
) -> tuple[ClaimVerification, ...]:
    """Compose existing NLI over exact source slices, retaining unresolved claims."""
    from openmed.core.offline import network_blocked_if_offline

    from .nli_backends import LocalNLIError, resolve_nli_backend

    if len(claims) != len(source_offsets):
        raise LocalNLIError("verification span alignment is invalid")
    outputs = (
        tuple(claim_offsets) if claim_offsets is not None else (None,) * len(claims)
    )
    reviews = (
        tuple(review_flags) if review_flags is not None else (False,) * len(claims)
    )
    if len(outputs) != len(claims) or len(reviews) != len(claims):
        raise LocalNLIError("verification span alignment is invalid")
    try:
        backend = resolve_nli_backend(
            get_default_backend() if option is True else option
        )
    except Exception as error:
        raise _verification_refusal(
            error, "local NLI backend resolution failed"
        ) from None
    results: list[ClaimVerification] = []
    for index, (claim, offset, output, review) in enumerate(
        zip(claims, source_offsets, outputs, reviews, strict=True)
    ):
        premise = None
        if source is not None and offset is not None:
            start, end = offset
            if (
                type(start) is int
                and type(end) is int
                and 0 <= start < end <= len(source)
            ):
                premise = source[start:end]
        usable = bool(premise and premise.strip() and claim and claim.strip())
        if usable:
            # Always block network for NLI, including caller-owned backends.
            try:
                with network_blocked_if_offline(local_only=True):
                    decision = verify([claim], premise, backend=backend)[0]
            except Exception as error:
                raise _verification_refusal(
                    error, "local NLI inference failed"
                ) from None
            label, score, backend_id = (
                decision["label"],
                decision["score"],
                decision["backend_id"],
            )
        else:
            label, score, backend_id = "abstention", 0.0, "unresolved-span"
        results.append(
            ClaimVerification(
                claim_index=index,
                label=label,
                score=score,
                backend_id=backend_id,
                source_offset=offset if usable else None,
                claim_offset=output if claim else None,
                source_digest=_verification_digest(premise) if usable else None,
                claim_digest=_verification_digest(claim) if claim else None,
                review_required=review or label != "entailment",
            )
        )
    return tuple(results)


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

    backend_id = "heuristic"

    def predict(self, premise: str, hypothesis: str) -> _RawNLIResult:
        """Return a deterministic three-way classification."""

        return _heuristic_prediction(premise, hypothesis)


HEURISTIC_NLI_BACKEND = HeuristicNLIBackend()
DEFAULT_NLI_BACKEND: NLIBackendLike | str = "local"


def get_default_backend() -> NLIBackendLike | str:
    """Return the process-wide backend used when none is passed to :func:`nli`."""

    return DEFAULT_NLI_BACKEND


def set_default_backend(backend: NLIBackendLike | str) -> None:
    """Replace the default backend used by :func:`nli`.

    Dependency injection through the ``backend=`` argument is preferred for
    request-scoped or concurrent applications.  This setter is provided for a
    process that installs one local model at startup.
    """

    if not isinstance(backend, str):
        _validate_backend(backend)
    global DEFAULT_NLI_BACKEND
    DEFAULT_NLI_BACKEND = backend


def nli(
    premise: str,
    hypothesis: str,
    *,
    backend: NLIBackendLike | str | None = None,
) -> NLIResult:
    """Classify a premise and hypothesis using a swappable NLI backend.

    Args:
        premise: Source span or other evidence text.
        hypothesis: Generated or grounded claim to check.
        backend: A local registry alias, ``"local"``, ``"heuristic"``, or a
            local backend implementing :class:`NLIBackend`.

    Returns:
        A value-free mapping with ``label``, ``score``, and ``backend_id``.
        ``label`` has four states, including explicit ``abstention``.

    Raises:
        TypeError: If either text is not a string or the backend is invalid.
        ValueError: If either text is empty or the backend returns an invalid
            label or score.
    """

    premise = _required_text(premise, "premise")
    hypothesis = _required_text(hypothesis, "hypothesis")
    selected_backend = DEFAULT_NLI_BACKEND if backend is None else backend
    from .nli_backends import resolve_nli_backend

    selected_backend = resolve_nli_backend(selected_backend)
    _validate_backend(selected_backend)
    raw_result = _call_backend(selected_backend, premise, hypothesis)
    backend_id = getattr(selected_backend, "backend_id", "custom-local")
    return _normalize_result(raw_result, backend_id)


def verify(
    claims: Iterable[Any] | Any,
    source: Any,
    *,
    backend: NLIBackendLike | str | None = None,
) -> list[VerificationResult]:
    """Verify claims against source text or source spans.

    A string source is reused for every claim.  A sequence of source spans is
    paired positionally when it has one item per claim, or its single item is
    reused for all claims.  Claim and source records may be strings, mappings,
    or objects exposing ``text``; mappings may use ``claim``/``hypothesis`` or
    ``source``/``evidence`` aliases.  A claim may also be a ``(source, claim)``
    pair, which is useful when a caller already has aligned spans.

    Each result contains only an index, label, score, backend id, and review
    flags. Contradicted claims remain visible by their index without retaining
    source or claim text in the returned metadata.

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
    selected_backend: NLIBackendLike | str | None = None
    for index, (raw_claim, fallback_source) in enumerate(
        zip(claim_items, aligned_sources, strict=True)
    ):
        claim_value, claim_text, claim_source = _claim_parts(raw_claim)
        source_value = fallback_source if claim_source is None else claim_source
        source_text = _text_from_record(source_value, "source")
        result = _structured_precheck(source_value, claim_value)
        if result is None:
            if selected_backend is None:
                from .nli_backends import resolve_nli_backend

                selected_backend = resolve_nli_backend(
                    DEFAULT_NLI_BACKEND if backend is None else backend
                )
            result = nli(source_text, claim_text, backend=selected_backend)
        results.append(
            {
                "claim_index": index,
                "label": result["label"],
                "score": result["score"],
                "backend_id": result["backend_id"],
                "contradicted": result["label"] == "contradiction",
                "review_required": result["label"] == "abstention",
            }
        )
    return results


def _call_backend(
    backend: NLIBackendLike,
    premise: str,
    hypothesis: str,
) -> Mapping[str, Any]:
    predictor = getattr(backend, "predict", None)
    try:
        if callable(predictor):
            result = predictor(premise, hypothesis)
        elif callable(backend):
            result = backend(premise, hypothesis)
        else:  # pragma: no cover - guarded by _validate_backend
            raise TypeError("NLI backend must implement predict or be callable")
    except Exception:
        from .nli_backends import LocalNLIError

        raise LocalNLIError("local NLI inference failed") from None
    if not isinstance(result, Mapping):
        raise TypeError("NLI backend must return a mapping")
    return result


def _normalize_result(result: Mapping[str, Any], backend_id: object) -> NLIResult:
    label = result.get("label")
    if not isinstance(label, str):
        raise TypeError("NLI backend result label must be a string")
    normalized_label = label.strip().casefold()
    if normalized_label not in NLI_LABELS:
        allowed = ", ".join(NLI_LABELS)
        raise ValueError(f"NLI backend returned an invalid label; expected {allowed}")

    score = result.get("score")
    if isinstance(score, bool) or not isinstance(score, int | float):
        raise TypeError("NLI backend result score must be a number")
    if not 0.0 <= score <= 1.0:
        raise ValueError("NLI backend result score must be finite and in [0, 1]")
    normalized_score = float(score)
    if not math.isfinite(normalized_score):
        raise ValueError("NLI backend result score must be finite and in [0, 1]")
    if (
        not isinstance(backend_id, str)
        or re.fullmatch(r"[a-z][a-z0-9-]{0,63}", backend_id) is None
    ):
        raise ValueError("NLI backend id must be a short safe token")
    return {
        "label": normalized_label,  # type: ignore[typeddict-item]
        "score": normalized_score,
        "backend_id": backend_id,
    }


def _structured_precheck(source: Any, claim: Any) -> NLIResult | None:
    if not isinstance(source, Mapping) or not isinstance(claim, Mapping):
        return None
    for field, module, function in (
        ("numeric", "nli_numeric_precheck", "numeric_contradiction_precheck"),
        (
            "medication_status",
            "nli_medication_status",
            "medication_status_contradiction_precheck",
        ),
    ):
        if field not in source or field not in claim:
            continue
        try:
            from importlib import import_module

            precheck = getattr(import_module(f"openmed.clinical.{module}"), function)
            outcome = precheck(source[field], claim[field])
        except Exception:
            return {"label": "abstention", "score": 1.0, "backend_id": "precheck"}
        if outcome.status.value == "contradiction":
            return {"label": "contradiction", "score": 1.0, "backend_id": "precheck"}
        if outcome.status.value == "review_required":
            return {"label": "abstention", "score": 1.0, "backend_id": "precheck"}
    return None


def _validate_backend(backend: object) -> None:
    if not callable(getattr(backend, "predict", None)) and not callable(backend):
        raise TypeError("NLI backend must implement predict or be callable")


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


def _heuristic_prediction(premise: str, hypothesis: str) -> _RawNLIResult:
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


def _classify_pair(premise: str, hypothesis: str) -> _RawNLIResult:
    premise_tokens = _content_tokens(premise)
    hypothesis_tokens = _content_tokens(hypothesis)
    if not premise_tokens or not hypothesis_tokens:
        return {"label": "neutral", "score": 0.5}

    shared = premise_tokens & hypothesis_tokens
    if not shared:
        return {"label": "neutral", "score": 0.5}

    if _has_opposite_terms(premise_tokens, hypothesis_tokens):
        return {"label": "contradiction", "score": 0.95}

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
    "ClaimVerification",
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
