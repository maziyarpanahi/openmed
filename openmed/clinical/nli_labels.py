"""Stable, backend-neutral labels for clinical natural-language inference.

Clinical NLI providers do not agree on label spelling or class indexes.  This
module gives callers one deterministic four-state vocabulary and a small
validation boundary for adapting provider output:

``entailment`` / ``contradiction`` / ``neutral`` / ``abstention``.

The adapter accepts an explicit backend-to-canonical mapping.  It never
guesses the meaning of numeric or provider-specific labels, and an unmapped
label is rejected before it can enter a clinical result.  Scores are retained
only as finite numbers in the closed interval ``[0, 1]``.  Result payloads,
metadata, and errors contain labels, backend identifiers, and bounded numeric
values only; source text and other caller payloads are deliberately outside
this contract.

The implementation is local-first and dependency-free.  It performs no model
loading, network access, logging, persistence, or clinical decision-making.
Results are assistive metadata for qualified human review.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Literal, TypeAlias

NLI_LABEL_CONTRACT_SCHEMA_VERSION = 1
NLI_LABEL_SCHEMA_VERSION = NLI_LABEL_CONTRACT_SCHEMA_VERSION

NliLabelValue: TypeAlias = Literal[
    "entailment",
    "contradiction",
    "neutral",
    "abstention",
]


class NliLabel(str, Enum):
    """Canonical four-state clinical NLI label.

    The uppercase member names are the stable API.  Lowercase and ``ENTAIL`` /
    ``ABSTAIN`` names are enum aliases for integrations that mirror provider
    vocabulary; iteration still exposes exactly four states.
    """

    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"
    ABSTENTION = "abstention"

    # Compatibility aliases for common backend terminology.
    ENTAIL = "entailment"
    ABSTAIN = "abstention"
    entailment = "entailment"
    contradiction = "contradiction"
    neutral = "neutral"
    abstention = "abstention"


ClinicalNliLabel = NliLabel
NLIState = NliLabel

NLI_LABELS: tuple[NliLabelValue, ...] = (
    "entailment",
    "contradiction",
    "neutral",
    "abstention",
)
NLI_LABEL_VALUES = NLI_LABELS
NLI_STATES = NLI_LABELS

NLI_LABEL_CONTRACT_ADVISORY = (
    "Clinical NLI labels are deterministic assistive metadata for qualified "
    "human review. They are not a diagnosis, treatment decision, or autonomous "
    "clinical judgment."
)
CLINICAL_NLI_LABEL_ADVISORY = NLI_LABEL_CONTRACT_ADVISORY

_CANONICAL_ALIASES = {
    "entail": "entailment",
    "abstain": "abstention",
}


class NliLabelContractError(ValueError):
    """Base error for malformed or unsafe clinical NLI contract data."""


class UnknownNliLabelError(NliLabelContractError):
    """Raised when a backend label is absent from its declared mapping."""


class NliLabelValidationError(NliLabelContractError):
    """Raised when a mapping or result violates the label contract."""


class NliScoreValidationError(NliLabelValidationError):
    """Raised when a score is not a finite probability in ``[0, 1]``."""


# Descriptive aliases keep integrations from having to know the internal error
# hierarchy while preserving one canonical exception type for callers to catch.
NLIValidationError = NliLabelValidationError
NLIContractError = NliLabelContractError
UnknownNLIlabelError = UnknownNliLabelError

_TOKEN_RE = re.compile(r"^[\w][\w.:/-]{0,63}$", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_token(value: object, *, field_name: str) -> str:
    """Normalize a short identifier without echoing it in validation errors."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    if len(normalized) > 64 or not _TOKEN_RE.fullmatch(normalized):
        raise ValueError(f"{field_name} must be a short label token")
    return normalized


def _unknown_label() -> UnknownNliLabelError:
    """Build a PHI-safe unknown-label error with no caller value."""

    return UnknownNliLabelError("backend NLI label is not declared in the mapping")


def _canonical_label(value: object) -> NliLabel:
    """Coerce an already-canonical value or reject it without echoing input."""

    try:
        normalized = _normalize_token(value, field_name="NLI label")
    except (TypeError, ValueError):
        raise NliLabelValidationError(
            "NLI label must be one of the four states"
        ) from None
    normalized = _CANONICAL_ALIASES.get(normalized, normalized)
    try:
        return NliLabel(normalized)
    except ValueError:
        raise NliLabelValidationError(
            "NLI label must be one of the four states"
        ) from None


def _bounded_score(value: object, *, field_name: str = "score") -> float:
    """Validate and normalize one bounded numeric score."""

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field_name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise NliScoreValidationError(f"{field_name} must be finite and in [0, 1]")
    return number


def _normalize_backend_labels(
    labels: Mapping[object, object],
) -> Mapping[str, NliLabel]:
    """Validate a backend-to-canonical map and freeze its normalized copy."""

    if not isinstance(labels, Mapping):
        raise TypeError("backend label mapping must be a mapping")
    if not labels:
        raise NliLabelValidationError("backend label mapping must not be empty")

    normalized: dict[str, NliLabel] = {}
    for raw_label, canonical in labels.items():
        try:
            key = _normalize_token(raw_label, field_name="backend label")
        except (TypeError, ValueError) as exc:
            raise NliLabelValidationError(
                "backend label mapping keys must be short string tokens"
            ) from exc
        if key in normalized:
            raise NliLabelValidationError(
                "backend label mapping contains duplicate normalized keys"
            )
        normalized[key] = _canonical_label(canonical)
    return MappingProxyType(dict(sorted(normalized.items())))


def _normalize_canonical_scores(scores: Mapping[object, object]) -> Mapping[str, float]:
    """Validate scores whose keys are already canonical labels."""

    if not isinstance(scores, Mapping):
        raise TypeError("NLI scores must be a mapping")
    normalized: dict[str, float] = {}
    for key, value in scores.items():
        label = _canonical_label(key).value
        if label in normalized:
            raise NliLabelValidationError("NLI scores contain duplicate labels")
        normalized[label] = _bounded_score(value, field_name="NLI score")
    return MappingProxyType(dict(sorted(normalized.items())))


def _normalize_mapped_scores(
    scores: Mapping[object, object],
    mapping: "BackendLabelMapping",
) -> Mapping[str, float]:
    """Map backend score keys to canonical labels and validate their values."""

    if not isinstance(scores, Mapping):
        raise TypeError("NLI scores must be a mapping")
    normalized: dict[str, float] = {}
    for raw_label, value in scores.items():
        label = mapping.resolve(raw_label).value
        score = _bounded_score(value, field_name="NLI score")
        if label in normalized and normalized[label] != score:
            raise NliLabelValidationError(
                "multiple backend scores map to one canonical label"
            )
        normalized[label] = score
    return MappingProxyType(dict(sorted(normalized.items())))


def _normalize_metadata(
    metadata: Mapping[object, object] | None,
    mapping: "BackendLabelMapping | None" = None,
) -> Mapping[str, object]:
    """Keep metadata to bounded numbers, with an optional raw-score mapping."""

    if metadata is None:
        return MappingProxyType({})
    if not isinstance(metadata, Mapping):
        raise TypeError("NLI metadata must be a mapping")
    normalized: dict[str, object] = {}
    for raw_key, value in metadata.items():
        try:
            key = _normalize_token(raw_key, field_name="metadata key")
        except (TypeError, ValueError) as exc:
            raise NliLabelValidationError(
                "NLI metadata keys must be short string tokens"
            ) from exc
        if key in normalized:
            raise NliLabelValidationError("NLI metadata contains duplicate keys")
        if key == "raw_scores":
            if mapping is None:
                normalized[key] = MappingProxyType(
                    dict(_normalize_canonical_scores(value))
                )
            else:
                normalized[key] = MappingProxyType(
                    dict(_normalize_mapped_scores(value, mapping))
                )
        else:
            normalized[key] = _bounded_score(value, field_name="NLI metadata value")
    return MappingProxyType(dict(sorted(normalized.items())))


@dataclass(frozen=True, slots=True)
class BackendLabelMapping(Mapping[str, NliLabel]):
    """Validated mapping from one backend's labels to canonical NLI states.

    Args:
        backend: Short stable backend identifier (for example ``"mlx"`` or
            ``"transformers"``). It is metadata only and does not load a
            backend.
        labels: Mapping whose keys are provider labels and whose values are one
            of the four :class:`NliLabel` values. Numeric class indexes must be
            declared by the caller; this type never guesses their order.
        schema_version: Contract schema version. Only version ``1`` is accepted.
    """

    backend: str
    labels: Mapping[str, NliLabel | NliLabelValue | str]
    schema_version: int = NLI_LABEL_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        try:
            backend = _normalize_token(self.backend, field_name="backend")
        except (TypeError, ValueError) as exc:
            raise NliLabelValidationError(
                "backend must be a short string token"
            ) from exc
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise NliLabelValidationError("unsupported NLI label contract version")
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "labels", _normalize_backend_labels(self.labels))

    @classmethod
    def from_mapping(
        cls,
        labels: Mapping[object, object],
        *,
        backend: str = "custom",
        schema_version: int = NLI_LABEL_CONTRACT_SCHEMA_VERSION,
    ) -> "BackendLabelMapping":
        """Build a mapping while keeping the caller's mutable mapping private."""

        return cls(backend=backend, labels=labels, schema_version=schema_version)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "BackendLabelMapping":
        """Rebuild and validate a serialized backend label map."""

        if not isinstance(payload, Mapping):
            raise TypeError("backend label mapping must be a mapping")
        return cls(
            backend=payload.get("backend", "custom"),  # type: ignore[arg-type]
            labels=payload.get("labels", {}),  # type: ignore[arg-type]
            schema_version=payload.get(
                "schema_version", NLI_LABEL_CONTRACT_SCHEMA_VERSION
            ),  # type: ignore[arg-type]
        )

    @property
    def backend_to_canonical(self) -> Mapping[str, NliLabel]:
        """Return the immutable normalized backend-to-canonical mapping."""

        return self.labels

    def __getitem__(self, key: str) -> NliLabel:
        """Expose mapping-style access to normalized backend labels."""

        return self.labels[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate normalized backend labels in stable order."""

        return iter(self.labels)

    def __len__(self) -> int:
        """Return the number of declared backend labels."""

        return len(self.labels)

    @property
    def canonical_labels(self) -> tuple[NliLabelValue, ...]:
        """Return canonical states represented by this mapping in stable order."""

        represented = {label.value for label in self.labels.values()}
        return tuple(label for label in NLI_LABELS if label in represented)

    def resolve(self, raw_label: object) -> NliLabel:
        """Resolve one provider label, rejecting unknown labels fail-closed."""

        try:
            key = _normalize_token(raw_label, field_name="backend label")
        except (TypeError, ValueError) as exc:
            raise _unknown_label() from exc
        try:
            return self.labels[key]
        except KeyError as exc:
            raise _unknown_label() from exc

    map_label = resolve

    def require_complete(self) -> "BackendLabelMapping":
        """Require that a backend map declares all four canonical states."""

        missing = set(NLI_LABELS) - set(self.canonical_labels)
        if missing:
            raise NliLabelValidationError(
                "backend label mapping must cover all four canonical states"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, PHI-safe mapping representation."""

        return {
            "schema_version": self.schema_version,
            "backend": self.backend,
            "labels": {key: value.value for key, value in sorted(self.labels.items())},
        }

    def to_json(self) -> str:
        """Serialize the mapping with stable key ordering."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )


NliLabelMapping = BackendLabelMapping
ClinicalNliLabelMapping = BackendLabelMapping
NliBackendLabelMap = BackendLabelMapping
BackendNliLabelMap = BackendLabelMapping


DEFAULT_BACKEND_LABEL_MAPPING = BackendLabelMapping(
    backend="canonical",
    labels={label: label for label in NLI_LABELS},
)
CANONICAL_NLI_LABEL_MAPPING = DEFAULT_BACKEND_LABEL_MAPPING
DEFAULT_NLI_LABEL_MAPPING = DEFAULT_BACKEND_LABEL_MAPPING
NLI_LABEL_MAPPING = DEFAULT_BACKEND_LABEL_MAPPING


def validate_backend_label_mapping(
    labels: BackendLabelMapping | Mapping[object, object],
    *,
    backend: str = "custom",
    require_complete: bool = False,
) -> BackendLabelMapping:
    """Validate and freeze a backend label map.

    ``labels`` may already be a :class:`BackendLabelMapping` or a plain
    backend-to-canonical mapping.  ``require_complete`` is opt-in because some
    providers expose only three classes and represent abstention by policy.
    """

    if isinstance(labels, BackendLabelMapping):
        result = labels
    else:
        result = BackendLabelMapping.from_mapping(labels, backend=backend)
    return result.require_complete() if require_complete else result


validate_nli_label_mapping = validate_backend_label_mapping
validate_label_mapping = validate_backend_label_mapping


def normalize_nli_label(
    raw_label: object,
    mapping: BackendLabelMapping | Mapping[object, object] | None = None,
    *,
    backend: str = "canonical",
) -> NliLabel:
    """Resolve a provider label to one canonical :class:`NliLabel` value."""

    if mapping is None:
        active = DEFAULT_BACKEND_LABEL_MAPPING
    elif isinstance(mapping, BackendLabelMapping):
        active = mapping
    else:
        active = BackendLabelMapping.from_mapping(mapping, backend=backend)
    return active.resolve(raw_label)


map_backend_label = normalize_nli_label
resolve_nli_label = normalize_nli_label
coerce_nli_label = normalize_nli_label


@dataclass(frozen=True, slots=True)
class ClinicalNliResult:
    """Typed, serializable result under the four-state NLI contract.

    ``scores`` contains only canonical label keys and bounded probabilities.
    ``metadata`` may contain additional bounded numeric values and, under the
    reserved ``raw_scores`` key, another bounded score mapping.  No source,
    premise, hypothesis, or arbitrary provider payload is accepted or emitted.
    """

    label: NliLabel | NliLabelValue | str
    backend_label: str | None = None
    backend: str = "canonical"
    scores: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)
    schema_version: int = NLI_LABEL_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        label = _canonical_label(self.label)
        try:
            backend = _normalize_token(self.backend, field_name="backend")
        except (TypeError, ValueError) as exc:
            raise NliLabelValidationError(
                "backend must be a short string token"
            ) from exc
        if self.backend_label is None:
            backend_label = None
        else:
            try:
                backend_label = _normalize_token(
                    self.backend_label, field_name="backend label"
                )
            except (TypeError, ValueError) as exc:
                raise NliLabelValidationError(
                    "backend label must be a short string token"
                ) from exc
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise NliLabelValidationError("unsupported NLI label contract version")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "backend_label", backend_label)
        object.__setattr__(self, "scores", _normalize_canonical_scores(self.scores))
        object.__setattr__(self, "metadata", _normalize_metadata(self.metadata))

    @classmethod
    def from_backend(
        cls,
        raw_label: object,
        *,
        mapping: BackendLabelMapping | Mapping[object, object] | None = None,
        backend: str = "custom",
        scores: Mapping[object, object] | None = None,
        raw_scores: Mapping[object, object] | None = None,
        score: object | None = None,
        metadata: Mapping[object, object] | None = None,
    ) -> "ClinicalNliResult":
        """Build a result from one backend label and optional score metadata.

        ``scores`` and ``raw_scores`` are aliases; supplying both is rejected.
        If a mapping is omitted, only canonical labels are accepted.  An
        explicit mapping is therefore required for numeric class indexes or
        provider-specific labels.
        """

        if scores is not None and raw_scores is not None:
            raise NliLabelValidationError(
                "scores and raw_scores are aliases; supply one"
            )
        if mapping is None:
            active = DEFAULT_BACKEND_LABEL_MAPPING
            selected_backend = "canonical" if backend == "custom" else backend
        elif isinstance(mapping, BackendLabelMapping):
            active = mapping
            selected_backend = active.backend
        else:
            active = BackendLabelMapping.from_mapping(mapping, backend=backend)
            selected_backend = active.backend

        label = active.resolve(raw_label)
        provided_scores = raw_scores if raw_scores is not None else scores
        normalized_scores: dict[str, float] = {}
        if provided_scores is not None:
            normalized_scores.update(_normalize_mapped_scores(provided_scores, active))
        if score is not None:
            selected_score = _bounded_score(score, field_name="score")
            existing = normalized_scores.get(label.value)
            if existing is not None and existing != selected_score:
                raise NliLabelValidationError(
                    "score conflicts with the mapped label score"
                )
            normalized_scores[label.value] = selected_score

        normalized_metadata = _normalize_metadata(metadata, active)
        return cls(
            label=label,
            backend_label=_normalize_token(raw_label, field_name="backend label"),
            backend=selected_backend,
            scores=normalized_scores,
            metadata=normalized_metadata,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ClinicalNliResult":
        """Rebuild and validate a result from :meth:`to_dict` output."""

        if not isinstance(payload, Mapping):
            raise TypeError("NLI result must be a mapping")
        return cls(
            label=payload.get("label"),
            backend_label=payload.get("backend_label"),
            backend=payload.get("backend", "canonical"),  # type: ignore[arg-type]
            scores=payload.get("scores", {}),  # type: ignore[arg-type]
            metadata=payload.get("metadata", {}),  # type: ignore[arg-type]
            schema_version=payload.get(
                "schema_version", NLI_LABEL_CONTRACT_SCHEMA_VERSION
            ),  # type: ignore[arg-type]
        )

    @property
    def state(self) -> NliLabel:
        """Alias for the canonical result label."""

        return self.label

    @property
    def canonical_label(self) -> NliLabel:
        """Return the canonical label under a descriptive field name."""

        return self.label

    @property
    def raw_label(self) -> str | None:
        """Return the normalized backend label, when one was supplied."""

        return self.backend_label

    @property
    def raw_scores(self) -> Mapping[str, float]:
        """Return the immutable bounded score metadata mapping."""

        return self.scores

    @property
    def score(self) -> float | None:
        """Return the score for the selected label when one was supplied."""

        return self.scores.get(self.label.value)

    @property
    def is_abstention(self) -> bool:
        """Whether this result explicitly requests abstention/review."""

        return self.label is NliLabel.ABSTENTION

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible, source-free result metadata."""

        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "label": self.label.value,
            "backend": self.backend,
            "backend_label": self.backend_label,
            "scores": dict(sorted(self.scores.items())),
            "metadata": _plain_metadata(self.metadata),
        }
        return payload

    def to_json(self) -> str:
        """Serialize the result with stable key ordering and no NaN values."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )


NliResult = ClinicalNliResult
NLIResult = ClinicalNliResult
NliLabelResult = ClinicalNliResult
ClinicalNLIResult = ClinicalNliResult


def _plain_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    """Copy frozen metadata into JSON-compatible plain containers."""

    result: dict[str, object] = {}
    for key, value in sorted(metadata.items()):
        if isinstance(value, Mapping):
            result[key] = dict(sorted((str(k), float(v)) for k, v in value.items()))
        else:
            result[key] = float(value)
    return result


def build_nli_result(
    raw_label: object,
    *,
    mapping: BackendLabelMapping | Mapping[object, object] | None = None,
    backend: str = "custom",
    scores: Mapping[object, object] | None = None,
    raw_scores: Mapping[object, object] | None = None,
    score: object | None = None,
    metadata: Mapping[object, object] | None = None,
) -> ClinicalNliResult:
    """Convenience wrapper around :meth:`ClinicalNliResult.from_backend`."""

    return ClinicalNliResult.from_backend(
        raw_label,
        mapping=mapping,
        backend=backend,
        scores=scores,
        raw_scores=raw_scores,
        score=score,
        metadata=metadata,
    )


make_nli_result = build_nli_result
result_from_backend = build_nli_result
create_nli_result = build_nli_result


def validate_nli_result(
    result: ClinicalNliResult | Mapping[str, object],
) -> ClinicalNliResult:
    """Validate an existing result or source-free serialized result mapping."""

    if isinstance(result, ClinicalNliResult):
        return result
    return ClinicalNliResult.from_dict(result)


validate_clinical_nli_result = validate_nli_result


__all__ = [
    "BACKEND_NLI_LABEL_MAP",
    "BackendLabelMapping",
    "BackendNliLabelMap",
    "CANONICAL_NLI_LABEL_MAPPING",
    "ClinicalNLIResult",
    "ClinicalNliLabel",
    "ClinicalNliLabelMapping",
    "ClinicalNliResult",
    "CLINICAL_NLI_LABEL_ADVISORY",
    "DEFAULT_BACKEND_LABEL_MAPPING",
    "DEFAULT_NLI_LABEL_MAPPING",
    "NLIContractError",
    "NLI_LABELS",
    "NLI_LABEL_CONTRACT_ADVISORY",
    "NLI_LABEL_CONTRACT_SCHEMA_VERSION",
    "NLI_LABEL_MAPPING",
    "NLI_LABEL_SCHEMA_VERSION",
    "NLI_LABEL_VALUES",
    "NLIResult",
    "NLIState",
    "NLI_STATES",
    "NLIValidationError",
    "NliBackendLabelMap",
    "NliLabel",
    "NliLabelContractError",
    "NliLabelMapping",
    "NliLabelResult",
    "NliLabelValidationError",
    "NliLabelValue",
    "NliResult",
    "NliScoreValidationError",
    "UnknownNLIlabelError",
    "UnknownNliLabelError",
    "build_nli_result",
    "coerce_nli_label",
    "create_nli_result",
    "make_nli_result",
    "map_backend_label",
    "normalize_nli_label",
    "resolve_nli_label",
    "result_from_backend",
    "validate_backend_label_mapping",
    "validate_clinical_nli_result",
    "validate_label_mapping",
    "validate_nli_label_mapping",
    "validate_nli_result",
]

# Compatibility name used by a few registry integrations.  It intentionally
# points at the immutable canonical map rather than exposing a mutable dict.
BACKEND_NLI_LABEL_MAP = DEFAULT_BACKEND_LABEL_MAPPING
