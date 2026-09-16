"""Deterministic structured-rule cascade before local clinical NLI inference."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Final, Protocol

NLI_CASCADE_SCHEMA_VERSION: Final[int] = 1
_CANONICAL_LABELS: Final[frozenset[str]] = frozenset(
    {"entailment", "contradiction", "neutral", "abstention"}
)
_IDENTIFIER_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_.:/-]{1,128}$")


class NliCascadeError(ValueError):
    """Raised when cascade data violates the value-free execution contract."""


class NliCascadeStage(str, Enum):
    """Ordered stages in the generative-last clinical NLI cascade."""

    ASSERTION = "assertion"
    TEMPORALITY = "temporality"
    EXPERIENCER = "experiencer"
    NUMERIC = "numeric"
    MODEL = "model"


class NliRuleStatus(str, Enum):
    """Normalized outcome supplied by one deterministic structured rule."""

    COMPATIBLE = "compatible"
    CONFLICT = "conflict"
    UNRESOLVED = "unresolved"
    NOT_APPLICABLE = "not_applicable"

    @classmethod
    def from_precheck(cls, value: object) -> NliRuleStatus:
        """Adapt a controlled status from an existing structured precheck.

        Objects exposing a ``status`` attribute and mappings containing a
        ``status`` key are accepted. The adapter recognizes the controlled
        vocabularies used by OpenMed pair and precheck components; unknown
        values fail closed without being echoed.
        """

        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            value = value.get("status")
        elif not isinstance(value, str) and hasattr(value, "status"):
            value = getattr(value, "status")
        if isinstance(value, Enum):
            value = value.value
        if not isinstance(value, str):
            raise NliCascadeError("rule status is invalid")

        normalized = value.strip().casefold()
        aliases = {
            "compatible": cls.COMPATIBLE,
            "pass": cls.COMPATIBLE,
            "contradiction": cls.CONFLICT,
            "conflict": cls.CONFLICT,
            "incompatible": cls.CONFLICT,
            "review_required": cls.UNRESOLVED,
            "unresolved": cls.UNRESOLVED,
            "not_applicable": cls.NOT_APPLICABLE,
        }
        try:
            return aliases[normalized]
        except KeyError:
            raise NliCascadeError("rule status is invalid") from None


@dataclass(frozen=True, repr=False)
class NliCascadePair:
    """One local model pair with normalized structured-rule outcomes.

    Premise and hypothesis text are retained only for the explicit model-call
    boundary. They are excluded from representations, results, and audit data.
    """

    pair_id: str = field(repr=False)
    premise: str = field(repr=False)
    hypothesis: str = field(repr=False)
    assertion: NliRuleStatus = NliRuleStatus.NOT_APPLICABLE
    temporality: NliRuleStatus = NliRuleStatus.NOT_APPLICABLE
    experiencer: NliRuleStatus = NliRuleStatus.NOT_APPLICABLE
    numeric: NliRuleStatus = NliRuleStatus.NOT_APPLICABLE

    def __post_init__(self) -> None:
        _validate_pair_id(self.pair_id)
        if type(self.premise) is not str or not self.premise:
            raise NliCascadeError("premise must be a non-empty string")
        if type(self.hypothesis) is not str or not self.hypothesis:
            raise NliCascadeError("hypothesis must be a non-empty string")
        for status in self.rule_statuses.values():
            if not isinstance(status, NliRuleStatus):
                raise NliCascadeError("rule status is invalid")

    @property
    def rule_statuses(self) -> Mapping[NliCascadeStage, NliRuleStatus]:
        """Return rule outcomes keyed by their fixed cascade stage."""

        return {
            NliCascadeStage.ASSERTION: self.assertion,
            NliCascadeStage.TEMPORALITY: self.temporality,
            NliCascadeStage.EXPERIENCER: self.experiencer,
            NliCascadeStage.NUMERIC: self.numeric,
        }

    def model_inputs(self) -> tuple[str, str]:
        """Return raw text only at the caller-controlled local model boundary."""

        return self.premise, self.hypothesis

    def __repr__(self) -> str:
        """Return a value-free representation of rule state only."""

        statuses = ", ".join(
            f"{stage.value}={status.value}"
            for stage, status in self.rule_statuses.items()
        )
        return f"NliCascadePair({statuses})"


@dataclass(frozen=True)
class LocalNliModelOutput:
    """Privacy-safe output returned by a caller-supplied local NLI model."""

    label: str
    score: float | None = None
    backend_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _normalize_label(self.label))
        if self.score is not None:
            if (
                isinstance(self.score, bool)
                or not isinstance(self.score, int | float)
                or not math.isfinite(float(self.score))
                or not 0.0 <= float(self.score) <= 1.0
            ):
                raise NliCascadeError("model score must be finite and in [0, 1]")
            object.__setattr__(self, "score", float(self.score))
        if self.backend_id is not None and (
            type(self.backend_id) is not str
            or not _IDENTIFIER_RE.fullmatch(self.backend_id)
        ):
            raise NliCascadeError("backend identifier is invalid")

    @classmethod
    def from_obj(
        cls, value: LocalNliModelOutput | str | Mapping[str, object]
    ) -> LocalNliModelOutput:
        """Coerce a canonical model output without accepting source text."""

        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(label=value)
        if not isinstance(value, Mapping):
            raise NliCascadeError("local model output is invalid")
        if "label" not in value:
            raise NliCascadeError("local model output is missing a label")
        return cls(
            label=value["label"],  # type: ignore[arg-type]
            score=value.get("score"),  # type: ignore[arg-type]
            backend_id=value.get("backend_id"),  # type: ignore[arg-type]
        )


class LocalNliModel(Protocol):
    """Callable protocol for an already-resolved local NLI backend."""

    def __call__(
        self, premise: str, hypothesis: str
    ) -> LocalNliModelOutput | str | Mapping[str, object]:
        """Infer one pair locally and return canonical value-free metadata."""


@dataclass(frozen=True, repr=False)
class NliCascadeResult:
    """One cascade decision with the exact deciding stage."""

    pair_id: str = field(repr=False)
    label: str
    deciding_stage: NliCascadeStage
    model_invoked: bool
    reason_code: str
    score: float | None = None
    backend_id: str | None = None
    schema_version: int = NLI_CASCADE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_pair_id(self.pair_id)
        object.__setattr__(self, "label", _normalize_label(self.label))
        if self.schema_version != NLI_CASCADE_SCHEMA_VERSION:
            raise NliCascadeError("unsupported cascade schema")
        if not isinstance(self.deciding_stage, NliCascadeStage):
            raise NliCascadeError("deciding cascade stage is invalid")
        if type(self.model_invoked) is not bool:
            raise NliCascadeError("model invocation flag is invalid")
        if not _IDENTIFIER_RE.fullmatch(self.reason_code):
            raise NliCascadeError("cascade reason code is invalid")
        if self.model_invoked != (self.deciding_stage is NliCascadeStage.MODEL):
            raise NliCascadeError("model invocation flag disagrees with cascade stage")
        if self.score is not None and (
            isinstance(self.score, bool)
            or not isinstance(self.score, int | float)
            or not math.isfinite(float(self.score))
            or not 0.0 <= float(self.score) <= 1.0
        ):
            raise NliCascadeError("model score must be finite and in [0, 1]")
        if self.backend_id is not None and (
            type(self.backend_id) is not str
            or not _IDENTIFIER_RE.fullmatch(self.backend_id)
        ):
            raise NliCascadeError("backend identifier is invalid")

    def to_audit_dict(self) -> dict[str, object]:
        """Return a JSON-safe report without source text or raw identifiers."""

        return {
            "schema_version": self.schema_version,
            "pair_fingerprint": _fingerprint(self.pair_id),
            "label": self.label,
            "deciding_stage": self.deciding_stage.value,
            "model_invoked": self.model_invoked,
            "reason_code": self.reason_code,
            "score": self.score,
            "backend_id": self.backend_id,
        }

    def __repr__(self) -> str:
        """Return a result representation that omits the pair identifier."""

        return (
            "NliCascadeResult("
            f"label={self.label!r}, deciding_stage={self.deciding_stage.value!r}, "
            f"model_invoked={self.model_invoked}, reason_code={self.reason_code!r}, "
            f"score={self.score!r}, backend_id={self.backend_id!r})"
        )


def evaluate_nli_pair(
    pair: NliCascadePair,
    model: LocalNliModel,
) -> NliCascadeResult:
    """Resolve the first exact structured conflict or invoke the local model.

    Assertion, temporality, experiencer, and numeric stages are evaluated in
    that fixed order. Only a normalized ``conflict`` is decisive. Compatible,
    unresolved, and non-applicable rule outcomes leave the semantic pair for
    the local model. Model exceptions are replaced with a value-free error.
    """

    if not isinstance(pair, NliCascadePair):
        raise NliCascadeError("cascade pair is invalid")
    if not callable(model):
        raise NliCascadeError("local model must be callable")

    for stage, status in pair.rule_statuses.items():
        if status is NliRuleStatus.CONFLICT:
            return NliCascadeResult(
                pair_id=pair.pair_id,
                label="contradiction",
                deciding_stage=stage,
                model_invoked=False,
                reason_code=f"{stage.value}_conflict",
            )

    try:
        raw_output = model(*pair.model_inputs())
    except Exception:
        raise NliCascadeError("local model inference failed") from None
    output = LocalNliModelOutput.from_obj(raw_output)
    return NliCascadeResult(
        pair_id=pair.pair_id,
        label=output.label,
        deciding_stage=NliCascadeStage.MODEL,
        model_invoked=True,
        reason_code="model_decision",
        score=output.score,
        backend_id=output.backend_id,
    )


def run_nli_cascade(
    pairs: Iterable[NliCascadePair],
    model: LocalNliModel,
) -> tuple[NliCascadeResult, ...]:
    """Evaluate unique pairs sequentially in deterministic caller order."""

    collected = tuple(pairs)
    seen: set[str] = set()
    for pair in collected:
        if not isinstance(pair, NliCascadePair):
            raise NliCascadeError("cascade pair is invalid")
        if pair.pair_id in seen:
            raise NliCascadeError("cascade pair identifiers must be unique")
        seen.add(pair.pair_id)
    return tuple(evaluate_nli_pair(pair, model) for pair in collected)


def _normalize_label(value: object) -> str:
    if not isinstance(value, str):
        raise NliCascadeError("NLI label must be canonical")
    normalized = value.strip().casefold()
    if normalized not in _CANONICAL_LABELS:
        raise NliCascadeError("NLI label must be canonical")
    return normalized


def _validate_pair_id(value: object) -> None:
    if type(value) is not str or not value.strip():
        raise NliCascadeError("pair identifier must be a non-empty string")


def _fingerprint(pair_id: str) -> str:
    return hashlib.sha256(
        b"openmed:nli-cascade:v1\0" + pair_id.encode("utf-8")
    ).hexdigest()


__all__ = [
    "NLI_CASCADE_SCHEMA_VERSION",
    "LocalNliModel",
    "LocalNliModelOutput",
    "NliCascadeError",
    "NliCascadePair",
    "NliCascadeResult",
    "NliCascadeStage",
    "NliRuleStatus",
    "evaluate_nli_pair",
    "run_nli_cascade",
]
