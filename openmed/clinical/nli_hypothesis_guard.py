"""Deterministic complexity guard for clinical NLI hypotheses."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Final

NLI_HYPOTHESIS_GUARD_SCHEMA_VERSION: Final[int] = 1
DEFAULT_MAX_HYPOTHESIS_CHARACTERS: Final[int] = 240
DEFAULT_MAX_HYPOTHESIS_CLAUSES: Final[int] = 2
DEFAULT_MAX_HYPOTHESIS_ASSERTIONS: Final[int] = 1

_MAJOR_BOUNDARY_RE = re.compile(r"(?:[!?;]+|\.(?!\d)|(?:\r?\n)+)")
_SUBORDINATE_BOUNDARY_RE = re.compile(
    r"\b(?:although|because|unless|whereas|while)\b",
    flags=re.IGNORECASE,
)
_EXPLICIT_SUBJECT_ASSERTION_RE = re.compile(
    r"\b(?:and|but)\s+(?=(?:(?:the\s+)?(?:he|it|patient|person|she|subject|they|"
    r"there)|the\s+[a-z][\w'-]*)\s+(?:denied|denies|developed|had|has|is|received|"
    r"remained|remains|reported|reports|required|requires|showed|shows|took|takes|"
    r"was|were)\b)",
    flags=re.IGNORECASE,
)
_ELIDED_SUBJECT_ASSERTION_RE = re.compile(
    r"\b(?:and|but)\s+(?=(?:also\s+)?(?:denied|denies|developed|had|has|is|"
    r"received|remained|remains|reported|reports|required|requires|showed|shows|"
    r"took|takes|was|were)\b)",
    flags=re.IGNORECASE,
)


class HypothesisGuardError(ValueError):
    """Raised when the hypothesis guard contract is invalid."""


class HypothesisGuardStatus(str, Enum):
    """Outcome of the pre-inference hypothesis guard."""

    READY = "ready"
    SEGMENTATION_REQUIRED = "segmentation_required"


class HypothesisComplexityReason(str, Enum):
    """Value-free reason why a hypothesis needs segmentation."""

    CHARACTER_LIMIT_EXCEEDED = "character_limit_exceeded"
    CLAUSE_LIMIT_EXCEEDED = "clause_limit_exceeded"
    ASSERTION_LIMIT_EXCEEDED = "assertion_limit_exceeded"


@dataclass(frozen=True)
class HypothesisComplexity:
    """Metadata-only complexity counts measured before NLI inference."""

    character_count: int
    clause_count: int
    assertion_count: int

    def __post_init__(self) -> None:
        if (
            type(self.character_count) is not int
            or type(self.clause_count) is not int
            or type(self.assertion_count) is not int
            or self.character_count < 1
            or self.clause_count < 1
            or self.assertion_count < 1
        ):
            raise HypothesisGuardError("invalid hypothesis complexity")

    def to_dict(self) -> dict[str, int]:
        """Return counts without retaining or rendering hypothesis text."""

        return {
            "character_count": self.character_count,
            "clause_count": self.clause_count,
            "assertion_count": self.assertion_count,
        }


@dataclass(frozen=True)
class HypothesisGuardResult:
    """Fail-closed result for a clinical NLI hypothesis.

    The hypothesis is returned only for a ready result. It is excluded from
    ``repr`` and :meth:`to_dict`, and is discarded when segmentation is needed.
    """

    status: HypothesisGuardStatus
    complexity: HypothesisComplexity
    reasons: tuple[HypothesisComplexityReason, ...]
    hypothesis: str | None = field(default=None, repr=False)
    schema_version: int = NLI_HYPOTHESIS_GUARD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != NLI_HYPOTHESIS_GUARD_SCHEMA_VERSION
        ):
            raise HypothesisGuardError("unsupported hypothesis-guard schema")
        if not isinstance(self.status, HypothesisGuardStatus):
            raise HypothesisGuardError("invalid hypothesis-guard status")
        if not isinstance(self.complexity, HypothesisComplexity):
            raise HypothesisGuardError("invalid hypothesis complexity")
        if not isinstance(self.reasons, tuple) or any(
            not isinstance(reason, HypothesisComplexityReason)
            for reason in self.reasons
        ):
            raise HypothesisGuardError("invalid hypothesis-guard reasons")
        if self.status is HypothesisGuardStatus.READY:
            if type(self.hypothesis) is not str or not self.hypothesis or self.reasons:
                raise HypothesisGuardError("invalid ready hypothesis result")
        elif self.hypothesis is not None or not self.reasons:
            raise HypothesisGuardError("invalid segmentation-required result")

    @property
    def inference_allowed(self) -> bool:
        """Return whether the hypothesis may be sent to an NLI verifier."""

        return self.status is HypothesisGuardStatus.READY

    def to_dict(self) -> dict[str, object]:
        """Return a metadata-only guard report with no hypothesis text."""

        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "inference_allowed": self.inference_allowed,
            "complexity": self.complexity.to_dict(),
            "reasons": [reason.value for reason in self.reasons],
        }


def measure_hypothesis_complexity(hypothesis: str) -> HypothesisComplexity:
    """Measure deterministic character, clause, and assertion counts.

    Decimal points are not sentence boundaries. Clauses add subordinate links
    (for example ``because`` or ``whereas``), while assertion counts add major
    punctuation boundaries and coordinated predicates with an explicit or
    safely recognized elided subject.
    """

    if type(hypothesis) is not str or not hypothesis.strip():
        raise HypothesisGuardError("invalid hypothesis text")

    major_segments = tuple(
        segment for segment in _MAJOR_BOUNDARY_RE.split(hypothesis) if segment.strip()
    )
    major_assertions = max(1, len(major_segments))
    explicit_assertions = len(_EXPLICIT_SUBJECT_ASSERTION_RE.findall(hypothesis))
    elided_assertions = len(_ELIDED_SUBJECT_ASSERTION_RE.findall(hypothesis))
    subordinate_clauses = len(_SUBORDINATE_BOUNDARY_RE.findall(hypothesis))

    return HypothesisComplexity(
        character_count=len(hypothesis),
        clause_count=major_assertions
        + subordinate_clauses
        + explicit_assertions
        + elided_assertions,
        assertion_count=major_assertions + explicit_assertions + elided_assertions,
    )


def guard_hypothesis(
    hypothesis: str,
    *,
    max_characters: int = DEFAULT_MAX_HYPOTHESIS_CHARACTERS,
    max_clauses: int = DEFAULT_MAX_HYPOTHESIS_CLAUSES,
    max_assertions: int = DEFAULT_MAX_HYPOTHESIS_ASSERTIONS,
) -> HypothesisGuardResult:
    """Allow one bounded hypothesis or require claim segmentation.

    Args:
        hypothesis: Candidate claim to verify.
        max_characters: Inclusive character-count limit.
        max_clauses: Inclusive deterministic clause-count limit.
        max_assertions: Inclusive assertion-count limit.

    Returns:
        A ready result retaining the input hypothesis, or a
        ``segmentation_required`` result containing counts and reason codes only.

    Raises:
        HypothesisGuardError: If the input or configured limits are invalid.
            Error messages never include submitted text or values.
    """

    _validate_limit(max_characters)
    _validate_limit(max_clauses)
    _validate_limit(max_assertions)
    complexity = measure_hypothesis_complexity(hypothesis)

    reasons: list[HypothesisComplexityReason] = []
    if complexity.character_count > max_characters:
        reasons.append(HypothesisComplexityReason.CHARACTER_LIMIT_EXCEEDED)
    if complexity.clause_count > max_clauses:
        reasons.append(HypothesisComplexityReason.CLAUSE_LIMIT_EXCEEDED)
    if complexity.assertion_count > max_assertions:
        reasons.append(HypothesisComplexityReason.ASSERTION_LIMIT_EXCEEDED)

    if reasons:
        return HypothesisGuardResult(
            status=HypothesisGuardStatus.SEGMENTATION_REQUIRED,
            complexity=complexity,
            reasons=tuple(reasons),
        )
    return HypothesisGuardResult(
        status=HypothesisGuardStatus.READY,
        complexity=complexity,
        reasons=(),
        hypothesis=hypothesis,
    )


def guard_nli_hypothesis(
    hypothesis: str,
    *,
    max_characters: int = DEFAULT_MAX_HYPOTHESIS_CHARACTERS,
    max_clauses: int = DEFAULT_MAX_HYPOTHESIS_CLAUSES,
    max_assertions: int = DEFAULT_MAX_HYPOTHESIS_ASSERTIONS,
) -> HypothesisGuardResult:
    """Alias for :func:`guard_hypothesis`."""

    return guard_hypothesis(
        hypothesis,
        max_characters=max_characters,
        max_clauses=max_clauses,
        max_assertions=max_assertions,
    )


def _validate_limit(value: object) -> None:
    if type(value) is not int or value < 1:
        raise HypothesisGuardError("invalid hypothesis complexity limit")


__all__ = [
    "DEFAULT_MAX_HYPOTHESIS_ASSERTIONS",
    "DEFAULT_MAX_HYPOTHESIS_CHARACTERS",
    "DEFAULT_MAX_HYPOTHESIS_CLAUSES",
    "NLI_HYPOTHESIS_GUARD_SCHEMA_VERSION",
    "HypothesisComplexity",
    "HypothesisComplexityReason",
    "HypothesisGuardError",
    "HypothesisGuardResult",
    "HypothesisGuardStatus",
    "guard_hypothesis",
    "guard_nli_hypothesis",
    "measure_hypothesis_complexity",
]
