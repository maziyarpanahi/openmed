"""Deterministic assertion resolution for social-determinant needs.

SDOH extractors commonly emit a determinant span for both a need and a
screening denial.  A document such as ``"no food insecurity, but a
transportation barrier is reported"`` therefore needs a narrower assertion
layer than a sentence-level polarity flag: the denial belongs to the food
finding only.

This module resolves each candidate independently in its local sentence and
clause.  It keeps only source offsets, controlled labels, and input indexes in
the returned records; candidate text and cue text are used transiently and are
never serialized.  Nested negation, contradictory polarity cues, and
uncertainty cues abstain to ``"unknown"`` and require human review.  The
implementation is rules-only, deterministic, and offline.

The records are review metadata rather than clinical decisions.  A caller can
join a record back to its protected in-memory finding with ``input_index`` and
``source_offsets`` inside an authorized workflow without placing the finding's
value in a report or log.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

from .context import AFFIRMED, NEGATED

UNKNOWN: Final[str] = "unknown"
NEED_PRESENT: Final[str] = "present"
NEED_ABSENT: Final[str] = "absent"
NEED_UNKNOWN: Final[str] = UNKNOWN

SDOH_NEGATED_NEED_SCHEMA_VERSION: Final[int] = 1
SDOH_NEGATED_NEED_ADVISORY: Final[str] = (
    "SDOH need assertions are deterministic, value-free review metadata. "
    "Negated and unresolved findings must be verified by qualified clinical "
    "review; this layer is not a clinical decision or billing rule."
)

SDOH_NEGATED_NEED_ASSERTIONS: Final[tuple[str, ...]] = (
    AFFIRMED,
    NEGATED,
    UNKNOWN,
)
SDOH_NEGATED_NEED_STATUSES: Final[tuple[str, ...]] = (
    NEED_PRESENT,
    NEED_ABSENT,
    NEED_UNKNOWN,
)
SDOH_DETERMINANTS: Final[tuple[str, ...]] = (
    "food_insecurity",
    "housing_instability",
    "transportation_barrier",
    "employment",
    "financial_strain",
    "utilities",
    "childcare",
    "social_isolation",
    "safety",
    "health_insurance",
    "education",
    "unknown",
)

SpanOffset = tuple[int, int]
# The shared clinical constants are intentionally typed as ``str`` for
# backwards compatibility. Runtime validation below still restricts this
# internal value to the three controlled assertion labels.
AssertionValue = str
ResolutionSource = Literal["cue", "provided", "default"]
ReviewReason = Literal[
    "double_negation",
    "contradictory_cues",
    "uncertain_cue",
    "provided_conflict",
]

_FORWARD = "forward"
_BACKWARD = "backward"
_BIDIRECTIONAL = "bidirectional"
_MAX_SCOPE_TOKENS = 64
_TOKEN_RE = re.compile(r"(?<!\w)\w+(?!\w)", re.UNICODE)
_HARD_BOUNDARY_RE = re.compile(
    r"(?:[.!?;:\n。！？；：]|(?<!\w)(?:but|however|although|whereas|"
    r"yet|except|apart from|aside from|while|because|unless)(?!\w))",
    re.IGNORECASE,
)
_COORDINATOR_RE = re.compile(r"(?<!\w)(?:and|or)(?!\w)", re.IGNORECASE)
_STRUCTURAL_BOUNDARY_RE = re.compile(
    r"(?:,|[.!?;:\n。！？；：]|(?<!\w)(?:and|or|but|however|although|"
    r"whereas|yet|except|apart from|aside from|while|because|unless)(?!\w))",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class _CueSpec:
    expression: str
    assertion: AssertionValue
    direction: str
    kind: str


@dataclass(frozen=True, slots=True)
class _CueHit:
    assertion: AssertionValue
    direction: str
    kind: str
    start: int
    end: int
    expression: str


def _literal_pattern(expression: str) -> re.Pattern[str]:
    normalized = " ".join(expression.split())
    escaped = r"\s+".join(re.escape(part) for part in normalized.split())
    return re.compile(rf"(?<!\w)(?:{escaped})(?!\w)", re.IGNORECASE)


# Longer phrases precede generic ``no``/``not`` cues.  Direction is explicit
# because ``denied`` and ``absent`` can occur on either side of a finding,
# while ``no`` and ``denies`` conventionally govern text to their right.
_NEGATION_SPECS: tuple[_CueSpec, ...] = (
    _CueSpec("no evidence of", NEGATED, _FORWARD, "negation"),
    _CueSpec("no evidence", NEGATED, _FORWARD, "negation"),
    _CueSpec("no indication of", NEGATED, _FORWARD, "negation"),
    _CueSpec("no signs of", NEGATED, _FORWARD, "negation"),
    _CueSpec("no sign of", NEGATED, _FORWARD, "negation"),
    _CueSpec("negative screen for", NEGATED, _FORWARD, "negation"),
    _CueSpec("screen negative for", NEGATED, _FORWARD, "negation"),
    _CueSpec("negative for", NEGATED, _FORWARD, "negation"),
    _CueSpec("absence of", NEGATED, _FORWARD, "negation"),
    _CueSpec("does not report", NEGATED, _FORWARD, "negation"),
    _CueSpec("do not report", NEGATED, _FORWARD, "negation"),
    _CueSpec("doesn't report", NEGATED, _FORWARD, "negation"),
    _CueSpec("don't report", NEGATED, _FORWARD, "negation"),
    _CueSpec("does not have", NEGATED, _FORWARD, "negation"),
    _CueSpec("do not have", NEGATED, _FORWARD, "negation"),
    _CueSpec("doesn't have", NEGATED, _FORWARD, "negation"),
    _CueSpec("don't have", NEGATED, _FORWARD, "negation"),
    _CueSpec("did not have", NEGATED, _FORWARD, "negation"),
    _CueSpec("didn't have", NEGATED, _FORWARD, "negation"),
    _CueSpec("no need for", NEGATED, _FORWARD, "negation"),
    _CueSpec("no concern for", NEGATED, _FORWARD, "negation"),
    _CueSpec("no difficulty with", NEGATED, _FORWARD, "negation"),
    _CueSpec("no issue with", NEGATED, _FORWARD, "negation"),
    _CueSpec("not an issue", NEGATED, _BACKWARD, "negation"),
    _CueSpec("not a concern", NEGATED, _BACKWARD, "negation"),
    _CueSpec("not concerning", NEGATED, _BACKWARD, "negation"),
    _CueSpec("not present", NEGATED, _BACKWARD, "negation"),
    _CueSpec("not detected", NEGATED, _BACKWARD, "negation"),
    _CueSpec("not identified", NEGATED, _BACKWARD, "negation"),
    _CueSpec("free from", NEGATED, _FORWARD, "negation"),
    _CueSpec("free of", NEGATED, _FORWARD, "negation"),
    _CueSpec("no longer", NEGATED, _FORWARD, "negation"),
    _CueSpec("denies", NEGATED, _FORWARD, "negation"),
    _CueSpec("denied", NEGATED, _BIDIRECTIONAL, "negation"),
    _CueSpec("deny", NEGATED, _FORWARD, "negation"),
    _CueSpec("ruled out", NEGATED, _BACKWARD, "negation"),
    _CueSpec("absent", NEGATED, _BIDIRECTIONAL, "negation"),
    _CueSpec("none", NEGATED, _BIDIRECTIONAL, "negation"),
    _CueSpec("never", NEGATED, _FORWARD, "negation"),
    _CueSpec("without", NEGATED, _FORWARD, "negation"),
    _CueSpec("not", NEGATED, _FORWARD, "negation"),
    _CueSpec("no", NEGATED, _FORWARD, "negation"),
)

# These phrases are intentionally masked before negation matching.  They
# contain negative words but do not assert that the determinant need is absent.
# ``no known``/``not documented`` are separately reported as uncertainty.
_PSEUDO_NEGATION_EXPRESSIONS: tuple[str, ...] = (
    "no change",
    "no increase",
    "no interval increase",
    "no significant increase",
    "no known",
    "not known",
    "not documented",
    "not specified",
    "not ruled out",
    "not yet ruled out",
    "not completely ruled out",
    "not been ruled out",
    "not denied",
    "cannot be excluded",
    "can't be excluded",
    "cannot exclude",
    "can't exclude",
    "unable to rule out",
    "not only",
    "not necessarily",
    "without a doubt",
)

_UNCERTAINTY_SPECS: tuple[_CueSpec, ...] = (
    _CueSpec("cannot be excluded", UNKNOWN, _BIDIRECTIONAL, "uncertainty"),
    _CueSpec("can't be excluded", UNKNOWN, _BIDIRECTIONAL, "uncertainty"),
    _CueSpec("cannot exclude", UNKNOWN, _BIDIRECTIONAL, "uncertainty"),
    _CueSpec("can't exclude", UNKNOWN, _BIDIRECTIONAL, "uncertainty"),
    _CueSpec("unable to rule out", UNKNOWN, _BIDIRECTIONAL, "uncertainty"),
    _CueSpec("not ruled out", UNKNOWN, _BIDIRECTIONAL, "uncertainty"),
    _CueSpec("not yet ruled out", UNKNOWN, _BIDIRECTIONAL, "uncertainty"),
    _CueSpec(
        "not completely ruled out",
        UNKNOWN,
        _BIDIRECTIONAL,
        "uncertainty",
    ),
    _CueSpec("not been ruled out", UNKNOWN, _BIDIRECTIONAL, "uncertainty"),
    _CueSpec("not denied", UNKNOWN, _BIDIRECTIONAL, "uncertainty"),
    _CueSpec("no known", UNKNOWN, _FORWARD, "uncertainty"),
    _CueSpec("not known", UNKNOWN, _BACKWARD, "uncertainty"),
    _CueSpec("not documented", UNKNOWN, _BACKWARD, "uncertainty"),
    _CueSpec("not specified", UNKNOWN, _BACKWARD, "uncertainty"),
    _CueSpec("not reported", UNKNOWN, _BACKWARD, "uncertainty"),
    _CueSpec("unable to determine", UNKNOWN, _FORWARD, "uncertainty"),
    _CueSpec("unable to assess", UNKNOWN, _FORWARD, "uncertainty"),
    _CueSpec("status unclear", UNKNOWN, _BACKWARD, "uncertainty"),
    _CueSpec("unclear", UNKNOWN, _BIDIRECTIONAL, "uncertainty"),
    _CueSpec("unknown", UNKNOWN, _BIDIRECTIONAL, "uncertainty"),
    _CueSpec("uncertain", UNKNOWN, _BIDIRECTIONAL, "uncertainty"),
    _CueSpec("possible", UNKNOWN, _FORWARD, "uncertainty"),
    _CueSpec("possibly", UNKNOWN, _FORWARD, "uncertainty"),
    _CueSpec("probable", UNKNOWN, _FORWARD, "uncertainty"),
    _CueSpec("probably", UNKNOWN, _FORWARD, "uncertainty"),
    _CueSpec("likely", UNKNOWN, _FORWARD, "uncertainty"),
    _CueSpec("unlikely", UNKNOWN, _FORWARD, "uncertainty"),
    _CueSpec("suspected", UNKNOWN, _FORWARD, "uncertainty"),
    _CueSpec("suspect", UNKNOWN, _FORWARD, "uncertainty"),
    _CueSpec("concern for", UNKNOWN, _FORWARD, "uncertainty"),
    _CueSpec("concerned about", UNKNOWN, _FORWARD, "uncertainty"),
    _CueSpec("worried about", UNKNOWN, _FORWARD, "uncertainty"),
    _CueSpec("may have", UNKNOWN, _FORWARD, "uncertainty"),
    _CueSpec("might have", UNKNOWN, _FORWARD, "uncertainty"),
    _CueSpec("could have", UNKNOWN, _FORWARD, "uncertainty"),
)

_AFFIRMATION_SPECS: tuple[_CueSpec, ...] = (
    _CueSpec("screen positive for", AFFIRMED, _FORWARD, "affirmation"),
    _CueSpec("positive for", AFFIRMED, _FORWARD, "affirmation"),
    _CueSpec("struggles with", AFFIRMED, _FORWARD, "affirmation"),
    _CueSpec("difficulty with", AFFIRMED, _FORWARD, "affirmation"),
    _CueSpec("cannot afford", AFFIRMED, _FORWARD, "affirmation"),
    _CueSpec("unable to", AFFIRMED, _FORWARD, "affirmation"),
    _CueSpec("lack of", AFFIRMED, _FORWARD, "affirmation"),
    _CueSpec("reports", AFFIRMED, _FORWARD, "affirmation"),
    _CueSpec("reported", AFFIRMED, _BIDIRECTIONAL, "affirmation"),
    _CueSpec("states", AFFIRMED, _FORWARD, "affirmation"),
    _CueSpec("describes", AFFIRMED, _FORWARD, "affirmation"),
    _CueSpec("endorses", AFFIRMED, _FORWARD, "affirmation"),
    _CueSpec("experiences", AFFIRMED, _FORWARD, "affirmation"),
    _CueSpec("experiencing", AFFIRMED, _FORWARD, "affirmation"),
    _CueSpec("needs", AFFIRMED, _FORWARD, "affirmation"),
    _CueSpec("requires", AFFIRMED, _FORWARD, "affirmation"),
    _CueSpec("has", AFFIRMED, _FORWARD, "affirmation"),
    _CueSpec("have", AFFIRMED, _FORWARD, "affirmation"),
    _CueSpec("lacks", AFFIRMED, _FORWARD, "affirmation"),
    _CueSpec("lack", AFFIRMED, _FORWARD, "affirmation"),
    _CueSpec("present", AFFIRMED, _BACKWARD, "affirmation"),
    _CueSpec("confirmed", AFFIRMED, _BACKWARD, "affirmation"),
    _CueSpec("identified", AFFIRMED, _BACKWARD, "affirmation"),
    _CueSpec("noted", AFFIRMED, _BACKWARD, "affirmation"),
    _CueSpec("exists", AFFIRMED, _BACKWARD, "affirmation"),
    _CueSpec("active", AFFIRMED, _BACKWARD, "affirmation"),
    _CueSpec("documented", AFFIRMED, _BACKWARD, "affirmation"),
)

_PSEUDO_NEGATION_RE = re.compile(
    "|".join(
        _literal_pattern(expression).pattern
        for expression in sorted(
            _PSEUDO_NEGATION_EXPRESSIONS,
            key=lambda expression: (-len(expression), expression),
        )
    ),
    re.IGNORECASE,
)
_COMPILED_NEGATION_SPECS = tuple(
    (spec, _literal_pattern(spec.expression)) for spec in _NEGATION_SPECS
)
_COMPILED_UNCERTAINTY_SPECS = tuple(
    (spec, _literal_pattern(spec.expression)) for spec in _UNCERTAINTY_SPECS
)
_COMPILED_AFFIRMATION_SPECS = tuple(
    (spec, _literal_pattern(spec.expression)) for spec in _AFFIRMATION_SPECS
)


_DETERMINANT_ALIASES: dict[str, str] = {
    "food": "food_insecurity",
    "food access": "food_insecurity",
    "food insecurity": "food_insecurity",
    "food_insecurity": "food_insecurity",
    "food insecure": "food_insecurity",
    "housing": "housing_instability",
    "housing insecurity": "housing_instability",
    "housing instability": "housing_instability",
    "housing_instability": "housing_instability",
    "homelessness": "housing_instability",
    "homeless": "housing_instability",
    "transport": "transportation_barrier",
    "transportation": "transportation_barrier",
    "transportation barrier": "transportation_barrier",
    "transportation barriers": "transportation_barrier",
    "transport barriers": "transportation_barrier",
    "transportation_barrier": "transportation_barrier",
    "transportation insecurity": "transportation_barrier",
    "employment": "employment",
    "unemployment": "employment",
    "job": "employment",
    "financial": "financial_strain",
    "financial strain": "financial_strain",
    "financial_strain": "financial_strain",
    "economic": "financial_strain",
    "utility": "utilities",
    "utilities": "utilities",
    "utility needs": "utilities",
    "child care": "childcare",
    "childcare": "childcare",
    "child_care": "childcare",
    "social isolation": "social_isolation",
    "social_isolation": "social_isolation",
    "isolation": "social_isolation",
    "safety": "safety",
    "intimate partner violence": "safety",
    "intimate_partner_violence": "safety",
    "insurance": "health_insurance",
    "health insurance": "health_insurance",
    "health_insurance": "health_insurance",
    "education": "education",
    "literacy": "education",
}
_COMPILED_DETERMINANT_ALIASES = tuple(
    (
        alias,
        canonical,
        _literal_pattern(alias),
    )
    for alias, canonical in sorted(
        _DETERMINANT_ALIASES.items(),
        key=lambda item: (-len(item[0]), item[0]),
    )
)


@dataclass(frozen=True, slots=True)
class SDOHNegatedNeedEvidence:
    """Value-free assertion metadata for one SDOH candidate.

    Args:
        source_offsets: Half-open offsets of the SDOH candidate in the source.
        assertion: ``affirmed`` for an asserted need, ``negated`` for explicit
            negative evidence, or ``unknown`` when the resolver abstains.
        cue_offsets: Offsets of polarity or uncertainty cues used for the
            result. Cue text is intentionally not retained.
        conflicting_assertions: Distinct polarity values that could not be
            reconciled. Double negation records both ``affirmed`` and
            ``negated`` because parity is deliberately not used to guess.
        source: Whether the result came from local cues, caller-provided
            metadata, or the conservative finding default.
        review_required: Whether qualified human review is required.
        input_index: Stable input position for joining back to caller-owned
            findings inside an authorized process.
        determinant: Canonical determinant label, never an arbitrary source
            value.
        double_negation: Whether multiple nested negation cues were observed.
        review_reasons: Controlled reasons for an abstention.
        negation_cue_count: Number of non-redundant negation cues governing the
            candidate.
    """

    source_offsets: SpanOffset
    assertion: str
    cue_offsets: tuple[SpanOffset, ...] = ()
    conflicting_assertions: tuple[str, ...] = ()
    source: ResolutionSource = "default"
    review_required: bool = False
    input_index: int | None = None
    determinant: str = "unknown"
    double_negation: bool = False
    review_reasons: tuple[str, ...] = ()
    negation_cue_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_offsets",
            _validate_offset(self.source_offsets, "source offsets"),
        )
        if self.assertion not in SDOH_NEGATED_NEED_ASSERTIONS:
            raise ValueError("unsupported SDOH need assertion")
        if self.source not in {"cue", "provided", "default"}:
            raise ValueError("unsupported SDOH need resolution source")
        object.__setattr__(
            self,
            "determinant",
            _normalize_determinant(self.determinant),
        )

        cue_offsets = tuple(
            sorted(
                {_validate_offset(offset, "cue offsets") for offset in self.cue_offsets}
            )
        )
        object.__setattr__(self, "cue_offsets", cue_offsets)

        conflicts = set(self.conflicting_assertions)
        if any(value not in SDOH_NEGATED_NEED_ASSERTIONS for value in conflicts):
            raise ValueError("unsupported conflicting SDOH need assertion")
        if self.assertion in conflicts:
            raise ValueError("conflicting assertions must differ from the result")
        ordered_conflicts = tuple(
            value for value in SDOH_NEGATED_NEED_ASSERTIONS if value in conflicts
        )
        object.__setattr__(self, "conflicting_assertions", ordered_conflicts)

        reasons = set(self.review_reasons)
        allowed_reasons = {
            "double_negation",
            "contradictory_cues",
            "uncertain_cue",
            "provided_conflict",
        }
        if any(reason not in allowed_reasons for reason in reasons):
            raise ValueError("unsupported SDOH need review reason")
        ordered_reasons = tuple(
            reason
            for reason in (
                "double_negation",
                "contradictory_cues",
                "uncertain_cue",
                "provided_conflict",
            )
            if reason in reasons
        )
        object.__setattr__(self, "review_reasons", ordered_reasons)

        if type(self.review_required) is not bool:
            raise TypeError("review_required must be a boolean")
        if type(self.double_negation) is not bool:
            raise TypeError("double_negation must be a boolean")
        if (
            isinstance(self.negation_cue_count, bool)
            or not isinstance(self.negation_cue_count, int)
            or self.negation_cue_count < 0
        ):
            raise ValueError("negation cue count must be a non-negative integer")
        if self.double_negation and self.negation_cue_count < 2:
            raise ValueError("double negation requires at least two negation cues")
        if self.assertion == UNKNOWN and not self.review_required:
            raise ValueError("unknown SDOH need assertions require human review")
        if self.conflicting_assertions and (
            self.assertion != UNKNOWN or not self.review_required
        ):
            raise ValueError("conflicting SDOH need assertions require human review")
        if self.double_negation and not self.review_required:
            raise ValueError("double negation requires human review")
        if self.review_reasons and not self.review_required:
            raise ValueError("review reasons require human review")
        if self.input_index is not None and (
            isinstance(self.input_index, bool)
            or not isinstance(self.input_index, int)
            or self.input_index < 0
        ):
            raise ValueError("input index must be a non-negative integer")

    @property
    def source_span(self) -> SpanOffset:
        """Return the candidate's source offsets."""

        return self.source_offsets

    @property
    def source_offset(self) -> SpanOffset:
        """Return the candidate's source offsets."""

        return self.source_offsets

    @property
    def need_status(self) -> str:
        """Return ``present``, ``absent``, or ``unknown`` for the need."""

        return {
            AFFIRMED: NEED_PRESENT,
            NEGATED: NEED_ABSENT,
            UNKNOWN: NEED_UNKNOWN,
        }[self.assertion]

    @property
    def status(self) -> str:
        """Return the assertion label as a compact status alias."""

        return self.assertion

    @property
    def resolution(self) -> str:
        """Return the resolved assertion label."""

        return self.assertion

    @property
    def assertion_result(self) -> str:
        """Return the assertion label using issue terminology."""

        return self.assertion

    @property
    def negation(self) -> str:
        """Return the assertion label for polarity-oriented callers."""

        return self.assertion

    @property
    def need_present(self) -> bool | None:
        """Return whether a need is present, or ``None`` when unresolved."""

        if self.assertion == AFFIRMED:
            return True
        if self.assertion == NEGATED:
            return False
        return None

    @property
    def is_negated(self) -> bool:
        """Return true only for a resolved explicit negation."""

        return self.assertion == NEGATED and not self.review_required

    @property
    def has_conflict(self) -> bool:
        """Return whether polarity could not be resolved without guessing."""

        return bool(self.conflicting_assertions or self.double_negation)

    @property
    def needs_review(self) -> bool:
        """Alias for :attr:`review_required`."""

        return self.review_required

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic metadata without candidate or cue values."""

        return {
            "schema_version": SDOH_NEGATED_NEED_SCHEMA_VERSION,
            "source_offsets": {
                "start": self.source_offsets[0],
                "end": self.source_offsets[1],
            },
            "determinant": self.determinant,
            "assertion": self.assertion,
            "need_status": self.need_status,
            "cue_offsets": [list(offset) for offset in self.cue_offsets],
            "conflicting_assertions": list(self.conflicting_assertions),
            "double_negation": self.double_negation,
            "review_reasons": list(self.review_reasons),
            "review_required": self.review_required,
            "source": self.source,
            "negation_cue_count": self.negation_cue_count,
            "input_index": self.input_index,
        }

    def to_json(self) -> str:
        """Return compact deterministic JSON containing safe metadata only."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SDOHNegatedNeedEvidence":
        """Rebuild a value-free record from :meth:`to_dict` output."""

        if not isinstance(payload, Mapping):
            raise TypeError("SDOH need evidence payload must be a mapping")
        if payload.get("schema_version") != SDOH_NEGATED_NEED_SCHEMA_VERSION:
            raise ValueError("unsupported SDOH need evidence schema version")
        source_payload = payload.get("source_offsets")
        if not isinstance(source_payload, Mapping):
            raise TypeError("SDOH need source offsets are required")
        cue_offsets = payload.get("cue_offsets", ())
        conflicts = payload.get("conflicting_assertions", ())
        reasons = payload.get("review_reasons", ())
        for value, message in (
            (cue_offsets, "SDOH need cue offsets are invalid"),
            (conflicts, "SDOH need conflicts are invalid"),
            (reasons, "SDOH need review reasons are invalid"),
        ):
            if not isinstance(value, Sequence) or isinstance(value, str | bytes):
                raise TypeError(message)
        assertion = payload.get("assertion")
        source = payload.get("source", "default")
        determinant = payload.get("determinant", "unknown")
        review_required = payload.get("review_required")
        double_negation = payload.get("double_negation", False)
        negation_cue_count = payload.get("negation_cue_count", 0)
        input_index = payload.get("input_index")
        if not isinstance(assertion, str) or not isinstance(source, str):
            raise TypeError("SDOH need assertion or source is invalid")
        if not isinstance(determinant, str):
            raise TypeError("SDOH need determinant is invalid")
        if type(review_required) is not bool or type(double_negation) is not bool:
            raise TypeError("SDOH need review flags are invalid")
        if any(not isinstance(value, str) for value in (*conflicts, *reasons)):
            raise TypeError("SDOH need conflict metadata is invalid")
        return cls(
            source_offsets=_validate_offset(
                (source_payload.get("start"), source_payload.get("end")),
                "SDOH need source offsets",
            ),
            assertion=assertion,
            cue_offsets=tuple(cue_offsets),
            conflicting_assertions=tuple(conflicts),
            source=source,  # type: ignore[arg-type]
            review_required=review_required,
            input_index=input_index,
            determinant=determinant,
            double_negation=double_negation,
            review_reasons=tuple(reasons),
            negation_cue_count=negation_cue_count,
        )


# Descriptive aliases keep the record discoverable under both evidence and
# resolution terminology without introducing multiple representations.
SDOHNegatedNeed = SDOHNegatedNeedEvidence
SDOHNegatedNeedResolution = SDOHNegatedNeedEvidence
NegatedNeedResolution = SDOHNegatedNeedEvidence


def resolve_sdoh_negated_needs(
    text: str,
    findings: Iterable[Any] | Any = (),
    *,
    evidence: Iterable[Any] | Any | None = None,
) -> list[SDOHNegatedNeedEvidence]:
    """Attach determinant-scoped need assertions to SDOH findings.

    Args:
        text: Source document text used transiently for local cue matching.
        findings: Candidate SDOH findings exposing ``span`` or ``start`` /
            ``end`` offsets. ``SDOHFinding`` values, mappings, and two-item
            offset sequences are accepted.
        evidence: Keyword alias for ``findings`` used by generic SDOH callers.
            When provided it takes precedence over the positional collection.

    Returns:
        Value-free records sorted by source offsets and then input order. An
        unqualified candidate defaults to an affirmed need because it already
        came from a caller's finding extractor. Explicit negation is marked
        ``negated``; nested, contradictory, or uncertain local cues are
        ``unknown`` and require review.

    Raises:
        TypeError: If the source text, collection, or offsets are malformed.
        ValueError: If a candidate offset is empty or outside the source.
    """

    if not isinstance(text, str):
        raise TypeError("SDOH need source text must be a string")
    items = _evidence_items(findings if evidence is None else evidence)

    records: list[tuple[int, SDOHNegatedNeedEvidence]] = []
    for index, item in enumerate(items):
        offsets = _offset_from_item(item, text)
        records.append(
            (
                index,
                _resolve_one(text, item, offsets, input_index=index),
            )
        )
    records.sort(key=lambda item: (item[1].source_offsets, item[0]))
    return [record for _, record in records]


def resolve_sdoh_negated_need(
    text: str,
    finding: Any = None,
    *,
    evidence: Any = None,
) -> SDOHNegatedNeedEvidence:
    """Resolve exactly one SDOH finding and return its safe assertion record."""

    item = finding if evidence is None else evidence
    records = resolve_sdoh_negated_needs(text, item)
    if len(records) != 1:
        raise ValueError("single SDOH need resolution requires exactly one finding")
    return records[0]


def attach_negated_need_assertions(
    text: str,
    findings: Iterable[Any] | Any = (),
    *,
    evidence: Iterable[Any] | Any | None = None,
) -> list[SDOHNegatedNeedEvidence]:
    """Alias for :func:`resolve_sdoh_negated_needs`."""

    return resolve_sdoh_negated_needs(text, findings, evidence=evidence)


def attach_sdoh_assertions(
    text: str,
    findings: Iterable[Any] | Any = (),
    *,
    evidence: Iterable[Any] | Any | None = None,
) -> list[SDOHNegatedNeedEvidence]:
    """Attach value-free assertion metadata to SDOH findings."""

    return resolve_sdoh_negated_needs(text, findings, evidence=evidence)


def resolve_sdoh_negation(
    text: str,
    findings: Iterable[Any] | Any = (),
    *,
    evidence: Iterable[Any] | Any | None = None,
) -> list[SDOHNegatedNeedEvidence]:
    """Alias using the shorter negation terminology."""

    return resolve_sdoh_negated_needs(text, findings, evidence=evidence)


def resolve_sdoh_evidence(
    text: str,
    evidence: Iterable[Any] | Any = (),
) -> list[SDOHNegatedNeedEvidence]:
    """Resolve SDOH evidence using a generic evidence parameter name."""

    return resolve_sdoh_negated_needs(text, evidence=evidence)


def _resolve_one(
    text: str,
    item: Any,
    offsets: SpanOffset,
    *,
    input_index: int,
) -> SDOHNegatedNeedEvidence:
    determinant = _determinant_from_item(item, text, offsets)
    negative, affirmative, uncertain = _scoped_polarity_hits(
        text,
        offsets,
        determinant,
    )
    negative = _collapse_redundant_negations(text, negative)
    provided = _provided_assertion(item)

    cue_hits = tuple(
        sorted(
            {
                (hit.start, hit.end, hit.kind): hit
                for hit in (*negative, *affirmative, *uncertain)
            }.values(),
            key=lambda hit: (hit.start, hit.end, hit.kind),
        )
    )
    cue_offsets = tuple((hit.start, hit.end) for hit in cue_hits)
    reasons: list[ReviewReason] = []
    conflicts: tuple[str, ...] = ()

    if len(negative) >= 2:
        reasons.append("double_negation")
    if negative and affirmative:
        reasons.append("contradictory_cues")
    if uncertain:
        reasons.append("uncertain_cue")

    local_ambiguous = bool(reasons)
    if local_ambiguous:
        assertion: AssertionValue = UNKNOWN
        conflicts = (AFFIRMED, NEGATED) if negative else ()
        source: ResolutionSource = "cue"
    elif negative:
        assertion = NEGATED
        source = "cue"
    elif affirmative:
        assertion = AFFIRMED
        source = "cue"
    else:
        assertion = AFFIRMED
        source = "default"

    if provided is not None:
        if assertion == AFFIRMED and source == "default":
            assertion = provided
            source = "provided"
        elif provided == UNKNOWN:
            assertion = UNKNOWN
            source = "provided"
            if "uncertain_cue" not in reasons:
                reasons.append("uncertain_cue")
        elif provided != assertion:
            assertion = UNKNOWN
            source = "provided" if not reasons else "cue"
            if "provided_conflict" not in reasons:
                reasons.append("provided_conflict")
            conflicts = (AFFIRMED, NEGATED)

    review_required = assertion == UNKNOWN or bool(reasons)
    if assertion == UNKNOWN and not conflicts and provided in (AFFIRMED, NEGATED):
        conflicts = ()

    return SDOHNegatedNeedEvidence(
        source_offsets=offsets,
        assertion=assertion,
        cue_offsets=cue_offsets,
        conflicting_assertions=conflicts,
        source=source,
        review_required=review_required,
        input_index=input_index,
        determinant=determinant,
        double_negation="double_negation" in reasons,
        review_reasons=tuple(reasons),
        negation_cue_count=len(negative),
    )


def _scoped_polarity_hits(
    text: str,
    target: SpanOffset,
    determinant: str,
) -> tuple[tuple[_CueHit, ...], tuple[_CueHit, ...], tuple[_CueHit, ...]]:
    scope_start, scope_end = _sentence_bounds(text, *target)
    masked = _mask_pseudo_negations(text)
    negative_all = tuple(
        hit
        for hit in _find_hits(
            masked,
            _COMPILED_NEGATION_SPECS,
        )
        if scope_start <= hit.start and hit.end <= scope_end
    )
    uncertain_all = tuple(
        hit
        for hit in _find_hits(text, _COMPILED_UNCERTAINTY_SPECS)
        if scope_start <= hit.start and hit.end <= scope_end
    )
    affirmative_all = tuple(
        hit
        for hit in _find_hits(text, _COMPILED_AFFIRMATION_SPECS)
        if scope_start <= hit.start and hit.end <= scope_end
    )

    negative = tuple(
        hit
        for hit in negative_all
        if _cue_reaches_target(text, hit, target)
        or _contrastive_negation_reaches_target(
            text,
            hit,
            target,
            determinant,
            affirmative_all,
        )
    )
    uncertain = tuple(
        hit
        for hit in uncertain_all
        if _cue_reaches_target(text, hit, target)
        and not any(_overlaps(hit, negative_hit) for negative_hit in negative)
    )

    all_polarity_hits = (*negative_all, *affirmative_all, *uncertain_all)
    negative = tuple(
        hit
        for hit in negative
        if not _cue_crosses_independent_determinant(
            text,
            hit,
            target,
            determinant,
            all_polarity_hits,
        )
    )
    uncertain = tuple(
        hit
        for hit in uncertain
        if not _cue_crosses_independent_determinant(
            text,
            hit,
            target,
            determinant,
            all_polarity_hits,
        )
    )

    affirmative = tuple(
        hit
        for hit in affirmative_all
        if not _overlaps(hit, target)
        and not any(_overlaps(hit, negative_hit) for negative_hit in negative)
        and not any(_overlaps(hit, uncertain_hit) for uncertain_hit in uncertain)
        and (
            _affirmation_reaches_target(text, hit, target, negative_all)
            or _contrastive_affirmation_reaches_target(
                text,
                hit,
                target,
                determinant,
                negative_all,
            )
        )
        and not _cue_crosses_independent_determinant(
            text,
            hit,
            target,
            determinant,
            all_polarity_hits,
        )
    )
    affirmative = tuple(
        hit
        for hit in affirmative
        if not any(
            _affirmation_is_subordinate(text, hit, negative_hit)
            for negative_hit in negative
        )
    )

    # A positive cue between a forward denial and the candidate starts a new
    # determinant assertion (for example, ``denies food and reports transport``)
    # and must stop the earlier denial from leaking into the new candidate.
    negative = tuple(
        hit
        for hit in negative
        if not any(
            _forward_negation_blocked_by_affirmation(text, hit, target, positive)
            for positive in affirmative
        )
        and not _forward_negation_starts_new_comma_clause(
            text,
            hit,
            target,
            affirmative_all,
        )
        and not any(
            _backward_negation_blocked_by_affirmation(
                text,
                hit,
                target,
                positive,
                determinant,
            )
            for positive in affirmative
        )
    )

    # Recompute positive cues after removing a denial that was only shared with
    # a different determinant.  This keeps the returned provenance aligned with
    # the assertion actually used.
    affirmative = tuple(
        hit
        for hit in affirmative
        if not any(
            _affirmation_is_subordinate(text, hit, negative_hit)
            for negative_hit in negative
        )
    )
    return negative, affirmative, uncertain


def _find_hits(
    text: str,
    compiled_specs: Sequence[tuple[_CueSpec, re.Pattern[str]]],
) -> tuple[_CueHit, ...]:
    candidates: list[_CueHit] = []
    for spec, pattern in compiled_specs:
        for match in pattern.finditer(text):
            start, end = match.span()
            candidates.append(
                _CueHit(
                    assertion=spec.assertion,
                    direction=spec.direction,
                    kind=spec.kind,
                    start=start,
                    end=end,
                    expression=spec.expression,
                )
            )

    # Matching each specification separately makes direction explicit but can
    # produce overlapping hits (``no`` inside ``no evidence of``).  Keep the
    # longest match at an offset, then sort by source position.
    candidates.sort(
        key=lambda hit: (
            hit.start,
            -(hit.end - hit.start),
            hit.expression,
        )
    )
    accepted: list[_CueHit] = []
    for hit in candidates:
        if any(_overlaps(hit, previous) for previous in accepted):
            continue
        accepted.append(hit)
    return tuple(sorted(accepted, key=lambda hit: (hit.start, hit.end, hit.kind)))


def _mask_pseudo_negations(text: str) -> str:
    """Mask pseudo-negation and unknown phrases while preserving offsets."""

    return _PSEUDO_NEGATION_RE.sub(
        lambda match: " " * len(match.group(0)),
        text,
    )


def _cue_reaches_target(
    text: str,
    cue: _CueHit,
    target: SpanOffset,
) -> bool:
    if _overlaps(cue, target):
        return True
    target_start, target_end = target
    if cue.direction in {_FORWARD, _BIDIRECTIONAL} and cue.end <= target_start:
        between = text[cue.end : target_start]
        return _bounded_between(between)
    if cue.direction in {_BACKWARD, _BIDIRECTIONAL} and target_end <= cue.start:
        between = text[target_end : cue.start]
        return _bounded_between(between)
    return False


def _affirmation_reaches_target(
    text: str,
    cue: _CueHit,
    target: SpanOffset,
    all_negative: Sequence[_CueHit],
) -> bool:
    if not _cue_reaches_target(text, cue, target):
        return False
    target_start, target_end = target
    if cue.start >= target_end:
        between = text[target_end : cue.start]
        structural = _STRUCTURAL_BOUNDARY_RE.search(between)
        if structural is None:
            return True
        # In ``finding was denied and present`` the positive cue is the
        # contradiction after the negative cue, not a new assertion.  Keep it
        # so the resolver abstains rather than hiding the disagreement.
        return any(
            negative.end <= cue.start
            and target_end <= negative.start
            and _STRUCTURAL_BOUNDARY_RE.search(text[negative.end : cue.start])
            is not None
            for negative in all_negative
        )
    if cue.end <= target_start:
        between = text[cue.end : target_start]
        if any(
            cue.end <= negative.start and negative.end <= target_start
            for negative in all_negative
        ):
            return False
        # A previous determinant assertion separated by a coordinator must not
        # be borrowed when a later denial is attached to the target.
        return not any(
            negative.start >= target_end
            and _STRUCTURAL_BOUNDARY_RE.search(between) is not None
            for negative in all_negative
        )
    return True


def _contrastive_negation_reaches_target(
    text: str,
    cue: _CueHit,
    target: SpanOffset,
    determinant: str,
    affirmations: Sequence[_CueHit],
) -> bool:
    """Retain a same-target denial after a contrastive positive cue."""

    if cue.direction not in {_BACKWARD, _BIDIRECTIONAL} or cue.start < target[1]:
        return False
    if _contains_other_determinant(text, target[1], cue.start, determinant):
        return False
    return any(
        target[1] <= affirmation.start
        and affirmation.end <= cue.start
        and _STRUCTURAL_BOUNDARY_RE.search(text[affirmation.end : cue.start])
        is not None
        for affirmation in affirmations
    )


def _contrastive_affirmation_reaches_target(
    text: str,
    cue: _CueHit,
    target: SpanOffset,
    determinant: str,
    negative: Sequence[_CueHit],
) -> bool:
    """Retain a same-target positive cue after a contrastive denial."""

    if cue.start < target[1]:
        return False
    if _contains_other_determinant(text, target[1], cue.start, determinant):
        return False
    return any(
        target[1] <= negation.start
        and negation.end <= cue.start
        and _STRUCTURAL_BOUNDARY_RE.search(text[negation.end : cue.start]) is not None
        for negation in negative
    )


def _cue_crosses_independent_determinant(
    text: str,
    cue: _CueHit,
    target: SpanOffset,
    determinant: str,
    all_polarity_hits: Sequence[_CueHit],
) -> bool:
    """Reject a cue crossing another determinant with a local assertion.

    A cue can intentionally govern a coordinated list, so an intervening
    determinant is not sufficient to stop its scope.  If the target side of
    that intervening determinant has its own polarity cue, however, treating
    the earlier cue as shared would leak one determinant's assertion into the
    next one.  This symmetric check covers both forward denials and
    post-coordinated backward denials.
    """

    if (
        cue.direction == _BIDIRECTIONAL
        and cue.end <= target[0]
        and _bidirectional_cue_follows_determinant(text, cue, target)
    ):
        return True

    if (
        cue.direction == _BIDIRECTIONAL
        and target[1] <= cue.start
        and _bidirectional_cue_precedes_determinant(text, cue, target)
    ):
        return True

    if cue.end <= target[0]:
        if not _contains_other_determinant(
            text,
            cue.end,
            target[0],
            determinant,
        ):
            return False
        if any(
            cue.end <= hit.start < target[0]
            and hit.start != cue.start
            and _cue_reaches_target(text, hit, target)
            for hit in all_polarity_hits
        ):
            return True
        return any(
            hit.start >= target[1]
            and hit.start != cue.start
            and _cue_reaches_target(text, hit, target)
            for hit in all_polarity_hits
        )

    if target[1] <= cue.start:
        if not _contains_other_determinant(
            text,
            target[1],
            cue.start,
            determinant,
        ):
            return False
        if cue.kind == "affirmation" and any(
            target[1] <= hit.start < cue.start
            for hit in all_polarity_hits
            if hit.start != cue.start
        ):
            return True
        return any(
            hit.end <= target[0]
            and hit.start != cue.start
            and _cue_reaches_target(text, hit, target)
            for hit in all_polarity_hits
        )
    return False


def _bidirectional_cue_follows_determinant(
    text: str,
    cue: _CueHit,
    target: SpanOffset,
) -> bool:
    """Stop a post-target cue from becoming a new target's forward cue."""

    scope_start = _sentence_bounds(text, *target)[0]
    mentions = _determinant_mentions(text, scope_start, cue.start)
    if not mentions:
        return False
    previous_start, previous_end, _previous_determinant = mentions[-1]
    between = text[previous_end : cue.start]
    return _STRUCTURAL_BOUNDARY_RE.search(between) is None


def _bidirectional_cue_precedes_determinant(
    text: str,
    cue: _CueHit,
    target: SpanOffset,
) -> bool:
    """Stop a cue before a new determinant from reaching the prior one."""

    scope_end = _sentence_bounds(text, *target)[1]
    if not _contains_other_determinant(text, cue.end, scope_end, "unknown"):
        return False
    between = text[target[1] : cue.start]
    return _STRUCTURAL_BOUNDARY_RE.search(between) is not None


def _backward_negation_blocked_by_affirmation(
    text: str,
    negation: _CueHit,
    target: SpanOffset,
    affirmation: _CueHit,
    determinant: str,
) -> bool:
    """Keep a later denial from reaching across a positive other determinant."""

    if negation.direction not in {_BACKWARD, _BIDIRECTIONAL}:
        return False
    if not target[1] <= affirmation.start <= affirmation.end <= negation.start:
        return False
    between = text[affirmation.end : negation.start]
    if _COORDINATOR_RE.search(between) is None:
        return False
    return _contains_other_determinant(
        text,
        affirmation.end,
        negation.start,
        determinant,
    )


def _affirmation_is_subordinate(
    text: str,
    affirmation: _CueHit,
    negation: _CueHit,
) -> bool:
    if _overlaps(affirmation, negation):
        return True
    if affirmation.end <= negation.start:
        between = text[affirmation.end : negation.start]
    elif negation.end <= affirmation.start:
        between = text[negation.end : affirmation.start]
    else:
        return True
    # No structural break means constructions such as ``not experiencing X``,
    # ``reports no X``, and ``no X is present``; the positive verb/adjective is
    # part of the negated construction rather than contradictory evidence.
    return _STRUCTURAL_BOUNDARY_RE.search(between) is None


def _forward_negation_blocked_by_affirmation(
    text: str,
    negation: _CueHit,
    target: SpanOffset,
    affirmation: _CueHit,
) -> bool:
    if negation.end > target[0] or affirmation.end > target[0]:
        return False
    if not negation.end <= affirmation.start:
        return False
    if not affirmation.end <= target[0]:
        return False
    return True


def _forward_negation_starts_new_comma_clause(
    text: str,
    negation: _CueHit,
    target: SpanOffset,
    affirmations: Sequence[_CueHit],
) -> bool:
    if negation.direction not in {_FORWARD, _BIDIRECTIONAL}:
        return False
    if negation.end > target[0]:
        return False
    between = text[negation.end : target[0]]
    if "," not in between:
        return False
    return any(affirmation.start >= target[1] for affirmation in affirmations)


def _collapse_redundant_negations(
    text: str,
    hits: Sequence[_CueHit],
) -> tuple[_CueHit, ...]:
    ordered = sorted(hits, key=lambda hit: (hit.start, hit.end))
    retained: list[_CueHit] = []
    for index, hit in enumerate(ordered):
        redundant = False
        for later in ordered[index + 1 :]:
            between = text[hit.end : later.start]
            if _HARD_BOUNDARY_RE.search(between) is not None:
                break
            if _COORDINATOR_RE.search(between) is not None:
                redundant = True
                break
        if not redundant:
            retained.append(hit)
    return tuple(retained)


def _sentence_bounds(text: str, start: int, end: int) -> SpanOffset:
    boundaries = ".!?;:\n。！？；："
    left = max((text.rfind(boundary, 0, start) for boundary in boundaries), default=-1)
    right_candidates = [text.find(boundary, end) for boundary in boundaries]
    right_candidates = [candidate for candidate in right_candidates if candidate >= 0]
    return left + 1, min(right_candidates, default=len(text))


def _bounded_between(text: str) -> bool:
    if _HARD_BOUNDARY_RE.search(text) is not None:
        return False
    return len(_TOKEN_RE.findall(text)) <= _MAX_SCOPE_TOKENS


def _evidence_items(evidence: Iterable[Any] | Any) -> tuple[Any, ...]:
    if isinstance(evidence, Mapping) or _has_offset_field(evidence):
        return (evidence,)
    if _is_offset_sequence(evidence):
        return (evidence,)
    if isinstance(evidence, str | bytes | bytearray):
        raise TypeError("SDOH need evidence must contain source offsets")
    try:
        return tuple(evidence)
    except TypeError:
        raise TypeError("SDOH need evidence must be iterable") from None


def _has_offset_field(value: Any) -> bool:
    return any(
        hasattr(value, name)
        for name in (
            "span",
            "source_span",
            "source_offsets",
            "source_offset",
            "offsets",
            "start",
        )
    )


def _is_offset_sequence(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, str | bytes | bytearray)
        and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    )


def _offset_from_item(item: Any, text: str) -> SpanOffset:
    candidate: Any = item
    for key in (
        "source_offsets",
        "source_offset",
        "offsets",
        "span",
        "source_span",
    ):
        value = _field(item, (key,))
        if value is not None:
            candidate = value
            break

    if isinstance(candidate, Mapping):
        start = _field(candidate, ("start", "source_start", "begin"))
        end = _field(candidate, ("end", "source_end", "stop"))
    elif _is_offset_sequence(candidate):
        start, end = candidate
    else:
        start = _field(item, ("start", "source_start", "start_char", "begin"))
        end = _field(item, ("end", "source_end", "end_char", "stop"))

    if start is None or end is None:
        value = _field(item, ("text", "surface", "term", "value"))
        if isinstance(value, str) and value:
            matches: list[SpanOffset] = []
            cursor = 0
            haystack = text.casefold()
            needle = value.casefold()
            while (position := haystack.find(needle, cursor)) >= 0:
                matches.append((position, position + len(value)))
                cursor = position + 1
            if len(matches) == 1:
                return _validate_offset(matches[0], "SDOH need offsets", len(text))
        raise TypeError("SDOH need evidence requires source offsets")
    return _validate_offset((start, end), "SDOH need offsets", len(text))


def _field(item: Any, names: Sequence[str]) -> Any:
    if isinstance(item, Mapping):
        for name in names:
            if name in item:
                return item[name]
        return None
    for name in names:
        value = getattr(item, name, None)
        if value is not None:
            return value
    return None


def _determinant_from_item(
    item: Any,
    text: str,
    offsets: SpanOffset,
) -> str:
    value = _field(item, ("determinant", "category", "domain", "label", "type"))
    normalized = _normalize_determinant(value)
    if normalized != UNKNOWN:
        return normalized
    target_text = text[offsets[0] : offsets[1]]
    return _normalize_determinant(target_text)


def _normalize_determinant(value: Any) -> str:
    if not isinstance(value, str):
        return UNKNOWN
    normalized = " ".join(value.casefold().replace("-", " ").split())
    if normalized in _DETERMINANT_ALIASES:
        return _DETERMINANT_ALIASES[normalized]
    underscored = normalized.replace(" ", "_")
    if underscored in _DETERMINANT_ALIASES:
        return _DETERMINANT_ALIASES[underscored]
    return UNKNOWN


def _determinant_mentions(
    text: str,
    start: int = 0,
    end: int | None = None,
) -> tuple[tuple[int, int, str], ...]:
    """Return non-overlapping canonical determinant mentions in an interval."""

    upper = len(text) if end is None else end
    candidates: list[tuple[int, int, str]] = []
    for _alias, canonical, pattern in _COMPILED_DETERMINANT_ALIASES:
        candidates.extend(
            (match.start(), match.end(), canonical)
            for match in pattern.finditer(text, start, upper)
        )

    candidates.sort(
        key=lambda mention: (
            mention[0],
            -(mention[1] - mention[0]),
            mention[2],
        )
    )
    accepted: list[tuple[int, int, str]] = []
    for mention in candidates:
        if any(
            mention[0] < previous[1] and previous[0] < mention[1]
            for previous in accepted
        ):
            continue
        accepted.append(mention)
    return tuple(sorted(accepted, key=lambda mention: (mention[0], mention[1])))


def _contains_other_determinant(
    text: str,
    start: int,
    end: int,
    determinant: str,
) -> bool:
    return any(
        canonical != determinant
        for _mention_start, _mention_end, canonical in _determinant_mentions(
            text,
            start,
            end,
        )
    )


def _provided_assertion(item: Any) -> AssertionValue | None:
    value: Any = None
    assertion = _field(item, ("assertion",))
    if isinstance(assertion, Mapping):
        value = _field(assertion, ("assertion", "negation", "polarity", "status"))
    elif assertion is not None:
        value = assertion
    if value is None:
        value = _field(item, ("negation", "polarity", "need_status", "status"))
    if value is None:
        return None
    if hasattr(value, "negation"):
        value = getattr(value, "negation")
    if isinstance(value, bool):
        return NEGATED if value else AFFIRMED
    if not isinstance(value, str):
        return None
    normalized = value.casefold().replace("-", "_").strip()
    if normalized in {
        "negated",
        "negative",
        "absent",
        "no_need",
        "not_present",
        "refuted",
        "none",
        "never",
    }:
        return NEGATED
    if normalized in {"affirmed", "positive", "present", "need", "needed"}:
        return AFFIRMED
    if normalized in {"unknown", "uncertain", "review", "needs_review"}:
        return UNKNOWN
    return None


def _validate_offset(
    value: Any,
    field_name: str,
    text_length: int | None = None,
) -> SpanOffset:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str | bytes)
        or len(value) != 2
    ):
        raise TypeError(f"{field_name} must be a two-item offset sequence")
    start, end = value
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
    ):
        raise TypeError(f"{field_name} must contain integer offsets")
    if start < 0 or end <= start:
        raise ValueError(f"{field_name} must form a non-empty half-open span")
    if text_length is not None and end > text_length:
        raise ValueError(f"{field_name} must be within the source text")
    return start, end


def _overlaps(left: _CueHit, right: SpanOffset | _CueHit) -> bool:
    if isinstance(right, _CueHit):
        right_start, right_end = right.start, right.end
    else:
        right_start, right_end = right
    return left.start < right_end and right_start < left.end


__all__ = [
    "AFFIRMED",
    "NEGATED",
    "UNKNOWN",
    "NEED_PRESENT",
    "NEED_ABSENT",
    "NEED_UNKNOWN",
    "SDOH_DETERMINANTS",
    "SDOH_NEGATED_NEED_ADVISORY",
    "SDOH_NEGATED_NEED_ASSERTIONS",
    "SDOH_NEGATED_NEED_SCHEMA_VERSION",
    "SDOH_NEGATED_NEED_STATUSES",
    "SDOHNegatedNeed",
    "SDOHNegatedNeedEvidence",
    "SDOHNegatedNeedResolution",
    "NegatedNeedResolution",
    "attach_negated_need_assertions",
    "attach_sdoh_assertions",
    "resolve_sdoh_evidence",
    "resolve_sdoh_negated_need",
    "resolve_sdoh_negated_needs",
    "resolve_sdoh_negation",
]
