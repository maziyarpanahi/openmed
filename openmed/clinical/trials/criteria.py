"""Deterministic parsing of public trial eligibility criteria."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from openmed.clinical.journey_contracts import canonical_digest, derived_opaque_id

from .contracts import TrialContractError, TrialStudyRecord

TRIAL_CRITERIA_SCHEMA_VERSION: Final = "1.0.0"
TRIAL_CRITERIA_COMPATIBILITY_POLICY: Final = "same_major"
TRIAL_CRITERIA_PARSER_VERSION: Final = "openmed.trial-criteria/1.0.0"

_CONTROLLED_RE = re.compile(r"^[a-z][a-z0-9_.:/-]{0,127}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_OPAQUE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}_[A-Za-z0-9_-]{16,128}$")
_NCT_ID_RE = re.compile(r"^NCT[0-9]{8}$")
_BULLET_RE = re.compile(r"^\s*(?:[-*•]+|[0-9]+[.)])\s*")
_CONCEPT_RE = re.compile(
    r"^(?P<kind>condition|medication|procedure|observation)\s*:\s*"
    r"(?P<concept>[^;]+?)\s*(?:;\s*within\s+(?P<days>[0-9]+)\s+days?)?$",
    re.IGNORECASE,
)
_COMPARISON_RE = re.compile(
    r"^(?P<concept>[A-Za-z][A-Za-z0-9 _./-]{0,127}?)\s*"
    r"(?P<operator>>=|<=|!=|=|>|<)\s*"
    r"(?P<value>-?[0-9]+(?:\.[0-9]+)?)"
    r"(?:\s+(?P<unit>[A-Za-z%][A-Za-z0-9%_./-]{0,31}))?"
    r"(?:\s+within\s+(?P<days>[0-9]+)\s+days?)?$",
    re.IGNORECASE,
)
_AGE_OLDER_RE = re.compile(
    r"^age\s+(?P<value>[0-9]+(?:\.[0-9]+)?)\s*"
    r"(?P<unit>years?|months?)?\s+(?:or\s+)?older$",
    re.IGNORECASE,
)
_AGE_YOUNGER_RE = re.compile(
    r"^age\s+(?P<value>[0-9]+(?:\.[0-9]+)?)\s*"
    r"(?P<unit>years?|months?)?\s+(?:or\s+)?younger$",
    re.IGNORECASE,
)


class TrialCriterionKind(str, Enum):
    """Eligibility section semantics."""

    INCLUSION = "inclusion"
    EXCLUSION = "exclusion"


class TrialCriterionOperator(str, Enum):
    """Supported deterministic criterion operators."""

    EXISTS = "exists"
    EQ = "eq"
    NE = "ne"
    LT = "lt"
    LTE = "lte"
    GT = "gt"
    GTE = "gte"


_OPERATOR_MAP = {
    "=": TrialCriterionOperator.EQ,
    "!=": TrialCriterionOperator.NE,
    "<": TrialCriterionOperator.LT,
    "<=": TrialCriterionOperator.LTE,
    ">": TrialCriterionOperator.GT,
    ">=": TrialCriterionOperator.GTE,
}


@dataclass(frozen=True, slots=True)
class TrialCriterionValue:
    """Typed public threshold extracted from one criterion."""

    value_type: str
    normalized_value: str
    unit: str | None = None

    def __post_init__(self) -> None:
        _controlled(self.value_type, "value_type")
        _bounded_text(self.normalized_value, "normalized_value", 256)
        if self.unit is not None:
            _controlled(self.unit.casefold(), "unit")
            object.__setattr__(self, "unit", self.unit.casefold())

    def to_dict(self) -> dict[str, str | None]:
        """Return the typed public threshold."""

        return {
            "normalized_value": self.normalized_value,
            "unit": self.unit,
            "value_type": self.value_type,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrialCriterionValue":
        """Parse one strict threshold."""

        data = _mapping(value, "criterion value")
        _exact_keys(data, {"normalized_value", "unit", "value_type"}, "criterion value")
        return cls(
            value_type=_text(data["value_type"], "value_type"),
            normalized_value=_text(data["normalized_value"], "normalized_value"),
            unit=_optional_text(data["unit"], "unit"),
        )


@dataclass(frozen=True, slots=True)
class TrialCriterionWindow:
    """Public recency constraint relative to the named Journey snapshot."""

    within_days: int

    def __post_init__(self) -> None:
        if type(self.within_days) is not int or not 1 <= self.within_days <= 36_500:
            raise TrialContractError("within_days must be between 1 and 36500")

    def to_dict(self) -> dict[str, int]:
        """Return the recency window."""

        return {"within_days": self.within_days}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrialCriterionWindow":
        """Parse one strict recency window."""

        data = _mapping(value, "criterion window")
        _exact_keys(data, {"within_days"}, "criterion window")
        return cls(within_days=_integer(data["within_days"], "within_days"))


@dataclass(frozen=True, slots=True)
class TrialCriterion:
    """One parsed or explicitly unsupported public criterion fragment."""

    criterion_id: str
    kind: TrialCriterionKind
    source_fragment: str
    concept_kind: str | None
    concept: str | None
    operator: TrialCriterionOperator | None
    value: TrialCriterionValue | None
    window: TrialCriterionWindow | None
    unsupported_reason: str | None

    def __post_init__(self) -> None:
        _opaque_id(self.criterion_id, "criterion_id")
        object.__setattr__(self, "kind", _kind(self.kind))
        _bounded_text(self.source_fragment, "source_fragment", 16_384)
        if self.unsupported_reason is not None:
            _controlled(self.unsupported_reason, "unsupported_reason")
            if any(
                item is not None
                for item in (
                    self.concept_kind,
                    self.concept,
                    self.operator,
                    self.value,
                    self.window,
                )
            ):
                raise TrialContractError(
                    "unsupported criterion cannot declare executable semantics"
                )
            return
        if self.concept_kind is None or self.concept is None or self.operator is None:
            raise TrialContractError(
                "supported criterion requires concept and operator"
            )
        _controlled(self.concept_kind, "concept_kind")
        _bounded_text(self.concept, "concept", 2048)
        operator = _operator(self.operator)
        if operator is TrialCriterionOperator.EXISTS and self.value is not None:
            raise TrialContractError("exists criterion cannot declare a value")
        if operator is not TrialCriterionOperator.EXISTS and self.value is None:
            raise TrialContractError("comparison criterion requires a value")
        if self.value is not None and not isinstance(self.value, TrialCriterionValue):
            raise TypeError("value must be TrialCriterionValue")
        if self.value is not None and self.value.value_type != "number":
            raise TrialContractError("comparison criterion value_type is unsupported")
        if self.window is not None and not isinstance(
            self.window, TrialCriterionWindow
        ):
            raise TypeError("window must be TrialCriterionWindow")
        object.__setattr__(self, "operator", operator)

    @property
    def supported(self) -> bool:
        """Return whether the fragment has deterministic executable semantics."""

        return self.unsupported_reason is None

    def to_dict(self) -> dict[str, Any]:
        """Return the complete public criterion."""

        return {
            "concept": self.concept,
            "concept_kind": self.concept_kind,
            "criterion_id": self.criterion_id,
            "kind": self.kind.value,
            "operator": self.operator.value if self.operator else None,
            "source_fragment": self.source_fragment,
            "supported": self.supported,
            "unsupported_reason": self.unsupported_reason,
            "value": self.value.to_dict() if self.value else None,
            "window": self.window.to_dict() if self.window else None,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrialCriterion":
        """Parse one strict criterion and verify derived state."""

        data = _mapping(value, "trial criterion")
        _exact_keys(
            data,
            {
                "concept",
                "concept_kind",
                "criterion_id",
                "kind",
                "operator",
                "source_fragment",
                "supported",
                "unsupported_reason",
                "value",
                "window",
            },
            "trial criterion",
        )
        result = cls(
            criterion_id=_text(data["criterion_id"], "criterion_id"),
            kind=_kind(data["kind"]),
            source_fragment=_text(data["source_fragment"], "source_fragment"),
            concept_kind=_optional_text(data["concept_kind"], "concept_kind"),
            concept=_optional_text(data["concept"], "concept"),
            operator=None if data["operator"] is None else _operator(data["operator"]),
            value=None
            if data["value"] is None
            else TrialCriterionValue.from_dict(_mapping(data["value"], "value")),
            window=None
            if data["window"] is None
            else TrialCriterionWindow.from_dict(_mapping(data["window"], "window")),
            unsupported_reason=_optional_text(
                data["unsupported_reason"], "unsupported_reason"
            ),
        )
        if data["supported"] is not result.supported:
            raise TrialContractError("persisted criterion support state differs")
        return result


@dataclass(frozen=True, slots=True)
class ParsedTrialCriteria:
    """Version-bound criteria parsed from one public study version."""

    study_id: str
    study_version_id: str
    study_version_digest: str
    criteria: tuple[TrialCriterion, ...]
    parser_version: str = TRIAL_CRITERIA_PARSER_VERSION
    schema_version: str = TRIAL_CRITERIA_SCHEMA_VERSION
    compatibility_policy: str = TRIAL_CRITERIA_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        _contract(self.schema_version, self.compatibility_policy)
        _opaque_id(self.study_version_id, "study_version_id")
        _digest(self.study_version_digest, "study_version_digest")
        if (
            not isinstance(self.study_id, str)
            or _NCT_ID_RE.fullmatch(self.study_id) is None
        ):
            raise TrialContractError("study_id must be an NCT identifier")
        _bounded_text(self.parser_version, "parser_version", 128)
        criteria = tuple(self.criteria)
        if any(not isinstance(item, TrialCriterion) for item in criteria):
            raise TypeError("criteria must contain TrialCriterion")
        if len({item.criterion_id for item in criteria}) != len(criteria):
            raise TrialContractError("criterion identifiers must be unique")
        object.__setattr__(self, "criteria", criteria)

    @property
    def parse_digest(self) -> str:
        """Return the exact parser output digest."""

        return canonical_digest(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "compatibility_policy": self.compatibility_policy,
            "criteria": [item.to_dict() for item in self.criteria],
            "parser_version": self.parser_version,
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "study_version_digest": self.study_version_digest,
            "study_version_id": self.study_version_id,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic parsed criteria with source custody."""

        return {**self._payload(), "parse_digest": self.parse_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ParsedTrialCriteria":
        """Parse and verify a strict criteria artifact."""

        data = _mapping(value, "parsed trial criteria")
        expected = {
            "compatibility_policy",
            "criteria",
            "parse_digest",
            "parser_version",
            "schema_version",
            "study_id",
            "study_version_digest",
            "study_version_id",
        }
        _exact_keys(data, expected, "parsed trial criteria")
        result = cls(
            study_id=_text(data["study_id"], "study_id"),
            study_version_id=_text(data["study_version_id"], "study_version_id"),
            study_version_digest=_text(
                data["study_version_digest"], "study_version_digest"
            ),
            criteria=tuple(
                TrialCriterion.from_dict(_mapping(item, "criterion"))
                for item in _sequence(data["criteria"], "criteria")
            ),
            parser_version=_text(data["parser_version"], "parser_version"),
            schema_version=_text(data["schema_version"], "schema_version"),
            compatibility_policy=_text(
                data["compatibility_policy"], "compatibility_policy"
            ),
        )
        if data["parse_digest"] != result.parse_digest:
            raise TrialContractError("parsed criteria digest differs")
        return result


def parse_trial_criteria(study: TrialStudyRecord) -> ParsedTrialCriteria:
    """Parse supported fragments and retain every unsupported fragment."""

    if not isinstance(study, TrialStudyRecord):
        raise TypeError("study must be TrialStudyRecord")
    fragments = _criterion_fragments(study.eligibility_text)
    criteria = tuple(
        _parse_fragment(kind, fragment, index, study_version_id=study.version_id)
        for index, (kind, fragment) in enumerate(fragments)
    )
    if not criteria:
        criteria = (
            _unsupported(
                TrialCriterionKind.INCLUSION,
                "Eligibility criteria not supplied",
                0,
                "criteria_missing",
                study_version_id=study.version_id,
            ),
        )
    return ParsedTrialCriteria(
        study_id=study.study_id,
        study_version_id=study.version_id,
        study_version_digest=study.version_digest,
        criteria=criteria,
    )


def _criterion_fragments(text: str) -> tuple[tuple[TrialCriterionKind, str], ...]:
    section = TrialCriterionKind.INCLUSION
    output: list[tuple[TrialCriterionKind, str]] = []
    for raw_line in text.splitlines():
        line = _BULLET_RE.sub("", raw_line).strip()
        if not line:
            continue
        lowered = line.casefold().rstrip(":")
        if lowered in {"inclusion", "inclusion criteria"}:
            section = TrialCriterionKind.INCLUSION
            continue
        if lowered in {"exclusion", "exclusion criteria"}:
            section = TrialCriterionKind.EXCLUSION
            continue
        output.append((section, line))
    return tuple(output)


def _parse_fragment(
    kind: TrialCriterionKind,
    fragment: str,
    index: int,
    *,
    study_version_id: str,
) -> TrialCriterion:
    concept_match = _CONCEPT_RE.fullmatch(fragment)
    if concept_match:
        window = _window(concept_match.group("days"))
        return _criterion(
            kind,
            fragment,
            index,
            concept_kind=concept_match.group("kind").casefold(),
            concept=concept_match.group("concept").strip(),
            operator=TrialCriterionOperator.EXISTS,
            value=None,
            window=window,
            study_version_id=study_version_id,
        )
    comparison = _COMPARISON_RE.fullmatch(fragment)
    if comparison:
        normalized = _normalize_number(comparison.group("value"))
        return _criterion(
            kind,
            fragment,
            index,
            concept_kind="observation",
            concept=comparison.group("concept").strip(),
            operator=_OPERATOR_MAP[comparison.group("operator")],
            value=TrialCriterionValue(
                value_type="number",
                normalized_value=normalized,
                unit=comparison.group("unit"),
            ),
            window=_window(comparison.group("days")),
            study_version_id=study_version_id,
        )
    for pattern, operator in (
        (_AGE_OLDER_RE, TrialCriterionOperator.GTE),
        (_AGE_YOUNGER_RE, TrialCriterionOperator.LTE),
    ):
        age = pattern.fullmatch(fragment)
        if age:
            return _criterion(
                kind,
                fragment,
                index,
                concept_kind="observation",
                concept="age",
                operator=operator,
                value=TrialCriterionValue(
                    value_type="number",
                    normalized_value=_normalize_number(age.group("value")),
                    unit=(age.group("unit") or "years").casefold(),
                ),
                window=None,
                study_version_id=study_version_id,
            )
    return _unsupported(
        kind,
        fragment,
        index,
        "fragment_unsupported",
        study_version_id=study_version_id,
    )


def _criterion(
    kind: TrialCriterionKind,
    fragment: str,
    index: int,
    *,
    concept_kind: str,
    concept: str,
    operator: TrialCriterionOperator,
    value: TrialCriterionValue | None,
    window: TrialCriterionWindow | None,
    study_version_id: str,
) -> TrialCriterion:
    return TrialCriterion(
        criterion_id=derived_opaque_id(
            "trialcriterion", study_version_id, kind.value, index, fragment
        ),
        kind=kind,
        source_fragment=fragment,
        concept_kind=concept_kind,
        concept=concept,
        operator=operator,
        value=value,
        window=window,
        unsupported_reason=None,
    )


def _unsupported(
    kind: TrialCriterionKind,
    fragment: str,
    index: int,
    reason: str,
    *,
    study_version_id: str,
) -> TrialCriterion:
    return TrialCriterion(
        criterion_id=derived_opaque_id(
            "trialcriterion", study_version_id, kind.value, index, fragment
        ),
        kind=kind,
        source_fragment=fragment,
        concept_kind=None,
        concept=None,
        operator=None,
        value=None,
        window=None,
        unsupported_reason=reason,
    )


def _window(days: str | None) -> TrialCriterionWindow | None:
    return None if days is None else TrialCriterionWindow(within_days=int(days))


def _normalize_number(value: str) -> str:
    return format(float(value), ".15g")


def _contract(schema_version: str, compatibility_policy: str) -> None:
    if compatibility_policy != TRIAL_CRITERIA_COMPATIBILITY_POLICY:
        raise TrialContractError("unsupported trial criteria compatibility policy")
    if not isinstance(schema_version, str) or not schema_version.startswith(
        f"{TRIAL_CRITERIA_SCHEMA_VERSION.split('.', 1)[0]}."
    ):
        raise TrialContractError("unsupported trial criteria schema version")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TrialContractError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TrialContractError(f"{name} must be an array")
    return value


def _exact_keys(data: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(data) != expected:
        raise TrialContractError(f"{name} fields do not match the contract")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrialContractError(f"{name} must be non-empty text")
    return value


def _optional_text(value: Any, name: str) -> str | None:
    return None if value is None else _text(value, name)


def _bounded_text(value: Any, name: str, limit: int) -> str:
    text = _text(value, name)
    if len(text.encode("utf-8")) > limit:
        raise TrialContractError(f"{name} exceeds the byte limit")
    return text


def _integer(value: Any, name: str) -> int:
    if type(value) is not int:
        raise TrialContractError(f"{name} must be an integer")
    return value


def _controlled(value: Any, name: str) -> str:
    if not isinstance(value, str) or _CONTROLLED_RE.fullmatch(value) is None:
        raise TrialContractError(f"{name} must be controlled text")
    return value


def _opaque_id(value: Any, name: str) -> str:
    if not isinstance(value, str) or _OPAQUE_ID_RE.fullmatch(value) is None:
        raise TrialContractError(f"{name} must be an opaque identifier")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise TrialContractError(f"{name} must be a normalized SHA-256 digest")
    return value


def _kind(value: Any) -> TrialCriterionKind:
    try:
        return (
            value
            if isinstance(value, TrialCriterionKind)
            else TrialCriterionKind(value)
        )
    except (TypeError, ValueError):
        raise TrialContractError("criterion kind is unsupported") from None


def _operator(value: Any) -> TrialCriterionOperator:
    try:
        return (
            value
            if isinstance(value, TrialCriterionOperator)
            else TrialCriterionOperator(value)
        )
    except (TypeError, ValueError):
        raise TrialContractError("criterion operator is unsupported") from None
