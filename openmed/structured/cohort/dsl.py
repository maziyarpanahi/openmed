"""Stable declarative phenotype definitions for local cohort resolution.

The schema intentionally models a small, auditable subset of cohort queries:
concept sets, occurrence thresholds, assertion filters, temporal windows, and
recursive boolean composition.  It is not an OHDSI ATLAS/Circe compatibility
layer and never contains executable SQL.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Final, Literal

PHENOTYPE_SCHEMA_VERSION: Final = "openmed.cohort.phenotype.v1"

ExpressionOperator = Literal["criterion", "and", "or", "not"]

_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_NEGATION_VALUES = frozenset({"affirmed", "negated"})
_TEMPORALITY_VALUES = frozenset({"recent", "historical", "hypothetical"})
_CERTAINTY_VALUES = frozenset({"certain", "uncertain"})


class PhenotypeDefinitionError(ValueError):
    """Raised when a phenotype definition violates the stable schema."""


def _identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise PhenotypeDefinitionError(
            f"{field_name} must match {_IDENTIFIER_RE.pattern}"
        )
    return value


def _nonblank(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PhenotypeDefinitionError(f"{field_name} must be a non-blank string")
    return value.strip()


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PhenotypeDefinitionError(f"{field_name} must be an object")
    return value


def _sequence(value: Any, field_name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PhenotypeDefinitionError(f"{field_name} must be an array")
    return value


def _reject_extra(
    payload: Mapping[str, Any], allowed: frozenset[str], field_name: str
) -> None:
    extra = sorted(set(payload).difference(allowed))
    if extra:
        raise PhenotypeDefinitionError(
            f"{field_name} contains unknown fields: {', '.join(extra)}"
        )


def _choice_tuple(
    values: Sequence[str],
    *,
    field_name: str,
    allowed: frozenset[str] | None = None,
) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(_nonblank(value, field_name) for value in values))
    if allowed is not None:
        unknown = sorted(set(normalized).difference(allowed))
        if unknown:
            raise PhenotypeDefinitionError(
                f"{field_name} contains unsupported values: {', '.join(unknown)}"
            )
    return normalized


def _iso_date(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = _nonblank(value, field_name)
    try:
        date.fromisoformat(normalized)
    except ValueError as exc:
        raise PhenotypeDefinitionError(
            f"{field_name} must be an ISO YYYY-MM-DD date"
        ) from exc
    return normalized


@dataclass(frozen=True)
class ConceptSet:
    """A named set of OMOP concept identifiers in one vocabulary."""

    id: str
    vocabulary: str
    concept_ids: tuple[int, ...]
    include_descendants: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _identifier(self.id, "concept_set.id"))
        object.__setattr__(
            self,
            "vocabulary",
            _nonblank(self.vocabulary, "concept_set.vocabulary"),
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in self.concept_ids
        ):
            raise PhenotypeDefinitionError(
                "concept_set.concept_ids must contain integers"
            )
        concept_ids = tuple(sorted(set(self.concept_ids)))
        if not concept_ids or any(value <= 0 for value in concept_ids):
            raise PhenotypeDefinitionError(
                "concept_set.concept_ids must contain positive integers"
            )
        object.__setattr__(self, "concept_ids", concept_ids)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible representation."""

        return {
            "id": self.id,
            "vocabulary": self.vocabulary,
            "concept_ids": list(self.concept_ids),
            "include_descendants": self.include_descendants,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConceptSet":
        """Build a concept set from a strict JSON object."""

        _reject_extra(
            payload,
            frozenset({"id", "vocabulary", "concept_ids", "include_descendants"}),
            "concept_set",
        )
        try:
            concept_ids = _sequence(payload["concept_ids"], "concept_ids")
            include_descendants = payload.get("include_descendants", False)
            if not isinstance(include_descendants, bool):
                raise PhenotypeDefinitionError(
                    "concept_set.include_descendants must be a boolean"
                )
            return cls(
                id=payload["id"],
                vocabulary=payload["vocabulary"],
                concept_ids=tuple(concept_ids),
                include_descendants=include_descendants,
            )
        except KeyError as exc:
            raise PhenotypeDefinitionError(
                f"concept_set is missing required field {exc.args[0]}"
            ) from exc


@dataclass(frozen=True)
class OccurrenceCount:
    """Inclusive occurrence-count bounds for one criterion."""

    minimum: int = 1
    maximum: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.minimum, int) or isinstance(self.minimum, bool):
            raise PhenotypeDefinitionError("occurrence.minimum must be an integer")
        if self.minimum < 1:
            raise PhenotypeDefinitionError("occurrence.minimum must be at least 1")
        minimum = self.minimum
        maximum = self.maximum
        if maximum is not None:
            if not isinstance(maximum, int) or isinstance(maximum, bool):
                raise PhenotypeDefinitionError("occurrence.maximum must be an integer")
            if maximum < minimum:
                raise PhenotypeDefinitionError(
                    "occurrence.maximum must be at least occurrence.minimum"
                )
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    def to_dict(self) -> dict[str, int]:
        """Return canonical inclusive count bounds."""

        payload = {"minimum": self.minimum}
        if self.maximum is not None:
            payload["maximum"] = self.maximum
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OccurrenceCount":
        """Build occurrence bounds from a strict JSON object."""

        _reject_extra(payload, frozenset({"minimum", "maximum"}), "occurrence")
        return cls(
            minimum=payload.get("minimum", 1),
            maximum=payload.get("maximum"),
        )


@dataclass(frozen=True)
class AssertionFilter:
    """Clinical context axes allowed to satisfy one criterion.

    Affirmed mentions are the safe default.  Empty non-negation axes mean that
    the definition does not constrain that axis.
    """

    negation: tuple[str, ...] = ("affirmed",)
    temporality: tuple[str, ...] = ()
    certainty: tuple[str, ...] = ()
    experiencer: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        negation = _choice_tuple(
            self.negation,
            field_name="assertion.negation",
            allowed=_NEGATION_VALUES,
        )
        if not negation:
            raise PhenotypeDefinitionError("assertion.negation must not be empty")
        object.__setattr__(self, "negation", negation)
        object.__setattr__(
            self,
            "temporality",
            _choice_tuple(
                self.temporality,
                field_name="assertion.temporality",
                allowed=_TEMPORALITY_VALUES,
            ),
        )
        object.__setattr__(
            self,
            "certainty",
            _choice_tuple(
                self.certainty,
                field_name="assertion.certainty",
                allowed=_CERTAINTY_VALUES,
            ),
        )
        object.__setattr__(
            self,
            "experiencer",
            _choice_tuple(
                self.experiencer,
                field_name="assertion.experiencer",
            ),
        )

    def to_dict(self) -> dict[str, list[str]]:
        """Return only constrained assertion axes."""

        payload = {"negation": list(self.negation)}
        if self.temporality:
            payload["temporality"] = list(self.temporality)
        if self.certainty:
            payload["certainty"] = list(self.certainty)
        if self.experiencer:
            payload["experiencer"] = list(self.experiencer)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AssertionFilter":
        """Build assertion filters from a strict JSON object."""

        _reject_extra(
            payload,
            frozenset({"negation", "temporality", "certainty", "experiencer"}),
            "assertion",
        )
        return cls(
            negation=tuple(
                _sequence(payload.get("negation", ["affirmed"]), "negation")
            ),
            temporality=tuple(_sequence(payload.get("temporality", []), "temporality")),
            certainty=tuple(_sequence(payload.get("certainty", []), "certainty")),
            experiencer=tuple(_sequence(payload.get("experiencer", []), "experiencer")),
        )


@dataclass(frozen=True)
class TemporalWindow:
    """Absolute and/or criterion-relative inclusive temporal bounds."""

    start_date: str | None = None
    end_date: str | None = None
    anchor_criterion: str | None = None
    days_before: int | None = None
    days_after: int | None = None

    def __post_init__(self) -> None:
        start_date = _iso_date(self.start_date, "temporal.start_date")
        end_date = _iso_date(self.end_date, "temporal.end_date")
        if start_date is not None and end_date is not None and start_date > end_date:
            raise PhenotypeDefinitionError(
                "temporal.start_date must not be after temporal.end_date"
            )
        object.__setattr__(self, "start_date", start_date)
        object.__setattr__(self, "end_date", end_date)

        anchor = self.anchor_criterion
        if anchor is not None:
            anchor = _identifier(anchor, "temporal.anchor_criterion")
            object.__setattr__(self, "anchor_criterion", anchor)
        if (self.days_before is not None or self.days_after is not None) and not anchor:
            raise PhenotypeDefinitionError(
                "temporal day bounds require temporal.anchor_criterion"
            )
        if anchor and self.days_before is None and self.days_after is None:
            raise PhenotypeDefinitionError(
                "temporal.anchor_criterion requires days_before or days_after"
            )
        for field_name in ("days_before", "days_after"):
            value = getattr(self, field_name)
            if value is not None:
                if not isinstance(value, int) or isinstance(value, bool):
                    raise PhenotypeDefinitionError(
                        f"temporal.{field_name} must be an integer"
                    )
                if value < 0:
                    raise PhenotypeDefinitionError(
                        f"temporal.{field_name} must be a non-negative integer"
                    )
        if start_date is None and end_date is None and anchor is None:
            raise PhenotypeDefinitionError(
                "temporal window must contain an absolute or relative bound"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical temporal representation."""

        values = {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "anchor_criterion": self.anchor_criterion,
            "days_before": self.days_before,
            "days_after": self.days_after,
        }
        return {key: value for key, value in values.items() if value is not None}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TemporalWindow":
        """Build a temporal window from a strict JSON object."""

        _reject_extra(
            payload,
            frozenset(
                {
                    "start_date",
                    "end_date",
                    "anchor_criterion",
                    "days_before",
                    "days_after",
                }
            ),
            "temporal",
        )
        return cls(
            start_date=payload.get("start_date"),
            end_date=payload.get("end_date"),
            anchor_criterion=payload.get("anchor_criterion"),
            days_before=payload.get("days_before"),
            days_after=payload.get("days_after"),
        )


@dataclass(frozen=True)
class Criterion:
    """One concept-set match with count, assertion, and temporal constraints."""

    id: str
    concept_set: str
    occurrence: OccurrenceCount = field(default_factory=OccurrenceCount)
    assertion: AssertionFilter = field(default_factory=AssertionFilter)
    temporal: TemporalWindow | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _identifier(self.id, "criterion.id"))
        object.__setattr__(
            self,
            "concept_set",
            _identifier(self.concept_set, "criterion.concept_set"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical criterion representation."""

        payload: dict[str, Any] = {
            "id": self.id,
            "concept_set": self.concept_set,
            "occurrence": self.occurrence.to_dict(),
            "assertion": self.assertion.to_dict(),
        }
        if self.temporal is not None:
            payload["temporal"] = self.temporal.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Criterion":
        """Build a criterion from a strict JSON object."""

        _reject_extra(
            payload,
            frozenset({"id", "concept_set", "occurrence", "assertion", "temporal"}),
            "criterion",
        )
        try:
            occurrence = _mapping(payload.get("occurrence", {}), "occurrence")
            assertion = _mapping(payload.get("assertion", {}), "assertion")
            temporal_value = payload.get("temporal")
            temporal = (
                None
                if temporal_value is None
                else TemporalWindow.from_dict(_mapping(temporal_value, "temporal"))
            )
            return cls(
                id=payload["id"],
                concept_set=payload["concept_set"],
                occurrence=OccurrenceCount.from_dict(occurrence),
                assertion=AssertionFilter.from_dict(assertion),
                temporal=temporal,
            )
        except KeyError as exc:
            raise PhenotypeDefinitionError(
                f"criterion is missing required field {exc.args[0]}"
            ) from exc


@dataclass(frozen=True)
class Expression:
    """A recursive boolean expression whose leaves are criteria."""

    operator: ExpressionOperator
    criterion: Criterion | None = None
    children: tuple["Expression", ...] = ()

    def __post_init__(self) -> None:
        if self.operator not in {"criterion", "and", "or", "not"}:
            raise PhenotypeDefinitionError(
                "expression.operator must be criterion, and, or, or not"
            )
        children = tuple(self.children)
        object.__setattr__(self, "children", children)
        if self.operator == "criterion":
            if self.criterion is None or children:
                raise PhenotypeDefinitionError(
                    "criterion expressions require one criterion and no children"
                )
            return
        if self.criterion is not None:
            raise PhenotypeDefinitionError(
                f"{self.operator} expressions must not contain criterion"
            )
        if self.operator == "not" and len(children) != 1:
            raise PhenotypeDefinitionError("not expressions require exactly one child")
        if self.operator in {"and", "or"} and len(children) < 2:
            raise PhenotypeDefinitionError(
                f"{self.operator} expressions require at least two children"
            )

    @classmethod
    def leaf(cls, criterion: Criterion) -> "Expression":
        """Create a criterion leaf."""

        return cls(operator="criterion", criterion=criterion)

    @classmethod
    def all_of(cls, *children: "Expression") -> "Expression":
        """Create an AND expression."""

        return cls(operator="and", children=tuple(children))

    @classmethod
    def any_of(cls, *children: "Expression") -> "Expression":
        """Create an OR expression."""

        return cls(operator="or", children=tuple(children))

    @classmethod
    def exclude(cls, child: "Expression") -> "Expression":
        """Create a NOT expression relative to the store's patient universe."""

        return cls(operator="not", children=(child,))

    def iter_criteria(self) -> Iterator[Criterion]:
        """Yield criteria in deterministic depth-first definition order."""

        if self.criterion is not None:
            yield self.criterion
        for child in self.children:
            yield from child.iter_criteria()

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical recursive expression representation."""

        if self.operator == "criterion":
            if self.criterion is None:  # pragma: no cover - constructor invariant
                raise PhenotypeDefinitionError("criterion expression is incomplete")
            return {"operator": self.operator, "criterion": self.criterion.to_dict()}
        return {
            "operator": self.operator,
            "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Expression":
        """Build a recursive expression from a strict JSON object."""

        _reject_extra(
            payload,
            frozenset({"operator", "criterion", "children"}),
            "expression",
        )
        operator = payload.get("operator")
        if operator == "criterion":
            criterion = Criterion.from_dict(
                _mapping(payload.get("criterion"), "expression.criterion")
            )
            return cls(operator="criterion", criterion=criterion)
        children = tuple(
            cls.from_dict(_mapping(child, "expression child"))
            for child in _sequence(payload.get("children", []), "expression.children")
        )
        return cls(operator=operator, children=children)  # type: ignore[arg-type]


@dataclass(frozen=True)
class PhenotypeDefinition:
    """A complete, versioned, JSON-stable computable phenotype."""

    id: str
    name: str
    concept_sets: tuple[ConceptSet, ...]
    expression: Expression
    description: str | None = None
    schema_version: str = PHENOTYPE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _identifier(self.id, "phenotype.id"))
        object.__setattr__(self, "name", _nonblank(self.name, "phenotype.name"))
        if self.description is not None:
            object.__setattr__(
                self,
                "description",
                _nonblank(self.description, "phenotype.description"),
            )
        if self.schema_version != PHENOTYPE_SCHEMA_VERSION:
            raise PhenotypeDefinitionError(
                f"unsupported phenotype schema_version {self.schema_version!r}"
            )
        concept_sets = tuple(self.concept_sets)
        object.__setattr__(self, "concept_sets", concept_sets)
        concept_set_ids = [item.id for item in concept_sets]
        if not concept_sets or len(concept_set_ids) != len(set(concept_set_ids)):
            raise PhenotypeDefinitionError(
                "phenotype concept_sets must be non-empty with unique ids"
            )
        criteria = tuple(self.expression.iter_criteria())
        criterion_ids = [item.id for item in criteria]
        if not criteria or len(criterion_ids) != len(set(criterion_ids)):
            raise PhenotypeDefinitionError(
                "phenotype criteria must be non-empty with unique ids"
            )
        unknown_sets = sorted(
            {item.concept_set for item in criteria}.difference(concept_set_ids)
        )
        if unknown_sets:
            raise PhenotypeDefinitionError(
                f"criteria reference unknown concept sets: {', '.join(unknown_sets)}"
            )
        known_criteria = set(criterion_ids)
        for criterion in criteria:
            temporal = criterion.temporal
            if temporal is None or temporal.anchor_criterion is None:
                continue
            if temporal.anchor_criterion not in known_criteria:
                raise PhenotypeDefinitionError(
                    f"criterion {criterion.id} references unknown temporal anchor "
                    f"{temporal.anchor_criterion}"
                )
            if temporal.anchor_criterion == criterion.id:
                raise PhenotypeDefinitionError(
                    f"criterion {criterion.id} cannot anchor its own temporal window"
                )

    def criteria(self) -> tuple[Criterion, ...]:
        """Return all criterion leaves in stable definition order."""

        return tuple(self.expression.iter_criteria())

    def concept_set(self, concept_set_id: str) -> ConceptSet:
        """Return one named concept set."""

        for concept_set in self.concept_sets:
            if concept_set.id == concept_set_id:
                return concept_set
        raise KeyError(concept_set_id)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible definition."""

        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "id": self.id,
            "name": self.name,
            "concept_sets": [item.to_dict() for item in self.concept_sets],
            "expression": self.expression.to_dict(),
        }
        if self.description is not None:
            payload["description"] = self.description
        return payload

    def to_json(self) -> str:
        """Serialize to a byte-stable, whitespace-free JSON string."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def to_json_bytes(self) -> bytes:
        """Serialize to canonical UTF-8 bytes."""

        return self.to_json().encode("utf-8")

    @property
    def sha256(self) -> str:
        """Return the canonical definition digest used by provenance reports."""

        return hashlib.sha256(self.to_json_bytes()).hexdigest()

    def write(self, path: str | Path) -> Path:
        """Write canonical bytes to *path* and return the expanded path."""

        target = Path(path).expanduser()
        target.write_bytes(self.to_json_bytes())
        return target

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PhenotypeDefinition":
        """Build a phenotype from a strict JSON object."""

        _reject_extra(
            payload,
            frozenset(
                {
                    "schema_version",
                    "id",
                    "name",
                    "description",
                    "concept_sets",
                    "expression",
                }
            ),
            "phenotype",
        )
        try:
            concept_sets = tuple(
                ConceptSet.from_dict(_mapping(item, "concept_set"))
                for item in _sequence(payload["concept_sets"], "concept_sets")
            )
            expression = Expression.from_dict(
                _mapping(payload["expression"], "expression")
            )
            return cls(
                id=payload["id"],
                name=payload["name"],
                description=payload.get("description"),
                concept_sets=concept_sets,
                expression=expression,
                schema_version=payload.get("schema_version", PHENOTYPE_SCHEMA_VERSION),
            )
        except KeyError as exc:
            raise PhenotypeDefinitionError(
                f"phenotype is missing required field {exc.args[0]}"
            ) from exc

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> "PhenotypeDefinition":
        """Parse a phenotype JSON document."""

        try:
            payload = json.loads(value)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise PhenotypeDefinitionError("phenotype is not valid JSON") from exc
        return cls.from_dict(_mapping(payload, "phenotype"))

    @classmethod
    def load(cls, path: str | Path) -> "PhenotypeDefinition":
        """Load a phenotype definition from a local JSON file."""

        return cls.from_json(Path(path).expanduser().read_bytes())


def phenotype_to_json(definition: PhenotypeDefinition) -> str:
    """Serialize a phenotype to canonical JSON."""

    return definition.to_json()


def phenotype_from_json(value: str | bytes | bytearray) -> PhenotypeDefinition:
    """Parse canonical or human-formatted phenotype JSON."""

    return PhenotypeDefinition.from_json(value)


__all__ = [
    "PHENOTYPE_SCHEMA_VERSION",
    "AssertionFilter",
    "ConceptSet",
    "Criterion",
    "Expression",
    "OccurrenceCount",
    "PhenotypeDefinition",
    "PhenotypeDefinitionError",
    "TemporalWindow",
    "phenotype_from_json",
    "phenotype_to_json",
]
