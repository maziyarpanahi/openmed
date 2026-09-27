"""Deterministic directionality gates for guarded clinical relations.

Guarded relation producers must establish the semantic source and target before
they calculate confidence or render a review item.  This module provides the
small, local-only contract for that boundary.  It validates canonical relation
types, endpoint semantic types, and the explicitly declared source-to-target
direction while retaining no endpoint text or arbitrary candidate metadata.

The registry covers the causal, treatment, procedure-indication, and
medication-indication relation classes used by the guarded clinical roadmap.
Unknown relation types and unsupported endpoint pairs fail closed so a new
producer cannot silently inherit a plausible but unsafe orientation.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Final, Literal

DIRECTIONALITY_SCHEMA_VERSION: Final = 1
DIRECTIONALITY_ADVISORY: Final = (
    "Guarded relation directionality is a deterministic input-validation gate. "
    "Validate endpoint roles before confidence scoring or clinician review; "
    "this output is assistive and is not a clinical decision."
)

RelationDirection = Literal["forward", "reverse"]


class DirectionalityError(ValueError):
    """Base error for a guarded relation directionality validation failure.

    Error messages and ``to_dict`` output contain only canonical relation
    metadata.  Submitted endpoint labels, relation values, and source text are
    never echoed because callers may pass untrusted clinical records.
    """

    code: ClassVar[str] = "directionality_error"

    def __init__(
        self,
        *,
        relation_type: str | None = None,
        relation_class: str | None = None,
        endpoint: str | None = None,
        expected_types: Iterable[str] = (),
        observed_type: str | None = None,
        expected_direction: RelationDirection | None = None,
        observed_direction: RelationDirection | None = None,
    ) -> None:
        self.relation_type = relation_type
        self.relation_class = relation_class
        self.endpoint = endpoint
        self.expected_types = tuple(sorted(set(expected_types)))
        self.observed_type = observed_type
        self.expected_direction = expected_direction
        self.observed_direction = observed_direction
        super().__init__(self._message())

    def _message(self) -> str:
        """Build a stable, value-free human-readable error message."""

        relation = self.relation_class or "guarded relation"
        if self.code == "unknown_relation_type":
            return "unknown guarded relation type; validation failed closed"
        if self.code == "invalid_relation_shape":
            return "guarded relation must provide a relation type and two endpoints"
        if self.code == "invalid_direction":
            expected = self.expected_direction or "forward"
            observed = self.observed_direction or "unknown"
            return (
                f"{relation} relation direction is invalid; expected {expected}, "
                f"received {observed}"
            )
        endpoint = self.endpoint or "relation"
        expected = ", ".join(self.expected_types) or "a registered type"
        if self.observed_type is None:
            return (
                f"{relation} relation has an unrecognized {endpoint} endpoint "
                f"type; expected one of {expected}"
            )
        return (
            f"{relation} relation has an invalid {endpoint} endpoint type "
            f"{self.observed_type}; expected one of {expected}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic error payload without submitted values."""

        payload: dict[str, Any] = {"code": self.code}
        if self.relation_type is not None:
            payload["relation_type"] = self.relation_type
        if self.relation_class is not None:
            payload["relation_class"] = self.relation_class
        if self.endpoint is not None:
            payload["endpoint"] = self.endpoint
        if self.expected_types:
            payload["expected_types"] = list(self.expected_types)
        if self.observed_type is not None:
            payload["observed_type"] = self.observed_type
        if self.expected_direction is not None:
            payload["expected_direction"] = self.expected_direction
        if self.observed_direction is not None:
            payload["observed_direction"] = self.observed_direction
        return payload


class UnknownRelationTypeError(DirectionalityError):
    """Raised when a relation is not in the guarded direction registry."""

    code = "unknown_relation_type"


class InvalidEndpointTypeError(DirectionalityError):
    """Raised when a source or target endpoint type is not permitted."""

    code = "invalid_endpoint_type"


class InvalidRelationDirectionError(DirectionalityError):
    """Raised when a relation declares the reverse of its guarded direction."""

    code = "invalid_direction"


class RelationShapeError(DirectionalityError):
    """Raised when a relation or endpoint does not expose the required shape."""

    code = "invalid_relation_shape"


# Compatibility aliases make the typed failure contract easy to discover while
# keeping one canonical class hierarchy for callers that catch a base error.
RelationDirectionError = DirectionalityError
DirectionalityValidationError = DirectionalityError
EndpointTypeError = InvalidEndpointTypeError
InvalidDirectionError = InvalidRelationDirectionError


@dataclass(frozen=True, slots=True)
class RelationDirectionRule:
    """Immutable endpoint and direction contract for one relation type.

    Attributes:
        relation_type: Canonical relation predicate.
        relation_class: Guarded family such as ``"causal"`` or ``"treatment"``.
        source_types: Canonical semantic endpoint types allowed on the source.
        target_types: Canonical semantic endpoint types allowed on the target.
        direction: Required orientation, relative to the relation source and
            target fields. Guarded predicates currently require ``"forward"``.
    """

    relation_type: str
    relation_class: str
    source_types: frozenset[str]
    target_types: frozenset[str]
    direction: RelationDirection = "forward"

    def __post_init__(self) -> None:
        if not self.relation_type or not self.relation_class:
            raise ValueError("direction rules require relation metadata")
        if not self.source_types or not self.target_types:
            raise ValueError("direction rules require source and target types")
        if self.direction not in {"forward", "reverse"}:
            raise ValueError("direction rules require a supported direction")
        object.__setattr__(self, "source_types", frozenset(self.source_types))
        object.__setattr__(self, "target_types", frozenset(self.target_types))

    def accepts(self, source_type: str, target_type: str) -> bool:
        """Return whether the ordered endpoint types satisfy this rule."""

        return source_type in self.source_types and target_type in self.target_types

    def to_dict(self) -> dict[str, Any]:
        """Return the value-free rule representation used by audit tooling."""

        return {
            "relation_type": self.relation_type,
            "relation_class": self.relation_class,
            "source_types": sorted(self.source_types),
            "target_types": sorted(self.target_types),
            "direction": self.direction,
            "schema_version": DIRECTIONALITY_SCHEMA_VERSION,
        }


@dataclass(frozen=True, slots=True)
class ValidatedRelationDirection:
    """Validated, privacy-safe direction metadata for a guarded relation.

    The result intentionally contains endpoint types rather than endpoint
    values.  It is suitable for passing to a later scorer or review renderer
    after this gate has succeeded.
    """

    relation_type: str
    relation_class: str
    source_type: str
    target_type: str
    direction: RelationDirection = "forward"
    schema_version: int = DIRECTIONALITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (
                self.relation_type,
                self.relation_class,
                self.source_type,
                self.target_type,
            )
        ):
            raise ValueError("validated relation direction metadata is incomplete")
        if self.direction not in {"forward", "reverse"}:
            raise ValueError("validated relation direction is unsupported")
        if self.schema_version != DIRECTIONALITY_SCHEMA_VERSION:
            raise ValueError("unsupported directionality schema version")

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic direction metadata without source text."""

        return {
            "schema_version": self.schema_version,
            "relation_type": self.relation_type,
            "relation_class": self.relation_class,
            "source_type": self.source_type,
            "target_type": self.target_type,
            "direction": self.direction,
        }


RelationDirectionValidation = ValidatedRelationDirection
RelationDirectionResult = ValidatedRelationDirection


_CONDITION = frozenset({"CONDITION"})
_CAUSES = frozenset({"CAUSE", "CONDITION"})
_EFFECTS = frozenset(
    {"COMPLICATION", "CONDITION", "EFFECT", "FINDING", "OTHER", "SYMPTOM"}
)
_FACTORS = frozenset(
    {
        "AGE",
        "BODY_SITE",
        "CAUSE",
        "CONDITION",
        "GENDER",
        "MEDICATION",
        "MICROORGANISM",
        "OTHER",
        "PROCEDURE",
    }
)
_TREATMENTS = frozenset({"CARE_INTERVENTION", "MEDICATION", "PROCEDURE", "TREATMENT"})


def _rule(
    relation_type: str,
    relation_class: str,
    source_types: frozenset[str],
    target_types: frozenset[str],
) -> RelationDirectionRule:
    return RelationDirectionRule(
        relation_type=relation_type,
        relation_class=relation_class,
        source_types=source_types,
        target_types=target_types,
    )


_RULES: tuple[RelationDirectionRule, ...] = (
    _rule("causal", "causal", _CAUSES, _EFFECTS),
    _rule("caused_by", "causal", _EFFECTS, _CAUSES),
    _rule("causes", "causal", _CAUSES, _EFFECTS),
    _rule("complication", "causal", _CONDITION, _CONDITION),
    _rule("etiology", "causal", _CONDITION, _FACTORS),
    _rule("genetic_factor", "causal", _CONDITION, _FACTORS),
    _rule("high_risk_factor", "causal", _CONDITION, _FACTORS),
    _rule("pathogenesis", "causal", _CONDITION, _FACTORS),
    _rule("risk_assessment_factor", "causal", _CONDITION, _FACTORS),
    _rule("transforms_to", "causal", _CONDITION, _CONDITION),
    _rule("transmission_route", "causal", _CONDITION, _FACTORS),
    _rule("adjuvant_treatment", "treatment", _CONDITION, _TREATMENTS),
    _rule("chemotherapy", "treatment", _CONDITION, _TREATMENTS),
    _rule("diagnosis_treatment", "treatment", _CONDITION, _TREATMENTS),
    _rule("drug_treatment", "treatment", _CONDITION, frozenset({"MEDICATION"})),
    _rule("prevention", "treatment", _CONDITION, _TREATMENTS),
    _rule("radiation_treatment", "treatment", _CONDITION, _TREATMENTS),
    _rule("surgical_treatment", "treatment", _CONDITION, frozenset({"PROCEDURE"})),
    _rule("treatment", "treatment", _CONDITION, _TREATMENTS),
    _rule(
        "procedure_indication",
        "procedure_indication",
        frozenset({"PROCEDURE"}),
        frozenset({"CONDITION", "INDICATION"}),
    ),
    _rule(
        "drug_to_indication",
        "medication_indication",
        frozenset({"MEDICATION"}),
        frozenset({"CONDITION", "INDICATION"}),
    ),
)

RELATION_DIRECTION_RULES: Mapping[str, RelationDirectionRule] = MappingProxyType(
    {rule.relation_type: rule for rule in _RULES}
)
"""Immutable canonical direction rules keyed by guarded relation type."""

GUARDED_RELATION_TYPES: tuple[str, ...] = tuple(sorted(RELATION_DIRECTION_RULES))
"""Canonical relation predicates covered by the directionality gate."""

GUARDED_RELATION_CLASSES: tuple[str, ...] = tuple(
    sorted({rule.relation_class for rule in RELATION_DIRECTION_RULES.values()})
)
"""Guarded relation families represented by :data:`RELATION_DIRECTION_RULES`."""

RELATION_DIRECTION_REGISTRY = RELATION_DIRECTION_RULES

_RELATION_TYPE_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "cause_effect": "causes",
        "causal_relation": "causal",
        "diagnosis_to_treatment": "diagnosis_treatment",
        "drug_indication": "drug_to_indication",
        "medication_indication": "drug_to_indication",
        "procedure_to_indication": "procedure_indication",
        "treatment_relation": "treatment",
    }
)

_ENDPOINT_TYPE_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "age": "AGE",
        "antibiotic": "MEDICATION",
        "body_site": "BODY_SITE",
        "care": "CARE_INTERVENTION",
        "care_intervention": "CARE_INTERVENTION",
        "cause": "CAUSE",
        "causes": "CAUSE",
        "chemical": "MEDICATION",
        "clinical_finding": "FINDING",
        "complication": "COMPLICATION",
        "condition": "CONDITION",
        "diagnosis": "CONDITION",
        "disease": "CONDITION",
        "drug": "MEDICATION",
        "effect": "EFFECT",
        "etiology": "CAUSE",
        "factor": "CAUSE",
        "finding": "FINDING",
        "gender": "GENDER",
        "healthcare_intervention": "CARE_INTERVENTION",
        "indication": "INDICATION",
        "intervention": "TREATMENT",
        "lab_test": "LAB_TEST",
        "medical_condition": "CONDITION",
        "medication": "MEDICATION",
        "microbe": "MICROORGANISM",
        "microorganism": "MICROORGANISM",
        "medicine": "MEDICATION",
        "observation": "FINDING",
        "operation": "PROCEDURE",
        "other": "OTHER",
        "pathogen": "CAUSE",
        "procedure": "PROCEDURE",
        "problem": "CONDITION",
        "risk_factor": "CAUSE",
        "surgery": "PROCEDURE",
        "symptom": "SYMPTOM",
        "therapeutic_procedure": "PROCEDURE",
        "therapy": "TREATMENT",
        "treatment": "TREATMENT",
    }
)

_DIRECTION_ALIASES: Mapping[str, RelationDirection] = MappingProxyType(
    {
        "backward": "reverse",
        "canonical": "forward",
        "direct": "forward",
        "forward": "forward",
        "reverse": "reverse",
        "source_target": "forward",
        "source_to_target": "forward",
        "target_source": "reverse",
        "target_to_source": "reverse",
    }
)

_ENDPOINT_KEYS: tuple[str, ...] = (
    "endpoint_type",
    "semantic_type",
    "semantic_label",
    "semantic_role",
    "canonical_label",
    "source_type",
    "target_type",
    "label",
    "entity",
    "entity_type",
    "category",
    "concept_type",
    "type",
    "kind",
    "role",
)


def _normalized_key(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().casefold()).strip("_")
    if normalized[:2] in {"b_", "e_", "i_", "s_"}:
        normalized = normalized[2:]
    return normalized or None


def _canonical_relation_type(value: Any) -> str | None:
    key = _normalized_key(value)
    if key is None:
        return None
    if key in RELATION_DIRECTION_RULES:
        return key
    return _RELATION_TYPE_ALIASES.get(key)


def _canonical_endpoint_type(value: Any) -> str | None:
    key = _normalized_key(value)
    if key is None:
        return None
    return _ENDPOINT_TYPE_ALIASES.get(key)


def _canonical_direction(value: Any) -> RelationDirection | None:
    if value is None:
        return "forward"
    return _DIRECTION_ALIASES.get(_normalized_key(value) or "")


def _value_from_mapping_or_object(value: Any, keys: Iterable[str]) -> Any:
    if isinstance(value, Mapping):
        for key in keys:
            candidate = value.get(key)
            if candidate is not None:
                return candidate
        return None
    for key in keys:
        try:
            candidate = getattr(value, key, None)
        except Exception:
            candidate = None
        if candidate is not None:
            return candidate
    return None


def _endpoint_type(value: Any) -> str | None:
    if isinstance(value, str):
        return _canonical_endpoint_type(value)
    raw = _value_from_mapping_or_object(value, _ENDPOINT_KEYS)
    return _canonical_endpoint_type(raw)


def get_relation_direction_rule(relation_type: str) -> RelationDirectionRule:
    """Return the immutable rule for a guarded relation predicate.

    Args:
        relation_type: Canonical predicate or a supported spelling alias.

    Raises:
        UnknownRelationTypeError: If the predicate is not guarded by this
            contract. The submitted value is not included in the error.
    """

    canonical = _canonical_relation_type(relation_type)
    if canonical is None:
        raise UnknownRelationTypeError()
    return RELATION_DIRECTION_RULES[canonical]


def relation_direction_rules() -> tuple[RelationDirectionRule, ...]:
    """Return all built-in direction rules in canonical order."""

    return tuple(RELATION_DIRECTION_RULES[key] for key in GUARDED_RELATION_TYPES)


def validate_relation_direction(
    relation_type: str,
    source: Any,
    target: Any,
    *,
    direction: str | None = None,
) -> ValidatedRelationDirection:
    """Validate ordered endpoint types and direction before relation scoring.

    ``source`` and ``target`` may be endpoint labels, mappings, or span-like
    objects exposing one of ``endpoint_type``, ``semantic_type``, ``label``,
    ``entity_type``, or ``type``. Other fields, including source text and
    confidence, are ignored. The validator performs no inference and no
    network access.

    Args:
        relation_type: Guarded canonical predicate or supported alias.
        source: Semantic source endpoint or span-like object.
        target: Semantic target endpoint or span-like object.
        direction: Optional declared orientation. ``"forward"`` and
            ``"source_to_target"`` are equivalent; reverse declarations fail
            for guarded predicates.

    Returns:
        Value-free validated direction metadata for downstream scoring/review.

    Raises:
        UnknownRelationTypeError: If ``relation_type`` is not registered.
        InvalidEndpointTypeError: If an endpoint is missing or not allowed.
        InvalidRelationDirectionError: If the declared direction is reverse or
            otherwise not the rule's required orientation.
    """

    rule = get_relation_direction_rule(relation_type)
    canonical_source = _endpoint_type(source)
    if canonical_source is None or canonical_source not in rule.source_types:
        raise InvalidEndpointTypeError(
            relation_type=rule.relation_type,
            relation_class=rule.relation_class,
            endpoint="source",
            expected_types=rule.source_types,
            observed_type=canonical_source,
        )
    canonical_target = _endpoint_type(target)
    if canonical_target is None or canonical_target not in rule.target_types:
        raise InvalidEndpointTypeError(
            relation_type=rule.relation_type,
            relation_class=rule.relation_class,
            endpoint="target",
            expected_types=rule.target_types,
            observed_type=canonical_target,
        )
    canonical_direction = _canonical_direction(direction)
    if canonical_direction is None or canonical_direction != rule.direction:
        raise InvalidRelationDirectionError(
            relation_type=rule.relation_type,
            relation_class=rule.relation_class,
            expected_direction=rule.direction,
            observed_direction=canonical_direction,
        )
    return ValidatedRelationDirection(
        relation_type=rule.relation_type,
        relation_class=rule.relation_class,
        source_type=canonical_source,
        target_type=canonical_target,
        direction=canonical_direction,
    )


def validate_guarded_relation(
    relation: Any,
    *,
    relation_type: str | None = None,
    source: Any | None = None,
    target: Any | None = None,
    direction: str | None = None,
) -> ValidatedRelationDirection:
    """Validate a relation object without reading confidence or source text.

    The relation may be a mapping or an object with ``relation_type`` (or
    ``relation_class``), ``source``/``head``, ``target``/``tail``, and an
    optional ``direction``. Explicit keyword arguments override fields on the
    object. This makes the function a suitable first boundary in a guarded
    producer: invalid directionality raises before any score or review payload
    is inspected.

    Raises:
        RelationShapeError: If the relation type or endpoint fields are absent.
        DirectionalityError: A typed directionality failure from
            :func:`validate_relation_direction`.
    """

    candidate_relation_type = relation_type
    candidate_source = source
    candidate_target = target
    candidate_direction = direction

    if isinstance(relation, str):
        if candidate_relation_type is None:
            candidate_relation_type = relation
    elif isinstance(relation, Mapping):
        if candidate_relation_type is None:
            candidate_relation_type = _value_from_mapping_or_object(
                relation,
                (
                    "relation_type",
                    "predicate",
                    "relation",
                    "relation_class",
                    "type",
                    "label",
                ),
            )
        if candidate_source is None:
            candidate_source = _value_from_mapping_or_object(
                relation, ("source", "subject", "head", "cause", "from")
            )
        if candidate_target is None:
            candidate_target = _value_from_mapping_or_object(
                relation,
                ("target", "object", "tail", "attribute", "effect", "to"),
            )
        if candidate_direction is None:
            candidate_direction = _value_from_mapping_or_object(
                relation, ("direction", "orientation", "relation_direction")
            )
    elif relation is not None:
        if candidate_relation_type is None:
            candidate_relation_type = _value_from_mapping_or_object(
                relation,
                (
                    "relation_type",
                    "predicate",
                    "relation",
                    "relation_class",
                    "type",
                    "label",
                ),
            )
        if candidate_source is None:
            candidate_source = _value_from_mapping_or_object(
                relation, ("source", "subject", "head", "cause", "from")
            )
        if candidate_target is None:
            candidate_target = _value_from_mapping_or_object(
                relation,
                ("target", "object", "tail", "attribute", "effect", "to"),
            )
        if candidate_direction is None:
            candidate_direction = _value_from_mapping_or_object(
                relation, ("direction", "orientation", "relation_direction")
            )

    if (
        candidate_relation_type is None
        or candidate_source is None
        or candidate_target is None
    ):
        raise RelationShapeError()
    return validate_relation_direction(
        candidate_relation_type,
        candidate_source,
        candidate_target,
        direction=candidate_direction,
    )


def validate_guarded_relations(
    relations: Iterable[Any],
) -> tuple[ValidatedRelationDirection, ...]:
    """Validate a batch and return value-free results in deterministic order."""

    if isinstance(relations, (str, bytes, bytearray, Mapping)):
        raise RelationShapeError()
    try:
        validated = tuple(validate_guarded_relation(relation) for relation in relations)
    except DirectionalityError:
        raise
    except Exception:
        raise RelationShapeError() from None
    return tuple(
        sorted(
            validated,
            key=lambda result: (
                result.relation_class,
                result.relation_type,
                result.source_type,
                result.target_type,
                result.direction,
            ),
        )
    )


__all__ = [
    "DIRECTIONALITY_ADVISORY",
    "DIRECTIONALITY_SCHEMA_VERSION",
    "DirectionalityError",
    "DirectionalityValidationError",
    "EndpointTypeError",
    "GUARDED_RELATION_CLASSES",
    "GUARDED_RELATION_TYPES",
    "InvalidDirectionError",
    "InvalidEndpointTypeError",
    "InvalidRelationDirectionError",
    "RelationDirection",
    "RelationDirectionError",
    "RelationDirectionResult",
    "RelationDirectionRule",
    "RelationDirectionValidation",
    "RELATION_DIRECTION_REGISTRY",
    "RELATION_DIRECTION_RULES",
    "RelationShapeError",
    "ValidatedRelationDirection",
    "UnknownRelationTypeError",
    "get_relation_direction_rule",
    "relation_direction_rules",
    "validate_guarded_relation",
    "validate_guarded_relations",
    "validate_relation_direction",
]
