"""Synthetic offline tests for guarded relation directionality."""

from __future__ import annotations

import json
from types import MappingProxyType, SimpleNamespace

import pytest

from openmed.clinical import (
    GUARDED_RELATION_CLASSES,
    GUARDED_RELATION_TYPES,
    RELATION_DIRECTION_RULES,
    InvalidEndpointTypeError,
    InvalidRelationDirectionError,
    UnknownRelationTypeError,
    get_relation_direction_rule,
    relation_direction_rules,
    validate_guarded_relation,
    validate_guarded_relations,
    validate_relation_direction,
)
from openmed.clinical.relations.directionality import RelationShapeError


def test_registry_is_complete_immutable_and_deterministic() -> None:
    rules = relation_direction_rules()

    assert tuple(rule.relation_type for rule in rules) == GUARDED_RELATION_TYPES
    assert (
        tuple(sorted({rule.relation_class for rule in rules}))
        == GUARDED_RELATION_CLASSES
    )
    assert all(rule.direction == "forward" for rule in rules)
    assert all(isinstance(rule.source_types, frozenset) for rule in rules)
    assert all(isinstance(rule.target_types, frozenset) for rule in rules)
    assert isinstance(RELATION_DIRECTION_RULES, MappingProxyType)
    canonical = sorted(
        (rule.to_dict() for rule in rules),
        key=lambda item: item["relation_type"],
    )
    reversed_input = sorted(
        (rule.to_dict() for rule in reversed(rules)),
        key=lambda item: item["relation_type"],
    )
    assert json.dumps(canonical, sort_keys=True) == json.dumps(
        reversed_input, sort_keys=True
    )


@pytest.mark.parametrize(
    ("relation_type", "source", "target", "expected_type"),
    [
        ("causes", "CAUSE", "EFFECT", "causes"),
        ("caused_by", "effect", "cause", "caused_by"),
        ("etiology", "diagnosis", "risk-factor", "etiology"),
        ("treatment", "diagnosis", "therapy", "treatment"),
        ("drug_treatment", "condition", "drug", "drug_treatment"),
        (
            "procedure_to_indication",
            "procedure",
            "diagnosis",
            "procedure_indication",
        ),
        (
            "medication_indication",
            "medication",
            "indication",
            "drug_to_indication",
        ),
    ],
)
def test_registered_relation_directions_accept_canonical_endpoint_roles(
    relation_type: str,
    source: str,
    target: str,
    expected_type: str,
) -> None:
    result = validate_relation_direction(
        relation_type,
        source,
        target,
        direction="source_to_target",
    )

    assert result.relation_type == expected_type
    assert result.direction == "forward"
    assert result.relation_class in GUARDED_RELATION_CLASSES
    assert "text" not in result.to_dict()


def test_every_registered_relation_rejects_reverse_direction() -> None:
    for relation_type in GUARDED_RELATION_TYPES:
        rule = get_relation_direction_rule(relation_type)
        source = next(iter(sorted(rule.source_types)))
        target = next(iter(sorted(rule.target_types)))

        with pytest.raises(InvalidRelationDirectionError) as error:
            validate_relation_direction(
                relation_type,
                source,
                target,
                direction="reverse",
            )

        assert error.value.code == "invalid_direction"
        assert error.value.to_dict()["expected_direction"] == "forward"
        assert error.value.to_dict()["observed_direction"] == "reverse"


@pytest.mark.parametrize(
    ("relation_type", "source", "target", "endpoint"),
    [
        ("treatment", "MEDICATION", "CONDITION", "source"),
        ("procedure_indication", "CONDITION", "PROCEDURE", "source"),
        ("drug_to_indication", "PROCEDURE", "CONDITION", "source"),
        ("causes", "EFFECT", "CAUSE", "source"),
        ("drug_treatment", "CONDITION", "PROCEDURE", "target"),
    ],
)
def test_reversed_endpoint_roles_fail_with_a_typed_error(
    relation_type: str,
    source: str,
    target: str,
    endpoint: str,
) -> None:
    with pytest.raises(InvalidEndpointTypeError) as error:
        validate_relation_direction(relation_type, source, target)

    assert error.value.code == "invalid_endpoint_type"
    assert error.value.endpoint == endpoint
    assert error.value.observed_type in {
        "CONDITION",
        "EFFECT",
        "MEDICATION",
        "PROCEDURE",
    }


def test_guarded_mapping_is_validated_before_score_access_and_omits_surface_text() -> (
    None
):
    class Candidate:
        relation_type = "diagnosis_to_treatment"
        source = {"label": "diagnosis", "text": "SYNTHETIC_SOURCE_SURFACE"}
        target = {"label": "drug", "text": "SYNTHETIC_TARGET_SURFACE"}
        direction = "forward"

        @property
        def score(self) -> float:
            raise AssertionError("directionality must run before confidence scoring")

    result = validate_guarded_relation(Candidate())
    serialized = json.dumps(result.to_dict(), sort_keys=True)

    assert result.relation_type == "diagnosis_treatment"
    assert "SYNTHETIC_SOURCE_SURFACE" not in serialized
    assert "SYNTHETIC_TARGET_SURFACE" not in serialized
    assert '"text"' not in serialized


def test_mapping_and_span_like_object_shapes_are_supported() -> None:
    result = validate_guarded_relation(
        {
            "relation_type": "procedure_indication",
            "head": SimpleNamespace(label="PROCEDURE", start=10, end=20),
            "tail": {"entity_type": "CONDITION", "start": 24, "end": 32},
            "direction": "canonical",
            "confidence": "SYNTHETIC_SCORE_NOT_READ",
        }
    )

    assert result.to_dict() == {
        "schema_version": 1,
        "relation_type": "procedure_indication",
        "relation_class": "procedure_indication",
        "source_type": "PROCEDURE",
        "target_type": "CONDITION",
        "direction": "forward",
    }


def test_subject_object_and_entity_aliases_are_supported() -> None:
    result = validate_guarded_relation(
        {
            "predicate": "diagnosis-treatment",
            "subject": {"entity": "medical_condition"},
            "object": {"category": "drug"},
            "orientation": "direct",
        }
    )

    assert result.relation_type == "diagnosis_treatment"
    assert result.source_type == "CONDITION"
    assert result.target_type == "MEDICATION"


def test_unknown_relation_and_endpoint_errors_do_not_echo_untrusted_values() -> None:
    relation_sentinel = "SYNTHETIC_PRIVATE_RELATION_SURFACE"
    endpoint_sentinel = "SYNTHETIC_PRIVATE_ENDPOINT_SURFACE"

    with pytest.raises(UnknownRelationTypeError) as relation_error:
        validate_relation_direction(
            relation_sentinel,
            {"label": endpoint_sentinel},
            {"label": "CONDITION"},
        )
    assert relation_sentinel not in str(relation_error.value)
    assert relation_sentinel not in json.dumps(relation_error.value.to_dict())

    with pytest.raises(InvalidEndpointTypeError) as endpoint_error:
        validate_relation_direction(
            "treatment",
            {"label": endpoint_sentinel},
            {"label": "MEDICATION"},
        )
    assert endpoint_sentinel not in str(endpoint_error.value)
    assert endpoint_sentinel not in json.dumps(endpoint_error.value.to_dict())


def test_batch_validation_is_input_order_independent() -> None:
    relations = [
        {
            "relation_type": "procedure_indication",
            "source": {"label": "procedure"},
            "target": {"label": "condition"},
        },
        {
            "relation_type": "causes",
            "source": {"label": "cause"},
            "target": {"label": "effect"},
        },
    ]

    first = validate_guarded_relations(relations)
    second = validate_guarded_relations(reversed(relations))

    assert first == second
    assert [result.relation_type for result in first] == [
        "causes",
        "procedure_indication",
    ]


def test_missing_relation_shape_fails_closed() -> None:
    with pytest.raises(RelationShapeError, match="two endpoints"):
        validate_guarded_relation({"relation_type": "causes"})
