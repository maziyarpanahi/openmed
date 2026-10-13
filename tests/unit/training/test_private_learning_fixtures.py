"""Synthetic metadata fixtures integrate with the existing envelope validators."""

import json
from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator

from openmed.training.private_training_schemas import build_schema
from tests.fixtures.private_learning import (
    generate_private_learning_fixture,
    render_private_learning_fixture,
)


def test_reproducibility_and_seed_separation():
    first = render_private_learning_fixture(7)
    assert first == render_private_learning_fixture(7)
    assert first != render_private_learning_fixture(8)
    assert json.loads(first)["synthetic"] is True


def test_current_envelope_schemas_accept_valid_generated_cases():
    for seed in (0, 7, 2**64 - 1):
        case = generate_private_learning_fixture(seed)
        for update in case["updates"]:
            Draft202012Validator(build_schema("federated_update_metadata")).validate(
                update
            )
        Draft202012Validator(build_schema("federated_round_schedule")).validate(
            case["schedule"]
        )
        Draft202012Validator(build_schema("federated_aggregate_metric")).validate(
            case["aggregate_result"]
        )


@pytest.mark.parametrize(
    "variant,field",
    [
        ("capability", "capabilities"),
        ("schedule", "schedule"),
        ("update", "updates"),
        ("dropout", "dropout"),
        ("aggregate", "aggregate_result"),
    ],
)
def test_named_corruptions_are_independent(variant, field):
    clean = generate_private_learning_fixture(7)
    bad = generate_private_learning_fixture(7, variant=variant)
    assert bad["variant"] == variant
    assert clean[field] != bad[field]
    for key in clean.keys() - {field, "variant"}:
        assert clean[key] == bad[key]


@pytest.mark.parametrize(
    "variant,field,schema",
    [
        ("schedule", "schedule", "federated_round_schedule"),
        ("update", "updates", "federated_update_metadata"),
        ("aggregate", "aggregate_result", "federated_aggregate_metric"),
    ],
)
def test_corruptions_are_rejected_by_independent_schemas(variant, field, schema):
    bad = generate_private_learning_fixture(7, variant=variant)[field]
    if field == "updates":
        bad = bad[0]
    if variant == "schedule":
        from openmed.training.federated_schedule import FederatedRoundSchedule

        with pytest.raises(ValueError):
            FederatedRoundSchedule.from_dict(bad)
    else:
        if variant != "update":
            assert list(Draft202012Validator(build_schema(schema)).iter_errors(bad))
        if variant == "update":
            from openmed.training.federated_update_metadata import (
                FederatedParameterMetadata,
                FederatedUpdateMetadata,
                FederatedUpdatePolicy,
            )

            policy = FederatedUpdatePolicy(
                bad["model_digest"],
                (FederatedParameterMetadata("adapter.weight", (2, 8), "float32"),),
            )
            with pytest.raises(ValueError):
                FederatedUpdateMetadata.from_dict(bad, policy=policy)


def test_fresh_results_and_coarse_heterogeneity():
    result = generate_private_learning_fixture(7)
    before = deepcopy(result)
    result["capabilities"][0]["backend"] = "modified"
    assert generate_private_learning_fixture(7) == before
    assert {x["backend"] for x in before["capabilities"]} == {"cpu", "numpy"}


def test_output_contains_no_forbidden_client_or_payload_fields():
    forbidden = {
        "site_id",
        "client_id",
        "patient_count",
        "tensors",
        "gradients",
        "examples",
        "endpoint",
        "path",
        "local_metrics",
    }

    def visit(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for variant in (
        "valid",
        "capability",
        "schedule",
        "update",
        "dropout",
        "aggregate",
    ):
        visit(generate_private_learning_fixture(7, variant=variant))


@pytest.mark.parametrize("seed", [-1, True, 1.5, 2**64])
def test_seed_is_bounded(seed):
    with pytest.raises(ValueError):
        generate_private_learning_fixture(seed)


@pytest.mark.parametrize("size", [0, 129, True, "8"])
def test_cohort_is_bounded_before_generation(size):
    with pytest.raises(ValueError):
        generate_private_learning_fixture(7, cohort_size=size)


def test_unknown_variant_rejected_without_echo():
    with pytest.raises(ValueError) as error:
        generate_private_learning_fixture(7, variant="synthetic-private-value")
    assert "synthetic-private-value" not in str(error.value)


def test_capability_and_dropout_controls_violate_only_their_own_contract():
    valid = generate_private_learning_fixture(7)
    assert all(c["backend"] in {"cpu", "numpy"} for c in valid["capabilities"])
    assert 0 <= valid["dropout"]["dropout_count"] <= valid["dropout"]["submitted_count"]
    capability = generate_private_learning_fixture(7, variant="capability")
    assert any(c["backend"] not in {"cpu", "numpy"} for c in capability["capabilities"])
    dropout = generate_private_learning_fixture(7, variant="dropout")
    assert dropout["dropout"]["dropout_count"] > dropout["dropout"]["submitted_count"]


def test_small_cohort_aggregate_is_suppressed():
    result = generate_private_learning_fixture(7, cohort_size=1)
    assert result["aggregate_result"]["aggregate_value"] is None
    assert result["aggregate_result"]["participant_count_band"] == "suppressed"
