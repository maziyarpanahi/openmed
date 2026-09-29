"""Tests for deterministic, content-free client-update clipping."""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import random

import pytest

import openmed.training as training
from openmed.training.federated import update_clipping as clipping
from tests.fixtures.private_learning_forbidden import FORBIDDEN_FIELD_CASES

_CASE_IDS = [case.reason_code for case in FORBIDDEN_FIELD_CASES]
_LAYER = "adapter.lora_A.weight"
_OTHER_LAYER = "adapter.lora_B.weight"
_SLACK = 1.0 + 1.0e-12


def _policy(**overrides: object) -> clipping.FederatedClippingPolicy:
    arguments: dict[str, object] = {"global_norm_bound": 1.0}
    arguments.update(overrides)
    return clipping.FederatedClippingPolicy(**arguments)  # type: ignore[arg-type]


def _norm(values: tuple[float, ...]) -> float:
    return math.sqrt(math.fsum(value * value for value in values))


def test_schema_version_is_stable() -> None:
    assert (
        clipping.UPDATE_CLIPPING_SCHEMA_VERSION
        == "openmed.training.federated.update_clipping.v1"
    )


def test_reason_codes_are_frozen_and_complete() -> None:
    assert isinstance(clipping.CLIPPING_REASON_CODES, frozenset)
    assert clipping.CLIPPING_REASON_CODES == {
        "within_bound",
        "zero_norm",
        "scaled_to_global_bound",
        "scaled_to_layer_bound",
    }


def test_limits_are_positive_and_ordered() -> None:
    assert 0 < clipping.MAX_CLIPPING_ELEMENTS_PER_LAYER
    assert (
        clipping.MAX_CLIPPING_ELEMENTS_PER_LAYER <= clipping.MAX_CLIPPING_TOTAL_ELEMENTS
    )
    assert 0 < clipping.MAX_CLIPPING_LAYERS
    assert 0.0 < clipping.MAX_CLIPPING_NORM_BOUND
    assert 0.0 < clipping.MAX_CLIPPING_VALUE


def test_policy_parses_bounds_and_applies_the_global_default() -> None:
    policy = _policy(per_layer_bounds=((_LAYER, 4.0),))

    assert policy.global_norm_bound == 1.0
    assert policy.layer_names == (_LAYER,)
    assert policy.bound_for(_LAYER) == 4.0
    assert policy.bound_for(_OTHER_LAYER) == 1.0


def test_policy_canonicalizes_layer_order_and_numeric_bounds() -> None:
    policy = _policy(
        per_layer_bounds=((_OTHER_LAYER, 2), (_LAYER, 4.0)),
    )

    assert policy.layer_names == (_LAYER, _OTHER_LAYER)
    assert type(policy.bound_for(_LAYER)) is float
    assert policy.to_dict()["per_layer_bounds"] == [
        {"layer": _LAYER, "norm_bound": 4.0},
        {"layer": _OTHER_LAYER, "norm_bound": 2.0},
    ]


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        "1.0",
        None,
        0,
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
        float("-inf"),
        clipping.MAX_CLIPPING_NORM_BOUND * 10.0,
        1 << 200,
        10**400,
    ],
)
def test_policy_rejects_malformed_norm_bounds(value: object) -> None:
    with pytest.raises(clipping.FederatedUpdateClippingError) as error:
        _policy(global_norm_bound=value)

    assert str(error.value) == "invalid clipping norm bound"


@pytest.mark.parametrize(
    "name",
    [
        "",
        " ",
        "a b",
        "a/b",
        "a-b",
        "adapter lora",
        ".a",
        "a.",
        "1a",
        5,
        None,
        "a" * 257,
    ],
)
def test_policy_rejects_malformed_layer_names(name: object) -> None:
    with pytest.raises(clipping.FederatedUpdateClippingError) as error:
        _policy(per_layer_bounds=((name, 1.0),))

    assert str(error.value) == "invalid layer name"


def test_policy_rejects_duplicate_layer_bounds() -> None:
    with pytest.raises(clipping.FederatedUpdateClippingError) as error:
        _policy(per_layer_bounds=((_LAYER, 1.0), (_LAYER, 2.0)))

    assert str(error.value) == "duplicate clipping policy layers"


@pytest.mark.parametrize(
    "entries",
    [
        "adapter",
        5,
        None,
        ((_LAYER, 1.0, 2.0),),
        ({"layer": _LAYER, "norm_bound": 1.0},),
        ([_LAYER],),
        ((_LAYER,),),
    ],
)
def test_policy_rejects_malformed_bound_entries(entries: object) -> None:
    with pytest.raises(clipping.FederatedUpdateClippingError) as error:
        _policy(per_layer_bounds=entries)

    assert str(error.value) == "invalid clipping policy fields"


def test_policy_rejects_more_than_the_layer_limit() -> None:
    entries = tuple((f"layer_{index}", 1.0) for index in range(1025))

    with pytest.raises(clipping.FederatedUpdateClippingError) as error:
        _policy(per_layer_bounds=entries)

    assert str(error.value) == "invalid clipping policy fields"


def test_bound_for_requires_a_valid_layer_name() -> None:
    with pytest.raises(clipping.FederatedUpdateClippingError) as error:
        _policy().bound_for("adapter/lora")

    assert str(error.value) == "invalid layer name"


def test_policy_is_frozen() -> None:
    policy = _policy()

    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.global_norm_bound = 2.0  # type: ignore[misc]


def test_policy_dict_and_json_round_trip() -> None:
    policy = _policy(
        per_layer_bounds=(
            (_LAYER, 4.0),
            (_OTHER_LAYER, 2.0),
        )
    )

    assert clipping.FederatedClippingPolicy.from_dict(policy.to_dict()) == policy
    assert clipping.FederatedClippingPolicy.from_json(policy.to_json()) == policy


def test_policy_json_output_is_canonical() -> None:
    policy = _policy(per_layer_bounds=((_OTHER_LAYER, 3.0),))
    text = policy.to_json()

    assert text.endswith("\n")
    assert json.loads(text) == policy.to_dict()
    assert text == json.dumps(policy.to_dict(), indent=2, sort_keys=True) + "\n"


def test_policy_dict_round_trip_does_not_mutate_the_payload() -> None:
    payload = _policy(per_layer_bounds=((_LAYER, 4.0),)).to_dict()
    original = copy.deepcopy(payload)

    clipping.FederatedClippingPolicy.from_dict(payload)

    assert payload == original


@pytest.mark.parametrize(
    "payload",
    [
        None,
        5,
        "policy",
        ["policy"],
        {},
        {"schema_version": clipping.UPDATE_CLIPPING_SCHEMA_VERSION},
        {
            "schema_version": "openmed.training.federated.update_clipping.v2",
            "global_norm_bound": 1.0,
            "per_layer_bounds": [],
        },
        {
            "schema_version": clipping.UPDATE_CLIPPING_SCHEMA_VERSION,
            "global_norm_bound": 1.0,
            "per_layer_bounds": [],
            "extra": 1,
        },
        {
            "schema_version": clipping.UPDATE_CLIPPING_SCHEMA_VERSION,
            "global_norm_bound": 1.0,
            "per_layer_bounds": "adapter",
        },
        {
            "schema_version": clipping.UPDATE_CLIPPING_SCHEMA_VERSION,
            "global_norm_bound": 1.0,
            "per_layer_bounds": [[_LAYER, 1.0]],
        },
    ],
)
def test_policy_from_dict_rejects_malformed_documents(payload: object) -> None:
    with pytest.raises(clipping.FederatedUpdateClippingError) as error:
        clipping.FederatedClippingPolicy.from_dict(payload)

    assert str(error.value) in {
        "invalid clipping policy fields",
        "unsupported clipping policy version",
    }


@pytest.mark.parametrize(
    "payload",
    [
        5,
        None,
        "",
        "{",
        "[]",
        "null",
        '{"schema_version": "v1", "global_norm_bound": 1.0,'
        ' "global_norm_bound": 2.0, "per_layer_bounds": []}',
        '{"schema_version": "openmed.training.federated.update_clipping.v1",'
        ' "global_norm_bound": NaN, "per_layer_bounds": []}',
        '{"schema_version": "openmed.training.federated.update_clipping.v1",'
        ' "global_norm_bound": Infinity, "per_layer_bounds": []}',
    ],
)
def test_policy_from_json_rejects_malformed_text(payload: object) -> None:
    with pytest.raises(clipping.FederatedUpdateClippingError) as error:
        clipping.FederatedClippingPolicy.from_json(payload)  # type: ignore[arg-type]

    assert str(error.value) in {
        "invalid clipping policy JSON",
        "invalid clipping policy fields",
    }


def test_policy_from_json_enforces_the_byte_limit() -> None:
    payload = " " * (1024 * 1024 + 1)

    with pytest.raises(clipping.FederatedUpdateClippingError) as error:
        clipping.FederatedClippingPolicy.from_json(payload)

    assert str(error.value) == "invalid clipping policy JSON"


def test_policy_fingerprint_is_deterministic_and_canonical() -> None:
    unsorted = _policy(per_layer_bounds=((_OTHER_LAYER, 2.0), (_LAYER, 4.0)))
    sorted_ = _policy(per_layer_bounds=((_LAYER, 4.0), (_OTHER_LAYER, 2.0)))

    digest = clipping.fingerprint_clipping_policy(unsorted)

    assert digest == clipping.fingerprint_clipping_policy(sorted_)
    assert digest.startswith("sha256:")
    assert len(digest) == len("sha256:") + 64
    assert digest == clipping.fingerprint_clipping_policy(unsorted)


def test_policy_fingerprint_changes_with_the_bounds() -> None:
    assert clipping.fingerprint_clipping_policy(
        _policy()
    ) != clipping.fingerprint_clipping_policy(_policy(global_norm_bound=2.0))
    assert clipping.fingerprint_clipping_policy(
        _policy()
    ) != clipping.fingerprint_clipping_policy(
        _policy(per_layer_bounds=((_LAYER, 1.0),))
    )


@pytest.mark.parametrize("policy", [None, 5, {"global_norm_bound": 1.0}, object()])
def test_policy_fingerprint_rejects_non_policies(policy: object) -> None:
    with pytest.raises(clipping.FederatedUpdateClippingError) as error:
        clipping.fingerprint_clipping_policy(policy)  # type: ignore[arg-type]

    assert str(error.value) == "invalid clipping policy"


def test_layer_within_bound_is_returned_unchanged() -> None:
    result = clipping.clip_federated_update(
        {_LAYER: [0.3, 0.4]}, policy=_policy(global_norm_bound=1.0)
    )

    assert result.layer_deltas(_LAYER) == (0.3, 0.4)
    assert result.report.reason_codes == ("within_bound",)
    assert result.report.clipped is False
    assert result.report.to_dict()["layers"][0]["clipped"] is False


def test_layer_over_bound_is_scaled_exactly_to_the_bound() -> None:
    result = clipping.clip_federated_update(
        {_LAYER: [3.0, 4.0]}, policy=_policy(global_norm_bound=2.5)
    )

    assert result.layer_deltas(_LAYER) == (1.5, 2.0)
    assert _norm(result.layer_deltas(_LAYER)) == 2.5
    assert result.report.reason_codes == ("scaled_to_global_bound",)
    assert result.report.clipped is True
    assert result.report.clipped_layer_count == 1
    assert result.report.to_dict()["layers"][0]["norm_bound"] == 2.5


def test_per_layer_override_replaces_the_global_bound() -> None:
    result = clipping.clip_federated_update(
        {_LAYER: [3.0, 4.0], _OTHER_LAYER: [3.0, 4.0]},
        policy=_policy(global_norm_bound=10.0, per_layer_bounds=((_LAYER, 2.5),)),
    )

    assert result.layer_deltas(_LAYER) == (1.5, 2.0)
    assert result.layer_deltas(_OTHER_LAYER) == (3.0, 4.0)
    assert result.report.reason_codes == (
        "scaled_to_layer_bound",
        "within_bound",
    )
    assert result.report.to_dict()["layers"][0]["norm_bound"] == 2.5


def test_zero_layer_reports_zero_norm_without_scaling() -> None:
    result = clipping.clip_federated_update({_LAYER: [0.0, 0.0, 0.0]}, policy=_policy())

    assert result.layer_deltas(_LAYER) == (0.0, 0.0, 0.0)
    assert result.report.reason_codes == ("zero_norm",)
    assert result.report.clipped is False


def test_negative_values_keep_the_sign_after_scaling() -> None:
    result = clipping.clip_federated_update(
        {_LAYER: [-3.0, -4.0]}, policy=_policy(global_norm_bound=2.5)
    )

    assert result.layer_deltas(_LAYER) == (-1.5, -2.0)


def test_single_element_layer_scales_to_the_bound() -> None:
    result = clipping.clip_federated_update(
        {_LAYER: [10.0]}, policy=_policy(global_norm_bound=1.0)
    )

    assert result.layer_deltas(_LAYER) == (1.0,)


def test_integer_inputs_are_normalized_to_floats() -> None:
    result = clipping.clip_federated_update(
        {_LAYER: [3, 4]}, policy=_policy(global_norm_bound=2.5)
    )

    assert result.layer_deltas(_LAYER) == (1.5, 2.0)
    assert all(type(value) is float for value in result.layer_deltas(_LAYER))


def test_scaling_preserves_direction_and_stays_within_the_bound() -> None:
    generator = random.Random(20260930)
    for _ in range(50):
        size = generator.randint(1, 12)
        values = [generator.uniform(-50.0, 50.0) for _ in range(size)]
        bound = generator.uniform(0.5, 20.0)
        result = clipping.clip_federated_update(
            {_LAYER: values}, policy=_policy(global_norm_bound=bound)
        )
        clipped = result.layer_deltas(_LAYER)

        assert _norm(clipped) <= bound * _SLACK
        original = _norm(tuple(values))
        if original > bound:
            for before, after in zip(values, clipped):
                assert after == pytest.approx(before * bound / original, rel=1e-9)


def test_equal_inputs_in_any_order_produce_identical_output() -> None:
    policy = _policy(per_layer_bounds=((_LAYER, 2.0),))
    forward = clipping.clip_federated_update(
        {_LAYER: [3.0, 4.0], _OTHER_LAYER: [1.0]}, policy=policy
    )
    reverse = clipping.clip_federated_update(
        {_OTHER_LAYER: [1.0], _LAYER: [3.0, 4.0]}, policy=policy
    )
    pairs = clipping.clip_federated_update(
        [(_OTHER_LAYER, [1.0]), (_LAYER, [3.0, 4.0])], policy=policy
    )

    assert forward.to_json() == reverse.to_json() == pairs.to_json()
    assert forward.report.to_json() == pairs.report.to_json()
    assert forward.report.policy_digest == pairs.report.policy_digest


def test_repeated_calls_are_byte_stable() -> None:
    arguments = {_LAYER: [3.0, 4.0]}
    policy = _policy(global_norm_bound=2.5)

    first = clipping.clip_federated_update(dict(arguments), policy=policy)
    second = clipping.clip_federated_update(dict(arguments), policy=policy)

    assert first.to_json() == second.to_json()


def test_report_counts_and_reason_codes_are_derived_from_layers() -> None:
    result = clipping.clip_federated_update(
        {_LAYER: [3.0, 4.0], _OTHER_LAYER: [0.5, 0.5, 0.5]},
        policy=_policy(global_norm_bound=2.0),
    )
    report = result.report

    assert report.layer_count == 2
    assert report.element_count == 5
    assert report.clipped is True
    assert report.clipped_layer_count == 1
    assert report.reason_codes == ("scaled_to_global_bound", "within_bound")
    assert report.schema_version == clipping.UPDATE_CLIPPING_SCHEMA_VERSION


def test_clipped_update_exposes_layers_by_name_and_mapping() -> None:
    result = clipping.clip_federated_update(
        {_LAYER: [3.0, 4.0], _OTHER_LAYER: [1.0]}, policy=_policy(global_norm_bound=2.5)
    )

    mapping = result.to_mapping()

    assert mapping == {_OTHER_LAYER: (1.0,), _LAYER: (1.5, 2.0)}
    assert result.layer_deltas(_LAYER) == (1.5, 2.0)
    assert result.layer_deltas(_OTHER_LAYER) == (1.0,)

    mapping[_LAYER] = (9.0,)
    assert result.layer_deltas(_LAYER) == (1.5, 2.0)


def test_clipped_update_json_document_is_canonical() -> None:
    result = clipping.clip_federated_update(
        {_LAYER: [3.0, 4.0], _OTHER_LAYER: [0.25]},
        policy=_policy(global_norm_bound=2.5),
    )

    text = result.to_json()
    decoded = json.loads(text)

    assert text.endswith("\n")
    assert decoded["report"] == result.report.to_dict()
    assert [entry["layer"] for entry in decoded["layers"]] == [
        _LAYER,
        _OTHER_LAYER,
    ]
    assert decoded["layers"][0]["values"] == pytest.approx([1.5, 2.0])
    assert decoded["layers"][1]["values"] == pytest.approx([0.25])


def test_clipped_update_rejects_an_unknown_layer() -> None:
    result = clipping.clip_federated_update({_LAYER: [1.0]}, policy=_policy())

    with pytest.raises(clipping.FederatedUpdateClippingError) as error:
        result.layer_deltas(_OTHER_LAYER)

    assert str(error.value) == "unknown layer name"


def test_clipped_update_rejects_an_invalid_layer_lookup() -> None:
    result = clipping.clip_federated_update({_LAYER: [1.0]}, policy=_policy())

    with pytest.raises(clipping.FederatedUpdateClippingError) as error:
        result.layer_deltas("adapter/lora")

    assert str(error.value) == "invalid layer name"


@pytest.mark.parametrize(
    "policy",
    [None, 5, {"global_norm_bound": 1.0}, object(), "policy"],
)
def test_clip_rejects_non_policies(policy: object) -> None:
    with pytest.raises(clipping.FederatedUpdateClippingError) as error:
        clipping.clip_federated_update({_LAYER: [1.0]}, policy=policy)  # type: ignore[arg-type]

    assert str(error.value) == "invalid clipping policy"


@pytest.mark.parametrize(
    "deltas",
    [
        5,
        None,
        "adapter",
        b"adapter",
        set(),
        [],
        {},
        [(_LAYER,)],
        [(_LAYER, [1.0], 2)],
        [(1, [1.0])],
        [[[1.0], _LAYER]],
        [(_LAYER, [1.0]), (_LAYER, [2.0])],
    ],
)
def test_clip_rejects_malformed_delta_containers(deltas: object) -> None:
    with pytest.raises(clipping.FederatedUpdateClippingError) as error:
        clipping.clip_federated_update(deltas, policy=_policy())  # type: ignore[arg-type]

    assert str(error.value) in {
        "invalid update layers",
        "invalid layer name",
        "duplicate layer names",
    }


@pytest.mark.parametrize(
    "values",
    [
        "abc",
        b"abc",
        {"a": 1.0},
        [],
        [None],
        [True],
        [False],
        ["1.0"],
        [[1.0]],
        [float("nan")],
        [float("inf")],
        [float("-inf")],
        [complex(1.0, 2.0)],
        [clipping.MAX_CLIPPING_VALUE * 10.0],
        [1 << 200],
        [10**400],
    ],
)
def test_clip_rejects_malformed_layer_values(values: object) -> None:
    with pytest.raises(clipping.FederatedUpdateClippingError) as error:
        clipping.clip_federated_update({_LAYER: values}, policy=_policy())  # type: ignore[arg-type]

    assert str(error.value) in {"invalid layer values", "invalid layer value"}


def test_clip_rejects_a_layer_missing_its_bounded_override() -> None:
    policy = _policy(per_layer_bounds=((_OTHER_LAYER, 1.0),))

    with pytest.raises(clipping.FederatedUpdateClippingError) as error:
        clipping.clip_federated_update({_LAYER: [1.0]}, policy=policy)

    assert str(error.value) == "update is missing a bounded layer"
    assert _OTHER_LAYER not in str(error.value)


def test_clip_enforces_the_layer_count_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(clipping, "MAX_CLIPPING_LAYERS", 1)

    with pytest.raises(clipping.FederatedUpdateClippingError) as error:
        clipping.clip_federated_update(
            {_LAYER: [1.0], _OTHER_LAYER: [1.0]}, policy=_policy()
        )

    assert str(error.value) == "invalid update layers"


def test_clip_enforces_the_elements_per_layer_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(clipping, "MAX_CLIPPING_ELEMENTS_PER_LAYER", 2)

    with pytest.raises(clipping.FederatedUpdateClippingError) as error:
        clipping.clip_federated_update({_LAYER: [1.0, 2.0, 3.0]}, policy=_policy())

    assert str(error.value) == "invalid layer values"


def test_clip_enforces_the_total_element_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(clipping, "MAX_CLIPPING_TOTAL_ELEMENTS", 3)

    with pytest.raises(clipping.FederatedUpdateClippingError) as error:
        clipping.clip_federated_update(
            {_LAYER: [1.0, 2.0], _OTHER_LAYER: [1.0, 2.0]}, policy=_policy()
        )

    assert str(error.value) == "total element count exceeds limit"


def test_clip_enforces_the_scalar_magnitude_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(clipping, "MAX_CLIPPING_VALUE", 10.0)

    with pytest.raises(clipping.FederatedUpdateClippingError) as error:
        clipping.clip_federated_update({_LAYER: [10.5]}, policy=_policy())

    assert str(error.value) == "invalid layer value"


@pytest.mark.parametrize(
    "reason_code",
    ["within_bound", "zero_norm"],
)
def test_diagnostics_reject_a_missing_clipped_flag(reason_code: str) -> None:
    diagnostics = clipping.FederatedLayerClipDiagnostics(
        layer_name=_LAYER,
        norm_bound=1.0,
        element_count=1,
        clipped=False,
        reason_code=reason_code,
    )

    assert diagnostics.to_dict()["reason_code"] == reason_code


def test_diagnostics_reject_inconsistent_clipping_state() -> None:
    with pytest.raises(clipping.FederatedUpdateClippingError) as error:
        clipping.FederatedLayerClipDiagnostics(
            layer_name=_LAYER,
            norm_bound=1.0,
            element_count=1,
            clipped=True,
            reason_code="within_bound",
        )

    assert str(error.value) == "inconsistent clipping diagnostics"

    with pytest.raises(clipping.FederatedUpdateClippingError) as error:
        clipping.FederatedLayerClipDiagnostics(
            layer_name=_LAYER,
            norm_bound=1.0,
            element_count=1,
            clipped=False,
            reason_code="scaled_to_global_bound",
        )

    assert str(error.value) == "inconsistent clipping diagnostics"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("layer_name", "adapter/lora", "invalid layer name"),
        ("norm_bound", 0.0, "invalid clipping norm bound"),
        ("norm_bound", float("nan"), "invalid clipping norm bound"),
        ("element_count", 0, "invalid layer element count"),
        ("element_count", True, "invalid layer element count"),
        ("clipped", 1, "invalid clipping status"),
        ("reason_code", "nope", "invalid clipping reason code"),
        ("reason_code", None, "invalid clipping reason code"),
    ],
)
def test_diagnostics_reject_malformed_fields(
    field: str, value: object, message: str
) -> None:
    arguments: dict[str, object] = {
        "layer_name": _LAYER,
        "norm_bound": 1.0,
        "element_count": 1,
        "clipped": False,
        "reason_code": "within_bound",
    }
    arguments[field] = value

    with pytest.raises(clipping.FederatedUpdateClippingError) as error:
        clipping.FederatedLayerClipDiagnostics(**arguments)  # type: ignore[arg-type]

    assert str(error.value) == message


def test_diagnostics_and_report_are_frozen() -> None:
    diagnostics = clipping.FederatedLayerClipDiagnostics(
        layer_name=_LAYER,
        norm_bound=1.0,
        element_count=1,
        clipped=False,
        reason_code="within_bound",
    )
    report = clipping.FederatedClippingReport(
        policy_digest=clipping.fingerprint_clipping_policy(_policy()),
        layers=(diagnostics,),
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        diagnostics.clipped = True  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.layers = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    "policy_digest",
    ["", "sha256:", "sha256:" + "A" * 64, "sha256:" + "a" * 63, 5, None],
)
def test_report_rejects_an_invalid_policy_digest(policy_digest: object) -> None:
    with pytest.raises(clipping.FederatedUpdateClippingError) as error:
        clipping.FederatedClippingReport(
            policy_digest=policy_digest,  # type: ignore[arg-type]
            layers=(
                clipping.FederatedLayerClipDiagnostics(
                    layer_name=_LAYER,
                    norm_bound=1.0,
                    element_count=1,
                    clipped=False,
                    reason_code="within_bound",
                ),
            ),
        )

    assert str(error.value) == "invalid SHA-256 digest reference"


def test_report_rejects_a_foreign_schema_version() -> None:
    diagnostics = clipping.FederatedLayerClipDiagnostics(
        layer_name=_LAYER,
        norm_bound=1.0,
        element_count=1,
        clipped=False,
        reason_code="within_bound",
    )

    with pytest.raises(clipping.FederatedUpdateClippingError) as error:
        clipping.FederatedClippingReport(
            policy_digest="sha256:" + "a" * 64,
            layers=(diagnostics,),
            schema_version="openmed.training.federated.update_clipping.v2",
        )

    assert str(error.value) == "unsupported clipping report version"


def _diagnostics(
    layer_name: str, *, clipped: bool = False, bound: float = 1.0
) -> clipping.FederatedLayerClipDiagnostics:
    return clipping.FederatedLayerClipDiagnostics(
        layer_name=layer_name,
        norm_bound=bound,
        element_count=1,
        clipped=clipped,
        reason_code="scaled_to_global_bound" if clipped else "within_bound",
    )


@pytest.mark.parametrize(
    "layers",
    [
        (_diagnostics(_OTHER_LAYER), _diagnostics(_LAYER)),
        (_diagnostics(_LAYER), _diagnostics(_LAYER)),
        (),
        "layers",
        (None,),
        ({"layer": _LAYER},),
    ],
)
def test_report_rejects_unsorted_duplicate_or_foreign_layers(
    layers: object,
) -> None:
    with pytest.raises(clipping.FederatedUpdateClippingError) as error:
        clipping.FederatedClippingReport(
            policy_digest="sha256:" + "a" * 64,
            layers=layers,  # type: ignore[arg-type]
        )

    assert str(error.value) in {
        "invalid clipping report layers",
        "clipping report layers must be unique and ordered",
    }


def test_clipped_update_rejects_a_foreign_report() -> None:
    with pytest.raises(clipping.FederatedUpdateClippingError) as error:
        clipping.ClippedFederatedUpdate(
            layers=((_LAYER, (1.0,)),),
            report={"policy_digest": "sha256:" + "a" * 64},  # type: ignore[arg-type]
        )

    assert str(error.value) == "invalid clipping report"


def test_clipped_update_rejects_layers_that_do_not_match_the_report() -> None:
    report = clipping.FederatedClippingReport(
        policy_digest="sha256:" + "a" * 64,
        layers=(_diagnostics(_LAYER),),
    )

    with pytest.raises(clipping.FederatedUpdateClippingError) as error:
        clipping.ClippedFederatedUpdate(layers=((_OTHER_LAYER, (1.0,)),), report=report)

    assert str(error.value) == "clipped layers do not match the report"


def test_clipped_update_rejects_values_over_their_bound() -> None:
    report = clipping.FederatedClippingReport(
        policy_digest="sha256:" + "a" * 64,
        layers=(_diagnostics(_LAYER),),
    )

    with pytest.raises(clipping.FederatedUpdateClippingError) as error:
        clipping.ClippedFederatedUpdate(layers=((_LAYER, (5.0,)),), report=report)

    assert str(error.value) == "clipped layer exceeds its norm bound"


@pytest.mark.parametrize(
    "layers",
    [
        [[(_LAYER, (1.0,))]],
        ((_LAYER, [1.0]),),
        ((_LAYER, (1.0,), 2),),
        ((("adapter/lora"), (1.0,)),),
        ((_LAYER, (float("nan"),)),),
        ((_LAYER, ()),),
        (),
    ],
)
def test_clipped_update_rejects_malformed_layers(layers: object) -> None:
    report = clipping.FederatedClippingReport(
        policy_digest="sha256:" + "a" * 64,
        layers=(_diagnostics(_LAYER),),
    )

    with pytest.raises(clipping.FederatedUpdateClippingError) as error:
        clipping.ClippedFederatedUpdate(layers=layers, report=report)  # type: ignore[arg-type]

    assert str(error.value) in {
        "invalid clipped layers",
        "invalid layer name",
        "invalid layer values",
        "invalid layer value",
    }


def test_report_excludes_raw_magnitudes_and_values() -> None:
    result = clipping.clip_federated_update(
        {_LAYER: [98765.4321, -12345.6789]},
        policy=_policy(global_norm_bound=1.0),
    )
    document = result.report.to_dict()
    rendered = result.report.to_json()

    assert "98765" not in rendered
    assert "12345" not in rendered
    assert "values" not in document
    assert set(document) == {
        "schema_version",
        "policy_digest",
        "layer_count",
        "element_count",
        "clipped",
        "clipped_layer_count",
        "layers",
    }
    assert set(document["layers"][0]) == {
        "layer",
        "norm_bound",
        "element_count",
        "clipped",
        "reason_code",
    }


def test_update_errors_never_echo_submitted_names_or_magnitudes() -> None:
    policy = _policy()

    with pytest.raises(clipping.FederatedUpdateClippingError) as name_error:
        clipping.clip_federated_update({"placeholder-site-id": [1.0]}, policy=policy)
    assert str(name_error.value) == "invalid layer name"

    with pytest.raises(clipping.FederatedUpdateClippingError) as value_error:
        clipping.clip_federated_update({_LAYER: [float("nan")]}, policy=policy)
    assert str(value_error.value) == "invalid layer value"

    with pytest.raises(clipping.FederatedUpdateClippingError) as range_error:
        clipping.clip_federated_update(
            {_LAYER: [clipping.MAX_CLIPPING_VALUE * 10.0]}, policy=policy
        )
    assert str(range_error.value) == "invalid layer value"


@pytest.mark.parametrize("case", FORBIDDEN_FIELD_CASES, ids=_CASE_IDS)
def test_policy_errors_never_echo_placeholder_fields(case: object) -> None:
    payload = _policy().to_dict()
    payload[case.field] = copy.deepcopy(case.value)  # type: ignore[attr-defined]

    with pytest.raises(clipping.FederatedUpdateClippingError) as error:
        clipping.FederatedClippingPolicy.from_dict(payload)

    assert str(error.value) == "invalid clipping policy fields"
    assert case.field not in str(error.value)  # type: ignore[attr-defined]
    if case.marker is not None:  # type: ignore[attr-defined]
        assert case.marker not in str(error.value)  # type: ignore[attr-defined]


def test_package_surface_re_exports_the_clipping_contract() -> None:
    from openmed.training import federated as federated_package

    for name in clipping.__all__:
        assert name in training.__all__
        assert getattr(training, name) is getattr(clipping, name)
        assert getattr(federated_package, name) is getattr(clipping, name)
    assert training.__all__.count("FederatedClippingPolicy") == 1
    assert training.clip_federated_update is clipping.clip_federated_update
