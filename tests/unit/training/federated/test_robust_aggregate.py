"""Synthetic, offline contracts for poisoning-resistant adapter aggregation."""

from __future__ import annotations

import ast
import json
import math
import traceback
from dataclasses import FrozenInstanceError
from itertools import permutations
from pathlib import Path

import pytest

from openmed.training.federated import robust_aggregate as robust_module
from openmed.training.federated.robust_aggregate import (
    MAX_ROBUST_UPDATES,
    ROBUST_AGGREGATION_REASON_CODES,
    ROBUST_AGGREGATION_SCHEMA_VERSION,
    ROBUST_AGGREGATION_STRATEGIES,
    ROBUST_AGGREGATION_STRATEGY_VERSION,
    ROBUST_NORM_BANDS,
    RobustAdapterUpdate,
    RobustAggregate,
    RobustAggregationConfig,
    RobustAggregationError,
    aggregate_adapter_updates,
)
from openmed.training.federated_update_metadata import (
    FederatedParameterMetadata,
    FederatedUpdateMetadata,
    FederatedUpdatePolicy,
)

MODEL = "sha256:" + "a" * 64
OTHER_MODEL = "sha256:" + "c" * 64
NAME_A = "adapter.lora_A.weight"
NAME_B = "adapter.lora_B.weight"
TRIMMED = "coordinate_trimmed_mean"
CLIPPED = "norm_clipped_mean"
# A distinctive synthetic magnitude used to prove values never leak.
SENTINEL = 987654.321


def _policy(
    shape: tuple[int, ...] = (1, 3),
    *,
    dtype: str = "float64",
    model: str = MODEL,
    extra: bool = False,
) -> FederatedUpdatePolicy:
    parameters = [FederatedParameterMetadata(NAME_A, shape, dtype)]
    if extra:
        parameters.append(FederatedParameterMetadata(NAME_B, (2,), dtype))
    total = sum(parameter.element_count for parameter in parameters)
    return FederatedUpdatePolicy(
        model_digest=model,
        parameters=tuple(parameters),
        max_total_elements=total,
    )


def _digest(index: int) -> str:
    return "sha256:" + format(index + 1, "064x")


def _metadata(policy: FederatedUpdatePolicy, index: int) -> FederatedUpdateMetadata:
    return FederatedUpdateMetadata(
        model_digest=policy.model_digest,
        adapter_format="dense",
        parameters=policy.parameters,
        total_elements=sum(p.element_count for p in policy.parameters),
        update_digest=_digest(index),
        clipped=True,
        policy=policy,
    )


def _update(
    policy: FederatedUpdatePolicy, index: int, tensors: dict
) -> RobustAdapterUpdate:
    return RobustAdapterUpdate(metadata=_metadata(policy, index), tensors=tensors)


def _rows(policy: FederatedUpdatePolicy, rows: list) -> list[RobustAdapterUpdate]:
    return [_update(policy, i, {NAME_A: [row]}) for i, row in enumerate(rows)]


def _config(strategy: str = TRIMMED, f: int = 1) -> RobustAggregationConfig:
    return RobustAggregationConfig(strategy=strategy, max_suspected_adversaries=f)


def _code(excinfo: pytest.ExceptionInfo[RobustAggregationError]) -> str:
    assert str(excinfo.value) == excinfo.value.code
    assert excinfo.value.code in ROBUST_AGGREGATION_REASON_CODES
    return excinfo.value.code


def _plain_mean(rows: list[list[float]]) -> list[float]:
    return [math.fsum(column) / len(rows) for column in zip(*rows)]


HONEST_ROWS = [
    [1.0, -2.0, 0.5],
    [2.0, 0.0, 0.25],
    [3.0, 2.0, 0.75],
    [4.0, 4.0, -1.0],
]


# --- Reference vectors ----------------------------------------------------


def test_trimmed_mean_reference_vector_is_exact():
    policy = _policy()
    rows = [*HONEST_ROWS, [100.0, -50.0, 0.0]]
    result = aggregate_adapter_updates(
        _rows(policy, rows), policy=policy, config=_config(TRIMMED, 1)
    )
    # Per coordinate: sort five values, drop one per side, average three.
    assert result.tensor(NAME_A) == ((3.0, 0.0, 0.25),)


def test_trimmed_mean_with_zero_bound_is_the_plain_mean():
    policy = _policy()
    result = aggregate_adapter_updates(
        _rows(policy, HONEST_ROWS), policy=policy, config=_config(TRIMMED, 0)
    )
    assert result.tensor(NAME_A) == ((2.5, 1.0, 0.125),)
    assert result.diagnostics.trimmed_per_side == 0
    assert result.diagnostics.trimmed_fraction == 0.0


def test_norm_clipped_mean_reference_vector_is_exact():
    policy = _policy((2,))
    updates = [
        _update(policy, 0, {NAME_A: [0.0, 20.0]}),  # norm 20, scaled by 1/4
        _update(policy, 1, {NAME_A: [3.0, 4.0]}),  # norm 5, the median
        _update(policy, 2, {NAME_A: [0.0, 3.0]}),  # norm 3
    ]
    result = aggregate_adapter_updates(
        updates, policy=policy, config=_config(CLIPPED, 1)
    )
    assert result.tensor(NAME_A) == (1.0, 4.0)
    assert result.diagnostics.clipped_update_count == 1
    assert result.diagnostics.trimmed_per_side == 0


def test_multi_tensor_reference_vector_restores_shapes_in_name_order():
    policy = _policy((2, 2), extra=True)
    tensors = [
        {NAME_B: [1.0, -1.0], NAME_A: [[1.0, 2.0], [3.0, 4.0]]},
        {NAME_B: [2.0, -2.0], NAME_A: [[5.0, 6.0], [7.0, 8.0]]},
        {NAME_B: [9.0, -9.0], NAME_A: [[-9.0, 0.0], [0.0, 9.0]]},
    ]
    updates = [_update(policy, i, t) for i, t in enumerate(tensors)]
    result = aggregate_adapter_updates(
        updates, policy=policy, config=_config(TRIMMED, 1)
    )
    # With n=3 and f=1 the trimmed mean is the coordinate-wise median.
    assert result.tensor(NAME_A) == ((1.0, 2.0), (3.0, 8.0))
    assert result.tensor(NAME_B) == (2.0, -2.0)
    mapping = result.to_mapping()
    assert list(mapping) == [NAME_A, NAME_B]
    assert mapping[NAME_A] == [[1.0, 2.0], [3.0, 8.0]]
    mapping[NAME_A][0][0] = SENTINEL
    assert result.tensor(NAME_A) == ((1.0, 2.0), (3.0, 8.0))


def test_reference_diagnostics_are_pinned():
    policy = _policy()
    rows = [*HONEST_ROWS, [100.0, -50.0, 0.0]]
    result = aggregate_adapter_updates(
        _rows(policy, rows), policy=policy, config=_config(TRIMMED, 1)
    )
    payload = result.diagnostics.to_dict()
    digests = {
        key: payload.pop(key)
        for key in ("config_digest", "inputs_digest", "aggregate_digest")
    }
    assert payload == {
        "schema_version": ROBUST_AGGREGATION_SCHEMA_VERSION,
        "strategy": TRIMMED,
        "strategy_version": ROBUST_AGGREGATION_STRATEGY_VERSION,
        "max_suspected_adversaries": 1,
        "update_count": 5,
        "parameter_count": 1,
        "element_count": 3,
        "trimmed_per_side": 1,
        "trimmed_fraction": 0.4,
        "clipped_update_count": 0,
        "norm_quantile_bands": {
            "p25": "1e0_to_1e1",
            "p50": "1e0_to_1e1",
            "p75": "1e0_to_1e1",
        },
    }
    assert digests == {
        "config_digest": (
            "sha256:7228e94066de5bb3c741a1f5014e01c26a566b4e499e4d74199977e82e68185c"
        ),
        "inputs_digest": (
            "sha256:97e1026bd035b1c6c7e44b32afae5f9ab51cc87b7b0c0e1e510e91339f2a924f"
        ),
        "aggregate_digest": (
            "sha256:e7fc8a0f9cf4589857d39101791d3d0c48f2e2a0b530b10f02d0d111d448dbf7"
        ),
    }
    text = result.diagnostics.to_json()
    assert text.endswith("\n")
    assert json.loads(text) == result.diagnostics.to_dict()


# --- Poisoning resistance -------------------------------------------------


@pytest.mark.parametrize("extreme", [1.0e9, -1.0e9, 5.0e11])
def test_one_extreme_update_cannot_leave_the_honest_range(extreme):
    policy = _policy()
    malicious = [extreme, -extreme, extreme]
    rows = [*HONEST_ROWS, malicious]
    result = aggregate_adapter_updates(
        _rows(policy, rows), policy=policy, config=_config(TRIMMED, 1)
    )
    aggregate = result.tensor(NAME_A)[0]
    for coordinate, value in enumerate(aggregate):
        honest = [row[coordinate] for row in HONEST_ROWS]
        assert min(honest) <= value <= max(honest)
    # Negative control: the plain mean is dragged far outside that range.
    plain = _plain_mean(rows)
    for coordinate, value in enumerate(plain):
        honest = [row[coordinate] for row in HONEST_ROWS]
        assert not min(honest) <= value <= max(honest)


def test_f_adversaries_cannot_leave_the_honest_range():
    policy = _policy()
    honest = [*HONEST_ROWS, [2.5, 1.0, 0.0]]
    attackers = [[1.0e10, 1.0e10, -1.0e10], [-1.0e10, 1.0e10, 1.0e10]]
    result = aggregate_adapter_updates(
        _rows(policy, honest + attackers), policy=policy, config=_config(TRIMMED, 2)
    )
    for coordinate, value in enumerate(result.tensor(NAME_A)[0]):
        column = [row[coordinate] for row in honest]
        assert min(column) <= value <= max(column)


def test_norm_clipping_bounds_a_boosted_update_influence():
    policy = _policy()
    boosted = [SENTINEL, -SENTINEL, SENTINEL]
    rows = [*HONEST_ROWS, boosted]
    result = aggregate_adapter_updates(
        _rows(policy, rows), policy=policy, config=_config(CLIPPED, 1)
    )
    honest_mean = _plain_mean(HONEST_ROWS)
    aggregate = result.tensor(NAME_A)[0]
    norms = sorted(math.sqrt(math.fsum(v * v for v in row)) for row in rows)
    threshold = norms[(len(rows) - 1) // 2]
    shift = math.sqrt(math.fsum((a - b) ** 2 for a, b in zip(aggregate, honest_mean)))
    # Each clipped update moves an n-way mean by at most threshold / n, and
    # replacing one honest update changes the honest part by a bounded amount.
    assert shift <= 2 * threshold
    plain_shift = math.sqrt(
        math.fsum((a - b) ** 2 for a, b in zip(_plain_mean(rows), honest_mean))
    )
    assert plain_shift > 1000 * shift


# --- Adversary bound ------------------------------------------------------


@pytest.mark.parametrize("strategy", ROBUST_AGGREGATION_STRATEGIES)
@pytest.mark.parametrize(("count", "f"), [(4, 2), (2, 1), (1, 1), (6, 3)])
def test_exceeding_the_adversary_bound_fails_closed(strategy, count, f):
    policy = _policy()
    rows = [[float(i), 0.0, 1.0] for i in range(count)]
    with pytest.raises(RobustAggregationError) as excinfo:
        aggregate_adapter_updates(
            _rows(policy, rows), policy=policy, config=_config(strategy, f)
        )
    assert _code(excinfo) == "robust_adversary_bound_exceeded"


@pytest.mark.parametrize("strategy", ROBUST_AGGREGATION_STRATEGIES)
def test_exactly_two_f_plus_one_updates_are_accepted(strategy):
    policy = _policy()
    rows = [[float(i), 0.0, 1.0] for i in range(5)]
    config = _config(strategy, 2)
    assert config.minimum_updates == 5
    result = aggregate_adapter_updates(
        _rows(policy, rows), policy=policy, config=config
    )
    assert result.diagnostics.update_count == 5


def test_empty_batch_fails_closed():
    with pytest.raises(RobustAggregationError) as excinfo:
        aggregate_adapter_updates([], policy=_policy(), config=_config(TRIMMED, 0))
    assert _code(excinfo) == "robust_adversary_bound_exceeded"


# --- Compatibility --------------------------------------------------------


def test_adapter_identity_mismatch_fails_closed():
    policy = _policy()
    other = _policy(model=OTHER_MODEL)
    updates = _rows(policy, HONEST_ROWS[:2]) + [
        _update(other, 9, {NAME_A: [[1.0, 1.0, 1.0]]})
    ]
    with pytest.raises(RobustAggregationError) as excinfo:
        aggregate_adapter_updates(updates, policy=policy, config=_config(TRIMMED, 1))
    assert _code(excinfo) == "robust_adapter_mismatch"


@pytest.mark.parametrize(
    ("other", "tensors"),
    [
        (_policy((3, 1)), {NAME_A: [[1.0], [1.0], [1.0]]}),
        (_policy((1, 3), dtype="float32"), {NAME_A: [[1.0, 1.0, 1.0]]}),
        (_policy(extra=True), {NAME_A: [[1.0, 1.0, 1.0]], NAME_B: [1.0, 1.0]}),
    ],
    ids=["shape", "dtype", "names"],
)
def test_parameter_layout_mismatch_fails_closed(other, tensors):
    policy = _policy()
    updates = _rows(policy, HONEST_ROWS[:2]) + [_update(other, 9, tensors)]
    with pytest.raises(RobustAggregationError) as excinfo:
        aggregate_adapter_updates(updates, policy=policy, config=_config(TRIMMED, 1))
    assert _code(excinfo) == "robust_parameter_mismatch"


@pytest.mark.parametrize(
    "tensors",
    [
        {},
        {NAME_A: [[1.0, 2.0, 3.0]], NAME_B: [1.0]},
        {"adapter.other.weight": [[1.0, 2.0, 3.0]]},
    ],
    ids=["missing", "extra", "renamed"],
)
def test_tensor_names_must_match_metadata(tensors):
    with pytest.raises(RobustAggregationError) as excinfo:
        _update(_policy(), 0, tensors)
    assert _code(excinfo) == "robust_parameter_mismatch"


@pytest.mark.parametrize(
    "values",
    [
        [1.0, 2.0, 3.0],
        [[1.0, 2.0]],
        [[1.0, 2.0, 3.0, 4.0]],
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[[1.0], [2.0], [3.0]]],
        [[1.0, [2.0], 3.0]],
        "abc",
        1.0,
    ],
)
def test_values_must_match_the_declared_shape(values):
    with pytest.raises(RobustAggregationError) as excinfo:
        _update(_policy(), 0, {NAME_A: values})
    assert _code(excinfo) == "robust_shape_mismatch"


@pytest.mark.parametrize(
    ("dtype", "value"),
    [("float16", 0.1), ("float32", 0.1), ("bfloat16", 1.0 + 2.0**-10)],
)
def test_values_must_be_representable_in_the_declared_dtype(dtype, value):
    policy = _policy((2,), dtype=dtype)
    with pytest.raises(RobustAggregationError) as excinfo:
        _update(policy, 0, {NAME_A: [1.0, value]})
    assert _code(excinfo) == "robust_dtype_mismatch"


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "float64"])
def test_representable_low_precision_values_are_accepted(dtype):
    policy = _policy((2,), dtype=dtype)
    updates = [_update(policy, i, {NAME_A: [0.5 * i, -0.25 * i]}) for i in range(3)]
    result = aggregate_adapter_updates(
        updates, policy=policy, config=_config(TRIMMED, 1)
    )
    assert result.tensor(NAME_A) == (0.5, -0.25)


def test_duplicate_updates_fail_closed():
    policy = _policy()
    updates = _rows(policy, HONEST_ROWS[:3])
    with pytest.raises(RobustAggregationError) as excinfo:
        aggregate_adapter_updates(
            [*updates, updates[0]], policy=policy, config=_config(TRIMMED, 1)
        )
    assert _code(excinfo) == "robust_duplicate_update"
    same_values = _update(policy, 7, {NAME_A: [HONEST_ROWS[0]]})
    with pytest.raises(RobustAggregationError) as excinfo:
        aggregate_adapter_updates(
            [*updates, same_values], policy=policy, config=_config(TRIMMED, 1)
        )
    assert _code(excinfo) == "robust_duplicate_update"


@pytest.mark.parametrize("updates", ["abc", None, {"a": 1}, (1, 2, 3)])
def test_invalid_update_collections_fail_closed(updates):
    with pytest.raises(RobustAggregationError) as excinfo:
        aggregate_adapter_updates(updates, policy=_policy(), config=_config())
    assert _code(excinfo) == "robust_update_invalid"


def test_invalid_policy_and_config_fail_closed():
    updates = _rows(_policy(), HONEST_ROWS[:3])
    with pytest.raises(RobustAggregationError) as excinfo:
        aggregate_adapter_updates(updates, policy=object(), config=_config())
    assert _code(excinfo) == "robust_config_invalid"
    with pytest.raises(RobustAggregationError) as excinfo:
        aggregate_adapter_updates(updates, policy=_policy(), config=object())
    assert _code(excinfo) == "robust_config_invalid"


@pytest.mark.parametrize("f", [-1, True, 1.0, "1", None, MAX_ROBUST_UPDATES])
def test_invalid_adversary_bounds_are_rejected(f):
    with pytest.raises(RobustAggregationError) as excinfo:
        RobustAggregationConfig(strategy=TRIMMED, max_suspected_adversaries=f)
    assert _code(excinfo) == "robust_config_invalid"


@pytest.mark.parametrize("strategy", ["mean", "fedavg", "krum", "TRIMMED", "", None, 1])
def test_unsupported_strategies_are_rejected(strategy):
    with pytest.raises(RobustAggregationError) as excinfo:
        RobustAggregationConfig(strategy=strategy, max_suspected_adversaries=1)
    assert _code(excinfo) == "robust_strategy_unsupported"


def test_update_requires_validated_metadata():
    with pytest.raises(RobustAggregationError) as excinfo:
        RobustAdapterUpdate(metadata={"model_digest": MODEL}, tensors={})
    assert _code(excinfo) == "robust_update_invalid"
    with pytest.raises(RobustAggregationError) as excinfo:
        RobustAdapterUpdate(metadata=_metadata(_policy(), 0), tensors=[])
    assert _code(excinfo) == "robust_update_invalid"


def test_update_limit_is_enforced(monkeypatch):
    monkeypatch.setattr(robust_module, "MAX_ROBUST_UPDATES", 3)
    policy = _policy()
    with pytest.raises(RobustAggregationError) as excinfo:
        aggregate_adapter_updates(
            _rows(policy, HONEST_ROWS), policy=policy, config=_config(TRIMMED, 1)
        )
    assert _code(excinfo) == "robust_limit_exceeded"


# --- Values ---------------------------------------------------------------


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_values_are_rejected(bad):
    with pytest.raises(RobustAggregationError) as excinfo:
        _update(_policy(), 0, {NAME_A: [[1.0, bad, 3.0]]})
    assert _code(excinfo) == "robust_non_finite_value"


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_non_finite_values_are_rejected_through_tolist(bad):
    class FakeArray:
        def tolist(self):
            return [[1.0, 2.0, bad]]

    with pytest.raises(RobustAggregationError) as excinfo:
        _update(_policy(), 0, {NAME_A: FakeArray()})
    assert _code(excinfo) == "robust_non_finite_value"


@pytest.mark.parametrize("bad", [True, None, "1.0", b"1", complex(1, 0)])
def test_non_numeric_values_are_rejected(bad):
    with pytest.raises(RobustAggregationError) as excinfo:
        _update(_policy(), 0, {NAME_A: [[1.0, bad, 3.0]]})
    assert _code(excinfo) == "robust_update_invalid"


@pytest.mark.parametrize("bad", [1.0e13, -1.0e13, 10**13])
def test_out_of_range_values_are_rejected(bad):
    with pytest.raises(RobustAggregationError) as excinfo:
        _update(_policy(), 0, {NAME_A: [[1.0, bad, 3.0]]})
    assert _code(excinfo) == "robust_value_out_of_range"


def test_array_like_inputs_with_tolist_are_accepted():
    class FakeArray:
        def __init__(self, rows):
            self._rows = rows

        def tolist(self):
            return [list(row) for row in self._rows]

    policy = _policy()
    updates = [
        _update(policy, i, {NAME_A: FakeArray([row])})
        for i, row in enumerate(HONEST_ROWS[:3])
    ]
    plain = [
        _update(policy, i, {NAME_A: [list(row)]})
        for i, row in enumerate(HONEST_ROWS[:3])
    ]
    config = _config(TRIMMED, 1)
    assert aggregate_adapter_updates(updates, policy=policy, config=config).tensor(
        NAME_A
    ) == aggregate_adapter_updates(plain, policy=policy, config=config).tensor(NAME_A)


def test_integers_and_negative_zero_are_normalized():
    policy = _policy((2,))
    updates = [
        _update(policy, 0, {NAME_A: [-0.0, 1]}),
        _update(policy, 1, {NAME_A: [0.0, 2]}),
        _update(policy, 2, {NAME_A: [-0.0, 3]}),
    ]
    result = aggregate_adapter_updates(
        updates, policy=policy, config=_config(TRIMMED, 1)
    )
    value = result.tensor(NAME_A)
    assert value == (0.0, 2.0)
    assert math.copysign(1.0, value[0]) == 1.0
    assert all(type(v) is float for v in value)


def test_caller_mutation_after_construction_has_no_effect():
    policy = _policy()
    rows = [list(row) for row in HONEST_ROWS[:3]]
    tensors = [{NAME_A: [row]} for row in rows]
    updates = [_update(policy, i, t) for i, t in enumerate(tensors)]
    rows[0][0] = SENTINEL
    tensors[1][NAME_A] = [[SENTINEL, SENTINEL, SENTINEL]]
    result = aggregate_adapter_updates(
        updates, policy=policy, config=_config(TRIMMED, 1)
    )
    assert result.tensor(NAME_A) == ((2.0, 0.0, 0.5),)


# --- Determinism ----------------------------------------------------------


@pytest.mark.parametrize("strategy", ROBUST_AGGREGATION_STRATEGIES)
def test_aggregate_is_invariant_under_input_permutation(strategy):
    policy = _policy((2, 2), extra=True)
    raw = [
        {NAME_A: [[0.1, 0.2], [0.3, -0.7]], NAME_B: [1.0e-3, 3.3]},
        {NAME_A: [[1.1, -0.2], [0.33, 0.7]], NAME_B: [-2.0e-3, 0.1]},
        {NAME_A: [[0.01, 0.25], [-0.3, 0.17]], NAME_B: [7.0e-3, 1.7]},
        {NAME_A: [[SENTINEL, 0.0], [0.0, -SENTINEL]], NAME_B: [0.0, SENTINEL]},
        {NAME_A: [[0.7, 0.1], [0.2, 0.3]], NAME_B: [1.1e-3, 0.9]},
    ]
    updates = [_update(policy, i, t) for i, t in enumerate(raw)]
    config = _config(strategy, 2)
    baseline = aggregate_adapter_updates(updates, policy=policy, config=config)
    expected_json = baseline.diagnostics.to_json()
    expected_values = baseline.to_mapping()
    for order in permutations(updates):
        result = aggregate_adapter_updates(list(order), policy=policy, config=config)
        assert result.to_mapping() == expected_values
        assert result.diagnostics.to_json() == expected_json


def test_repeated_runs_are_byte_identical():
    policy = _policy()
    rows = [[0.1 * i, 0.3 / (i + 1), -0.7 * i] for i in range(7)]
    config = _config(CLIPPED, 3)
    first = aggregate_adapter_updates(_rows(policy, rows), policy=policy, config=config)
    second = aggregate_adapter_updates(
        _rows(policy, rows), policy=policy, config=config
    )
    assert first.to_mapping() == second.to_mapping()
    assert first.diagnostics.to_json() == second.diagnostics.to_json()


def test_digests_change_with_strategy_and_inputs():
    policy = _policy()
    rows = [*HONEST_ROWS, [9.0, 9.0, 9.0]]
    trimmed = aggregate_adapter_updates(
        _rows(policy, rows), policy=policy, config=_config(TRIMMED, 1)
    ).diagnostics
    clipped = aggregate_adapter_updates(
        _rows(policy, rows), policy=policy, config=_config(CLIPPED, 1)
    ).diagnostics
    other_inputs = aggregate_adapter_updates(
        _rows(policy, [*HONEST_ROWS, [8.0, 9.0, 9.0]]),
        policy=policy,
        config=_config(TRIMMED, 1),
    ).diagnostics
    assert trimmed.config_digest != clipped.config_digest
    assert trimmed.inputs_digest == clipped.inputs_digest
    assert trimmed.inputs_digest != other_inputs.inputs_digest


# --- Privacy of diagnostics, reprs, and errors ----------------------------


def _sentinel_batch():
    policy = _policy()
    rows = [*HONEST_ROWS, [SENTINEL, -SENTINEL, SENTINEL]]
    return policy, _rows(policy, rows)


@pytest.mark.parametrize("strategy", ROBUST_AGGREGATION_STRATEGIES)
def test_diagnostics_contain_no_per_client_values(strategy):
    policy, updates = _sentinel_batch()
    result = aggregate_adapter_updates(
        updates, policy=policy, config=_config(strategy, 1)
    )
    surfaces = [
        result.diagnostics.to_json(),
        repr(result.diagnostics),
        repr(result),
        *(repr(update) for update in updates),
    ]
    sentinel_norm = math.sqrt(3) * SENTINEL
    assert NAME_A not in surfaces[0]
    for text in surfaces:
        assert "987654" not in text
        assert str(int(sentinel_norm))[:6] not in text
        for index in range(len(updates)):
            assert _digest(index) not in text
    payload = result.diagnostics.to_dict()
    assert set(payload["norm_quantile_bands"]) == {"p25", "p50", "p75"}
    assert set(payload["norm_quantile_bands"].values()) <= set(ROBUST_NORM_BANDS)
    assert not {"client", "client_id", "site", "trimmed_clients", "norms"} & set(
        payload
    )


def test_errors_never_echo_submitted_values_or_names():
    policy = _policy()
    attempts = [
        lambda: _update(policy, 0, {NAME_A: [[SENTINEL, math.nan, 1.0]]}),
        lambda: _update(policy, 0, {NAME_A: [[SENTINEL, 1.0e13, 1.0]]}),
        lambda: _update(policy, 0, {"adapter.secret_site_name": [[SENTINEL]]}),
        lambda: _update(_policy((2,), dtype="float16"), 0, {NAME_A: [SENTINEL, 0.1]}),
    ]
    for attempt in attempts:
        with pytest.raises(RobustAggregationError) as excinfo:
            attempt()
        rendered = "".join(traceback.format_exception_only(excinfo.value))
        assert "987654" not in rendered
        assert "secret_site_name" not in rendered
        assert str(excinfo.value) in ROBUST_AGGREGATION_REASON_CODES
        assert excinfo.value.__cause__ is None


def test_tolist_failure_is_suppressed():
    class Exploding:
        def tolist(self):
            raise RuntimeError("leak 987654.321")

    with pytest.raises(RobustAggregationError) as excinfo:
        _update(_policy(), 0, {NAME_A: Exploding()})
    assert _code(excinfo) == "robust_update_invalid"
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__suppress_context__ is True


def test_results_and_updates_are_immutable():
    policy, updates = _sentinel_batch()
    result = aggregate_adapter_updates(
        updates, policy=policy, config=_config(TRIMMED, 1)
    )
    assert isinstance(result, RobustAggregate)
    with pytest.raises(FrozenInstanceError):
        result.diagnostics = None  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        updates[0].metadata = None  # type: ignore[misc]
    with pytest.raises(TypeError):
        result.tensors[NAME_A] = ()  # type: ignore[index]
    with pytest.raises(TypeError):
        updates[0].tensors[NAME_A] = ()  # type: ignore[index]


@pytest.mark.parametrize("strategy", ROBUST_AGGREGATION_STRATEGIES)
def test_norm_quantiles_never_report_extreme_norms(strategy):
    policy = _policy((1,))
    updates = [
        _update(policy, 0, {NAME_A: [1.0e-9]}),
        _update(policy, 1, {NAME_A: [5.0]}),
        _update(policy, 2, {NAME_A: [-SENTINEL]}),
    ]
    result = aggregate_adapter_updates(
        updates, policy=policy, config=_config(strategy, 1)
    )
    assert result.diagnostics.norm_quantile_bands == ("1e0_to_1e1",) * 3


def test_unclipped_metadata_is_refused_under_a_clipping_policy():
    policy = _policy()
    lax = FederatedUpdatePolicy(
        model_digest=MODEL,
        parameters=policy.parameters,
        require_clipped=False,
        max_total_elements=3,
    )
    unclipped = FederatedUpdateMetadata(
        model_digest=MODEL,
        adapter_format="dense",
        parameters=lax.parameters,
        total_elements=3,
        update_digest=_digest(9),
        clipped=False,
        policy=lax,
    )
    updates = _rows(policy, HONEST_ROWS[:2]) + [
        RobustAdapterUpdate(metadata=unclipped, tensors={NAME_A: [[0.0, 1.0, 2.0]]})
    ]
    with pytest.raises(RobustAggregationError) as excinfo:
        aggregate_adapter_updates(updates, policy=policy, config=_config(TRIMMED, 1))
    assert _code(excinfo) == "robust_update_invalid"


def test_norm_band_edges_are_stable():
    band = robust_module._norm_band
    assert band(0.0) == "zero"
    assert band(5.0e-7) == "below_1e-6"
    assert band(1.0e-6) == "1e-6_to_1e-5"
    assert band(0.999) == "1e-1_to_1e0"
    assert band(1.0) == "1e0_to_1e1"
    assert band(1000.0) == "1e3_to_1e4"
    assert band(999999.0) == "1e5_to_1e6"
    assert band(1.0e6) == "at_least_1e6"
    assert len(ROBUST_NORM_BANDS) == len(set(ROBUST_NORM_BANDS))


def test_reason_codes_are_a_closed_stable_set():
    assert ROBUST_AGGREGATION_REASON_CODES == frozenset(
        {
            "robust_adapter_mismatch",
            "robust_adversary_bound_exceeded",
            "robust_config_invalid",
            "robust_dtype_mismatch",
            "robust_duplicate_update",
            "robust_limit_exceeded",
            "robust_non_finite_value",
            "robust_parameter_mismatch",
            "robust_shape_mismatch",
            "robust_strategy_unsupported",
            "robust_update_invalid",
            "robust_value_out_of_range",
        }
    )
    assert ROBUST_AGGREGATION_STRATEGIES == (TRIMMED, CLIPPED)
    error = RobustAggregationError("not_a_code")
    assert error.code == "robust_update_invalid"


def test_module_is_stdlib_only_and_offline():
    source = Path(robust_module.__file__).read_text(encoding="utf-8")
    assert source.isascii()
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    forbidden = {
        "numpy",
        "torch",
        "socket",
        "urllib",
        "http",
        "requests",
        "httpx",
        "logging",
        "subprocess",
    }
    assert not imported & forbidden
    assert imported <= {
        "__future__",
        "collections",
        "dataclasses",
        "hashlib",
        "json",
        "math",
        "openmed",
        "struct",
        "types",
        "typing",
    }
