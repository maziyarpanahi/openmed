"""Synthetic, offline contracts for federated LoRA product-space aggregation."""

from __future__ import annotations

import json
import math
import re
from dataclasses import FrozenInstanceError
from fractions import Fraction
from itertools import permutations

import pytest

from openmed.training.federated_lora_aggregation import (
    LORA_AGGREGATION_EVIDENCE_SCHEMA_VERSION,
    LORA_AGGREGATION_MODES,
    LORA_AGGREGATION_REASON_CODES,
    LORA_RESIDUAL_BAND_BOUNDS,
    LORA_RESIDUAL_BANDS,
    LORA_ROUND_SCHEMA_VERSION,
    LORA_UPDATE_SCHEMA_VERSION,
    LoraAggregate,
    LoraAggregationError,
    LoraFactorPair,
    LoraModuleSpec,
    LoraRoundDeclaration,
    LoraUpdate,
    aggregate_lora_updates,
    lora_factor_digest,
)

QUERY = "encoder.layer.0.attention.query"
VALUE = "encoder.layer.0.attention.value"
SENTINEL = 1234.5678

# Rank-2 factors for a 4x3 module (out=4, in=3). Every value is dyadic so the
# exact weighted means used below stay representable in binary floating point.
SHARED_A = ((1.0, 0.5, -0.25), (0.0, 1.0, 0.75))
SITE_B = (
    ((1.0, 0.0), (0.5, -1.0), (0.25, 2.0), (-1.5, 0.5)),
    ((-0.5, 1.0), (1.5, 0.25), (0.0, -0.75), (2.0, 1.0)),
    ((0.75, -0.25), (-1.0, 0.5), (1.25, 0.0), (0.5, -2.0)),
)
SITE_A = (
    SHARED_A,
    ((0.5, -1.0, 0.25), (1.0, 0.0, -0.5)),
    ((-0.25, 0.75, 1.0), (0.5, 0.5, 0.0)),
)
VALUE_A = ((0.5, 0.25, 0.0), (0.0, -0.5, 1.0))
VALUE_B = ((1.0, 0.5), (0.0, 0.25), (-0.5, 1.0), (0.25, 0.0))


def _fractions(matrix):
    return [[Fraction(value) for value in row] for row in matrix]


def _matmul(left, right):
    left = _fractions(left)
    right = _fractions(right)
    return [
        [
            sum((a * b for a, b in zip(row, column)), Fraction(0))
            for column in zip(*right)
        ]
        for row in left
    ]


def _weighted_mean(matrices, weights):
    total = sum(weights, Fraction(0))
    rows, cols = len(matrices[0]), len(matrices[0][0])
    return [
        [
            sum(
                (
                    Fraction(weight) * Fraction(matrix[j][k])
                    for matrix, weight in zip(matrices, weights)
                ),
                Fraction(0),
            )
            / total
            for k in range(cols)
        ]
        for j in range(rows)
    ]


def _relative_residual(target, approx):
    diff = sum(
        (t - a) ** 2
        for t_row, a_row in zip(target, approx)
        for t, a in zip(t_row, a_row)
    )
    norm = sum(t**2 for row in target for t in row)
    return math.sqrt(diff / norm)


def _update(b, a, *, alpha=16, module=QUERY, extra=True):
    modules = [LoraFactorPair(module, factor_a=a, factor_b=b)]
    if extra:
        modules.append(LoraFactorPair(VALUE, factor_a=VALUE_A, factor_b=VALUE_B))
    return LoraUpdate(rank=len(a), alpha=alpha, modules=tuple(modules))


def _declaration(mode, *, rank=2, alpha=16, pinned=True, **overrides):
    frozen = mode == "frozen_a" and pinned
    modules = (
        LoraModuleSpec(
            QUERY,
            in_features=3,
            out_features=4,
            frozen_a_digest=lora_factor_digest(SHARED_A) if frozen else None,
        ),
        LoraModuleSpec(
            VALUE,
            in_features=3,
            out_features=4,
            frozen_a_digest=lora_factor_digest(VALUE_A) if frozen else None,
        ),
    )
    values = {"mode": mode, "rank": rank, "alpha": alpha, "modules": modules}
    values.update(overrides)
    return LoraRoundDeclaration(**values)


def _frozen_updates():
    return [_update(b, SHARED_A) for b in SITE_B]


def _product_updates():
    return [_update(b, a) for b, a in zip(SITE_B, SITE_A)]


def _module(aggregate: LoraAggregate, name: str) -> LoraFactorPair:
    return next(pair for pair in aggregate.update.modules if pair.target_module == name)


def _code(excinfo) -> str:
    assert str(excinfo.value) == excinfo.value.code
    return excinfo.value.code


# --- Frozen-A aggregation equals the weighted mean of B @ A exactly ---------


@pytest.mark.parametrize("weights", [None, (1, 1, 2), (Fraction(1, 2), 1, 2.5)])
def test_frozen_a_equals_weighted_mean_of_products_exactly(weights):
    updates = _frozen_updates()
    weight_map = (
        None if weights is None else {u.digest: w for u, w in zip(updates, weights)}
    )
    aggregate = aggregate_lora_updates(
        updates, _declaration("frozen_a"), weights=weight_map
    )
    pair = _module(aggregate, QUERY)
    expected = _weighted_mean(
        [_matmul(b, SHARED_A) for b in SITE_B],
        [Fraction(w) for w in (weights or (1, 1, 1))],
    )
    if weights is None:
        # Uniform weights over three sites are not dyadic: B is the correctly
        # rounded exact mean and the product matches within the recorded band.
        exact_b = _weighted_mean(SITE_B, [Fraction(1)] * 3)
        assert pair.factor_b == tuple(
            tuple(float(value) for value in row) for row in exact_b
        )
        band = aggregate.evidence.residual_band
        assert band in {"zero", "below_1e-12"}
        bound = LORA_RESIDUAL_BAND_BOUNDS[band]
        assert _relative_residual(expected, _matmul(pair.factor_b, SHARED_A)) <= bound
    else:
        assert _matmul(pair.factor_b, pair.factor_a) == expected
        assert aggregate.evidence.residual_band == "zero"
    assert pair.factor_a == SHARED_A
    assert aggregate.evidence.mode == "frozen_a"
    assert aggregate.evidence.rank == 2


def test_naive_per_factor_mean_is_not_the_mean_of_products():
    """Negative control: separate factor averages diverge from mean(B @ A)."""
    exact = _weighted_mean(
        [_matmul(b, a) for b, a in zip(SITE_B, SITE_A)], [Fraction(1)] * 3
    )
    naive = _matmul(
        _weighted_mean(SITE_B, [Fraction(1)] * 3),
        _weighted_mean(SITE_A, [Fraction(1)] * 3),
    )
    assert naive != exact
    assert _relative_residual(exact, naive) > 0.1


# --- Product-space aggregation stays within the recorded residual bound -----


def test_product_space_matches_mean_within_recorded_residual_bound():
    aggregate = aggregate_lora_updates(
        _product_updates(), _declaration("product_space")
    )
    pair = _module(aggregate, QUERY)
    assert len(pair.factor_a) == 2 and len(pair.factor_b[0]) == 2
    expected = _weighted_mean(
        [_matmul(b, a) for b, a in zip(SITE_B, SITE_A)], [Fraction(1)] * 3
    )
    residual = _relative_residual(expected, _matmul(pair.factor_b, pair.factor_a))
    band = aggregate.evidence.residual_band
    bound = LORA_RESIDUAL_BAND_BOUNDS[band]
    if bound is None:
        assert residual >= 0.1
    else:
        assert residual < bound
    naive = _matmul(
        _weighted_mean(SITE_B, [Fraction(1)] * 3),
        _weighted_mean(SITE_A, [Fraction(1)] * 3),
    )
    # The truncated factorization is the best rank-2 fit, so it never loses to
    # naive per-factor averaging on the same synthetic sites.
    assert residual <= _relative_residual(expected, naive)


def test_product_space_recovers_low_rank_mean_to_rounding_error():
    updates = _frozen_updates()
    aggregate = aggregate_lora_updates(updates, _declaration("product_space"))
    pair = _module(aggregate, QUERY)
    expected = _weighted_mean([_matmul(b, SHARED_A) for b in SITE_B], [Fraction(1)] * 3)
    residual = _relative_residual(expected, _matmul(pair.factor_b, pair.factor_a))
    assert aggregate.evidence.residual_band in {"zero", "below_1e-12", "below_1e-9"}
    assert residual < LORA_RESIDUAL_BAND_BOUNDS["below_1e-9"]
    # A' has orthonormal rows under the documented factor convention.
    gram = _matmul(pair.factor_a, tuple(zip(*pair.factor_a)))
    for i, row in enumerate(gram):
        for j, value in enumerate(row):
            assert abs(float(value) - (1.0 if i == j else 0.0)) < 1e-12


@pytest.mark.parametrize(("out_features", "in_features"), [(5, 3), (3, 5)])
def test_product_space_handles_tall_and_wide_modules(out_features, in_features):
    def matrix(rows, cols, offset):
        return tuple(
            tuple(((r * 7 + c * 3 + offset) % 9 - 4) / 4 for c in range(cols))
            for r in range(rows)
        )

    sites = [
        (matrix(out_features, 2, seed), matrix(2, in_features, seed + 5))
        for seed in range(3)
    ]
    declaration = LoraRoundDeclaration(
        mode="product_space",
        rank=2,
        alpha=8,
        modules=(LoraModuleSpec(QUERY, in_features, out_features),),
    )
    updates = [_update(b, a, alpha=8, extra=False) for b, a in sites]
    aggregate = aggregate_lora_updates(updates, declaration)
    pair = aggregate.update.modules[0]
    assert len(pair.factor_b) == out_features and len(pair.factor_a[0]) == in_features
    expected = _weighted_mean([_matmul(b, a) for b, a in sites], [Fraction(1)] * 3)
    residual = _relative_residual(expected, _matmul(pair.factor_b, pair.factor_a))
    bound = LORA_RESIDUAL_BAND_BOUNDS[aggregate.evidence.residual_band]
    assert residual < bound if bound is not None else residual >= 0.1


def test_zero_mean_product_space_records_zero_residual():
    zero_b = ((0.0, 0.0),) * 4
    updates = [_update(zero_b, a) for a in SITE_A]
    aggregate = aggregate_lora_updates(updates, _declaration("product_space"))
    assert aggregate.evidence.residual_band in {"zero", "below_1e-12"}
    pair = _module(aggregate, QUERY)
    assert all(value == 0.0 for row in pair.factor_b for value in row)


def test_residual_ceiling_refuses_coarse_refactorization():
    declaration = _declaration("product_space", residual_ceiling="below_1e-6")
    with pytest.raises(LoraAggregationError) as excinfo:
        aggregate_lora_updates(_product_updates(), declaration)
    assert _code(excinfo) == "lora_residual_exceeds_ceiling"
    # Negative control: a low-rank mean passes the same ceiling.
    aggregate = aggregate_lora_updates(_frozen_updates(), declaration)
    assert aggregate.evidence.residual_band != "at_least_1e-1"


# --- Heterogeneous ranks only when declared --------------------------------


def test_zero_pad_rank_policy_aggregates_scaled_products():
    low_a = (SHARED_A[0],)
    low_b = tuple((row[0],) for row in SITE_B[1])
    updates = [
        _update(SITE_B[0], SHARED_A),
        LoraUpdate(
            rank=1,
            alpha=16,
            modules=(
                LoraFactorPair(QUERY, factor_a=low_a, factor_b=low_b),
                LoraFactorPair(
                    VALUE,
                    factor_a=(VALUE_A[0],),
                    factor_b=tuple((row[0],) for row in VALUE_B),
                ),
            ),
        ),
    ]
    with pytest.raises(LoraAggregationError) as excinfo:
        aggregate_lora_updates(updates, _declaration("product_space"))
    assert _code(excinfo) == "lora_rank_mismatch"

    declaration = _declaration("product_space", rank_policy="zero_pad")
    aggregate = aggregate_lora_updates(updates, declaration)
    assert aggregate.evidence.rank_policy == "zero_pad"
    pair = _module(aggregate, QUERY)
    # Each site's alpha / rank scaled delta is preserved: rank-1 products are
    # multiplied by round_rank / site_rank before averaging.
    scaled = [
        _matmul(SITE_B[0], SHARED_A),
        [[2 * value for value in row] for row in _matmul(low_b, low_a)],
    ]
    expected = _weighted_mean(scaled, [Fraction(1)] * 2)
    residual = _relative_residual(expected, _matmul(pair.factor_b, pair.factor_a))
    assert residual < LORA_RESIDUAL_BAND_BOUNDS["below_1e-9"]


def test_zero_pad_rejects_site_rank_above_round_rank():
    declaration = _declaration("product_space", rank=1, rank_policy="zero_pad")
    with pytest.raises(LoraAggregationError) as excinfo:
        aggregate_lora_updates(_product_updates(), declaration)
    assert _code(excinfo) == "lora_rank_mismatch"


def test_zero_pad_requires_product_space_mode():
    with pytest.raises(LoraAggregationError) as excinfo:
        _declaration("frozen_a", rank_policy="zero_pad")
    assert _code(excinfo) == "lora_declaration_invalid"


# --- Fail-closed rejections with stable reason codes ------------------------


@pytest.mark.parametrize(
    "mode", ["per_factor_mean", "per_factor", "naive", "factor_mean", "fedavg_factors"]
)
def test_naive_per_factor_averaging_is_rejected(mode):
    with pytest.raises(LoraAggregationError) as excinfo:
        _declaration(mode)
    assert _code(excinfo) == "lora_naive_factor_average_rejected"


def test_unknown_mode_is_rejected():
    with pytest.raises(LoraAggregationError) as excinfo:
        _declaration("median")
    assert _code(excinfo) == "lora_mode_unsupported"


def test_reject_mode_refuses_every_batch():
    declaration = _declaration("reject")
    with pytest.raises(LoraAggregationError) as excinfo:
        aggregate_lora_updates(_frozen_updates(), declaration)
    assert _code(excinfo) == "lora_aggregation_rejected"


def test_unpaired_named_factors_are_rejected():
    factors = {
        f"{QUERY}.lora_A.weight": SHARED_A,
        f"{QUERY}.lora_B.weight": SITE_B[0],
        f"{VALUE}.lora_A.weight": VALUE_A,
    }
    with pytest.raises(LoraAggregationError) as excinfo:
        LoraUpdate.from_named_factors(factors, rank=2, alpha=16)
    assert _code(excinfo) == "lora_factor_unpaired"
    # Negative control: the complete pairing parses into the same update.
    factors[f"{VALUE}.lora_B.weight"] = VALUE_B
    parsed = LoraUpdate.from_named_factors(factors, rank=2, alpha=16)
    assert parsed.digest == _update(SITE_B[0], SHARED_A).digest


def test_factor_pair_with_incompatible_inner_dimension_is_unpaired():
    with pytest.raises(LoraAggregationError) as excinfo:
        LoraFactorPair(QUERY, factor_a=SHARED_A, factor_b=((1.0,),) * 4)
    assert _code(excinfo) == "lora_factor_unpaired"


def test_unrecognized_factor_names_are_rejected():
    with pytest.raises(LoraAggregationError) as excinfo:
        LoraUpdate.from_named_factors({f"{QUERY}.weight": SHARED_A}, rank=2, alpha=16)
    assert _code(excinfo) == "lora_update_invalid"


def test_declared_rank_must_match_factor_shapes():
    with pytest.raises(LoraAggregationError) as excinfo:
        LoraUpdate(
            rank=3,
            alpha=16,
            modules=(LoraFactorPair(QUERY, factor_a=SHARED_A, factor_b=SITE_B[0]),),
        )
    assert _code(excinfo) == "lora_rank_mismatch"


@pytest.mark.parametrize("mode", ["frozen_a", "product_space"])
def test_rank_mismatch_across_sites_is_rejected(mode):
    updates = _frozen_updates()
    with pytest.raises(LoraAggregationError) as excinfo:
        aggregate_lora_updates(updates, _declaration(mode, rank=1))
    assert _code(excinfo) == "lora_rank_mismatch"


@pytest.mark.parametrize("mode", ["frozen_a", "product_space"])
def test_scaling_mismatch_is_rejected(mode):
    updates = _frozen_updates()
    updates[1] = _update(SITE_B[1], SHARED_A, alpha=32)
    with pytest.raises(LoraAggregationError) as excinfo:
        aggregate_lora_updates(updates, _declaration(mode))
    assert _code(excinfo) == "lora_scaling_mismatch"


def test_frozen_a_rejects_diverged_a_factor():
    with pytest.raises(LoraAggregationError) as excinfo:
        aggregate_lora_updates(_product_updates(), _declaration("frozen_a"))
    assert _code(excinfo) == "lora_frozen_a_mismatch"


def test_frozen_a_requires_pinned_digest():
    with pytest.raises(LoraAggregationError) as excinfo:
        _declaration("frozen_a", pinned=False)
    assert _code(excinfo) == "lora_declaration_invalid"


def test_pinned_digest_is_refused_outside_frozen_a():
    module = LoraModuleSpec(
        QUERY,
        in_features=3,
        out_features=4,
        frozen_a_digest=lora_factor_digest(SHARED_A),
    )
    with pytest.raises(LoraAggregationError) as excinfo:
        LoraRoundDeclaration(mode="product_space", rank=2, alpha=16, modules=(module,))
    assert _code(excinfo) == "lora_declaration_invalid"


def test_declared_rank_above_module_dimensions_is_rejected():
    module = LoraModuleSpec(QUERY, in_features=1, out_features=4)
    with pytest.raises(LoraAggregationError) as excinfo:
        LoraRoundDeclaration(mode="product_space", rank=2, alpha=16, modules=(module,))
    assert _code(excinfo) == "lora_declaration_invalid"


def test_module_set_and_shape_mismatches_are_rejected():
    updates = [_update(b, SHARED_A, extra=False) for b in SITE_B]
    with pytest.raises(LoraAggregationError) as excinfo:
        aggregate_lora_updates(updates, _declaration("product_space"))
    assert _code(excinfo) == "lora_module_mismatch"
    wide_a = tuple(row + (0.0,) for row in SHARED_A)
    updates = [_update(SITE_B[0], wide_a)]
    with pytest.raises(LoraAggregationError) as excinfo:
        aggregate_lora_updates(updates, _declaration("product_space"))
    assert _code(excinfo) == "lora_module_mismatch"


def test_duplicate_updates_are_rejected():
    updates = [_update(SITE_B[0], SHARED_A), _update(SITE_B[0], SHARED_A)]
    with pytest.raises(LoraAggregationError) as excinfo:
        aggregate_lora_updates(updates, _declaration("frozen_a"))
    assert _code(excinfo) == "lora_duplicate_update"


@pytest.mark.parametrize(
    "bad", [0, -1, True, float("nan"), float("inf"), "1", Fraction(0)]
)
def test_invalid_weights_are_rejected(bad):
    updates = _frozen_updates()
    weights = {update.digest: 1 for update in updates}
    weights[updates[0].digest] = bad
    with pytest.raises(LoraAggregationError) as excinfo:
        aggregate_lora_updates(updates, _declaration("frozen_a"), weights=weights)
    assert _code(excinfo) == "lora_weight_invalid"


def test_weights_must_cover_exactly_the_submitted_updates():
    updates = _frozen_updates()
    weights = {update.digest: 1 for update in updates[:2]}
    with pytest.raises(LoraAggregationError) as excinfo:
        aggregate_lora_updates(updates, _declaration("frozen_a"), weights=weights)
    assert _code(excinfo) == "lora_weight_invalid"


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), True, "0.5", None, 2.0**40]
)
def test_invalid_factor_values_are_rejected(value):
    bad_a = ((value, 0.0, 0.0), SHARED_A[1])
    with pytest.raises(LoraAggregationError) as excinfo:
        LoraFactorPair(QUERY, factor_a=bad_a, factor_b=SITE_B[0])
    assert _code(excinfo) in {"lora_update_invalid", "lora_limit_exceeded"}


def test_ragged_and_oversized_factors_are_rejected():
    with pytest.raises(LoraAggregationError) as excinfo:
        LoraFactorPair(QUERY, factor_a=((1.0, 0.0), (1.0,)), factor_b=SITE_B[0])
    assert _code(excinfo) == "lora_update_invalid"
    with pytest.raises(LoraAggregationError) as excinfo:
        LoraFactorPair(QUERY, factor_a=((1.0,) * 65,), factor_b=((1.0,),))
    assert _code(excinfo) == "lora_limit_exceeded"


def test_empty_or_foreign_update_batches_are_rejected():
    declaration = _declaration("frozen_a")
    for batch in ([], [object()], "updates"):
        with pytest.raises(LoraAggregationError) as excinfo:
            aggregate_lora_updates(batch, declaration)
        assert _code(excinfo) == "lora_update_invalid"
    with pytest.raises(LoraAggregationError) as excinfo:
        aggregate_lora_updates(_frozen_updates(), object())
    assert _code(excinfo) == "lora_declaration_invalid"


def test_reason_codes_are_a_closed_stable_set():
    assert LORA_AGGREGATION_REASON_CODES == frozenset(
        {
            "lora_aggregation_rejected",
            "lora_declaration_invalid",
            "lora_duplicate_update",
            "lora_factor_unpaired",
            "lora_frozen_a_mismatch",
            "lora_limit_exceeded",
            "lora_mode_unsupported",
            "lora_module_mismatch",
            "lora_naive_factor_average_rejected",
            "lora_rank_mismatch",
            "lora_residual_exceeds_ceiling",
            "lora_scaling_mismatch",
            "lora_update_invalid",
            "lora_weight_invalid",
        }
    )
    assert LORA_AGGREGATION_MODES == ("frozen_a", "product_space", "reject")
    assert LORA_RESIDUAL_BANDS[0] == "zero"
    assert set(LORA_RESIDUAL_BAND_BOUNDS) == set(LORA_RESIDUAL_BANDS)


# --- Arrival-order independence --------------------------------------------


@pytest.mark.parametrize("mode", ["frozen_a", "product_space"])
def test_permuted_arrival_order_is_byte_identical(mode):
    updates = _frozen_updates() if mode == "frozen_a" else _product_updates()
    weights = {u.digest: w for u, w in zip(updates, (1, Fraction(1, 3), 2.5))}
    declaration = _declaration(mode)
    results = [
        aggregate_lora_updates(list(order), declaration, weights=weights)
        for order in permutations(updates)
    ]
    first = results[0]
    for result in results[1:]:
        assert result.to_bytes() == first.to_bytes()
        assert result.digest == first.digest
        assert result.evidence.to_json() == first.evidence.to_json()
    assert first.evidence.aggregate_digest == first.digest


def test_module_order_does_not_change_update_digest():
    first = LoraUpdate(
        rank=2,
        alpha=16,
        modules=(
            LoraFactorPair(QUERY, factor_a=SHARED_A, factor_b=SITE_B[0]),
            LoraFactorPair(VALUE, factor_a=VALUE_A, factor_b=VALUE_B),
        ),
    )
    second = LoraUpdate(rank=2, alpha=16, modules=tuple(reversed(first.modules)))
    assert first.digest == second.digest
    assert first.to_bytes() == second.to_bytes()


def test_negative_zero_is_canonicalized():
    negative = ((-0.0, 0.5, -0.25), SHARED_A[1])
    positive = ((0.0, 0.5, -0.25), SHARED_A[1])
    assert lora_factor_digest(negative) == lora_factor_digest(positive)


def test_frozen_a_aggregate_digest_golden_vector():
    aggregate = aggregate_lora_updates(_frozen_updates(), _declaration("frozen_a"))
    again = aggregate_lora_updates(
        list(reversed(_frozen_updates())), _declaration("frozen_a")
    )
    # Exact integer sums and one correctly rounded division per value make the
    # frozen-A aggregate bytes platform independent.
    assert (
        aggregate.digest
        == again.digest
        == ("sha256:07b6752f86a89b977521c58b2786e9e161db11ee736c7b8b432bfe12dbd2148c")
    )


# --- Evidence is value-free -------------------------------------------------


@pytest.mark.parametrize("mode", ["frozen_a", "product_space"])
def test_evidence_contains_only_mode_rank_band_and_digests(mode):
    b = tuple((SENTINEL, row[1]) for row in SITE_B[0])
    a = SHARED_A if mode == "frozen_a" else ((SENTINEL, 0.5, -0.25), SHARED_A[1])
    updates = [_update(b, a), *(_update(sb, SHARED_A) for sb in SITE_B[1:])]
    if mode == "frozen_a":
        updates[0] = _update(b, SHARED_A)
    weights = {u.digest: w for u, w in zip(updates, (7, 11, 13))}
    aggregate = aggregate_lora_updates(updates, _declaration(mode), weights=weights)
    evidence = aggregate.evidence
    payload = evidence.to_dict()
    assert set(payload) == {
        "schema_version",
        "mode",
        "rank",
        "rank_policy",
        "residual_band",
        "declaration_digest",
        "inputs_digest",
        "aggregate_digest",
    }
    assert payload["schema_version"] == LORA_AGGREGATION_EVIDENCE_SCHEMA_VERSION
    assert payload["residual_band"] in LORA_RESIDUAL_BANDS
    for key in ("declaration_digest", "inputs_digest", "aggregate_digest"):
        assert payload[key].startswith("sha256:")
    text = evidence.to_json() + repr(evidence)
    for update in updates:
        assert update.digest not in text
    stripped = re.sub(r"sha256:[0-9a-f]{64}", "", text)
    for forbidden in ("1234", "5678", QUERY, VALUE, "0.25", "11", "13"):
        assert forbidden not in stripped
    assert "1234" not in repr(aggregate)
    assert json.loads(evidence.to_json()) == payload


def test_reprs_and_errors_do_not_echo_factor_values():
    pair = LoraFactorPair(QUERY, factor_a=SHARED_A, factor_b=((SENTINEL, 0.0),) * 4)
    update = LoraUpdate(rank=2, alpha=16, modules=(pair,))
    assert "1234" not in repr(pair) + repr(update)
    descriptor = update.descriptor()
    assert "1234" not in json.dumps(descriptor)
    assert descriptor["schema_version"] == LORA_UPDATE_SCHEMA_VERSION
    assert descriptor["modules"][0]["factor_a_digest"] == lora_factor_digest(SHARED_A)
    with pytest.raises(LoraAggregationError) as excinfo:
        LoraFactorPair(
            "secret.module", factor_a=((SENTINEL, float("nan")),), factor_b=((1.0,),)
        )
    message = str(excinfo.value)
    assert "1234" not in message and "secret" not in message


def test_records_are_immutable_and_versioned():
    declaration = _declaration("product_space")
    assert declaration.to_dict()["schema_version"] == LORA_ROUND_SCHEMA_VERSION
    aggregate = aggregate_lora_updates(_product_updates(), declaration)
    with pytest.raises(FrozenInstanceError):
        aggregate.evidence.mode = "frozen_a"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        declaration.mode = "frozen_a"  # type: ignore[misc]


def test_numpy_factors_are_accepted_when_available():
    np = pytest.importorskip("numpy")
    from_numpy = LoraFactorPair(
        QUERY, factor_a=np.array(SHARED_A), factor_b=np.array(SITE_B[0])
    )
    from_tuples = LoraFactorPair(QUERY, factor_a=SHARED_A, factor_b=SITE_B[0])
    assert from_numpy == from_tuples
    assert from_numpy.factor_a_digest == from_tuples.factor_a_digest
