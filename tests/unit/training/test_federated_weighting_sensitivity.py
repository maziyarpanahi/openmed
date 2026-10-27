"""Weighting-versus-sensitivity guard over synthetic DP declarations."""

from __future__ import annotations

import json
import math
import re
from decimal import Decimal
from fractions import Fraction

import pytest

from openmed.training.federated_weighting_sensitivity import (
    FEDERATED_WEIGHTING_SENSITIVITY_SCHEMA_VERSION,
    REASON_CODES,
    AccountantDeclaration,
    UpdateMechanismDeclaration,
    WeightingCompatibilityReceipt,
    WeightingPolicyDeclaration,
    WeightingSensitivityAssessment,
    WeightingSensitivityError,
    check_weighting_sensitivity,
    derive_effective_sensitivity,
    require_weighting_sensitivity,
    weighted_sum_sensitivity,
)

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RECEIPT_KEYS = {
    "schema_version",
    "outcome",
    "reason_codes",
    "weighting_policy_digest",
    "mechanism_digest",
    "accountant_digest",
    "normalization",
    "adjacency",
    "receipt_digest",
}

# Synthetic declared volume bands with relative factors, never record counts.
_BANDS = {"small": 1, "medium": 2, "large": 8}


def _policy(**overrides: object) -> WeightingPolicyDeclaration:
    values: dict[str, object] = {
        "weight_source": "declared_band",
        "band_weights": _BANDS,
        "weight_cap": 4,
        "normalization": "fixed_denominator",
        "fixed_denominator": 40,
    }
    values.update(overrides)
    return WeightingPolicyDeclaration(**values)


def _mechanism(**overrides: object) -> UpdateMechanismDeclaration:
    # Effective sensitivity of _policy() with C = 3/2 is 3/2 * 4 / 40 = 3/20.
    values: dict[str, object] = {
        "mechanism": "gaussian",
        "privacy_unit": "client",
        "adjacency": "add_remove",
        "clip_norm": Fraction(3, 2),
        "sensitivity": Fraction(3, 20),
        "noise_multiplier": Fraction(11, 10),
    }
    values.update(overrides)
    return UpdateMechanismDeclaration(**values)


def _accountant(**overrides: object) -> AccountantDeclaration:
    values: dict[str, object] = {
        "accountant": "rdp",
        "privacy_unit": "client",
        "adjacency": "add_remove",
        "noise_multiplier": Fraction(11, 10),
    }
    values.update(overrides)
    return AccountantDeclaration(**values)


# Reference vectors for the L2 sensitivity of a weighted sum of clipped client
# updates divided by a fixed, data-independent denominator W:
#   estimate = (sum_i w_i * x_i) / W, with ||x_i|| <= C and 0 <= w_i <= w_max.
# Add or remove one client changes the numerator by at most w_max * C, so the
# sensitivity is C * w_max / W. Replacing one client changes it by at most
# ||w x - w' x'|| <= 2 * C * w_max, so the sensitivity is 2 * C * w_max / W.
_REFERENCE_VECTORS = (
    # (clip C, max weight w_max, denominator W, adjacency, expected)
    (1, 1, 10, "add_remove", Fraction(1, 10)),  # 1 * 1 / 10
    (1, 1, 10, "replace_one", Fraction(1, 5)),  # 2 * 1 * 1 / 10
    (Fraction(3, 2), 4, 40, "add_remove", Fraction(3, 20)),  # 3/2 * 4 / 40
    (Fraction(3, 2), 4, 40, "replace_one", Fraction(3, 10)),  # 2 * 3/2 * 4 / 40
    (0.1, 1, 3, "add_remove", Fraction(1, 30)),  # decimal 0.1 read as 1/10
    (2, Fraction(5, 2), 7, "add_remove", Fraction(5, 7)),  # 2 * 5/2 / 7
)


@pytest.mark.parametrize(
    ("clip", "weight", "denominator", "adjacency", "expected"), _REFERENCE_VECTORS
)
def test_weighted_sum_sensitivity_matches_reference_vectors(
    clip: object,
    weight: object,
    denominator: object,
    adjacency: str,
    expected: Fraction,
) -> None:
    result = weighted_sum_sensitivity(clip, weight, denominator, adjacency=adjacency)
    assert type(result) is Fraction
    assert result == expected


def test_effective_sensitivity_uses_capped_band_weight() -> None:
    # Bands 1, 2, 8 capped at 4 -> w_max = min(8, 4) = 4; 3/2 * 4 / 40 = 3/20.
    assert derive_effective_sensitivity(_policy(), _mechanism()) == Fraction(3, 20)
    # Raising the cap above every band leaves the largest band: 3/2 * 8 / 40.
    raised = _policy(weight_cap=16)
    assert derive_effective_sensitivity(raised, _mechanism()) == Fraction(3, 10)
    # Replace-one adjacency doubles it: 2 * 3/2 * 4 / 40 = 3/10.
    replace = _mechanism(adjacency="replace_one", sensitivity=Fraction(3, 10))
    assert derive_effective_sensitivity(_policy(), replace) == Fraction(3, 10)


def test_floors_below_cap_do_not_change_sensitivity() -> None:
    policy = _policy(weight_floor=2, protected_slice_floors={"rare_language": 3})
    assert derive_effective_sensitivity(policy, _mechanism()) == Fraction(3, 20)
    assessment = check_weighting_sensitivity(policy, _mechanism(), _accountant())
    assert assessment.compatible is True
    assert assessment.reason_codes == ()
    # A floor equal to the cap keeps w_max = 4 and is still compatible.
    at_cap = _policy(weight_floor=4)
    assert check_weighting_sensitivity(at_cap, _mechanism(), _accountant()).compatible


def test_capped_weights_with_fixed_denominator_pass() -> None:
    assessment = check_weighting_sensitivity(
        _policy(),
        _mechanism(),
        _accountant(),
        weight_inputs=("small", "large", "medium", "large"),
    )
    assert type(assessment) is WeightingSensitivityAssessment
    assert assessment.compatible is True
    assert assessment.reason_codes == ()
    assert assessment.effective_sensitivity == Fraction(3, 20)
    assert assessment.effective_noise_multiplier == Fraction(11, 10)
    assert assessment.receipt.outcome == "compatible"
    receipt = require_weighting_sensitivity(
        _policy(), _mechanism(), _accountant(), weight_inputs=("small",)
    )
    assert type(receipt) is WeightingCompatibilityReceipt
    assert receipt.to_dict() == assessment.receipt.to_dict()


def test_conservative_mechanism_and_accountant_pass() -> None:
    # Noise calibrated for a larger sensitivity, accounted with a smaller
    # multiplier, only strengthens the bound: sigma = 1 * 3/10, z_eff = 2.
    mechanism = _mechanism(sensitivity=Fraction(3, 10), noise_multiplier=1)
    assessment = check_weighting_sensitivity(
        _policy(), mechanism, _accountant(noise_multiplier=Fraction(3, 2))
    )
    assert assessment.compatible is True
    assert assessment.effective_noise_multiplier == 2


def test_uncapped_count_proportional_weights_fail() -> None:
    policy = _policy(weight_source="record_count", band_weights={}, weight_cap=None)
    assessment = check_weighting_sensitivity(policy, _mechanism(), _accountant())
    assert assessment.compatible is False
    assert assessment.reason_codes == (
        "weighting_count_proportional",
        "weighting_uncapped",
    )
    assert assessment.effective_sensitivity is None
    assert assessment.receipt.outcome == "incompatible"


def test_capped_count_proportional_weights_still_fail() -> None:
    policy = _policy(weight_source="record_count", band_weights={}, weight_cap=4)
    assessment = check_weighting_sensitivity(policy, _mechanism(), _accountant())
    assert assessment.reason_codes == ("weighting_count_proportional",)
    assert assessment.effective_sensitivity == Fraction(3, 20)


def test_uncapped_declared_bands_fail() -> None:
    policy = _policy(weight_cap=None)
    assessment = check_weighting_sensitivity(policy, _mechanism(), _accountant())
    # Without a cap the largest band (8) sets 3/2 * 8 / 40 = 3/10, which also
    # exceeds the 3/20 calibration and overstates the accountant multiplier.
    assert assessment.reason_codes == (
        "weighting_uncapped",
        "mechanism_sensitivity_too_small",
        "accountant_noise_multiplier_overstated",
    )
    assert assessment.effective_sensitivity == Fraction(3, 10)


def test_realized_sum_normalization_fails() -> None:
    policy = _policy(normalization="realized_sum", fixed_denominator=None)
    assessment = check_weighting_sensitivity(policy, _mechanism(), _accountant())
    assert assessment.compatible is False
    assert assessment.reason_codes == ("weighting_data_dependent_normalization",)
    assert assessment.effective_sensitivity is None
    assert assessment.effective_noise_multiplier is None
    assert derive_effective_sensitivity(policy, _mechanism()) is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"weight_floor": 5},
        {"protected_slice_floors": {"rare_language": Fraction(9, 2)}},
    ],
)
def test_floor_lifting_a_client_above_the_cap_fails(
    overrides: dict[str, object],
) -> None:
    policy = _policy(**overrides)
    assessment = check_weighting_sensitivity(
        policy,
        _mechanism(),
        _accountant(),
    )
    assert assessment.reason_codes[0] == "weighting_floor_exceeds_cap"
    assert "mechanism_sensitivity_too_small" in assessment.reason_codes
    # The floor becomes the effective maximum weight: 3/2 * floor / 40.
    floor = overrides.get("weight_floor") or Fraction(9, 2)
    assert assessment.effective_sensitivity == Fraction(3, 2) * floor / 40


def test_mechanism_calibrated_for_smaller_sensitivity_is_rejected() -> None:
    # Policy sensitivity is 3/20, mechanism calibrated for 1/10.
    # sigma = 11/10 * 1/10 = 11/100; z_eff = (11/100) / (3/20) = 11/15 < 11/10.
    mechanism = _mechanism(sensitivity=Fraction(1, 10))
    assessment = check_weighting_sensitivity(_policy(), mechanism, _accountant())
    assert assessment.compatible is False
    assert assessment.reason_codes == (
        "mechanism_sensitivity_too_small",
        "accountant_noise_multiplier_overstated",
    )
    assert assessment.effective_noise_multiplier == Fraction(11, 15)
    with pytest.raises(WeightingSensitivityError) as caught:
        require_weighting_sensitivity(_policy(), mechanism, _accountant())
    assert caught.value.code == "mechanism_sensitivity_too_small"
    assert caught.value.reason_codes == assessment.reason_codes


def test_accountant_overstating_noise_multiplier_is_rejected() -> None:
    accountant = _accountant(noise_multiplier=Fraction(6, 5))
    assessment = check_weighting_sensitivity(_policy(), _mechanism(), accountant)
    assert assessment.reason_codes == ("accountant_noise_multiplier_overstated",)


@pytest.mark.parametrize("count", [1234567, 98.0, Fraction(17), Decimal("42"), True])
def test_exact_record_counts_as_weight_inputs_are_rejected(count: object) -> None:
    assessment = check_weighting_sensitivity(
        _policy(),
        _mechanism(),
        _accountant(),
        weight_inputs=("small", count, "large"),
    )
    assert assessment.compatible is False
    assert assessment.reason_codes == ("weighting_exact_count_input",)
    rendered = assessment.receipt.to_json() + repr(assessment)
    assert "1234567" not in re.sub(r"sha256:[0-9a-f]{64}", "", rendered)
    with pytest.raises(WeightingSensitivityError) as caught:
        require_weighting_sensitivity(
            _policy(), _mechanism(), _accountant(), weight_inputs=(count,)
        )
    assert str(caught.value) == "weighting_exact_count_input"


def test_undeclared_band_inputs_are_rejected() -> None:
    assessment = check_weighting_sensitivity(
        _policy(),
        _mechanism(),
        _accountant(),
        weight_inputs=("small", "site_alpha_hospital"),
    )
    assert assessment.reason_codes == ("weighting_undeclared_band",)
    assert "site_alpha_hospital" not in assessment.receipt.to_json()
    assert "site_alpha_hospital" not in repr(assessment)


def test_record_count_policy_cannot_take_band_inputs() -> None:
    policy = _policy(weight_source="record_count", band_weights={})
    assessment = check_weighting_sensitivity(
        policy, _mechanism(), _accountant(), weight_inputs=("small",)
    )
    assert assessment.reason_codes == (
        "weighting_count_proportional",
        "weighting_undeclared_band",
    )


@pytest.mark.parametrize(
    ("mechanism", "accountant", "codes"),
    [
        (
            {"mechanism": "laplace"},
            {},
            ("mechanism_unsupported",),
        ),
        (
            {"privacy_unit": "record"},
            {"privacy_unit": "record"},
            ("mechanism_privacy_unit_unsupported",),
        ),
        (
            {},
            {"privacy_unit": "record"},
            ("accountant_privacy_unit_mismatch",),
        ),
        (
            {},
            {"adjacency": "replace_one"},
            ("accountant_adjacency_mismatch",),
        ),
    ],
)
def test_mechanism_and_accountant_declarations_must_agree(
    mechanism: dict[str, object],
    accountant: dict[str, object],
    codes: tuple[str, ...],
) -> None:
    assessment = check_weighting_sensitivity(
        _policy(), _mechanism(**mechanism), _accountant(**accountant)
    )
    assert assessment.reason_codes == codes


def test_reason_codes_follow_the_published_order() -> None:
    policy = _policy(
        weight_source="record_count",
        band_weights={},
        weight_cap=None,
        normalization="realized_sum",
        fixed_denominator=None,
    )
    assessment = check_weighting_sensitivity(
        policy,
        _mechanism(mechanism="laplace", privacy_unit="record"),
        _accountant(adjacency="replace_one"),
        weight_inputs=(7, "unknown_band"),
    )
    assert set(assessment.reason_codes) <= set(REASON_CODES)
    assert list(assessment.reason_codes) == sorted(
        assessment.reason_codes, key=REASON_CODES.index
    )
    assert assessment.reason_codes == (
        "weighting_count_proportional",
        "weighting_exact_count_input",
        "weighting_undeclared_band",
        "weighting_uncapped",
        "weighting_data_dependent_normalization",
        "mechanism_unsupported",
        "mechanism_privacy_unit_unsupported",
        "accountant_privacy_unit_mismatch",
        "accountant_adjacency_mismatch",
    )
    assert len(set(REASON_CODES)) == len(REASON_CODES)


def test_receipt_is_value_free_and_binds_digests() -> None:
    small = check_weighting_sensitivity(
        _policy(), _mechanism(), _accountant(), weight_inputs=("small",) * 3
    )
    large = check_weighting_sensitivity(
        _policy(),
        _mechanism(),
        _accountant(),
        weight_inputs=("large", "medium") * 50,
    )
    payload = small.receipt.to_dict()
    assert set(payload) == _RECEIPT_KEYS
    # Participation volume and per-site band assignments never change it.
    assert payload == large.receipt.to_dict()
    assert payload["schema_version"] == FEDERATED_WEIGHTING_SENSITIVITY_SCHEMA_VERSION
    assert payload["outcome"] == "compatible"
    assert payload["reason_codes"] == []
    assert payload["normalization"] == "fixed_denominator"
    assert payload["adjacency"] == "add_remove"
    for key in (
        "weighting_policy_digest",
        "mechanism_digest",
        "accountant_digest",
        "receipt_digest",
    ):
        assert _DIGEST.fullmatch(payload[key])
    assert payload["weighting_policy_digest"] == _policy().digest()
    assert payload["mechanism_digest"] == _mechanism().digest()
    assert payload["accountant_digest"] == _accountant().digest()
    rendered = small.receipt.to_json()
    assert rendered.endswith("\n")
    assert json.loads(rendered) == payload
    assert rendered.isascii()
    stripped = re.sub(r"sha256:[0-9a-f]{64}", "", rendered)
    assert not re.search(r"\d", stripped.replace(payload["schema_version"], ""))
    for band in _BANDS:
        assert band not in rendered


def test_receipt_digest_changes_with_each_bound_declaration() -> None:
    base = check_weighting_sensitivity(_policy(), _mechanism(), _accountant())
    variants = (
        check_weighting_sensitivity(
            _policy(fixed_denominator=80), _mechanism(), _accountant()
        ),
        check_weighting_sensitivity(
            _policy(), _mechanism(noise_multiplier=2), _accountant()
        ),
        check_weighting_sensitivity(
            _policy(), _mechanism(), _accountant(accountant="prv")
        ),
    )
    digests = {base.receipt.receipt_digest}
    for variant in variants:
        digests.add(variant.receipt.receipt_digest)
    assert len(digests) == 4


def test_declaration_digests_are_canonical() -> None:
    first = _policy(band_weights={"large": 8, "small": 1, "medium": 2})
    second = _policy(band_weights={"small": 1.0, "medium": 2, "large": Fraction(8)})
    assert first.digest() == second.digest()
    assert _policy(weight_cap=3).digest() != first.digest()
    assert _mechanism(clip_norm=1.5).digest() == _mechanism().digest()


def test_reprs_and_errors_hide_policy_values() -> None:
    policy = _policy(weight_cap=Fraction(37, 9), fixed_denominator=Fraction(911, 3))
    text = repr(policy) + repr(
        check_weighting_sensitivity(policy, _mechanism(), _accountant())
    )
    text = re.sub(r"sha256:[0-9a-f]{64}", "", text)
    for value in ("37", "911", "Fraction"):
        assert value not in text
    with pytest.raises(WeightingSensitivityError) as caught:
        _policy(weight_cap=-31337)
    assert "31337" not in str(caught.value)
    assert caught.value.code == "invalid_weighting_policy"


@pytest.mark.parametrize(
    "overrides",
    [
        {"weight_source": "volume_proportional"},
        {"band_weights": {}},
        {"band_weights": {"Bad Band": 1}},
        {"band_weights": {"small": 0}},
        {"band_weights": {"small": math.inf}},
        {"band_weights": {"small": True}},
        {"band_weights": ["small"]},
        {"weight_source": "record_count", "band_weights": {"small": 1}},
        {"weight_cap": 0},
        {"weight_floor": -1},
        {"protected_slice_floors": {"rare": "2"}},
        {"normalization": "per_round"},
        {"fixed_denominator": None},
        {"normalization": "realized_sum", "fixed_denominator": 40},
        {"fixed_denominator": Fraction(1, 10**13)},
    ],
)
def test_malformed_weighting_policies_reject(overrides: dict[str, object]) -> None:
    with pytest.raises(WeightingSensitivityError) as caught:
        _policy(**overrides)
    assert caught.value.code == "invalid_weighting_policy"


@pytest.mark.parametrize(
    ("factory", "overrides", "code"),
    [
        (_mechanism, {"mechanism": "exponential"}, "invalid_mechanism_declaration"),
        (_mechanism, {"privacy_unit": "site"}, "invalid_mechanism_declaration"),
        (_mechanism, {"adjacency": "swap"}, "invalid_mechanism_declaration"),
        (_mechanism, {"clip_norm": 0}, "invalid_mechanism_declaration"),
        (_mechanism, {"sensitivity": None}, "invalid_mechanism_declaration"),
        (_mechanism, {"noise_multiplier": math.nan}, "invalid_mechanism_declaration"),
        (_accountant, {"accountant": "RDP!"}, "invalid_accountant_declaration"),
        (_accountant, {"privacy_unit": None}, "invalid_accountant_declaration"),
        (_accountant, {"noise_multiplier": -1}, "invalid_accountant_declaration"),
    ],
)
def test_malformed_mechanism_and_accountant_reject(
    factory: object, overrides: dict[str, object], code: str
) -> None:
    with pytest.raises(WeightingSensitivityError) as caught:
        factory(**overrides)  # type: ignore[operator]
    assert caught.value.code == code


@pytest.mark.parametrize(
    "inputs", ["small", 7, ("small", None), ("small", ["large"]), ("x",) * 4097]
)
def test_malformed_weight_input_collections_reject(inputs: object) -> None:
    with pytest.raises(WeightingSensitivityError) as caught:
        check_weighting_sensitivity(
            _policy(), _mechanism(), _accountant(), weight_inputs=inputs
        )
    assert caught.value.code == "invalid_weight_inputs"


def test_check_rejects_wrong_declaration_types() -> None:
    with pytest.raises(WeightingSensitivityError) as caught:
        swapped: object = _mechanism()
        check_weighting_sensitivity(swapped, _policy(), _accountant())
    assert caught.value.code == "invalid_declaration"


@pytest.mark.parametrize(
    ("args", "adjacency"),
    [
        ((0, 1, 1), "add_remove"),
        ((1, 0, 1), "add_remove"),
        ((1, 1, 0), "add_remove"),
        ((1, 1, 1), "neighbor"),
    ],
)
def test_weighted_sum_sensitivity_rejects_invalid_inputs(
    args: tuple[int, int, int], adjacency: str
) -> None:
    with pytest.raises(WeightingSensitivityError) as caught:
        weighted_sum_sensitivity(*args, adjacency=adjacency)
    assert caught.value.code == "invalid_sensitivity_input"


@pytest.mark.integration
def test_guard_matches_gaussian_calibration_profile() -> None:
    """A mechanism the guard rejects really misses its declared delta."""
    from openmed.risk.differential_privacy import (
        gaussian_privacy_delta,
        gaussian_scale,
    )

    epsilon, delta = 1.0, 1e-5
    effective = derive_effective_sensitivity(_policy(), _mechanism())
    assert effective == Fraction(3, 20)
    # Noise calibrated for the effective sensitivity meets the target delta.
    sigma = gaussian_scale(float(effective), epsilon, delta)
    assert gaussian_privacy_delta(sigma, float(effective), epsilon) <= delta
    # Noise calibrated for a smaller sensitivity (1/10) misses it, and the
    # guard rejects that declaration before any update is released.
    small_sigma = gaussian_scale(0.1, epsilon, delta)
    assert gaussian_privacy_delta(small_sigma, float(effective), epsilon) > delta
    multiplier = round(small_sigma / 0.1, 6)
    mechanism = _mechanism(sensitivity=Fraction(1, 10), noise_multiplier=multiplier)
    accountant = _accountant(noise_multiplier=multiplier)
    assessment = check_weighting_sensitivity(_policy(), mechanism, accountant)
    assert "mechanism_sensitivity_too_small" in assessment.reason_codes
    assert assessment.receipt.outcome == "incompatible"
