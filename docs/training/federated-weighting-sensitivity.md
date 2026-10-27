# Federated weighting sensitivity

Site weighting changes how much one client can move a federated aggregate. A
client-level differential-privacy guarantee holds only when the noise and the
accountant cover that influence. The module
`openmed.training.federated_weighting_sensitivity` compares three metadata
declarations for a DP round and fails closed when the weighting policy breaks
the declared sensitivity bound:

- a `WeightingPolicyDeclaration` (weight source, declared band weights, cap,
  floors, normalization rule and denominator),
- an `UpdateMechanismDeclaration` (mechanism, privacy unit, adjacency, clip
  norm, calibrated sensitivity and noise multiplier), and
- an `AccountantDeclaration` (accountant code, privacy unit, adjacency and the
  noise multiplier it consumes).

The guard reads no tensors, site identities or record counts. It does not
compute site weights or apply noise; weighting strategies and the update
mechanism are separate components. All arithmetic uses exact
`fractions.Fraction` values. A float input is read by its shortest decimal
form, so `0.1` means exactly `1/10`.

## Sensitivity derivation

For clipped client updates `x_i` with `||x_i|| <= C` and weights
`0 <= w_i <= w_max`, the weighted estimate with a fixed, data-independent
denominator `W` is `sum_i w_i * x_i / W`.

| Adjacency | Effective L2 sensitivity |
| --- | --- |
| `add_remove` one client | `C * w_max / W` |
| `replace_one` client | `2 * C * w_max / W` |

Each client weight is `max(floors, min(band weight, cap))`, so `w_max` is the
larger of the largest capped band weight, the global floor, and every
protected-slice floor. `weighted_sum_sensitivity()` evaluates the formula and
`derive_effective_sensitivity()` applies it to a policy and mechanism.

With realized-sum normalization (`W = sum_i w_i`), one client's influence
depends on every other client's private weight. With uncapped count weights,
`w_max` follows the largest private site. In both cases the declaration does
not bound the sensitivity, and the derivation returns `None`.

The mechanism adds noise with standard deviation
`noise_multiplier * sensitivity`. The effective noise multiplier is that scale
divided by the effective sensitivity. The accountant may use a smaller
multiplier than the effective one, never a larger one.

## Example

```python
from fractions import Fraction

from openmed.training.federated_weighting_sensitivity import (
    AccountantDeclaration,
    UpdateMechanismDeclaration,
    WeightingPolicyDeclaration,
    check_weighting_sensitivity,
)

# Synthetic relative volume bands; never site record counts.
policy = WeightingPolicyDeclaration(
    weight_source="declared_band",
    band_weights={"small": 1, "medium": 2, "large": 8},
    weight_cap=4,
    normalization="fixed_denominator",
    fixed_denominator=40,
)
mechanism = UpdateMechanismDeclaration(
    mechanism="gaussian",
    privacy_unit="client",
    adjacency="add_remove",
    clip_norm=Fraction(3, 2),
    sensitivity=Fraction(3, 20),
    noise_multiplier=Fraction(11, 10),
)
accountant = AccountantDeclaration(
    accountant="rdp",
    privacy_unit="client",
    adjacency="add_remove",
    noise_multiplier=Fraction(11, 10),
)

assessment = check_weighting_sensitivity(
    policy, mechanism, accountant, weight_inputs=("small", "large")
)
assert assessment.compatible
# 3/2 * min(8, 4) / 40 == 3/20
assert assessment.effective_sensitivity == Fraction(3, 20)
receipt_json = assessment.receipt.to_json()
```

`require_weighting_sensitivity()` takes the same arguments, returns the receipt
for a compatible round, and otherwise raises `WeightingSensitivityError`. The
error's `code` is the first reason code and `reason_codes` holds every reason.

## Reason codes

Reasons are reported in this fixed order, published as `REASON_CODES`:

| Code | Meaning |
| --- | --- |
| `weighting_count_proportional` | Weights follow record counts, even when capped. |
| `weighting_exact_count_input` | A per-client weight input is a number, treated as an exact count. |
| `weighting_undeclared_band` | A per-client weight input is not a declared band code. |
| `weighting_uncapped` | The policy declares no weight cap. |
| `weighting_floor_exceeds_cap` | A global or protected-slice floor lifts a client above the cap. |
| `weighting_data_dependent_normalization` | The denominator is the realized total weight. |
| `mechanism_unsupported` | The mechanism is not Gaussian; this guard derives L2 sensitivity. |
| `mechanism_privacy_unit_unsupported` | The mechanism protects records, not clients. |
| `accountant_privacy_unit_mismatch` | Accountant and mechanism privacy units differ. |
| `accountant_adjacency_mismatch` | Accountant and mechanism adjacency differ. |
| `mechanism_sensitivity_too_small` | The noise was calibrated for a smaller sensitivity than derived. |
| `accountant_noise_multiplier_overstated` | The accountant consumes a larger multiplier than applied. |

Malformed declarations raise `WeightingSensitivityError` with a fixed code
(`invalid_weighting_policy`, `invalid_mechanism_declaration`,
`invalid_accountant_declaration`, `invalid_declaration`,
`invalid_weight_inputs` or `invalid_sensitivity_input`) and at most a field
name. Errors never echo submitted values.

## Compatibility receipt

The receipt binds the SHA-256 digests of the weighting-policy, mechanism and
accountant declarations, the outcome, the ordered reason codes, the
normalization code and the adjacency code. A `receipt_digest` binds every other
field. Each digest uses a domain-separated prefix and canonical sorted-key JSON
with exact numerator and denominator pairs.

The receipt contains no site identities, record or participation counts, band
assignments, per-site weights or numeric policy values. Per-client weight
inputs are checked locally and never enter the receipt, so the same
declarations yield the same receipt for any number of participating clients.
Numeric policy fields are also hidden from `repr`.

## Limits

The guard checks declarations, not behavior. It cannot prove that the
aggregator actually uses the declared denominator, that `W` was fixed before
the round without consulting private counts, or that updates were clipped.
Declared band weights are trusted to be coarse relative factors. Rounds without
differential privacy are out of scope. A compatible receipt is not a privacy
certification or a release authorization.

Run the offline checks with:

```sh
.venv/bin/python -m pytest tests/unit/training/test_federated_weighting_sensitivity.py -q
```
