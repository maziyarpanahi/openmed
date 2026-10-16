# Exact differential-privacy noise samplers

The original aggregate helpers in `openmed.risk` sample Laplace noise by
inverse CDF on a 53-bit float and Gaussian noise with floating-point `gauss`.
Mironov, "On Significance of the Least Significant Bits for Differential
Privacy" (CCS 2012), shows that this construction can leak: the set of
reachable floating-point outputs depends on the input, so an observer can
sometimes tell neighboring inputs apart regardless of the declared epsilon.

`openmed.risk.discrete_sampling` adds exact samplers that use only `int` and
`fractions.Fraction` arithmetic, and `openmed.risk` adds release helpers built
on them. The float helpers remain available and unchanged; prefer the helpers
below for new releases.

## Samplers

| Function | Distribution | Source |
| --- | --- | --- |
| `sample_discrete_laplace(scale)` | `P(Y = y) = tanh(1 / (2 * scale)) * exp(-abs(y) / scale)` | Canonne, Kamath and Steinke 2020, Algorithm 2 |
| `sample_discrete_gaussian(sigma_squared)` | `P(Y = y)` proportional to `exp(-y**2 / (2 * sigma_squared))` | Canonne, Kamath and Steinke 2020, Algorithm 3 |

Both use Bernoulli(exp(-gamma)) trials from Algorithm 1 of the same paper
([arXiv:2004.00010](https://arxiv.org/abs/2004.00010)). Parameters may be
`int`, `Fraction`, or finite `float`. A float is converted exactly through
`Fraction(value)`, so no rounding is introduced; pass a `Fraction` to use an
exact rational such as `Fraction(1, 3)`. Booleans, strings, zero, negative
and non-finite values are rejected.

The rejection loops take a constant expected number of rounds. Each loop is
bounded; reaching a bound raises `DiscreteSamplerError` with a fixed message
and releases nothing. This happens with probability far below `2**-1000` per
draw, so it does not measurably change the output distribution.

## Integer counts and histograms

```python
from openmed.risk import (
    PrivacyBudget,
    release_discrete_count,
    release_discrete_histogram,
)

budget = PrivacyBudget(epsilon=1.0, delta=1e-6)
count = release_discrete_count(120, epsilon=0.25, budget=budget)
bins = release_discrete_histogram(
    {"bin_a": 40, "bin_b": 12},
    epsilon=0.25,
    delta=1e-7,
    mechanism="gaussian",
    budget=budget,
)
```

Inputs and outputs are Python integers; floats and booleans are rejected.
Outputs may be negative, and clamping is left to the caller as
post-processing.

- **Laplace**: noise has scale `sensitivity / epsilon`. For an integer query
  whose L1 change between neighbors is at most `sensitivity`, the release is
  pure `epsilon`-DP (Canonne, Kamath and Steinke 2020, Lemma 29). The charged
  epsilon is the float you pass, and the scale is computed from its exact
  binary value, so the guarantee matches the charge exactly.
- **Gaussian**: `discrete_gaussian_sigma_squared(sensitivity, epsilon, delta)`
  returns an exact `Fraction`. The discrete Gaussian gives
  `rho = sensitivity**2 / (2 * sigma_squared)`-zCDP (Theorems 4 and 14 of the
  same paper, with `sensitivity` as the L2 bound for histograms), and
  `rho`-zCDP implies `(rho + 2 * sqrt(rho * log(1 / delta)), delta)`-DP
  (Bun and Steinke 2016, Proposition 1.3). The helper solves for `rho`, applies
  a `2**-30` relative safety margin, rounds the variance up to a multiple of
  `2**-32`, and re-checks the conversion. This calibration is conservative, so
  it adds more noise than the analytic continuous calibration in
  `gaussian_scale` for the same budget. `delta` must be greater than zero.

A histogram is one vector query and consumes one budget charge.

`openmed.risk.release_discrete_aggregate` provides the same discrete Laplace
path for an `AggregateDPBudgetLedger`, accepting one integer or a mapping of
names to integers and rejecting row-shaped input.

## Snapped real-valued releases

`release_snapped` releases a real value on a declared power-of-two grid:

```python
from openmed.risk import PrivacyBudget, release_snapped, snapped_epsilon

budget = PrivacyBudget(epsilon=1.0)
charge = snapped_epsilon(0.5, sensitivity=0.75, grid=0.5)  # about 0.667
value = release_snapped(
    12.3,
    epsilon=0.5,
    sensitivity=0.75,
    bound=100.0,
    grid=0.5,
    budget=budget,
)
```

With requested epsilon `eps`, L1 sensitivity `Delta`, grid `G` and bound `B`,
the helper:

1. clamps the value to `[-B, B]`;
2. rounds it half up to the nearest multiple of `G`, giving the integer
   `m = floor(value / G + 1/2)`, using exact rational arithmetic;
3. adds exact discrete Laplace noise with scale `Delta / (eps * G)` in grid
   units;
4. clamps the result to `[-B / G, B / G]` and multiplies by `G`.

Every output is an integer multiple of `G` inside `[-B, B]`.

**Epsilon adjustment.** Rounding can move neighboring inputs apart by up to
`ceil(Delta / G)` grid steps, while the noise scale stays tied to `Delta`.
The release is therefore pure `eps_adj`-DP with

```text
eps_adj = ceil(Delta / G) * G * eps / Delta
```

and the budget is charged the smallest float that is not below `eps_adj`.
`snapped_epsilon` returns that charge. `eps_adj` equals `eps` when `Delta / G`
is an integer and is always below `eps * (1 + G / Delta)`. Proof sketch: the
clamp is 1-Lipschitz; `floor(x + 1/2)` maps reals at distance at most
`Delta / G` to integers at distance at most `ceil(Delta / G)`; discrete
Laplace noise with scale `t` on an integer query with sensitivity `k` is
pure `k / t`-DP (Canonne, Kamath and Steinke 2020, Lemma 29); the final clamp
and scaling are post-processing.

**Preconditions.** `Delta` must bound the change of the exact real query value
between neighbors. `G` must be a power of two within the normal float range.
`B` must be positive, an integer multiple of `G`, and at most `2**52` grid
steps (`B / G <= 2**52`), so every output is exactly representable as a float.
Invalid grids and bounds are rejected before the budget is checked.

**Relation to Mironov's snapping mechanism.** Mironov's mechanism clamps, adds
floating-point Laplace noise, rounds to the power-of-two grid at least as large
as the noise scale, and clamps again. Its epsilon adjustment relies on
assumptions about the floating-point uniform sampler and logarithm that
Python's standard library does not document. This helper instead rounds before
adding exact integer noise, so no floating-point sampler or correctly rounded
math library is assumed. The cost is the `eps_adj` charge above, which favors
grids that are fine relative to `Delta`.

## Randomness and test mode

All discrete helpers take an optional `rng` that must be a `DPRandomSource`.
`DPRandomSource()` takes no arguments and draws from the operating system
through `secrets.randbelow`; this is the default when `rng` is omitted.
Seeded draws exist only through the explicitly named
`DPRandomSource.for_testing(seed)`, whose `test_mode` property is `True`.
Passing `random.Random`, `random.SystemRandom`, or any other object raises
`TypeError`. Seeded output is predictable and provides no privacy guarantee;
use it only with synthetic data. `release_discrete_aggregate` marks results
from a test-mode source `test_only: true`.

## Spend records

`PrivacySpend` and `DPBudgetSpend` now record a `sampler`:

| Sampler | Produced by |
| --- | --- |
| `float_laplace` | Float Laplace helpers, including `release_aggregate` |
| `float_gaussian` | Float Gaussian helpers |
| `discrete_laplace` | `release_discrete_count`, `release_discrete_histogram`, `release_discrete_aggregate` |
| `discrete_gaussian` | Discrete Gaussian count and histogram releases |
| `grid_discrete_laplace` | `release_snapped` |
| `unspecified` | `PrivacyBudget.spend` called directly |

`PrivacySpend.to_dict()` always includes `sampler`, and reports
`calibration_method` as `zcdp_discrete_gaussian` or `grid_snapped_laplace` for
the new Gaussian and snapped paths. `DPBudgetSpend` includes `sampler` only
when a release helper set it, so ledgers built from direct `spend()` calls keep
their previous records and digests.

Records, errors and logs never contain the source aggregate, the noise draw,
the released value, or a seed. The budget is checked before any noise is
drawn, so a rejected query draws nothing. The `PrivacyBudget` helpers charge
only after sampling succeeds. `release_discrete_aggregate`, like
`release_aggregate`, commits its ledger spend before drawing noise, so a
sampler failure there is still counted, which is conservative.

## Limits

- Sampler running time depends on the drawn value. Timing side channels are
  out of scope.
- Sensitivity, clipping and the adjacency relation remain the caller's
  responsibility.
- The legacy float helpers are unchanged and remain exposed to the
  floating-point attack; their spend records say so through their sampler.
- A noisy aggregate is not a clinical decision, compliance certification, or
  guarantee of zero disclosure risk.
