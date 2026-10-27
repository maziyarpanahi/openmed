# Federated distributed noise

With masked secure aggregation the coordinator sees only the sum of client
updates. Noise added centrally after unmasking still lets the coordinator see
the exact aggregate. To keep the aggregate private from an honest-but-curious
coordinator, each client adds a noise share to its encoded update before
masking. The coordinator then only ever sees the noisy modular sum.

`openmed.training.federated_distributed_noise` calibrates those shares so that
client dropout cannot push the aggregate noise below the declared level. It
also samples the shares exactly and declares the calibration in a value-free
mechanism record.

## Trust boundary

| Party | Assumption |
| --- | --- |
| Coordinator | Untrusted: honest but curious. Sees only the modular sum. |
| Clients | Honest and non-colluding. Each adds its own share locally. |
| Secure aggregation | Reveals only the sum modulo `m`. Owned by the masking work. |

Noise lives in the client shares. The record declares this as
`trust_boundary: coordinator_untrusted` and `noise_placement: client_shares`.
If clients collude with the coordinator, the shares those clients know stop
protecting the sum, and the bounds below do not cover that case.

The reported bound uses `zero_one_client_update` adjacency: replace one
client signal with zero while retaining that client's independent noise
share. It conditions on a fixed survivor set and data-independent dropout.
It does not hide participation, cover dropout chosen from private updates, or
cover replacing one arbitrary clipped update with another. The latter can
double the signal sensitivity and needs a separate calibration. These
assumptions are included in the mechanism record.

## Calibrating to the recovery threshold

For `n` enrolled clients and recovery threshold `t`, any unmaskable survivor
set has `s` survivors with `t <= s <= n`. Each share uses the discrete Gaussian
scale

```text
sigma**2 = target_aggregate_variance / t * (1 + 2**-20)
```

The scale is tied to `t`, not `n`. If it were `target / n`, a round that lost
`n - t` clients would fall short of the target. The scale must be at least 1
grid unit, otherwise the reason code is `share_scale_below_minimum`.

For `sigma**2 >= 1`, one share has variance at least
`sigma**2 * (1 - 2**-21)`. This follows from Poisson summation. The variance is
also strictly below `sigma**2` (KLS21 Lemma 15). Variances of independent
shares add exactly, so for every `s >= t`:

```text
Var[sum of s shares] >= s * sigma**2 * (1 - 2**-21) >= target_aggregate_variance
```

`aggregate_variance_lower_bound(s)` returns this bound, and
`aggregate_scale(s)` returns `s * sigma**2`. Both refuse `s < t`.

## Dropout abort

`unmask_noised_aggregate(calibration, survivor_count, unmask)` checks the
survivor count before it calls the injected `unmask` port. Below `t` it raises
`survivors_below_threshold` and the port is never called. A count that is not a
plain integer in `[0, n]` raises `invalid_survivor_count`. Once the quorum is
met, the port's modular sum is checked and mapped to centered representatives
in `[-(m // 2), (m - 1) // 2]`. A port failure becomes `unmask_failed` with no
exception text.

## Rounding, sum closeness and modular clipping

The bounds follow Kairouz, Liu and Steinke, "The Distributed Discrete Gaussian
Mechanism for Federated Learning with Secure Aggregation" (KLS21,
arXiv:2102.06387v4). All quantities are in integer grid units of the fixed-point
codec.

- **Rounding.** Randomized rounding can make the norm larger than the clip
  bound `c`. The sensitivity bound is `Delta_2**2 <= (c + ceil(sqrt(d)))**2`,
  the probability-one branch of Theorem 1, eq. (2). Before drawing any noise,
  `noised_client_share` refuses a signal with squared norm above this bound or
  any coordinate above `ceil(c)` (`signal_out_of_bounds`).
- **Sum of shares.** A sum of discrete Gaussians is not exactly a discrete
  Gaussian. The calibration takes a rational upper bound on
  `tau_n = 10 * sum_{k=1}^{n-1} exp(-2 pi^2 sigma^2 k / (k + 1))` from
  Theorem 1, eq. (3). It then reports
  `zcdp_rho_bound = (Delta_2**2 / (t sigma**2) + 2 tau_n d) / 2` from
  Proposition 14, eq. (19). The bound holds for every `t <= s <= n` because
  `tau_s <= tau_n`. Accountant binding stays with the central mechanism work.
- **Modular clipping.** The sum of at most `n` shares is subgaussian with
  variance proxy `n sigma**2` (Lemma 15). Its noise tail bound is the smallest
  integer `k` with `k**2 >= 2 n sigma**2 ln 2 (b + 1 + ceil(log2 d))`, so every
  coordinate stays within `k` except with probability at most `2**-b`. A
  modulus with `n * ceil(c) + k > (m - 1) // 2` is rejected with
  `modulus_wraparound_risk`. This is a fail-closed form of the wraparound
  condition in Proposition 35 and Theorem 36.

Rational constants are rounded in the conservative direction: a lower bound for
`pi**2` and `log2(e)`, and an upper bound for `ln 2`.

## Exact sampling and randomness

`DiscreteGaussianShareSampler` samples the discrete Gaussian exactly. It
follows Canonne, Kamath and Steinke (CKS20, arXiv:2004.00010): Algorithm 1 for
Bernoulli(exp(-gamma)), Algorithm 2 for the discrete Laplace and Algorithm 3
for the discrete Gaussian. Only `int` and `fractions.Fraction` arithmetic is
used, so there is no floating point in the sampling path. Every rejection loop
has a round bound and raises `sampler_round_limit` when it runs out. It never
returns a clipped or biased draw.

The discrete Gaussian was chosen over Skellam because CKS20 is exact for any
rational scale with a constant expected number of rounds. An exact Poisson
sampler for large means, built from integer arithmetic alone, is either slow or
needs floating point.

Operational construction accepts only an instance whose type is exactly
`random.SystemRandom`. Subclasses, `random.Random` and other generators raise
`randomness_not_operational`. `DiscreteGaussianShareSampler.for_testing(seed)`
builds a seeded sampler for synthetic tests. `noised_client_share` refuses it
(`test_sampler_refused`) unless the caller passes `allow_test_sampler=True`.

## Usage

```python
import random

from openmed.training.federated_distributed_noise import (
    DiscreteGaussianShareSampler,
    calibrate_distributed_noise,
    noised_client_share,
    unmask_noised_aggregate,
)

calibration = calibrate_distributed_noise(
    client_count=5,
    recovery_threshold=3,
    target_aggregate_variance=300,
    dimension=8,
    modulus=2**32,
    clip_l2_norm=16,
    tail_bound_bits=40,
)
sampler = DiscreteGaussianShareSampler(random.SystemRandom())
signals = [[1, -2, 0, 3, 0, 0, 1, -1] for _ in range(5)]
shares = [noised_client_share(calibration, s, sampler) for s in signals]

# Synthetic stand-in for secure aggregation over three survivors.
survivors = shares[:3]
modular_sum = [sum(column) % calibration.modulus for column in zip(*survivors)]
aggregate = unmask_noised_aggregate(calibration, 3, lambda: modular_sum)

assert calibration.aggregate_variance_lower_bound(3) >= 300
record = calibration.mechanism_record().to_dict()
assert record["trust_boundary"] == "coordinator_untrusted"
```

## Mechanism record

`mechanism_record()` returns a `DistributedNoiseMechanismRecord` with a
`sha256:` digest over its canonical JSON. The record contains:

- the trust boundary, threat model, sampler, survivor rule and wraparound rule;
- `n`, `t`, `d`, `m` and the tail bits `b`;
- the target variance, the share scale, the margin and deficit constants, and
  the sensitivity bound, as exact `p/q` strings;
- the signal and noise tail bounds, as integers;
- `tau` and `rho` upper bounds, as decimal strings rounded up;
- citations for each bound.

The record is computed from configuration alone. It holds no noise draws,
seeds, encoded updates, shares, survivor identities or per-client values.
Drawing shares does not change it. Reprs and error messages show only fixed
reason codes, never submitted values.

## Reason codes

| Code | Meaning |
| --- | --- |
| `invalid_client_count` | `n` is not an integer in `[2, 2**20]` |
| `invalid_recovery_threshold` | `t` is not an integer in `[1, n]` |
| `invalid_target_variance` | Target variance is not positive and finite |
| `invalid_dimension` | `d` is not a positive integer |
| `invalid_modulus` | `m` is not an integer in `[3, 2**512]` |
| `invalid_clip_bound` | Clip bound is not positive and finite |
| `invalid_tail_bound` | Tail bits are not an integer in `[20, 256]` |
| `share_scale_below_minimum` | Per-share `sigma**2` would fall below 1 |
| `modulus_wraparound_risk` | `m` cannot hold signal plus noise tail |
| `invalid_survivor_count` | Survivor count is not an integer in `[0, n]` |
| `survivors_below_threshold` | Fewer than `t` survivors; unmasking aborted |
| `invalid_calibration` | Object is not a calibration |
| `invalid_sampler` | Object is not a share sampler |
| `test_sampler_refused` | Seeded sampler used without explicit opt-in |
| `invalid_signal` | Signal is not a list or tuple of `d` integers |
| `signal_out_of_bounds` | Signal exceeds the coordinate or L2 bound |
| `randomness_not_operational` | Generator is not exactly `random.SystemRandom` |
| `invalid_test_seed` | Test seed is not an integer in `[0, 2**64)` |
| `invalid_sampler_scale` | Sampler scale is not positive and finite |
| `sampler_round_limit` | A bounded rejection loop was exhausted |
| `unmask_failed` | The injected unmask port raised |
| `invalid_unmasked_sum` | Unmasked sum has the wrong shape or range |

## Scope

Masking and recovery, the fixed-point codec and central mechanisms with
accountant binding are separate work in
[#2828](https://github.com/maziyarpanahi/openmed/issues/2828),
[#2829](https://github.com/maziyarpanahi/openmed/issues/2829),
[#3689](https://github.com/maziyarpanahi/openmed/issues/3689) and
[#3686](https://github.com/maziyarpanahi/openmed/issues/3686). This module
imports none of them. See also
[federated aggregate metrics](federated-metrics.md) and
[federated round lifecycle](federated-round-lifecycle.md).
