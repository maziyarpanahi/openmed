# Renyi privacy accounting for federated rounds

OpenMed accounts for the privacy cost of a federated training round before the
round is admitted to a federation schedule. The accountant is offline and
deterministic: it consumes sampling, clipping, and noise mechanics only, keeps
no participant identifiers, and imports nothing beyond the Python standard
library.

## Round bound

For a round that retains participants with probability `q` and adds Gaussian
noise with multiplier `z`, the Renyi divergence of order `alpha > 1` between
neighbouring round outputs is bounded by convexity of the Renyi divergence in
the retained/absent mixture:

```text
eps_round(alpha) = log(1 - q + q * exp((alpha - 1) * alpha / (2 * z ** 2)))
                   / (alpha - 1)
```

Three properties make this bound usable in a privacy gate:

- It holds for any real `alpha > 1`, not only for integer orders.
- At `q == 1` it is exact and reduces to `alpha / (2 * z ** 2)`, the Gaussian
  mechanism's RDP.
- The clipping norm cancels out: the additive noise standard deviation is
  `noise_multiplier * clipping_norm`, so the bound depends on the multiplier
  and the sampling rate, not on the clip size.

## Composition and conversion

Renyi divergences are additive, so a schedule's cost is the per-order sum of
its rounds:

```text
eps_total(alpha) = sum(eps_round(alpha) for each round)
```

The composed curve converts to a single `(epsilon, delta)` guarantee by taking
the best order, and an empty schedule reports `epsilon = 0` without applying a
conversion to a vacuous curve:

```text
epsilon = min over alpha of (eps_total(alpha) + log(1 / delta) / (alpha - 1))
```

## Usage

```python
from openmed.training.federated.rdp_accountant import (
    FederatedRenyiRound,
    RenyiFederationPolicy,
    account_federated_rounds,
    evaluate_federation_round,
)

schedule = [
    FederatedRenyiRound(
        round_index=0,
        sampling_rate=0.25,
        clipping_norm=1.0,
        noise_multiplier=1.0,
    )
]

report = account_federated_rounds(schedule, orders=(2, 4), delta=1e-5)
print(report.to_json())

policy = RenyiFederationPolicy(max_epsilon=8.0, max_delta=1e-5)
decision = evaluate_federation_round(policy, schedule, candidate_round)
print(decision.allowed, decision.reason_code)
```

`account_federated_rounds` returns a `RenyiAccountingReport` whose `to_dict()`
follows a fixed field order (`round_count`, `orders`, `epsilons`, `delta`,
`epsilon`) and whose `to_json()` is canonical, so byte-identical inputs always
produce byte-identical reports.

## Admitting a round

`evaluate_federation_round` charges a candidate round on top of the accepted
schedule and returns a `RenyiRoundDecision`. The checks run in a fixed order so
that the reported reason is stable:

| Reason code | Raised when |
| --- | --- |
| `round_limit_exceeded` | The schedule plus the candidate exceeds `max_rounds`. |
| `delta_exceeded` | The requested target delta exceeds the policy's `max_delta`. |
| `epsilon_exceeded` | The projected epsilon exceeds `max_epsilon`. |
| `within_policy` | The candidate may be admitted. |

A blocked round still returns a full report, computed against the policy delta,
so operators can see how far the federation is from its ceiling before the
round runs. The accountant decides admission only; it never runs training,
contacts a coordinator, or mutates the schedule.

## Determinism and privacy boundary

- Rounds are validated and then ordered by `(round_index, sampling_rate,
  clipping_norm, noise_multiplier)`, and duplicate round indices are rejected,
  so floating-point sums do not depend on caller iteration order.
- Reports describe mechanics and counts only: round count, order grid, epsilon
  curve, delta, epsilon. They contain no participant, site, patient, record, or
  update payload.
- `fingerprint_accounting_report` returns a domain-separated
  `sha256:<hex>` digest over the canonical report payload, suitable for audit
  records that must not embed raw accounting inputs.
- Validation errors name the rejected field but never echo the rejected value.

## Limits

| Constant | Value | Purpose |
| --- | --- | --- |
| `DEFAULT_RENYI_ORDERS` | `(2, 4, 8, 16, 32, 64)` | Default order grid. |
| `MAX_RENYI_ORDER` | `4096` | Largest admissible order. |
| `MAX_RENYI_ORDERS` | `64` | Largest admissible order grid. |
| `MAX_ACCOUNTED_ROUNDS` | `1000000` | Largest admissible schedule or round limit. |
| `MAX_CLIPPING_NORM` | `1e12` | Largest admissible clipping norm. |
| `MAX_NOISE_MULTIPLIER` | `1e6` | Largest admissible noise multiplier. |

## See also

- [Federated round lifecycle](../training/federated-round-lifecycle.md)
- [Federated aggregate metrics](../training/federated-metrics.md)
- [Federated update metadata](../training/federated-update-metadata.md)
- [Privacy budget](../security/privacy-budget.md)
