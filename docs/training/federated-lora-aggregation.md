# Federated LoRA aggregation

Averaging LoRA `A` and `B` factors separately does not average the low-rank
updates they encode: `mean(B) @ mean(A)` differs from `mean(B @ A)` whenever
sites diverge. `openmed.training.federated_lora_aggregation` combines paired
factor updates in the product space under a mode the round declares up front.

```python
from openmed.training.federated_lora_aggregation import (
    LoraFactorPair,
    LoraModuleSpec,
    LoraRoundDeclaration,
    LoraUpdate,
    aggregate_lora_updates,
    lora_factor_digest,
)

shared_a = ((1.0, 0.5, -0.25), (0.0, 1.0, 0.75))
declaration = LoraRoundDeclaration(
    mode="frozen_a",
    rank=2,
    alpha=16,
    modules=(
        LoraModuleSpec(
            "encoder.layer.0.attention.query",
            in_features=3,
            out_features=4,
            frozen_a_digest=lora_factor_digest(shared_a),
        ),
    ),
)
updates = [
    LoraUpdate.from_named_factors(
        {
            "encoder.layer.0.attention.query.lora_A.weight": shared_a,
            "encoder.layer.0.attention.query.lora_B.weight": factor_b,
        },
        rank=2,
        alpha=16,
    )
    for factor_b in (
        ((1.0, 0.0), (0.5, -1.0), (0.25, 2.0), (-1.5, 0.5)),
        ((-0.5, 1.0), (1.5, 0.25), (0.0, -0.75), (2.0, 1.0)),
    )
]
aggregate = aggregate_lora_updates(updates, declaration)
print(aggregate.evidence.to_json())
```

## Update descriptor

`LoraUpdate` is a versioned descriptor
(`openmed.training.federated_lora_update.v1`) holding one `LoraFactorPair`
per target module plus the update's rank and alpha. `A` is
`rank x in_features`, `B` is `out_features x rank`, and the effective update is
`(alpha / rank) * B @ A`, matching the adapter training recipe's LoRA rank and
alpha. `from_named_factors()` pairs `<module>.lora_A.weight` and
`<module>.lora_B.weight` entries. Values are normalized to float64, negative
zero becomes zero, and `repr` never shows factor values. `descriptor()`
returns value-free metadata with per-factor digests; `digest` is a content
digest over the canonical bytes.

## Aggregation modes

| Mode | Behavior |
|---|---|
| `frozen_a` | Each module's shared `A` is pinned by `lora_factor_digest()`. Every update's `A` must match the pin; only `B` is averaged. `mean(B) @ A` equals the weighted mean of `B @ A` exactly before one final rounding to float64. |
| `product_space` | The weighted mean of `B @ A` is computed exactly and re-factorized to the declared rank. The result spans the top singular subspace; `A` has orthonormal rows. |
| `reject` | Every batch is refused with `lora_aggregation_rejected`. |

Naive per-factor averaging (`per_factor_mean`, `per_factor`, `naive`,
`factor_mean`, `fedavg_factors`) is refused when the round is declared.

### Heterogeneous ranks

The default `rank_policy="uniform"` requires every update to use the round
rank. A `product_space` round can declare `rank_policy="zero_pad"`: an update
of rank `r_i <= rank` is treated as its factors zero-padded to the round rank
with its `alpha / r_i` scaled delta preserved, so its product is multiplied by
`rank / r_i` before averaging. Alpha must still match exactly.

## Determinism and limits

The whole batch is validated before any arithmetic. Updates are sorted by
content digest, every float64 value is converted to an exact integer ratio,
and weighted sums use exact integer arithmetic. Each aggregate value is
rounded to float64 once with correctly rounded division, so any arrival order
gives byte-identical aggregate bytes and digests.

Re-factorization forms the Gram matrix of the smaller side exactly, rounds it
once, and runs a cyclic Jacobi eigensolver in fixed order. Eigenvectors are
ordered by descending eigenvalue with ties broken by index, and each is signed
so its largest-magnitude component is positive. Bounds: at most 32 updates,
8 modules, 64 input or output features, rank 16, and absolute factor values up
to 2**20. The worst case at these bounds runs in a few seconds of pure Python.
numpy is not required; array-like inputs exposing `tolist()` are accepted.

Weights are injected as positive integers, fractions or finite floats keyed by
`LoraUpdate.digest` and must cover exactly the submitted updates; uniform
weights are the default.

## Residual bands and evidence

The relative Frobenius residual between the aggregate product and the exact
weighted mean is classified exactly into one band: `zero`, `below_1e-12`,
`below_1e-9`, `below_1e-6`, `below_1e-3`, `below_1e-1` or `at_least_1e-1`.
The coarsest module band is recorded. A round can declare `residual_ceiling`
to refuse coarser results with `lora_residual_exceeds_ceiling`.

`LoraAggregationEvidence` records only the schema version, mode, rank, rank
policy, residual band, declaration digest, a digest over the sorted update
digests, and the aggregate digest. It contains no tensor values, weights,
module names, individual update digests or per-site contributions.

## Reason codes

Errors carry exactly one stable code as their message: `lora_aggregation_rejected`,
`lora_declaration_invalid`, `lora_duplicate_update`, `lora_factor_unpaired`,
`lora_frozen_a_mismatch`, `lora_limit_exceeded`, `lora_mode_unsupported`,
`lora_module_mismatch`, `lora_naive_factor_average_rejected`,
`lora_rank_mismatch`, `lora_residual_exceeds_ceiling`, `lora_scaling_mismatch`,
`lora_update_invalid` and `lora_weight_invalid`.

This module owns factor semantics and deterministic arithmetic over plaintext
updates or an already-unmasked sum. Robust strategies, site weighting
policies, masking, fixed-point encoding and local materialization stay
separate.

Run the offline checks with:

```sh
uv run --frozen --extra dev python -m pytest tests/unit/training/test_federated_lora_aggregation.py -q
```
