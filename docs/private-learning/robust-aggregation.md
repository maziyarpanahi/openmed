# Robust adapter aggregation

Naive averaging lets one extreme or malicious client update move the federated
adapter arbitrarily far, which can damage critical clinical recall for every
site. `openmed.training.federated.robust_aggregate` combines compatible dense
adapter updates with deterministic, poisoning-resistant strategies and fails
closed when the declared bound on suspected adversaries cannot be honored.

```python
from openmed.training.federated.robust_aggregate import (
    RobustAdapterUpdate,
    RobustAggregationConfig,
    aggregate_adapter_updates,
)
from openmed.training.federated_update_metadata import (
    FederatedParameterMetadata,
    FederatedUpdateMetadata,
    FederatedUpdatePolicy,
)

# Synthetic digests, shapes and values for this example only.
name = "adapter.lora_B.weight"
policy = FederatedUpdatePolicy(
    model_digest="sha256:" + "a" * 64,
    parameters=(FederatedParameterMetadata(name, (2, 2), "float32"),),
    max_total_elements=4,
)


def update(index, values):
    metadata = FederatedUpdateMetadata(
        model_digest=policy.model_digest,
        adapter_format="dense",
        parameters=policy.parameters,
        total_elements=4,
        update_digest="sha256:" + format(index, "064x"),
        clipped=True,
        policy=policy,
    )
    return RobustAdapterUpdate(metadata=metadata, tensors={name: values})


batch = [
    update(1, [[1.0, -1.0], [0.5, 0.0]]),
    update(2, [[2.0, -2.0], [0.25, 0.5]]),
    update(3, [[3.0, -3.0], [0.75, -0.5]]),
    update(4, [[4.0, -4.0], [1.0, 1.0]]),
    update(5, [[9000.0, 9000.0], [-9000.0, 9000.0]]),  # poisoned
]
config = RobustAggregationConfig(
    strategy="coordinate_trimmed_mean", max_suspected_adversaries=1
)
result = aggregate_adapter_updates(batch, policy=policy, config=config)
assert result.tensor(name) == ((3.0, -2.0), (0.5, 0.5))
print(result.diagnostics.to_json())
```

The plain mean of the same batch would be `1802.0` in the first coordinate;
the trimmed mean stays inside the range of the four honest values.

## Strategies

| Strategy | Behavior | Requirement |
| --- | --- | --- |
| `coordinate_trimmed_mean` | For every coordinate, sort the `n` values, drop the `f` smallest and `f` largest, and average the rest. | `n >= 2f + 1` |
| `norm_clipped_mean` | Scale every update whose whole-update L2 norm exceeds the lower median norm down to that median, then average all `n` updates. | `n >= 2f + 1` |

`f` is `max_suspected_adversaries`, the coordinator's declared bound on the
number of adversarial updates in one batch. A batch smaller than `2f + 1` is
refused with `robust_adversary_bound_exceeded`; the call never falls back to a
weaker aggregate. `f = 0` reduces the trimmed mean to the plain mean and is
intended only for trusted federations.

The coordinate-wise trimmed mean follows Yin, Chen, Ramchandran and Bartlett,
"Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates"
(ICML 2018). With at most `f` adversarial updates, every kept value lies
between two honest values, so each aggregate coordinate stays inside the honest
range. The final rounding is clamped to the kept values, so this holds exactly
in floating point.

The norm-clipped mean is the norm-bounding defense of Sun, Kairouz, Suresh and
McMahan, "Can You Really Backdoor Federated Learning?" (arXiv:1911.07963,
2019), with the bound taken from the submitted norms. With `n >= 2f + 1` the
lower median is bracketed by honest norms, so a boosted update contributes at
most `median / n` in L2 norm. It keeps every update, which helps when honest
sites are heterogeneous, but unlike the trimmed mean it does not confine each
coordinate to the honest range.

Krum and multi-Krum (Blanchard, El Mhamdi, Guerraoui and Stainer, "Machine
Learning with Adversaries: Byzantine Tolerant Gradient Descent", NeurIPS 2017)
are not provided. They require `n > 2f + 2` and pairwise distances that are
quadratic in the number of updates.

## Compatibility

Each `RobustAdapterUpdate` pairs `FederatedUpdateMetadata` (see
[federated update metadata](../training/federated-update-metadata.md)) with its
tensor values. Updates are compatible only when the metadata matches the
trusted `FederatedUpdatePolicy` passed to `aggregate_adapter_updates()`:

| Check | Reason code |
| --- | --- |
| Base-model digest and adapter format match the policy | `robust_adapter_mismatch` |
| Parameter names, shapes and dtypes equal the policy allowlist | `robust_parameter_mismatch` |
| Tensor names equal the metadata parameter names | `robust_parameter_mismatch` |
| Nested values match each declared shape exactly | `robust_shape_mismatch` |
| Every value is exactly representable in the declared dtype | `robust_dtype_mismatch` |
| A policy that requires clipping receives only clipped declarations | `robust_update_invalid` |
| No update repeats a declared update digest or identical content | `robust_duplicate_update` |

Values may be nested lists or tuples of numbers, or array-like objects that
expose `tolist()`; numpy and torch are never imported. NaN and infinity are
rejected with `robust_non_finite_value`; magnitudes above `1e12` with
`robust_value_out_of_range`; booleans, strings and other types with
`robust_update_invalid`. At most 1,024 updates, `2**24` values per update and
`2**26` values per batch are accepted (`robust_limit_exceeded`).

The whole batch is validated before any arithmetic, and any failure refuses the
whole batch. Do not build the policy or the configuration from incoming
updates: a submitter could then choose a layout or bound that suits itself.

For LoRA factors, robust statistics over the separate `A` and `B` factors have
the same flaw as naive per-factor averaging. Aggregate the materialized dense
delta, or only `B` when the round pins a shared frozen `A`. Low-rank
product-space aggregation, tracked in
[#3847](https://github.com/maziyarpanahi/openmed/issues/3847), owns factor
semantics and leaves robust strategies to this module.

## Determinism

Values are normalized to float64, and negative zero becomes zero. Sums use
`math.fsum`, which is correctly rounded for the multiset of addends, and
every per-coordinate step depends only on the sorted values. Any arrival order
of the same updates therefore yields byte-identical aggregates, diagnostics and
digests. The module is stdlib-only and performs no file access, logging or
network calls.

## Diagnostics and privacy boundary

`RobustAggregationDiagnostics` contains only:

| Field | Meaning |
| --- | --- |
| `schema_version` | `openmed.training.federated.robust_aggregate.v1` |
| `strategy`, `strategy_version` | Strategy identifier and its version |
| `max_suspected_adversaries` | Declared bound `f` |
| `update_count`, `parameter_count`, `element_count` | Batch size and layout counts |
| `trimmed_per_side`, `trimmed_fraction` | Values dropped per side and `2f / n`, rounded to four decimals |
| `clipped_update_count` | How many updates were scaled, never which |
| `norm_quantile_bands` | Decade bands (`zero`, `below_1e-6`, `1e-6_to_1e-5`, ..., `at_least_1e6`) of the 25th, 50th and 75th percentile update norms |
| `config_digest`, `inputs_digest`, `aggregate_digest` | SHA-256 references of the policy layout and config, the sorted update content digests, and the aggregate values |

Norm percentiles use nearest rank, clamped to the window between the `f`-th
smallest and the `f`-th largest norm, so up to `f` suspected adversaries cannot
set a band alone and, for `f >= 1`, the minimum and maximum norms are never
reported. Diagnostics contain no tensor values, raw norms, client or site
identifiers, individual update digests, or any record of which update was
trimmed or clipped. `repr` of updates and aggregates shows only counts.

Errors are `RobustAggregationError` instances whose message is exactly one
stable reason code: `robust_adapter_mismatch`,
`robust_adversary_bound_exceeded`, `robust_config_invalid`,
`robust_dtype_mismatch`, `robust_duplicate_update`, `robust_limit_exceeded`,
`robust_non_finite_value`, `robust_parameter_mismatch`,
`robust_shape_mismatch`, `robust_strategy_unsupported`,
`robust_update_invalid` or `robust_value_out_of_range`. Submitted values,
names and digests are never echoed, and exceptions raised by `tolist()` are
suppressed.

Robust aggregation bounds the influence of a limited number of poisoned
updates. It is not a privacy mechanism, does not verify tensor provenance,
signatures or clipping claims, and is not a compliance certification or a
guarantee about clinical decisions. Choose `f` from the federation's threat
model, not from received updates.

Run the offline checks with:

```sh
.venv/bin/python -m pytest tests/unit/training/federated/test_robust_aggregate.py -q
```

See also [federated aggregate metrics](../training/federated-metrics.md) and
[federated round lifecycle](../training/federated-round-lifecycle.md).
