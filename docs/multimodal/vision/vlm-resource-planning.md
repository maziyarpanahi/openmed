# Edge-VLM Execution Memory Plans

`openmed.multimodal.vision.runtime.resource_planner` estimates the peak memory
of running an edge vision-language model over a declared preprocessing plan,
compares that estimate with the device tiers the caller registers, and selects
the **smallest tier that can hold it**. When the estimate cannot be trusted or
nothing fits, the planner abstains.

A plan is a declaration, not a measurement. It does not load weights, download
models, reserve memory, schedule inference, or contact a service. It holds only
under the documented arithmetic and the policy factors the caller supplies, and
it never proposes a cloud fallback: `CLOUD_FALLBACK_ALLOWED` is `False`, and
every plan reports `cloud_fallback_allowed` as `false`.

## Inputs

`plan_vlm_execution(policy, preprocessing)` takes two frozen declarations:

- `policy`: a `VLMExecutionPolicy` holding the registered device tiers and the
  five factors a preprocessing plan cannot carry.
- `preprocessing`: a `PreprocessingPlan` describing the image geometry and the
  tile assignment.

`VLMExecutionPolicy(tiers, parameter_count=None, weight_bytes_per_parameter=None,
activation_bytes_per_pixel=None, cache_bytes_per_tile=None, output_bytes=None,
runtime_overhead_bytes=0)`. Every factor is either `None` or a bounded integer.
None has a default, and none is guessed: a factor the caller does not supply
makes the run unevaluable.

| Factor | Bound | Used for |
| --- | --- | --- |
| `parameter_count` | 0 to 2^62 | model weights |
| `weight_bytes_per_parameter` | 1 to 64 | model weights |
| `activation_bytes_per_pixel` | 1 to 4096 | frame stack and tile buffer |
| `cache_bytes_per_tile` | 0 to 2^32 | per-tile cache |
| `output_bytes` | 0 to 2^40 | decoded output |
| `runtime_overhead_bytes` | 0 to 2^63 - 1 | fixed runtime overhead |

`PreprocessingPlan(width, height, tile_size, tile_overlap=0, frames=1)` derives
`stride = tile_size - tile_overlap`, `tiles_x = ceil(width / stride)`,
`tiles_y = ceil(height / stride)`, and `tile_count = tiles_x x tiles_y`.

## Device tiers

A tier is a `DeviceTier(name, memory_budget_bytes, reserved_bytes=0)`. The name
is a bounded opaque label matching `[a-z][a-z0-9]*([._-][a-z0-9]+)*`, so an
identifier can be reported in diagnostics without carrying free-form text.
`reserved_bytes` must stay below `memory_budget_bytes`; the difference is the
`available_bytes` a plan may use.

| Tier | Budget | Reserved | Available |
| --- | --- | --- | --- |
| `VLM_TIER_MOBILE_V1` (`mobile.v1`) | 6 GiB | 1 GiB | 5 GiB |
| `VLM_TIER_DESKTOP_V1` (`desktop.v1`) | 32 GiB | 4 GiB | 28 GiB |

Zero, negative, boolean, and fractional byte counts are rejected with
`VLMResourcePlannerError`, as are budgets above `MAX_MEMORY_BYTES`. A policy
holds between 1 and `MAX_DEVICE_TIERS` (64) tiers with unique names.

## Estimation

| Component | Formula |
| --- | --- |
| `model_bytes` | `parameter_count x weight_bytes_per_parameter` |
| `peak_activation_bytes` | `width x height x frames x activation_bytes_per_pixel` + `tile_size x tile_size x frames x activation_bytes_per_pixel` |
| `cache_bytes` | `tile_count x cache_bytes_per_tile` |
| `output_bytes` | `output_bytes` |
| `runtime_overhead_bytes` | `runtime_overhead_bytes` |
| `total_bytes` | the sum of the five components above |

Overlapping tiles are charged the full tile buffer, because the planner bounds
a decoder that materializes each tile independently.

## Outcomes

| Outcome | Condition |
| --- | --- |
| `safe` | every factor is present, nothing saturates, and at least one tier's `available_bytes` is at least `total_bytes` |
| `abstain` | a factor is missing, an estimate saturates, or no tier fits |

Every plan carries a content-free `VLMResourceEstimate` with the components
above, and each declared tier is reported as:
`TierCandidate(name, available_bytes, fits)`.

| Reason code | Meaning | `total_bytes` | Candidates |
| --- | --- | --- | --- |
| `insufficient_metadata` | a policy factor is missing; `field_name` names the first one | `null` | none |
| `estimate_saturated` | a component would pass the saturation ceiling; `field_name` names it | `null` | none |
| `memory_budget_exceeded` | no declared tier fits; every candidate reports `fits=False` | the estimate | all tiers |

A `safe` plan names the selected `tier_name` and its `tier_available_bytes`.
An `abstain` plan leaves both `null`. Selection is deterministic and ignores
declaration order: candidates are ordered by `(available_bytes, name)`, the
smallest fitting tier wins, and ties break on the lexicographically first name.

## Arithmetic

All arithmetic is integer and platform-independent. A component or total that
would exceed `MAX_MEMORY_BYTES`, 2^63 - 1, saturates, and a saturated estimate
always abstains, so saturation never produces a `safe` plan.

## Example

```python
from openmed.multimodal.vision.runtime import (
    VLM_TIER_MOBILE_V1,
    PreprocessingPlan,
    VLMExecutionPolicy,
    plan_vlm_execution,
)

policy = VLMExecutionPolicy(
    tiers=(VLM_TIER_MOBILE_V1,),
    parameter_count=1_000,
    weight_bytes_per_parameter=4,
    activation_bytes_per_pixel=4,
    cache_bytes_per_tile=16,
    output_bytes=256,
    runtime_overhead_bytes=64,
)
plan = plan_vlm_execution(
    policy,
    PreprocessingPlan(width=64, height=32, tile_size=16),
)
plan.outcome.value  # "safe"
plan.estimate.total_bytes  # 13664
plan.tier_name  # "mobile.v1"
```

`plan.to_dict()` and `plan.to_json()` serialize with a fixed key order and
carry only tier names, byte counts, geometry, and reason codes. They never
carry provider names, endpoints, regions, credentials, retry settings, or
media content: the planner has no remote path to report.

## See also

- [Multimodal batch memory estimates](../batch-memory-estimates.md) for the
  decoded-batch budget that runs upstream of inference.
- [Asset manifests](../asset-manifests.md) for the validated inputs a batch
  plan consumes.
