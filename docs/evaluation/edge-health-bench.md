# EdgeHealthBench device profiles

EdgeHealthBench measures an actually executed local workload against an immutable `DeviceProfile`. It keeps latency, throughput, process memory, allocator memory, thermal observations, an energy proxy, output parity, quality, utility, abstention and privacy separate. Critical or document-any-PHI leakage fails the profile even when it occurs only during a warmup.

## Tier and measurement boundaries

| Declared tier | Required local adapter | Evidence boundary |
| --- | --- | --- |
| CPU | A bounded CPU workload | Process RSS is not model-exclusive memory |
| Browser | A host bridge to the actual browser workload | Include bridge cost; supply browser memory/thermal observations explicitly |
| MLX | An installed, pinned native MLX workload | Separate allocator budget and readings are required |
| Core ML | An installed, pinned native Core ML workload | Separate allocator budget and readings are required |
| Optional accelerator | A caller-supplied local backend | Separate allocator budget and readings are required |

The generic harness does not install providers, download weights or substitute CPU execution for an unavailable tier. `workload=None` returns `not_run`, never `passed`. A tier label and host-supplied readings are declarations, not remote attestation. No browser or accelerator qualification is implied by the included CPU example.

A `DeviceWorkload` supplies immutable provenance, dataset/workload digests, its tier and a local-only declaration. `run()` executes the actual workload and returns a `DeviceTrial` containing aggregate counts and the actual output digest. Host adapters must derive counts from executed outputs, as `SyntheticCPUWorkload` does through [OpenPHI-Vision](openphi-vision.md). A declared score without execution is not benchmark evidence.

## Run the CPU development profile

Use a source checkout with `uv sync --extra dev`. These intentionally generous example budgets exercise the harness; they are not product or clinical release thresholds. The reference output below comes from a separate execution of the same known-layout control, so parity establishes repeatability only.

```python
from openmed.eval.suites.edge_health import (
    DeviceProfile,
    DeviceTier,
    SyntheticCPUWorkload,
    run_device_profile,
)
from openmed.eval.suites.openphi_vision import (
    VisionGatePolicy,
    synthetic_vision_cases,
)

cases = synthetic_vision_cases(cases_per_track=3)
policy = VisionGatePolicy(
    minimum_cases_per_track=3,
    minimum_utility_rate=1.0,
    minimum_quality_rate=1.0,
    maximum_abstention_rate=0.0,
)
workload = SyntheticCPUWorkload(cases, policy)
reference = workload.run()
profile = DeviceProfile(
    tier=DeviceTier.CPU,
    provenance=workload.provenance,
    dataset_digest=workload.dataset_digest,
    workload_digest=workload.workload_digest,
    reference_output_digest=reference.output_digest,
    units_per_trial=len(cases),
    trials=3,
    warmups=1,
    maximum_latency_ms=60_000,
    minimum_throughput=0.1,
    maximum_rss_bytes=8 * 1024**3,
    maximum_thermal_level=1,
    require_thermal=False,
    minimum_quality_rate=1.0,
    minimum_utility_rate=1.0,
    maximum_abstention_rate=0.0,
)
report = run_device_profile(profile, workload)
assert report["passed"]
assert report["measured_trials"] == 3
assert not report["thermal_measured"]
assert not report["energy_measured"]
assert not report["clinical_readiness"]
```

The default resource observer reads process-lifetime RSS high water on macOS/Linux. It does not reset process-global tracing. It has no thermal or accelerator sensor, so those values remain missing. Setting `require_thermal=False` permits this limited smoke profile; it does not qualify sustained thermal behavior.

## Bind a real device profile

Pin weight revision/file manifest, preprocessing configuration, dataset, workload/harness, hardware, runtime, and model/dataset license snapshots before execution. Retain the source manifests privately and pass their hashes in `ReleaseProvenance`. Changing a license snapshot changes provenance without adding a licensing allowlist. Independently written adapter code keeps its own license; users are responsible for the weights they choose.

Supply `observe()` returning fresh `ResourceSample` values on the harness's monotonic clock. The source digest must identify the actual sensor implementation and configuration. RSS scope is `process_current` or `process_lifetime_highwater`. Optional device memory scope is `allocator_current` or `allocator_peak`; it is not total physical GPU memory. MLX, Core ML and accelerator profiles require `maximum_device_memory_bytes` and non-missing allocator evidence. A browser GPU workload should also declare this budget. Thermal levels use the host's documented mapping from nominal (0) through critical (3).

Stale samples, changed sensor source/scope, missing required evidence, wrong execution tier and changed artifact bindings fail the profile. An unsafe pre-sample prevents execution; an unsafe post-sample prevents success. Before/after readings are not continuous peak coverage, especially for `process_current` and `allocator_current`. A blocked callback cannot be preempted by the harness; use bounded adapters and the relevant backend lifecycle controls.

For parity across devices, pin a separately qualified reference output digest from the same corpus and semantic output representation. The harness reports reference parity separately from within-run repeat parity. Exact digest equality is deliberately strict; it does not claim numeric or semantic equivalence for different outputs.

## Interpret metrics

Warmups participate in privacy, quality, utility, abstention and parity checks. Only the requested warm trials contribute latency and throughput statistics. The harness measures wall time with `perf_counter`, process CPU time with `process_time`, and observes resource values before and after each run.

Continuous metrics include sample count, mean, nearest-rank p95, maximum and a deterministic 512-resample descriptive bootstrap interval for the mean. Small repeated samples do not establish population-level or sustained performance. Ordinal thermal summaries are descriptive levels, not degrees Celsius. The process CPU-seconds proxy is not energy in joules, whole-system power or GPU energy.

Quality/privacy counts and descriptive Wilson intervals remain per trial. Repeated executions of the same corpus are explicitly **not independent quality samples** and are never pooled to narrow their intervals. Failed units and an entirely unassessed trial fail independently. Every report binds the profile, provenance and harness; it carries `requires_review=True`, `clinical_readiness=False` and `independent_validation=False` even when all declared gates pass.
