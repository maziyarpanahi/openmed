# Device Capability Resolution

Resolver contract version 1 (OpenMed v4.0). OpenMedKit for Swift, OpenMedKit
for Android, the `openmed` web package, and the Python package choose a model
artifact the same way. Each kit keeps its own native probe and maps the result
into one normalized device-capability profile. One resolver then combines that
profile with a verified [signed portable bundle](../models/portable-bundle.md)
and the caller's constraints. The output is an ordered execution plan, or an
explicit `unsupported` result with a reason code.

Resolution is local, deterministic, and offline. It opens no network
connection, loads no model, and runs no inference. A plan names only
`(runtime, provider)` steps that the signed bundle declares. It never names a
cloud service, a remote endpoint, or a model the bundle does not contain.

| Surface | Profile | Resolver |
| --- | --- | --- |
| Python | `openmed.core.device_capabilities.DeviceCapabilityProfile` | `openmed.core.execution_plan.resolve_execution_plan` |
| Swift | `DeviceCapabilityProfile.apple(...)`, `.currentApple(...)` | `ExecutionPlanResolver.resolve(...)` |
| Kotlin | `AndroidCapabilityProbe.profile(...)`, `.current(...)` | `ExecutionPlanResolver.resolve(...)` |
| TypeScript | `webDeviceCapabilityProfile(...)`, `probeWebDeviceCapabilities(...)` | `resolveExecutionPlan(...)` |

The JSON Schemas ship in `openmed/core/schemas/json/`:
`device_capability_profile`, `execution_constraints`, `execution_extension`,
and `execution_plan`.

## Normalized profile

A profile is a closed JSON document. Every member is required, unknown members
fail with `profile_invalid`, and lists are sets (sorted, without duplicates),
so equivalent profiles always encode to the same canonical bytes.

| Field | Values | Unknown behavior |
| --- | --- | --- |
| `platform` | `android`, `ios`, `linux`, `macos`, `visionos`, `watchos`, `web`, `windows` | Required |
| `sdk_name`, `sdk_version` | Reading SDK, matched against the bundle's `compatibility.sdks` | Required |
| `os_version` | `MAJOR` or `MAJOR.MINOR`; Android uses the API level | `null`. A runtime with an OS range is then refused (`os_version_unknown`) |
| `cpu_architecture` | `arm64`, `armv7`, `wasm32`, `x86_64`, `unknown` | `unknown` |
| `providers` | Execution providers available on the device (see below) | Absent means unavailable |
| `accelerators` | `cpu`, `gpu`, `npu`, derived from `providers` | Derived, never supplied |
| `runtimes` | Bundle formats the SDK can load on this platform | Absent means unsupported |
| `precisions` | `fp32`, `fp16`, `bf16`, `int8`, `int4`, `int2` | Absent means unsupported |
| `features` | `cpu_neon`, `cpu_avx2`, `cpu_fp16`, `wasm_simd`, `wasm_threads`, `webgpu_shader_f16` | Absent means unavailable |
| `operator_sets` | Highest operator-set version per format, such as ONNX opset | Missing format: a runtime that needs one is refused (`operator_set_unknown`) |
| `memory_budget_bytes` | Memory the runtime may use, on the budget ladder | `null`: not enforced, reported as `memory_budget_unknown` |
| `storage_budget_bytes` | Storage the model cache may use, on the budget ladder | `null`: not enforced, reported as `storage_budget_unknown` |
| `max_sequence_length` | Longest token sequence the platform accepts | `null`: not enforced, reported as `device_sequence_limit_unknown` |
| `max_model_tier` | `nano`, `tiny`, `base` | `null`: no tier ceiling, reported as `model_tier_limit_unknown` |
| `network` | `offline`, `online`, `unknown` | Informational only. It never changes a plan and is excluded from the profile digest |

The rule for unknown values: a missing capability is unavailable, and a
requirement checked against an unknown capability fails closed. A missing
budget or limit is not enforced, but the plan lists it under `assumptions` so
the caller can refuse the plan.

**Budget ladder.** Budgets are rounded down to `0` or a power of two from
16 MiB to 1 TiB. For example, Apple's 150 MB watchOS ceiling becomes 128 MiB,
and 3 GB of browser memory becomes 2 GiB. Rounding down is conservative and
keeps exact memory sizes out of profiles.

**Privacy.** A profile has no place for a device name, hardware model, serial
number, vendor or adapter string, user agent, or build fingerprint. Adapters
never read those values. The golden vectors plant such values in the probe
inputs, and every kit asserts that none of them reaches a profile or a plan.

### Providers and accelerators

| Provider | Accelerator | Platforms | Formats |
| --- | --- | --- | --- |
| `cpu` | cpu | android, ios, macos, watchos, visionos, linux, windows | onnx, ort, coreml (CPU-only compute units), mlx, pytorch |
| `wasm` | cpu | web | onnx, transformersjs |
| `webgpu` | gpu | web | onnx, transformersjs |
| `metal` | gpu | ios, macos | mlx |
| `coreml` | npu | ios, macos, watchos, visionos | coreml, onnx, ort (Core ML may dispatch to the Neural Engine, GPU, or CPU) |
| `nnapi` | npu | android | onnx, ort |
| `qnn` | npu | android | onnx, ort |

## Kit probes

Each adapter is a pure mapping from the kit's existing probe. Unit tests drive
it with fake probes, and the Python reference normalizers
(`profile_from_web_probe`, `profile_from_android_probe`,
`profile_from_apple_probe`) define the expected output for the shared vectors.

| Kit | Probe | Normalized as |
| --- | --- | --- |
| Web | `detectOrtWebCapabilities` / `probeOrtWebCapabilities` (`runtime/capability.ts`) | `webgpu` adds the `webgpu` provider and `fp16`; `wasm` adds the `wasm` provider; WebAssembly SIMD adds `wasm_simd`; SIMD plus `SharedArrayBuffer`, cross-origin isolation, and at least two hardware threads adds `wasm_threads`, the same rule as `selectOrtWebBackend`; any provider adds `fp32`, `int8`, and `int4`; the adapter's `shader-f16` feature bit adds `webgpu_shader_f16`; `navigator.onLine` sets `network`. The architecture is `wasm32` and there is no OS version or tier ceiling |
| Web | Caller options | `sdkVersion`, `runtimes` (default `onnx`), `memoryBudgetBytes`, `storageFreeBytes`, `maxSequenceLength`, `operatorSets` |
| Android | `AcceleratorSession` provider discovery | `CPU` always becomes `cpu`, `QNN` becomes `qnn`, and `NNAPI` becomes `nnapi` only from API 27, as in the session |
| Android | `Build.VERSION.SDK_INT`, `Build.SUPPORTED_ABIS[0]` | The API level is `os_version`. `arm64-v8a` becomes `arm64` and adds `cpu_neon`; `armeabi-v7a` becomes `armv7`; `x86_64` stays `x86_64`; anything else is `unknown` |
| Android | Caller inputs | `memoryClassMb` (from `ActivityManager.getMemoryClass()`) becomes the memory budget. `deviceTier` maps `ENTRY` to `nano`, `MID_RANGE` to `tiny`, `FLAGSHIP` to `base`, and `DEVICE_FARM_STUB` to no ceiling. Runtimes are `onnx` and `ort`; precisions are `fp32`, `fp16`, `int8`, and `int4` |
| Apple | `PlatformModelConfiguration` | `maximumResidentMemoryMB` becomes the memory budget, `maximumSequenceLength` the sequence limit, and `maximumTier` the tier ceiling. `requiresINT8` (watchOS, visionOS) limits precisions to `int8` |
| Apple | Compile-time platform, `ProcessInfo` OS version, CPU architecture, MLX linkage | `coreml` and `cpu` are always available. `metal`, the `mlx` runtime, `bf16`, and `int2` are added only on iOS and macOS when MLX is linked and not on the Simulator. `arm64` adds `cpu_neon` |

## Execution extension

Provider choice, required features, and qualification are part of the signed
manifest. They live in the `org.openmed.execution` extension, keyed by runtime
id:

```json
{
  "runtimes": {
    "onnx-webgpu-fp16": {
      "evidence": ["webgpu_parity"],
      "features": ["webgpu_shader_f16"],
      "max_sequence_length": 512,
      "min_operator_set": 17,
      "providers": ["webgpu"],
      "tier": null
    }
  },
  "schema_version": 1
}
```

- `providers`: ordered attempts for this artifact. Each provider must be able
  to execute the runtime format.
- `features`: profile features the runtime needs.
- `min_operator_set`: lowest operator-set version the runtime needs.
- `max_sequence_length`: longest sequence the artifact accepts.
- `tier`: model tier, checked against `max_model_tier`.
- `evidence`: evidence kinds from the manifest `evidence` list that qualify
  this runtime.

A runtime the extension does not describe gets its format's baseline provider
for the profile's platform: `wasm` for ONNX on the web, `cpu` for ONNX and ORT
elsewhere, `coreml` for Core ML, `metal` for MLX, `wasm` for Transformers.js,
and `cpu` for PyTorch. It declares no features, sequence limit, tier, or
evidence. A default never adds an accelerator or a fallback step.

A malformed extension makes the whole bundle `execution_metadata_invalid`.
That covers a missing member, an undeclared runtime id, an unknown provider
such as a remote service, a provider that cannot run the format, or evidence
the manifest does not contain. Bundles may list the extension in
`critical_extensions`, and the resolver accepts that. Pass
`org.openmed.execution` in the verifier's understood extensions as well.

## Caller constraints

| Member | Default | Effect |
| --- | --- | --- |
| `local_only` | `true` | Must be `true`. `false` returns `remote_execution_unsupported`, because there is no remote step to fall back to |
| `require_verified_files` | `true` | The verification must cover the file tree, not only the manifest and signatures |
| `allowed_precisions` | `null` (any) | Steps must use one of these precisions |
| `disallowed_providers` | `[]` | Steps may not use these providers |
| `max_memory_bytes` | `null` | Memory ceiling, combined with the profile budget (the lower one wins) |
| `min_sequence_length` | `null` | Every step must support at least this many tokens |
| `energy` | `balanced` | `low_power` excludes GPU providers (`webgpu`, `metal`) |
| `required_evidence` | `[]` | Every step's runtime must declare these evidence kinds |
| `revoked_artifacts` | `[]` | `sha256:` file digests no step may use |
| `max_steps` | `4` | Upper bound on plan steps, from 1 to 16 |
| `deadline_ms` | `null` | Recorded in the plan for the executor to enforce across steps. The resolver does not benchmark |

## Fallback rules

1. The caller verifies the bundle. The plan uses only runtimes in the
   verification's `eligible_runtimes`, and only a verification with `ok` true
   for these exact manifest bytes.
2. A *root* runtime is one that no other runtime names as its `fallback`.
   Roots are tried in manifest order.
3. A root's chain is the root followed by its declared `fallback` links. The
   plan follows the first chain that contains at least one qualified step, and
   only that chain. Runtimes on other chains are reported as
   `not_declared_fallback`.
4. Within one runtime, steps follow the extension's provider order. A later
   provider is a `provider_fallback` step, and the next runtime in the chain
   is an `artifact_fallback` step.
5. Steps beyond `max_steps` are reported as `fallback_bound_reached`.

In practice:

- **WebGPU to WASM** happens only when the WebGPU runtime declares the WASM
  runtime as its `fallback`. A bundle without that link has no WASM step.
- **Android accelerator to CPU** happens only when the runtime lists `cpu`
  after `qnn` or `nnapi`. A runtime that lists only `nnapi` is unsupported on a
  device without NNAPI.
- **Apple tier fallback** (base to tiny to nano) follows declared `fallback`
  links. A runtime that is too large for the tier ceiling or memory budget is
  skipped, and the plan moves down the declared chain.

## Reason codes

Checks run in this order. The plan-level `reason` is the first failure. Each
skipped runtime or pair reports its own first failure.

| Scope | Codes |
| --- | --- |
| Plan | `remote_execution_unsupported`, `bundle_revoked`, `bundle_unverified`, `bundle_files_unverified`, `sdk_incompatible`, `execution_metadata_invalid`, `no_eligible_runtime` |
| Runtime | `runtime_not_eligible`, `artifact_revoked`, `format_unsupported`, `platform_unsupported`, `os_version_unknown`, `os_version_unsupported`, `precision_unsupported`, `precision_disallowed`, `memory_insufficient`, `storage_insufficient`, `model_tier_exceeded`, `sequence_limit_exceeded`, `sequence_limit_undeclared`, `operator_set_unknown`, `operator_set_unsupported`, `feature_unavailable`, `evidence_missing` |
| Provider | `provider_unavailable`, `provider_disallowed`, `energy_policy_excluded` |
| Unused step | `not_declared_fallback`, `fallback_bound_reached` |

Malformed caller inputs raise errors rather than returning a plan:
`profile_invalid`, `probe_invalid`, and `constraints_invalid`.

## Plan document

A plan holds `status` (`planned` or `unsupported`), `reason`, `root_runtime`,
`steps`, `skipped`, `assumptions`, `bounds` (`max_steps`, `deadline_ms`, and
the effective `max_memory_bytes`), `local_only: true`, and the invalidation
digests. Each step names `runtime_id`, `runtime_index`, `format`, `precision`,
`provider`, `accelerator`, `artifact_sha256` (the entrypoint file digest),
`min_memory_bytes`, `max_sequence_length`, `tier`, `evidence`, and `role`. A
plan contains no file paths, device identifiers, or model inputs.

## Invalidation

A plan records `bundle_content_id`, `profile_digest`, `constraints_digest`,
`verification_digest`, and `plan_key`, the digest of all four plus the
resolver version. Before reusing a cached plan, re-probe the device, re-verify
the bundle, and call `plan_invalidation` (Python), `ExecutionPlanResolver.invalidation`
(Swift, Kotlin), or `executionPlanInvalidation` (TypeScript). The result lists
the components that changed:

- `bundle`: different manifest bytes.
- `profile`: an OS update, a new or lost provider, or a changed budget,
  feature, or limit. `network` is not included.
- `constraints`: different caller policy.
- `verification`: a revocation, expiry, eligibility, or file-tree change.
- `resolver`: the plan came from another resolver version.

An empty list means the cached plan is still current.

## Golden vectors

`tests/fixtures/device_capabilities/vectors.json` holds the shared corpus:
synthetic probes with planted identity sentinels and their expected profiles,
every named profile and constraint set with its digest, nine manifest
variants, 57 resolution cases with the exact expected plan, and the
invalidation cases. Every reason code and assumption has at least one case.
Python, Swift, Kotlin, and TypeScript tests all consume the file.

```bash
python tests/fixtures/device_capabilities/generate_vectors.py
python tests/fixtures/device_capabilities/generate_vectors.py --check
```
