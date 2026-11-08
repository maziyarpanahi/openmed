# OpenMedKit Device Qualification

The [OpenMedKit support matrix](openmedkit-support-matrix.md) only allows a
cell to be **supported** when it names an evidence record. At freeze, every
evidence record is a CI lane, and no lane runs a model on a phone. This page
describes the harness that produces the `device-run` evidence kind: it
collects results from runs on physical devices, simulators and emulators,
checks them against a pinned plan, and proposes matrix changes for review.

The harness does not run anything on a device itself. A maintainer runs the
kit on hardware and saves one evidence record per run. The harness checks
those records. It never edits the committed matrix.

The code is `openmed.eval.kit_qualification`. The command-line entry point is
`scripts/release/kit_qualification_report.py`.

## How it works

1. **Plan.** List the matrix cells to qualify. For each one, pin the model
   bundles, fixtures and policies by SHA-256 digest, and declare the gates.
2. **Targets.** The plan expands into one target per bundle × fixture × policy
   × platform version. Each target fixes a runtime and provider (from the
   cell), a precision (from the bundle), a fixture and a policy.
3. **Runs.** Each device, simulator or emulator run writes one evidence
   record: the execution type, coarse device class, platform and runtime
   versions, the digests it actually loaded, and its measurements.
4. **Report.** `collect_qualification` matches each record to a target. It
   rejects records that do not fit, marks stale ones, checks every gate on
   its own, and reports each target and cell.
5. **Proposal.** `propose_matrix_updates` turns qualified cells into a
   matrix change. `apply_matrix_proposal` applies it to a copy. A maintainer
   reviews the change, commits it, regenerates the matrix page and runs the
   coherence check.

## Physical-device and simulator evidence

Each record states how it ran in its `execution` field:

| `execution` | Device classes | Evidence class | Can qualify a target |
|---|---|---|---|
| `physical-device` | Physical classes only (`iphone-ipad`, `android-device`, `apple-silicon-mac`, ...) | device | yes |
| `simulator` | `ios-simulator`, `watchos-simulator`, `visionos-simulator` | simulator | no |
| `emulator` | `android-emulator` | simulator | no |
| `fake-runtime` | any | fake runtime | no |

A record whose `execution` does not match its device class is rejected.
Examples are a `physical-device` record on `ios-simulator`, or a `simulator`
record on `android-device`. A simulator record can cite the physical cell it
stands in for, for example `ios-simulator` evidence for an `iphone-ipad`
cell. The report counts it separately, and the target stays
`simulator_only`. Cells whose own device class is a simulator or emulator can
never be promoted by this harness.

## Plan schema

A plan (`openmed.kit_qualification.plan.v1`) is a JSON object:

| Key | Contents |
|---|---|
| `schema` | `openmed.kit_qualification.plan.v1` |
| `release` | The matrix release, for example `v4.0`. |
| `harness_digest` | Optional. The digest of the device runner build. If set, every record must carry the same digest. |
| `bundles` | `id`, `family`, `artifact`, `precision` and `digest` for each model bundle. |
| `fixtures`, `policies` | `id` and `digest` for each one. Fixtures must be synthetic or held locally; they are never committed with restricted data. |
| `gates` | The gate values for every planned cell (see below). |
| `cells` | `cell`, `cell_digest` and `bundles`. Optional `fixtures`, `policies` and `platform_versions` narrow the targets. Optional `gates` overrides values for that cell. |

`cell_digest` pins the fields that define what a cell runs: platform range,
device class, ABI, runtime pin and version, provider, artifact, families,
precisions, tier and condition. It leaves out status, evidence and notes, so
promoting a cell does not make the plan stale. Changing the cell's runtime
pin does, and every target for that cell becomes `stale` with
`cell_definition_changed`.

A plan is rejected when any of these holds:

- It names an unknown, unsupported or non-inference cell.
- A bundle does not match the cell's artifact, family or precision.
- A digest is missing.
- A required gate is missing.
- A memory budget is larger than the cell tier's RAM ceiling.
- The cell's tier keeps model memory outside process RSS and the plan
  declares no `maximum_device_memory_mb`.

Problem codes refer to unknown entries by position and never repeat the
submitted value.

`plan_skeleton(matrix)` (or `--plan-skeleton`) writes a plan that lists every
qualification cell with its current digest. Its artifact lists are empty and
its gate values are `null`, so it stays invalid until a maintainer fills it
in. Key-store and OCR cells have no model runtime. They are listed as out of
scope.

### Gates

No thresholds are built in. Clinical and product budgets belong to whoever
runs the qualification, as with EdgeHealthBench.

| Gate key | Required | Meaning |
|---|---|---|
| `minimum_direct_identifier_recall` | yes | Detected over expected direct identifiers. |
| `maximum_cold_latency_ms` | yes | First inference after load. |
| `maximum_warm_p95_latency_ms` | yes | Warm p95; needs at least 3 warm runs, the same floor as `DeviceProfile.trials`. |
| `maximum_sustained_p95_latency_ms` | yes | p95 across a sustained session. |
| `minimum_sustained_seconds` | yes | Minimum length of that session. |
| `maximum_cancel_latency_ms` | yes | p95 time from cancel request to stop. |
| `maximum_thermal_level` | yes | Highest ordinal thermal level allowed, from nominal (0) to critical (3). |
| `require_thermal` | yes | Whether a missing thermal reading fails. |
| `require_measured_energy` | yes | Whether energy must be measured in joules; a proxy then fails. |
| `maximum_memory_peak_mb` | no | Tighter process memory budget. The default and the upper limit is the tier RAM ceiling from `openmed.eval.tiers`. |
| `maximum_device_memory_mb` | for accelerator tiers | Allocator budget for GPU, Neural Engine, NPU or DSP memory. |

Leakage has no threshold. Any critical leak or any document PHI leak fails,
as in EdgeHealthBench.

## Evidence record schema

An evidence record (`openmed.kit_qualification.evidence.v1`) is a closed JSON
object. Unknown fields are rejected, so a device name, serial number or user
field cannot be added. All values are controlled codes, versions, counts,
timings or digests.

| Field | Contents |
|---|---|
| `schema` | `openmed.kit_qualification.evidence.v1` |
| `recorded_on` | `YYYY-MM-DD` |
| `cell` | Matrix cell id. |
| `execution` | `physical-device`, `simulator`, `emulator` or `fake-runtime`. |
| `device_class`, `abi`, `provider`, `precision` | Matrix vocabulary. |
| `platform` | `name` (matrix platform) and `version` (dotted numeric; Android API level). |
| `runtime` | `pin` (matrix runtime pin id) and the `version` that actually ran. |
| `bundle`, `fixture`, `policy` | `id` and the `digest` actually loaded. |
| `harness_digest` | Digest of the runner build. |
| `run` | `status` (`completed`, `failed`, `not_run`) and `failures`, a list of fixed codes. |
| `edge_health` | Optional summary of an EdgeHealthBench report: `schema`, `tier`, `status`, `failures`, `report_digest`. Build it with `edge_health_block(report)`, which rechecks the report digest. |
| `metrics` | Every metric below, each with a `status`. |

Every metric must be present. If a reading was not taken, the metric is
`{"status": "unavailable"}`. Leaving it out is an error, so a missing
reading is always stated rather than hidden.

| Metric | `measured` fields |
|---|---|
| `direct_identifier_recall` | `detected`, `expected` |
| `leakage` | `assessed`, `critical`, `any_phi` (`critical ≤ any_phi ≤ assessed`) |
| `cold_latency_ms` | `value` |
| `warm_latency_ms` | `n`, `p50`, `p95` |
| `sustained_latency_ms` | `n`, `p95`, `duration_s` |
| `cancel_latency_ms` | `n`, `p95` |
| `memory_peak_bytes` | `value`, `scope` (`process_current` or `process_lifetime_highwater`) |
| `device_memory_peak_bytes` | `value`, `scope` (`allocator_current` or `allocator_peak`) |
| `thermal` | `max_level` (0–3) |
| `energy` | `joules`; or `status: "proxy"` with `proxy` (`process_cpu_seconds`, `battery_percent_delta`, `battery_charge_delta`) and `value` |
| `provider_fallback` | `observed` (true if any part of the graph fell back to another provider) |

Battery percentage, battery charge counters and process CPU time are
**proxies**, not energy measurements. The report labels them, for example
`proxy:battery_percent_delta`, and sets `energy_measured: false`. Thermal
and allocator readings are reported as measured or unavailable in the same
way.

## Matching, staleness and gates

A record is **rejected** when it does not fit its cell and plan:

- Unknown, unsupported, out-of-scope or unplanned cell.
- Platform, platform version, runtime pin, provider, device class or ABI
  mismatch.
- A bundle, fixture, policy or platform version that is not planned.
- A precision that differs from the bundle's.
- A duplicate of another record.

Rejected records appear in the report only by digest and reason code.

A record that fits is **stale** when any of these differ from the plan or
matrix:

- `bundle_digest_mismatch`, `fixture_digest_mismatch` or
  `policy_digest_mismatch`
- `runtime_version_mismatch`: the version does not satisfy the cell's pin
  (exact, minimum or major).
- `harness_digest_mismatch`
- `cell_definition_changed`: the plan's `cell_digest` is out of date.

Stale records are never evaluated.

A fresh record with `run.status` `not_run`, or an EdgeHealthBench summary
with status `not_run`, is `not_run`. It never counts as a pass. Every other
fresh record is evaluated against each gate on its own:

| Gate | Fails with |
|---|---|
| `run` | the run's failure codes |
| `edge_health` | `edge_health_tier_mismatch` or the EdgeHealthBench failure codes |
| `recall` | `recall`, or `missing_privacy_evidence` |
| `leakage` | `critical_or_document_phi_leakage`, `unassessed_or_failed_units`, or `missing_privacy_evidence` |
| `cold_latency` | `cold_latency` |
| `warm_latency` | `latency`, or `incomplete_measurements` when fewer than 3 warm runs |
| `sustained_duration`, `sustained_latency` | `sustained_duration`, `sustained_latency` |
| `cancel_latency` | `cancel_latency` |
| any timing gate | `missing_latency_evidence` when unavailable |
| `memory`, `device_memory`, `thermal` | `resource_budget`, or `missing_resource_evidence` |
| `energy` | `missing_resource_evidence` when measured energy is required |
| `provider_binding` | `runtime_fallback`, or `missing_provider_evidence` for `coreml`, `metal`, `nnapi`, `qnn` and `webgpu` |

Each gate reports its status (`passed`, `failed`, `missing`,
`not_applicable` or `informational`), the measurement label, the observed
value and the limit.

The record is identified by the SHA-256 digest of its canonical JSON.

## Target and cell status

A target's status, worst first:

| Status | Meaning |
|---|---|
| `failed` | At least one fresh physical-device record failed a gate. One failure blocks qualification even if another run passed. |
| `stale` | Only stale physical-device records, or the plan's cell digest is out of date. |
| `missing` | No evidence at all. |
| `not_run` | Only `not_run` physical-device records. |
| `simulator_only` | Only simulator or emulator evidence. |
| `fake_runtime_only` | Only fake-runtime evidence. |
| `qualified` | At least one passing physical-device record and no failing one. |

A cell takes the worst status among its targets. If all targets qualify but
the plan leaves a family × precision pair of the cell unplanned, the cell is
`incomplete`. A cell is `qualified` only when every target qualifies and the
plan covers every family and precision the cell declares. Cells without a
plan entry are `unplanned`. The report also lists cells that are currently
`supported` but have no device qualification.

Each report carries `requires_review: true`, `clinical_readiness: false`,
`independent_validation: false` and `auto_promotion: false`. It also carries
the matrix and plan digests, the collector source digest and a
`report_digest` over its whole content.

## Matrix proposals

`propose_matrix_updates(matrix, report, evidence_refs=[...])` only considers
cells that the report marks `qualified`:

- It adds one `device-run` evidence record per qualified cell, with id
  `device-run-<cell>`. The record covers `runtime-model`, `memory-budget` and
  `device-qualification`.
- It proposes each experimental cell for `supported`.
- It switches the surface's `support_claim` when the surface was
  experimental-only.
- It attaches the evidence to cells that are already supported.

A supported cell whose planned targets do not qualify is listed for
`review`. It is never demoted automatically.

The function refuses a report whose digest does not match its content or
that was collected against a different matrix. It also checks its own
result: it applies the proposal to a copy and validates it with the matrix
validator.

`evidence_refs` must be repository paths where the qualification report and
records will be committed. The coherence check fails while those paths do
not exist.

## Command line

```bash
# List every qualification cell with its current digest.
python scripts/release/kit_qualification_report.py --plan-skeleton plan.json

# Collect evidence (files or directories of *.json records).
python scripts/release/kit_qualification_report.py --plan plan.json \
    --evidence runs/ --output report.json

# Propose matrix changes and write a proposed copy for review.
python scripts/release/kit_qualification_report.py --plan plan.json \
    --evidence runs/ --output report.json --proposal proposal.json \
    --evidence-ref eval/qualification/v4.0/report.json \
    --proposed-matrix proposed-matrix.json
```

The script reads files of up to 1 MiB each and at most 10,000 records. An
unreadable record file is reported as `evidence_unreadable`. Output lines
hold fixed codes, counts and controlled matrix ids. File names and record
values are never printed. The script refuses to write `--proposed-matrix`
over the matrix it read or over the packaged matrix.

`--require-qualified` exits with status 1 unless every planned cell is
qualified. The portable-kit release gate (#3932) is expected to use this
report.

After reviewing a proposed matrix, copy it over
`openmed/eval/data/kit_support_matrix.json` and commit the referenced
evidence. Then run:

```bash
python scripts/release/check_kit_support_matrix.py --write-docs
python -m pytest tests/unit/eval/test_kit_support_matrix.py -q
```

## EdgeHealthBench device tiers

`edge_health_tier(device_class, provider)` maps each cell to an
[EdgeHealthBench](evaluation/edge-health-bench.md) tier. The Android ONNX
Runtime tiers are `android_ort_cpu`, `android_ort_nnapi` and
`android_ort_qnn`. The iOS-family device tiers, which cover iPhone, iPad,
Apple Watch and Apple Vision hardware, are `ios_device_cpu`, `ios_device_mlx`
and `ios_device_core_ml`.

Apple simulators run on the host Mac, so they map to the host tiers. The
Android emulator still runs Android ONNX Runtime. NNAPI, QNN and the iOS MLX
and Core ML tiers need separate allocator evidence, like the existing MLX,
Core ML and accelerator tiers.

## What this harness does not do

- It does not run kits on devices, buy hardware or use device farms.
  Maintainers run the qualification and supply the records.
- It does not attest that a record came from the hardware it names. The
  device class and execution type are declarations by the runner.
- It does not qualify key-store or OCR cells, and it does not make clinical
  qualification claims.
