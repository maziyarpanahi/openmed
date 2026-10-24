# Language Adapter Distillation

Family-transfer adapters let one clinical backbone serve many languages, but
loading several adapters side by side can exceed a phone's memory and latency
budget. `openmed.training.languages.adapter_distillation` plans how to distill
selected adapters from one language family into a single compact student. It
then decides, from measured metrics, whether the student may replace its
parents on edge devices.

The module plans and gates only. It does not train a model, download weights or
open a network connection. Plans and gate reports carry identifiers, language
and label codes, `sha256:` digests and numbers. They never carry source text or
local paths.

## Plan a family student

`select_family_adapters` routes each requested language through the existing
`FamilyTransferRouter`. By default it selects every language of the family that
has its own installed adapter. A requested language without its own adapter is
refused unless `allow_donor_fallback=True`. In that case its family donor
becomes the parent, and the student is gated against the donor's zero-shot
results. Languages outside the family are refused. Teacher adapters must carry
a permissive license and be offline-runnable.

`build_adapter_distillation_plan` builds on the Mode-A distillation contract in
`openmed.training.distill`. The student backbone, critical labels,
critical-label loss weight, seed and device tier come from the `tiny_distill`
preset unless you override them.

```python
from openmed.training.adapters import AdapterMetadata
from openmed.training.languages.adapter_distillation import (
    DistillationSource,
    build_adapter_distillation_plan,
)

installed = {
    "hi": AdapterMetadata(adapter_id="local/indic-hi", provenance="local build"),
    "te": AdapterMetadata(adapter_id="local/indic-te", provenance="local build"),
}
plan = build_adapter_distillation_plan(
    plan_id="indic-edge-v1",
    family_id="indic",
    available_adapters=installed,
    sources=[
        DistillationSource(
            source_id="synthetic-indic-notes-v1",
            kind="synthetic",
            license="Apache-2.0",
            content_digest="sha256:" + "0" * 64,
            record_counts={"hi": 900, "te": 300},
        ),
    ],
    student_id="local/indic-edge-student",
    donor_teacher_weight=0.5,
)
plan.write_json("artifacts/indic-edge-plan.json")
```

The digest and identifiers above are placeholders. Use the digest of your
local source manifest.

The plan applies these rules:

- **Sources.** Each source is `synthetic` or `public` and is referenced by an
  opaque ID, a manifest digest and per-language record counts. Licenses must
  be permissive SPDX identifiers, `CC0-1.0` or `CC-BY-4.0`. Source IDs that
  name restricted or DUA-gated corpora (MIMIC, i2b2, n2c2, UMLS, SNOMED, CPT
  and the other gated DAPT corpora) are refused. Every language needs at least
  one synthetic source, and no source may declare records for a language
  outside the plan.
- **Teacher ensemble.** Each language is labeled by its parent adapter. With a
  positive `donor_teacher_weight`, installed family donors from the transfer
  graph join that language's ensemble as auxiliary teachers. Weights are
  normalized per language.
- **Mixture.** Languages are sampled in proportion to
  `records ** language_sampling_exponent`. The default exponent of 0.3 shifts
  mass toward low-resource languages, so high-resource parents do not dominate
  the shared student. `plan.shards()` lists every source-by-language slice with
  its sampling probability and teachers.
- **Gate policy.** The promotion thresholds are pinned into the plan and its
  `plan_digest` before any student is trained.

`compute_language_kd_loss(plan, language=..., ...)` averages the planned
teachers' temperature-scaled distributions with `ensemble_teacher_soft_labels`.
It then calls `openmed.training.distill.compute_kd_loss` with the plan's
temperature, alpha and critical-label weights. The function refuses teacher
outputs from any adapter not planned for that language, so one language's
adapter never silently supervises another language.

## Promotion gate

`evaluate_adapter_distillation_gate(plan, records)` consumes measured metric
records as data. Each record describes one model, either a parent adapter or
the student, on one language:

| Field | Meaning |
|---|---|
| `model_id` | Parent adapter ID or the plan's `student_id`. |
| `language` | Evaluated language. |
| `eval_set_digest` | `sha256:` digest of the local evaluation set. |
| `device_profile` | Opaque ID of the measurement device. |
| `label_recall` | Recall per label code; must include every critical label. |
| `f1` | Overall span F1 on the evaluation set. |
| `size_bytes` | Deployable artifact size; for a parent, backbone plus adapter. |
| `peak_memory_mb` | Peak resident memory while serving the language. |
| `latency_p50_ms`, `latency_p95_ms` | Latency for approximately one page. |

Each language is compared with the parent that serves it today, and every
check must pass:

| Check | Rule | Failure reason |
|---|---|---|
| `critical_label_recall` | Student recall stays within `critical_recall_tolerance` (default 0) of the parent for each critical label. | `critical_recall_drop`, `critical_label_unmeasured` |
| `cross_language_interference` | Student F1 per language drops at most `max_f1_drop` (default 0.02) below that language's parent. | `f1_drop_exceeds_budget` |
| `artifact_size_vs_parent` | Student artifact is at most `max_size_ratio` times each parent's size. | `size_exceeds_parent` |
| `peak_memory_budget` | Student memory fits the tier's `ram_mb_max`. | `memory_exceeds_tier_budget` |
| `latency_p50_budget`, `latency_p95_budget` | Student latency fits the tier's p50 and p95 budgets. | `latency_exceeds_tier_budget` |
| `latency_p95_vs_parent` | Student p95 is at most `max_latency_ratio` times the parent's p95. | `latency_exceeds_parent` |
| `measurement_integrity` | Every parent and student record exists, each pair shares one evaluation set, and all records share one device profile and consistent sizes. | `missing_parent_record`, `missing_student_record`, `eval_set_mismatch`, `device_profile_mismatch`, `inconsistent_size_measurement` |

Critical-label recall is non-compensable. Each language and label pair is
checked separately, so a gain on another label or another language never
offsets a loss. A student that is smaller and faster than every parent is
still rejected if it loses critical recall in one language. Device budgets
come from `openmed.eval.tiers.TIERS` through `DistillationGatePolicy.from_tier`
(the `Tiny` tier by default, with a `Nano` sub-tier available). Tier numbers
are never restated in code.

```python
from openmed.training.languages.adapter_distillation import (
    evaluate_adapter_distillation_gate,
)

report = evaluate_adapter_distillation_gate(plan, measured_records)
print(report.promoted, report.blocking_reasons)
report.write_json("artifacts/indic-edge-gate.json")
```

Malformed, duplicated or out-of-plan records raise
`AdapterDistillationError`. Missing or incomparable measurements fail the gate
instead of being skipped. The report embeds one
`openmed.training.distill.DistillationReport` per language as model-card
evidence.

A promotion means only that the student met the pinned evidence thresholds for
replacing its parent adapters on a device tier. It is not a clinical
validation, a compliance certification or a guarantee of de-identification.
Outputs remain decision support that qualified users must review.

```sh
.venv/bin/python -m pytest tests/unit/training/languages/test_adapter_distillation.py -q
```
