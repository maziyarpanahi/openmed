# Journey Specialist Pack

OpenMed v3 includes a reproducible control plane for four bounded clinical
specialists: assertion, temporality, relation-pair scoring, and fixed-option
classification. The control plane creates immutable data, split, recipe,
budget, evaluation, export, and promotion evidence. It does not download a
model or spend GPU budget by itself.

The bundled pack is deliberately conservative. It uses a small encoder pinned
to an immutable revision, parameter-efficient adapters, mixed precision,
predeclared stop rules, and one shared GPU ledger capped at USD 1,000.

## Current pack status

No trained artifact is promoted by the committed dry-run configuration. The
42-record bundled dataset is synthetic contract-test data. It proves offline
resolution, hashing, splitting, budgeting, and report generation; it is not a
clinical-quality training or validation corpus.

Before an artifact can be promoted, an actual run must use a separately pinned
permissively usable public or synthetic training dataset and pass a frozen
holdout. Credentialed or data-use-agreement corpora remain user supplied,
eval-only, and outside distributable training manifests.

## Pinned recipes and budget

All four recipes use `microsoft/deberta-v3-small` at revision
`a36c739020e01763fe789b4b85e2df55d6180012`, recorded as MIT licensed in the
pack configuration. The runtime never resolves that identifier during a dry
run; operators must acquire the asset explicitly and verify its digest before
training.

| Task | Head | PEFT | Precision | GPU-hour ceiling | Cost assumption | Estimate |
|---|---|---|---|---:|---:|---:|
| assertion/certainty/experiencer | sequence classifier | LoRA r8 | bf16 | 30 | $3/hour | $90 |
| temporal status | sequence classifier | LoRA r8 | bf16 | 36 | $3/hour | $108 |
| relation scoring | span-pair classifier | LoRA r16 | bf16 | 60 | $3/hour | $180 |
| fixed-option classification | sequence classifier | LoRA r8 | bf16 | 30 | $3/hour | $90 |

The aggregate reservation is USD 468. The hourly rate is a declared planning
input, not a live market-price claim. A run is denied if reservations or
recorded actual spend would take the shared ledger above USD 1,000.

Every recipe pins:

- model ID, immutable revision, architecture, and license;
- label set, output schema, adapter modules, rank, and trainable-ratio ceiling;
- seed, mixed precision, early-stop metric, patience, step ceiling, and GPU-hour
  ceiling;
- calibration method, abstention coverage floor, per-class and subgroup recall
  floors, baseline improvement, and quantized-delta ceiling;
- required `safetensors`, ONNX, and INT8 artifacts.

## Offline dry run

```python
from openmed.training import dry_run_journey_specialist_pack

result = dry_run_journey_specialist_pack(
    code_revision="0123456789abcdef0123456789abcdef01234567"
)
if not result.ok:
    raise RuntimeError(result.code)

report = result.value
print(report.ledger.committed_cost_usd)
```

The dry run reads package resources only. It verifies the data digest, creates
seeded stratified train/validation/holdout assignment digests, reserves all
four runs in one ledger, and emits versioned run manifests. No source text is
copied into a manifest.

Use a derived configuration with `bundled=false`, a local caller-supplied byte
payload, `usage_lane=distributable`, and a matching content digest for a larger
permissive training dataset. An `eval-only` lineage is rejected from training.

## Frozen holdout and calibration

`evaluate_journey_specialist_holdout(...)` accepts label probabilities and an
optional subgroup map for each frozen example. It reports:

- accuracy, macro F1, and per-class precision, recall, F1, and support;
- reliability bins and expected calibration error;
- abstention threshold, coverage, retained accuracy, and abstention rate;
- supported subgroup slices without source text or example identifiers;
- full-precision versus quantized macro-F1 and per-class recall deltas;
- count-only confusion, abstention, and quantization-change slices.

Missing predictions return `unknown`; incomplete quantized evidence returns
`partial`; mismatched labels or duplicate example IDs return `conflict`; a task
mismatch returns `unsupported`; policy and schema errors return `failure`.
None is converted into success.

## Promotion and fallback

A candidate is promoted only when every declared gate passes:

1. macro F1 meets the task floor;
2. macro F1 beats the named deterministic or existing-model baseline by the
   declared minimum improvement;
3. every class meets its recall floor;
4. expected calibration error stays below its ceiling;
5. retained coverage stays above its abstention floor;
6. supported subgroup recall meets its floor; and
7. the quantized macro-F1 delta stays below its ceiling.

Any failed gate produces `hold` and selects the declared fallback alias. The
completion step also refuses missing export digests, mismatched data/recipe
evidence, and spend that would breach the shared cap.

After a promoted completion,
`build_journey_specialist_model_pack_entry(...)` converts the measured INT8
artifact into the versioned local model-pack entry used by
`ClinicalTaskRouter`. Calibration, frozen-holdout, quantization, license,
artifact, and fallback evidence are carried forward. A held run is denied this
bridge and continues to select its deterministic or existing-model fallback.

## Model cards and limitations

`render_journey_specialist_model_card(...)` produces a deterministic task card
from a run manifest and measured holdout report. It includes lineage, metrics,
promotion status, and the selected runtime alias without raw examples. The
card digest, metrics digest, failure-slice digest, artifact digests, actual
hardware spend, and final promotion decision are pinned into the completed run
manifest.

These specialists are bounded components, not clinical decision makers. They
are not clinically validated and cannot autonomously diagnose, treat, enroll,
contact, order, or otherwise act on a patient. Ambiguous or insufficient
evidence must abstain, fall back, or enter review.

## Compatibility

Plans, recipes, dataset lineage, splits, ledgers, run manifests, and holdout
reports use schema version `1.0.0` with compatibility policy `same_major`. The
bundled `journey_specialist_training.schema.json` validates plans, dry runs,
individual run manifests, and frozen-holdout reports.
