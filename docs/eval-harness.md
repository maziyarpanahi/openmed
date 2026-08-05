# Eval Harness & Metrics

`run_benchmark` executes a model over a sequence of `BenchmarkFixture` objects and returns a
`BenchmarkReport` whose `metrics` dict contains the standard OM-018 metric bundle.

## Chinese clinical NER

The `chinese-clinical-ner` suite ships a tiny synthetic CMeEE-shaped fixture
for offline CI. It reports exact precision and recall per canonical label and
applies a zero-tolerance PHI-token leakage gate to injected synthetic
identifiers. Leakage findings contain hashes and offsets, never identifier
text.

CMeEE, CBLUE, eHealth corpora, and related model weights are not
redistributed: callers must provision licensed assets outside the repository
and pass an explicit local path to `load_cmeee`. Missing paths and
repository-internal real-data paths fail with license-boundary guidance.

### Model-card evidence for the Chinese route

`chinese_clinical_ner_metadata()` emits a `model_evidence` block instead of a
fixed claim about which checkpoint serves `zh`:

| Field | Meaning |
| --- | --- |
| `routed_default_model` | The checkpoint the `zh` language pack resolves to today |
| `dedicated_zh_model` | `False` only while `zh` still routes to the multilingual fallback |
| `weights_bundled` | Always `False`; OpenMed ships no weights |

Record that block verbatim in a model card. Two cases need explicit evidence:

- **Multilingual fallback.** If `dedicated_zh_model` is `False`, state that
  Chinese coverage comes from `OpenMed/privacy-filter-multilingual` and report
  per-label recall separately for Han-script fixtures, because a fallback
  encoder's Chinese recall is not implied by its aggregate score.
- **Dedicated local model.** If you substitute your own checkpoint, the card
  must carry its license, provenance, and training-corpus statement. A model
  trained on CMeEE or any other CBLUE task must not be evaluated on that same
  task's split without disclosing the overlap; record the split hashes you
  scored against.

The routed default is a PII checkpoint scored here for entity coverage, not a
CMeEE-trained clinical NER checkpoint. That distinction belongs in the card.

## CBLUE task coverage

The `cblue-task-coverage` suite extends the user-supplied benchmark interface
to the CBLUE task shapes that carry clinical entity annotations:

| Task | Shape | Path variable |
| --- | --- | --- |
| `chip-cdn` | Diagnosis-term normalization | `OPENMED_CHIP_CDN_PATH` |
| `imcs-v2-ner` | Medical-dialogue entity recognition | `OPENMED_IMCS_V2_NER_PATH` |

Relation decoding (`cmeie`) is deliberately out of scope. Requesting it raises
rather than silently returning an empty result.

The two committed fixtures contain no benchmark data. Every surface form is
composed from a closed, invented vocabulary of enumerator morphemes, and both
offsets and BIO tags are computed rather than hand-written. The generator is
deterministic by construction — it draws no random numbers and reads no
external state — so the fixtures can be reproduced byte for byte:

```bash
python scripts/benchmarks/generate_cblue_synthetic_fixtures.py --check
```

### Tested local-asset layouts

Both layouts below are exercised by the offline test suite. Either set one
variable per task, or set `OPENMED_CBLUE_PATH` to a root that contains one
directory per task id:

```text
$OPENMED_CBLUE_PATH/
  chip-cdn/          # CHIP-CDN rows: {"text": ..., "normalized_result": "a##b"}
  imcs-v2-ner/       # IMCS-V2-NER rows: {"sentence": ..., "BIO_label": ...}
```

`sentence` may be a string or a character list, and `BIO_label` a
whitespace-joined string or a tag list; character offsets are decoded from the
tag sequence rather than read from the row. CHIP-CDN standard terms are split
on `##`. Paths are resolved explicitly: the task variable wins over
`OPENMED_CBLUE_PATH`, and an argument wins over both.

### Evaluation commands

```python
from openmed.eval.suites import load_suite_fixtures, suite_metadata
from openmed.eval.suites.cblue_coverage import run_cblue_task_coverage

# Offline smoke run over the bundled synthetic fixtures.
fixtures = load_suite_fixtures("cblue-task-coverage")

# Licensed local data, one path per task.
fixtures = load_suite_fixtures(
    "cblue-task-coverage",
    paths={"chip-cdn": "/data/cblue/chip-cdn", "imcs-v2-ner": "/data/cblue/imcs"},
)

report = run_cblue_task_coverage(
    fixtures, model_name="your-local-model", runner=your_runner
)
print(suite_metadata("cblue-task-coverage")["tasks"])
```

### Interpreting the report

`metrics.tasks` holds one entry per task shape with `exact_span_f1`,
`span_count`, and `label_counts`; `chip-cdn` adds `normalization_accuracy`,
the share of mentions whose predicted standard-term set matched gold exactly.
`metrics.gate` fails closed when a fixture cannot prove its origin. Reason
codes are `missing_license_block`, `incomplete_license_block`,
`unexpected_redistribution`, `missing_source_path_hash`, `unexpected_script`,
`unexpected_language`, `unmapped_source_label`, `raw_source_path_in_metadata`,
`missing_normalized_terms`, `unknown_task`, and `no_task_fixtures`. Findings
carry fixture ids, reason codes, metadata key names, and hashes only; source
paths appear as SHA-256 digests so a report can be attached to a ticket
without disclosing where licensed data lives.

A `no_task_fixtures` failure means a task shape produced nothing. Treat it as a
configuration error, not a score of zero.

### Upgrade and rollback

Adding a task shape changes `DEFAULT_SUITES`, so pin the OpenMed version that
produced any archived report. To upgrade, re-run the offline smoke path first
(`run_synthetic_cblue_task_coverage_smoke()`); it needs no licensed data and
no network, so it isolates loader regressions from data problems. Then re-run
against local data and compare `metrics.tasks` per task rather than a single
aggregate, because label mixes differ sharply between the two shapes. To roll
back, reinstall the pinned version and re-run the same smoke path: it is
deterministic, so an unchanged report confirms the rollback rather than
merely a successful install. Reports are self-describing; `metadata.tasks`
records which path variable was configured for each task at run time.

### Optional local model assets

Set `OPENMED_ZH_CLINICAL_NER_MODEL_DIR` to a checkpoint directory you are
licensed to use, then run `run_chinese_clinical_ner_conformance`. It enforces
the span and canonical-label contract — a span out of bounds, inverted, or
misaligned to the text it claims raises `ChineseClinicalNerContractError`,
while nested and overlapping spans are allowed because nested entities are
normal in CMeEE data. On top of the suite's existing `per_label` and
`phi_token_leakage` metrics it adds latency percentiles and local-asset
evidence: a path fingerprint and whitelisted artifact descriptors, never the
local path or the directory basename. When the variable is unset the optional
check skips with configuration guidance.

Two limits are worth stating. Passing proves the contract and the skip
semantics, not the measured quality of any checkpoint — quality numbers only
exist for your own opt-in run. And the PHI-leakage gate judges
de-identification behaviour, so a CMeEE-trained entity model will
legitimately leave the synthetic identifiers standing; pass
`fail_on_leakage=False` for such a model and read leakage as evidence.

## Clinical PHI flagship SHIELD comparison

The `OpenMed/OpenMed-ClinicalPrivacy-tier0` SHIELD report is comparison
evidence, not a high-recall release gate. Its explicit
`metrics.shield_comparison` block contains aggregate recall, leakage, and exact
span scores plus recall and leakage for every SHIELD-mapped canonical label.
G1a, G2, and G3 certification remains a separate release-gate workflow.

The flagship runner requires a checkpoint manifest and a stable link to that
manifest. The report binds that checkpoint row to the committed clinical-PHI
dataset manifest, the public SHIELD corpus coordinates, the normalized fixture
set, and the eval code. It stores hashes instead of raw fixture identifiers and
does not vendor SHIELD rows.

```bash
openmed benchmark pii \
  --suite shield \
  --models OpenMed/OpenMed-ClinicalPrivacy-tier0 \
  --checkpoint-manifest models.jsonl \
  --checkpoint-manifest-ref \
    models.jsonl#OpenMed/OpenMed-ClinicalPrivacy-tier0 \
  --output clinical-privacy-shield.report.json
```

The checkpoint input may be a JSON object, a list of model rows, or JSONL. Its
flagship row must identify `OpenMed/OpenMed-ClinicalPrivacy-tier0`; when a
`reproducibility_hash` is present, it must be a lowercase `sha256:` digest.
Offline tests use only synthetic SHIELD-shaped rows.

## Clinical PHI flagship certification

`build_clinical_privacy_release` certifies the named
`OpenMed/OpenMed-ClinicalPrivacy-tier0` checkpoint from four inputs: its
checkpoint manifest row, verified mode-C training provenance, a held-out
clinical-PHI `BenchmarkReport`, and the manifest-linked SHIELD comparison
report. It reuses the release harness's v1.6 G1a and G2 floors, requires
explicit category coverage plus a reported zero critical-leakage count and
zero residual leakage for G3, then signs a report containing exactly those
three gate verdicts.

The signed checks bind hashes and stable references for the checkpoint,
clinical-PHI dataset manifest, held-out report, and SHIELD report. SHIELD stays
comparison-only evidence and cannot be promoted to the high-recall gate. A
quarantined candidate yields a signed failure report but no model card or model
manifest entry.

```python
from openmed.eval.clinical_privacy_release import (
    build_clinical_privacy_release,
)

release = build_clinical_privacy_release(
    held_out_report,
    shield_report,
    checkpoint_manifest=checkpoint_row,
    training_provenance=training_provenance,
    checkpoint_manifest_ref="models.jsonl#clinical-privacy-tier0",
    held_out_report_ref="release-evidence/held-out-report.json",
    shield_report_ref="release-evidence/shield-report.json",
    release_date="2026-08-01",
    signing_key=release_signing_key,
)
paths = release.write("release-evidence/clinical-privacy-tier0")
```

The output directory contains `gate-report.json`, `README.md`,
`model-datasheet.json`, `model-manifest-entry.json`, and
`release-manifest.json`. These artifacts contain aggregate metrics, hashes,
and provenance references only—never source notes, fixture identifiers, or raw
PHI. The generated card marks the checkpoint as assistive de-identification,
not diagnosis, treatment recommendation, or an automatic clinical-decision
trigger.

## DirectID Tiny certification

`build_directid_release` consumes the aggregate artifacts from the DirectID
dataset, Mode-A distillation, safety-sweep, and quantization workflows. It
reuses the release-gate harness and signs exactly G1b, G3, G4, and G5 for one
selected runtime format. The selected format is publishable only when the
certified evaluation covers every DirectID label, structured-ID recall is at
least 99.5%, critical and residual leakage are both zero, quantization recall
loss is within the format limit, and measured latency and RAM fit the Tiny
tier.

```python
from openmed.eval.directid_release import build_directid_release

release = build_directid_release(
    directid_evidence,
    candidate_checkpoint=candidate_checkpoint,
    training_report=training_report,
    run_manifest=run_manifest,
    training_provenance=training_provenance,
    dataset_evidence=dataset_evidence,
    quantization_evidence=quantization_evidence,
    release_format="mlx-8bit",
    release_date="2026-08-02",
    signing_key=release_signing_key,
)
paths = release.write("release-evidence/directid-tiny")
```

A releasable package contains `gate-report.json`,
`checkpoint-manifest.json`, `README.md`, `model-datasheet.json`, and
`release-manifest.json`. A failing package writes only the signed gate report
and release manifest; it has no checkpoint manifest, model card, or publish
target. Optional formats are isolated: for example, a rejected INT4 artifact
is listed under `quarantined_formats` without blocking a passing INT8 format.

All inputs and outputs are aggregate and offline-friendly. Dataset sources are
represented by IDs, licenses, revisions, and hashes; safety-sweep evidence uses
counts, offsets, hashes, and its patterns version. Raw identifiers, source
records, restricted dataset payloads, and signing secrets are never copied into
the release package.

## Metric Bundle

| Metric | Path | Gating? | Description |
| --- | --- | --- | --- |
| Latency p50 | `latency.p50_ms` | No | Median steady-state fixture latency in ms. |
| Latency p95 | `latency.p95_ms` | No | 95th-percentile steady-state fixture latency in ms. |
| Latency count | `latency.count` | No | Number of steady-state fixtures (excludes cold start). |
| Cold-start latency | `latency.cold_start_ms` | No | Wall-clock latency of the first fixture call in ms. |
| Peak RSS | `resources.peak_rss_bytes` | No | Peak resident set size in bytes during the run. |

## Signed relation scorecard contract

The committed synthetic relation gold is part of the normal suite-runner flow.
Selecting `suite="relations"` makes `run_suite` load the relation schema rather
than the span schema, run the relation gate, and return a signed
`RelationScorecard`. `run_relation_suite` exposes the same flow directly when a
caller needs to supply a signing key, key id, or baseline explicitly.

The fixture file is JSONL with one schema-version `1` document per line. Every
row requires a globally unique `id`, source `text`, `metadata.synthetic=true`,
and non-empty `entities` and `relations` arrays. Entity rows require unique
`id`, `start`, `end`, and `label` fields. Relation rows require unique `id`, a
normalized `type`, `head` and `tail` entity references, and a `scope` of either
`sentence` or `document`. Optional `traps` rows require an `id`, a `kind`
of `assertion` or `temporal`, one or more known `relation_ids`, and
`zero_tolerance=true`. The loader rejects duplicate fixture, entity, or relation
ids; invalid offsets; unknown references; unsupported versions; and
non-synthetic rows before scoring.

```python
from openmed.eval import run_relation_suite

scorecard = run_relation_suite(
    model_name="local-relation-model",
    runner=relation_runner,
    signing_key=local_release_key,
    output_json="evidence/relation-scorecard.json",
    output_markdown="evidence/relation-scorecard.md",
)

assert scorecard.verify(local_release_key)
model_card_block = scorecard.model_card_evidence()
```

The JSON artifact type is `openmed.eval.relation_scorecard`, schema version `1`.
Its signed payload contains:

- strict and relaxed precision, recall, and F1, with relation-type, scope, and
  language breakdowns;
- assertion- and temporal-consistency sub-scores computed over the trapped
  relations, alongside their evaluated and leaked relation counts;
- a SHA-256 hash of the fixture bytes after canonicalizing platform line endings
  to LF, plus a canonical hash for every validated fixture;
- configured assertion and temporal trap summaries, aggregate leak counts, and
  hashes for leaked trap relations rather than relation text;
- the complete relation regression gate result, reproducibility hash, and HMAC
  signature.

`model_card_evidence()` returns the complete signed artifact under the
`relation_scorecard` key. Existing `ModelScorecard` consumers can instead use
`scorecard.to_benchmark_report()`, which preserves fixture hashes and trap
summaries in report metadata. Consumers must verify the signature and require
`gate_passed` before publishing either form as passing evidence.

The runner is fail-closed. It writes requested JSON and Markdown failure
artifacts first, then raises `RelationGateFailure` whenever a pinned strict-F1
comparison, required baseline, or zero-tolerance trap check fails. The
exception carries the signed failure scorecard for archival; it is not a model
release authorization.

Every `(family, relation-type)` baseline is bound to the SHA-256 hash of the
fixture file that produced it. A missing, malformed, or different candidate
fixture hash quarantines the run, even when all reported F1 values are above
their pinned floors.

Future relation, event, and coreference suites should keep raw notes and mention
surfaces out of this evidence layer. Extensions should add typed aggregate
breakdowns and hashed fixture provenance, define task-specific trap summaries,
and version the artifact when binary head-tail relation semantics no longer
describe the scored object.

## Edge Metrics

### cold_start_ms

The harness records the wall-clock latency of the **first** fixture call separately. The default
runner keeps a shared model loader for the duration of the benchmark run, so that first call encloses
model and tokenizer loading plus the first forward pass. Later fixture calls reuse the warmed loader
and feed the steady-state latency summary. The value is surfaced at:

```
report.metrics['latency']['cold_start_ms']
```

It is **excluded** from the steady-state `p50_ms`, `p95_ms`, and `count` values.

!!! note "Reported, not gating"
    `cold_start_ms` does not participate in any release gate. It is an observability metric
    intended to track model-load overhead over time — not a pass/fail criterion.

```python
report = run_benchmark(fixtures, suite="my-suite", model_name="my-model", runner=runner)
cold_ms = report.metrics["latency"]["cold_start_ms"]
print(f"Cold-start latency: {cold_ms:.1f} ms")
```

## Family-transfer evaluation

`cross_lingual_family_transfer_report` scores the bundled synthetic Indic
donor/target gold in three modes: an untargeted multilingual baseline, Hindi
donor-adapter zero-shot inference for Telugu, and a Telugu-adapted path. The
same report re-scores the donor before and after adaptation so target F1 is
published alongside an explicit donor non-regression result.

The runner receives `transfer_mode`, `evaluation_role`, `family`,
`donor_language`, `target_language`, and `adapter_language` in fixture metadata.
It can therefore select locally provisioned model and adapter assets without a
network call. JSON and Markdown artifacts contain only aggregate metrics,
language codes, and donor-to-target deltas; synthetic fixture text and spans
are excluded.

For release qualification, pass measured `AdapterParameterAccounting` keyed by
the configured output adapter ID. The report then adds a `full_model` target
mode on the same synthetic gold and jointly requires the adapted target to
retain at least 90% of the full per-language model F1 while training no more
than 10% of its parameters. Both thresholds are explicit arguments, and the
aggregate JSON and Markdown evidence records the thresholds, observed ratios,
and pass/fail results.

```python
from openmed.eval import cross_lingual_family_transfer_report
from openmed.training.adapters import AdapterParameterAccounting

report = cross_lingual_family_transfer_report(
    "local-family-transfer-model",
    runner=local_family_transfer_runner,
    parameter_accounting_by_adapter={
        "family-transfer/indic-hi-to-te": AdapterParameterAccounting(
            shared_backbone_parameter_count=110_000_000,
            adapter_trainable_parameter_count=524_288,
            task_head_trainable_parameter_count=65_536,
            full_language_model_trainable_parameter_count=110_065_536,
        )
    },
)
print(report.to_markdown())
```

The counts above are synthetic examples; release evidence must use measured
counts from the locally provisioned backbone, adapter, task head, and full-model
reference.

## Grounding accuracy gate

`openmed/eval/grounding_accuracy.py` measures whether the sparse candidate
generator maps a clinical mention to the **right** coded concept. It ships a
fully synthetic gold set under `openmed/eval/golden/grounding/` (one JSONL per
permissive system: RxNorm, LOINC, ICD-10-CM) and reports **top-1** and
**top-5** accuracy per system and per language, plus a not-groundable
**abstention rate**, in the standard benchmark report schema.

Run the suite locally and print the per-system table:

```
openmed gates grounding            # prints the table
openmed gates grounding --strict   # non-zero exit when a floor is breached
```

`GroundingGateConfig` in `openmed/eval/release_gates.py` holds the floors:
English gold is held to `top-1 >= 0.90` and `top-5 >= 0.97` per permissive
system; `zh`/`hi` gold is held to lower, explicitly **provisional** floors
(`top-1 >= 0.80`, `top-5 >= 0.90`) while multilingual grounding coverage
matures. `evaluate_grounding_accuracy_gate` emits one `GateCheck` per system,
and `build_grounding_gate_report` wraps them in a signed `GateReport`.

!!! warning "Fail closed"
    The release-gates workflow runs `python -m openmed.eval.release_gates
    --grounding` on every tag build, uploads the signed gate report artifact,
    and fails the job when any per-system floor is breached — a silently wrong
    crosswalk or ranking regression cannot ship.

```python
from openmed.eval.grounding_accuracy import evaluate_grounding_accuracy
from openmed.eval.release_gates import evaluate_grounding_accuracy_gate

report = evaluate_grounding_accuracy()
rxnorm_en = report.system("rxnorm").language("en")
print(rxnorm_en.top1_accuracy, rxnorm_en.top5_accuracy)
checks = evaluate_grounding_accuracy_gate(report)
```

## Cross-lingual grounding capstone gate

The standalone `cross_lingual_grounding` suite verifies the complete lexical
grounding path for English, Spanish, French, German, and Chinese mentions. It
uses the bundled synthetic/permissive alias fixture for ICD-10-CM, RxNorm, and
HPO, requires top-1 accuracy of at least `0.80` in every non-English language,
and requires the unchanged English baseline to remain at `1.00`.

`assert_cross_lingual_grounding_gate` evaluates those accuracy requirements
together with synthetic provenance and the restricted-corpus marker scan. Its
report contains aggregate counts and scores, not raw mentions; failure
diagnostics likewise contain only language codes, scores, and known
restricted-marker names. Language-aware linkers accept the source-language code
used by the suite and stamp it on every grounded candidate for downstream audit.

```python
from openmed.eval.suites import assert_cross_lingual_grounding_gate

report = assert_cross_lingual_grounding_gate()
print(report.per_language_top1)
```

The gate is deterministic and offline. The default grounding path stays
lexical and does not import an embedding dependency. Applications may opt in to
a separately provisioned local multilingual embedding backend for zero-alias
fallback; remote model identifiers are rejected rather than downloaded.

!!! warning "Synthetic/permissive data only"
    The committed fixture must remain synthetic or permissively licensed.
    UMLS, SNOMED CT, MIMIC, i2b2, n2c2, CPT, and other restricted or DUA-bound
    assets must stay user supplied and out of process; do not copy their text
    into alias tables, fixtures, logs, or gate diagnostics.
