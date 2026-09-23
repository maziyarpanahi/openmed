# Annotation interchange and pipeline migration

OpenMed provides a clean-room, local-first interchange for moving reviewed
annotations into and out of labeling workflows. It also scans declarative
pipeline descriptions and produces an OpenMed-native configuration stub
without importing or executing referenced code.

The public Python modules are:

- `openmed.eval.annotation` for records, TSV/JSON interchange, offset
  conversion, loss reports, access policy, and bounded pagination;
- `openmed.interop.bridges.annotation_tools` for fact-correction and registry
  label rows; and
- `openmed.interop.bridges.pipeline_migration` for data-only pipeline scans.

All examples and committed fixtures are synthetic. The interchange carries
opaque identifiers, controlled results, digests, offsets, and optional numeric
embeddings. It never needs source text in a persisted record.

## Versioned annotation envelope

`AnnotationRecord` supports four annotation families:

| Type | Coordinates | Required result fields |
| --- | --- | --- |
| `entity` | Required | `label`, `surface_hash` |
| `relation` | Not allowed | `relation`, `source_annotation_id`, `target_annotation_id` |
| `fact_correction` | Not allowed | `fact_id`, `field`, `reason_code`, `replacement_code` |
| `registry_label` | Not allowed | `registry_id`, `record_id`, `label` |

Fact corrections and registry labels may also carry bounded `evidence_ids`.
Metadata is allow-listed: source format, review state, batch or annotator ID,
and model ID/version. Unknown fields fail closed rather than disappearing.

Every record and envelope declares schema `1.0.0` and compatibility policy
`same_major`. The bundled persisted schemas are:

- `annotation_record.schema.json`
- `annotation_envelope.schema.json`
- `annotation_loss_report.schema.json`
- `annotation_page.schema.json`

The canonical table is UTF-8 TSV with an exact header. Structured cells use
canonical JSON. Duplicate JSON keys, non-finite numbers, extra columns,
oversized cells, more than 10,000 rows, and inputs over 8 MiB are rejected.

```python
from openmed.eval.annotation import (
    AnnotationRecord,
    AnnotationState,
    AnnotationType,
    CoordinateConvention,
    build_annotation_envelope,
    export_annotation_tsv,
    import_annotation_tsv,
)

record = AnnotationRecord(
    annotation_id="annotation_aaaaaaaaaaaa",
    document_id="document_aaaaaaaaaaaa",
    namespace="default",
    annotation_type=AnnotationType.ENTITY,
    coordinate_convention=CoordinateConvention.UNICODE_CODEPOINT,
    start=0,
    end=4,
    state=AnnotationState.SUCCESS,
    result={
        "label": "condition",
        "surface_hash": "sha256:" + "0" * 64,
    },
    metadata={"source_format": "neutral_table"},
    embedding=(0.1, 0.2),
)
envelope = build_annotation_envelope([record])
exported = export_annotation_tsv(envelope)
restored = import_annotation_tsv(exported.text)
assert restored.records[0].record_digest == record.record_digest
```

`export_annotation_tsv(..., include_embeddings=False)` does not silently drop
the vector. It returns a `partial` `AnnotationLossReport` with one
`embedding_omitted` entry per affected record. A lossless export returns a
`success` report with no entries.

## Offset convention

Entity offsets are half-open: `start` is inclusive and `end` is exclusive.
Each entity declares one convention:

- `unicode_codepoint`
- `utf8_byte`
- `utf16_code_unit`
- `token_index`

`convert_record_offsets()` converts code-point, UTF-8, and UTF-16 boundaries
exactly from caller-held source text. The text is used only for the conversion
and is neither returned nor persisted. An offset that splits an encoded
character fails with `offset_not_boundary`. Token conversion fails with
`token_map_required`; OpenMed does not guess a tokenizer or its boundaries.

## Access and pagination

`AnnotationCatalog` provides an immutable read surface. An `AnnotationQuery`
binds the namespace, purpose, role, consent state, export policy, selected
annotation types, and page size. `AnnotationAccessPolicy` checks all of them
before returning records.

Pages are capped at 100 records. The opaque cursor commits to the full query
and envelope snapshot. Reusing it with another role, purpose, filter, or
snapshot returns a typed failure instead of returning an ambiguous page.
Denied pages contain no records.

## Fact-correction and registry-label rows

The neutral row adapters accept exact mappings and reject raw or unknown
fields:

```python
from openmed.interop.bridges.annotation_tools import (
    export_fact_correction_rows,
    import_fact_correction_rows,
)

rows = ({
    "annotation_id": "annotation_aaaaaaaaaaaa",
    "document_id": "document_aaaaaaaaaaaa",
    "evidence_ids": ["evidence_aaaaaaaaaaaa"],
    "fact_id": "fact_aaaaaaaaaaaa",
    "field": "assertion",
    "reason_code": "reviewed_correction",
    "replacement_code": "absent",
},)
envelope = import_fact_correction_rows(rows)
assert export_fact_correction_rows(envelope) == rows
```

These adapters exchange review decisions. They do not apply a correction to a
clinical store, authorize training, or make a clinical decision.

## Data-only pipeline scan

The scanner accepts JSON shaped as a top-level `stages` array. A stage has a
controlled `type` and an optional `config` object. Native stage names are
`supported`; documented neutral aliases such as `ingest`, `ner`, and
`terminology` are `mapped`; unknown types are `unsupported`.

Any nested executable-looking key—including `module`, `class`, `callable`,
`command`, `script`, `shell`, or `plugin`—makes that stage
`manual_review`. The stage is excluded from the generated stub. The scanner
contains no dynamic import, evaluation, compilation, or process-launch path.

Unsupported or manual-review stages remain in the report and force the overall
state to `unsupported`. Mapped stages produce `partial`. Only an entirely
native description produces `success` and `can_auto_migrate=True`.

```python
from openmed.interop.bridges import scan_pipeline_mapping

report = scan_pipeline_mapping({
    "stages": [
        {"type": "ingest", "config": {"format": "jsonl"}},
        {"type": "ner", "config": {"model_id": "model_local"}},
    ]
})
assert report.state.value == "partial"
assert report.can_auto_migrate is False
```

The report and native stub use schema `1.0.0` and `same_major`. The persisted
report schema is `pipeline_migration_report.schema.json`. Reports contain a
source digest and controlled stage classifications, not a copy of the source
description.

## CLI

Import canonical TSV to persisted JSON:

```console
openmed annotation import \
  --input annotations.tsv \
  --output annotations.json
```

Export JSON to canonical TSV and record declared loss:

```console
openmed annotation export \
  --input annotations.json \
  --output annotations.tsv \
  --omit-embeddings \
  --loss-report annotation-loss.json
```

Scan a declarative pipeline and write a report plus a safe native stub:

```console
openmed annotation scan-pipeline \
  --input pipeline.json \
  --report migration-report.json \
  --stub openmed-pipeline.json
```

Commands refuse to overwrite existing files unless `--force` is supplied.
Machine-readable mode emits counts, states, and digests, never output paths,
annotation rows, or source pipeline content. Failures use content-free messages
that do not echo a potentially identifying input or output path.

## Safety boundary

Annotation interchange is a transport and review primitive. A valid record or
migration report does not establish clinical correctness, consent for a new
use, model suitability, or permission to execute a migrated workflow. Keep
source documents in the caller's protected environment, review declared loss,
and validate the generated native configuration before running it.
