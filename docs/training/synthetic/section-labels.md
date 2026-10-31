# Section-label datasets

`build_section_dataset` writes a deterministic JSONL dataset of fictional
clinical notes whose section boundaries are tagged token by token (BIO), plus
a sidecar manifest of hashes and provenance. It trains and tests section and
document-type classifiers. The note text comes from repository-owned
templates; section boundaries come from the same clinical section detector
the runtime uses. See the [overview](index.md) for shared conventions.

## Build a dataset

```python
import tempfile
from pathlib import Path

from openmed.training.synthetic import build_section_dataset

with tempfile.TemporaryDirectory() as tmp:
    result = build_section_dataset(seed=11, n=3, output_path=Path(tmp) / "sections.jsonl")
    print(result.dataset_path.name, result.manifest_path.name)
    # sections.jsonl sections.manifest.json
    print(len(result.records), result.leakage_count)   # 3 0
    print(result.dataset_hash, result.manifest_hash)

    record = result.records[0]
    print(record["doc_type"], [s["label"] for s in record["sections"]])
```

`seed` and `n` are required; `n=0` writes an empty but valid dataset. The
manifest is written next to the dataset with the suffix `.manifest.json`.
`output_path` defaults to `DEFAULT_SECTION_DATASET_PATH`
(`section_labels.jsonl` in the working directory).

To restrict which sections appear, pass `section_labels`:

```python
with tempfile.TemporaryDirectory() as tmp:
    subset = build_section_dataset(
        seed=1, n=2, output_path=Path(tmp) / "plan.jsonl",
        section_labels=["plan", "impression"],
    )
    print({s["label"] for r in subset.records for s in r["sections"]})
    # {'impression', 'plan'}
```

## Record shape

Each record has `text`, `doc_type` (one of `DOCUMENT_TYPES`), `sections`
(`label`, `header`, `start`, `end` offsets), `tokens`, `token_offsets`, and
BIO `labels`/`section_tags` aligned with the tokens. Provenance fields mark
the record as `synthetic: True`, `contains_real_phi: False`,
`restricted_data: False`, `source` `SYNTHETIC_SECTION_SOURCE`, and `license`
`SYNTHETIC_SECTION_LICENSE` (Apache-2.0). `schema_version` is
`SECTION_RECORD_SCHEMA_VERSION`; the manifest's is
`SECTION_MANIFEST_SCHEMA_VERSION`.

`SECTION_LABELS` and `CANONICAL_SECTION_LABELS` list the 21 section labels in
the packaged lexicon (as a sorted tuple and a frozenset), and `DOCUMENT_TYPES`
and `CANONICAL_DOCUMENT_TYPES` list the seven document types.

## Load and validate

```python
from openmed.training.synthetic import (
    load_section_dataset,
    load_section_manifest,
    validate_section_labels,
    validate_section_record,
)

with tempfile.TemporaryDirectory() as tmp:
    result = build_section_dataset(seed=11, n=3, output_path=Path(tmp) / "s.jsonl")
    rows = load_section_dataset(result.dataset_path)        # validates every row
    manifest = load_section_manifest(result.manifest_path)  # checks schema, leakage
    print(len(rows), manifest["leakage_check"])
    # 3 {'fixture': 'eval/fixtures/section_multilingual.jsonl', 'overlap_count': 0}

validate_section_record(rows[0])
print(validate_section_labels(["plan"]))   # ('plan',)
```

`validate_section_labels` rejects unknown, duplicate, empty, or non-string
labels with `ValueError`; `validate_section_record` rejects records whose
schema version, provenance, or BIO alignment is wrong. The manifest carries
the dataset hash, record count, seed, and the evaluation fixture's path, never
its text.

## The leakage guard

Before writing, the builder compares every generated note to the held-out
fixture `DEFAULT_SECTION_EVAL_FIXTURE`
(`eval/fixtures/section_multilingual.jsonl`). Exact normalized matches and
containment of substantial fixture text raise `SectionDatasetLeakageError`
and nothing is written. The error names record and fixture indices only,
never clinical text. You can run the same check on your own records or texts:

```python
from openmed.training.synthetic import SectionDatasetLeakageError, assert_no_eval_overlap

print(assert_no_eval_overlap(rows))    # 0 overlaps against the default fixture
try:
    assert_no_eval_overlap(rows, eval_texts=[rows[0]["text"]])
except SectionDatasetLeakageError as error:
    print(error)
    # synthetic section dataset overlaps held-out eval text at record 0/fixture 0
```

## Seeds and determinism

The same `seed`, `n`, and `section_labels` produce identical records and the
same `dataset_hash`; the seed is recorded in the manifest and no global random
state is touched.

## Known limits

- English only, with a small set of templated sentences per section. The
  text is deliberately bland; it teaches header and boundary structure, not
  clinical language.
- The leakage guard only compares against the fixture you give it. It does
  not protect against overlap with evaluation data it has never seen.
- Labels reflect the packaged section lexicon. Headers outside it are not
  generated.
- Synthetic notes are augmentation data, not clinical ground truth.
