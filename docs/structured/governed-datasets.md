# Governed dataset snapshots

OpenMed can turn a saved cohort execution or a completed ingestion job into an
immutable `DatasetSnapshot`. The snapshot records the exact selection,
source snapshot, query, policy, schema, component and model versions,
vocabulary digest, input-license constraints, split membership, row digests,
and artifact hashes needed to reproduce a dataset build.

The default build is metadata-only. It emits opaque lineage, labels,
annotations, and digests, but not record values. Dataset construction never
changes a cohort decision and never initiates a clinical action.

## Build from an ingestion job

Create a value-free selection reference and a pinned build specification. All
identifiers in the example are synthetic.

```python
from openmed.structured.datasets import (
    DatasetBuildSpec,
    DatasetLicenseConstraint,
    DatasetRecord,
    DatasetSelection,
    LocalDatasetExporter,
    RedistributionPolicy,
    build_dataset_snapshot,
)

selection = DatasetSelection.from_ingestion_job(
    job_id="job_syntheticjob0001",
    job_digest="sha256:" + "1" * 64,
    source_snapshot_id="snapshot_syntheticsource1",
    source_snapshot_digest="sha256:" + "2" * 64,
)
spec = DatasetBuildSpec(
    dataset_id="dataset_syntheticdata001",
    created_at="2026-01-02T03:04:05Z",
    selection=selection,
    query_digest="sha256:" + "3" * 64,
    policy_digest="sha256:" + "4" * 64,
    schema_digest="sha256:" + "5" * 64,
    vocabulary_digest="sha256:" + "6" * 64,
    component_versions={"dataset_builder": "1.0.0"},
    model_versions={"clinical_encoder": "local-model-1"},
    licenses=(
        DatasetLicenseConstraint(
            source_id="synthetic_fixture",
            license_id="Apache-2.0",
            terms_digest="sha256:" + "7" * 64,
            redistribution=RedistributionPolicy.PERMITTED,
        ),
    ),
)
record = DatasetRecord(
    record_id="record_syntheticrecord1",
    patient_key="patient_syntheticpatient1",
    split="train",
    source_fact_ids=("fact_syntheticfact0001",),
    labels=("Condition",),
    values={"clinical_code": "synthetic-condition"},
)
result = build_dataset_snapshot(spec, (record,))
assert result.ok and result.value is not None

exported = LocalDatasetExporter(".openmed/datasets/safe").export(result.value)
assert exported.ok
```

The default formats are deterministic JSONL, annotation JSONL, and Parquet.
Parquet support uses the optional columnar dependency. Each Parquet row carries
the same canonical payload represented in JSONL so the two formats can be
checked directly.

## Build from a saved cohort

Use `DatasetSelection.from_cohort_execution(execution)` with an immutable saved
cohort execution. Only `met` memberships are eligible. Records associated with
an `unknown` or `conflict` membership cannot enter the snapshot, and supplying
a record outside the eligible set produces a typed conflict.

The selection records the cohort execution digest, the source snapshot and its
digest, the exact eligible opaque patient keys, and the number excluded for
review. This preserves the reason a row was or was not eligible without
exporting source text or direct identifiers.

## Split isolation

Snapshot construction fails closed when a patient or a source artifact, fact,
event, or evidence identifier appears in more than one split. The manifest
records both row-level digests and a digest for every split. Reordering input
records cannot change the manifest or output bytes.

Changing the source snapshot, query, policy, schema, component version, model
version, vocabulary digest, license terms, split assignment, row content, or
requested format changes the snapshot identity.

## Value and privacy profiles

There are three deliberately separate paths:

| Path | Values emitted | Required controls |
| --- | --- | --- |
| Default snapshot | no | opaque lineage and digest custody |
| De-identified snapshot | yes | `include_deidentified_values=True`; direct-identifier fields rejected |
| Identified export | yes | snapshot-bound approval plus a two-phase audit hook |

Token-vault material, credentials, secrets, passwords, encryption keys, and
token-like fields are rejected at record construction even when the default
value-free projection is requested.

An identified export requires `DatasetExportAuthorization` bound to the exact
snapshot and policy digest. `LocalDatasetExporter.export_identified` calls the
supplied `DatasetExportAuditHook.authorize` before writing and
`DatasetExportAuditHook.record` after writing. The request and receipt contain
only opaque identifiers, file hashes, the authorization digest, purpose, and
distribution posture. A refused authorization writes nothing; a failed
post-write audit is reported as `partial` with the digest-only receipt.

Use a separate, access-controlled output root for identified exports. The
exporter treats existing files as immutable: the same bytes are idempotent and
different bytes at the same path return `immutable_export_conflict`.

## License-aware distribution

Every input source needs a `DatasetLicenseConstraint` with a digest of the
terms used for the build and one of these redistribution states:

- `permitted`
- `restricted`
- `prohibited`
- `unknown`

Local export remains possible for restricted inputs. Setting
`distribution=True` succeeds only when every input is `permitted`; every other
state fails closed. The manifest stores license custody and policy state, not
the licensed vocabulary or dataset content itself.

## Compatibility and verification

Governed manifests declare schema version `1.0.0` with `same_major`
compatibility. The bundled `governed_dataset.schema.json` verifies the persisted
manifest shape. Loading a manifest also verifies its canonical digest, source
snapshot custody, file and split hashes, governed attributes, license tags, and
the underlying Journey `DatasetSnapshot` invariants.
