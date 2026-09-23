# Ingestion-to-fact pipeline

OpenMed provides a local-first orchestrator that composes source adaptation,
privacy policy, routing, clinical extraction, assertion, temporality, relation
extraction, grounding, fact normalization, validation, and durable Journey
writes. The pipeline is explicit about component and policy identity, preserves
evidence coordinates, and never converts a partial, unknown, conflict,
unsupported, denied, or failed stage into success.

The public entry point is `IngestionToFactPipeline` in
`openmed.interop.ingest`.

## Execution model

The ordered stages are:

1. `source_adaptation`
2. `privacy_policy`
3. `document_routing`
4. `extraction`
5. `assertion`
6. `temporality`
7. `relation_extraction`
8. `grounding`
9. `fact_normalization`
10. `validation`
11. `durable_writes`

The source adapter, fact normalizer, evidence validator, and Journey write are
built in. Every processing stage between adaptation and normalization requires
an explicit `PipelineComponent`. A component declares its stage, controlled
name, version, policy digest, and local operation. This makes an accidental
network fallback visible as an implementation error rather than a hidden
runtime choice.

`PipelineStageProduct` keeps component values in memory. Its durable surface is
limited to output digests, opaque record identifiers, a typed state, a
controlled reason code, and an optional committed revision. Components can add
`FactFragment` records; the built-in normalization stage converts their exact,
versioned outputs into evidence-bound `ClinicalFact` records.

## Manifests and idempotency

Build the pipeline before the `SourceManifest`, because both the component
graph and source adapter are part of the manifest identity:

```python
from openmed.clinical.journey_contracts import derived_opaque_id, sha256_digest
from openmed.interop.ingest import (
    EvidenceAdapterContext,
    IngestionToFactPipeline,
    SourceManifest,
    SQLiteIngestionStore,
    TextEvidenceAdapter,
)

store = SQLiteIngestionStore("./journey.sqlite3")
adapter = TextEvidenceAdapter()
pipeline = IngestionToFactPipeline(store, components)
source = "caller-held clinical text"
source_id = derived_opaque_id("source", sha256_digest(source))

manifest = SourceManifest(
    manifest_id=derived_opaque_id("manifest", source_id, "initial"),
    source_id=source_id,
    artifact_digests=(sha256_digest(source),),
    policy_digest=pipeline.policy_digest,
    pipeline_digest=pipeline.pipeline_digest(adapter),
    created_at="2026-01-02T03:04:05Z",
)

result = pipeline.run(
    manifest=manifest,
    source=source,
    adapter=adapter,
    adapter_context=EvidenceAdapterContext(
        source_id=source_id,
        subject_id="subject_aaaaaaaaaaaaaaaa",
        recorded_at="2026-01-02T03:04:05Z",
    ),
    subject_id="subject_aaaaaaaaaaaaaaaa",
    fact_profile="condition",
    recorded_at="2026-01-02T03:04:05Z",
    worker_id="worker_aaaaaaaaaaaaaaaa",
)
```

The example assumes `components` contains exactly one explicitly configured
component for each name in `PIPELINE_COMPONENT_STAGES`. A pipeline rejects a
manifest if its combined policy digest or complete pipeline digest differs.
Replaying an identical completed manifest returns the original stage lineage
and fact identifiers without running the model components or writing duplicate
facts.

A model version, component graph, or policy change produces a different
manifest and therefore a new append-only job and derivation. Existing facts
and lineage are not overwritten.

## Durable lineage

`PipelineStageManifest` uses schema `1.0.0` and compatibility policy
`same_major`. Each record contains:

- the stage and fixed sequence number;
- input and output digests;
- opaque input and output record identifiers;
- component name and version;
- policy digest;
- typed state and controlled reason code;
- parent stage-manifest identifiers; and
- a timezone-aware recording time.

SQLite and PostgreSQL ingestion stores persist the manifests, graph edges, and
append-only invalidations. The bundled JSON Schemas are
`ingestion_pipeline_stage.schema.json` and
`ingestion_stage_invalidation.schema.json`. `ingestion_integrity_check()`
verifies their payload hashes along with the resumable control-plane records.

The lineage tables are audit metadata. They cannot accept source text,
component values, credentials, or arbitrary metadata keys. Normalized clinical
values are written only to the protected Journey tables during the final
transaction.

## Failure and quarantine

Every non-success stage stops the pipeline before downstream execution. The
result retains its exact `StoreState` and controlled reason code. Untrusted or
incomplete outputs enter the existing ingestion quarantine path, so the job
cannot be completed without an explicit review and promotion. An exception is
reduced to a value-free retry classification; exception text is not persisted.

The `durable_writes` stage commits the artifact metadata, evidence locators,
and facts in one Journey-store transaction. If any dependency or policy check
fails, the transaction rolls back. A failed earlier stage therefore commits no
downstream facts and cannot produce a successful job state.

## Selected-stage reprocessing

Pass both `reprocess_from` and `previous_job_id` with a new manifest. The new
run reuses the prior ancestors as lineage parents, reconstructs their protected
in-memory context from the caller-held source, and persists only the selected
stage and its descendants. The previous selected-stage output and graph
descendants receive append-only `PipelineStageInvalidation` records. Ancestors
remain valid and are never invalidated merely because a downstream component
changed.

Reprocessing with the identical manifest is rejected: a changed model or
policy must have a new pipeline or policy digest so its output cannot overwrite
the prior derivation.

## Supported source slice

The synthetic integration suite exercises text, FHIR R4 JSON, HL7v2, CDA R2,
and CSV end to end. XLSX uses the same table adapter contract. Existing
normalized documents can use `ExistingDocumentEvidenceAdapter`. Image/OCR and
DICOM profiles require their dedicated evidence-coordinate adapters before
they are enabled in this orchestrator.

This pipeline provides evidence and review controls; it is not clinical
validation and does not authorize autonomous diagnosis, treatment, enrollment,
outreach, ordering, or another patient-care action.
