# Clinical fact projection to OMOP 5.4

OpenMed can project evidence-bound `ClinicalFact` records into a deterministic
OMOP 5.4 current view. The projection is local-only: it does not download a
vocabulary, infer a mapping from network state, retain source note text, or
bundle licensed vocabulary content.

The public record contains:

- `person`, `visit_occurrence`, and value-free `note` rows whose `note_text` is
  an empty string while source custody stays in provenance;
- `condition_occurrence`, `drug_exposure`, `procedure_occurrence`,
  `measurement`, and `observation` rows;
- `source_to_concept_map` rows for every current mapping decision;
- fact-to-row provenance with evidence keys and source revision digests;
- visible mapped, unmapped, ambiguous, or rejected mapping outcomes;
- append-only ETL runs and count-only summaries; and
- explicit records of fields or fact types that could not be represented.

## Project a resolved batch

Mappings and vocabulary snapshots are references supplied by the caller. A
snapshot identifies the exact vocabulary version and content digest without
embedding any vocabulary records.

```python
from openmed.clinical.journey_contracts import ClinicalFact, sha256_digest
from openmed.interop.omop import (
    OmopConceptMapping,
    OmopFactProjectionInput,
    OmopVocabularySnapshot,
    project_clinical_facts_to_omop,
)

snapshot = OmopVocabularySnapshot(
    snapshot_id="local.synthetic.v1",
    version="2026-09-21",
    digest=sha256_digest("caller-owned-vocabulary-snapshot"),
    license="apache-2.0",
    usage_lane="redistributable",
)

fact = ClinicalFact(
    fact_id="fact_aaaaaaaaaaaaaaaa",
    subject_id="subject_aaaaaaaaaaaaaaaa",
    encounter_id="encounter_aaaaaaaaaaaaaaaa",
    fact_type="condition",
    value={"system": "local", "code": "synthetic-condition"},
    status="active",
    evidence_ids=("evidence_aaaaaaaaaaaaaaaa",),
    derivation_hash=sha256_digest("synthetic-derivation"),
    effective_time={"start": "2026-01-01T08:00:00Z"},
)

mapping = OmopConceptMapping(
    state="mapped",
    source_system="local",
    source_code="synthetic-condition",
    source_concept_id=1001,
    standard_concept_id=2001,
    standard_vocabulary="synthetic",
    standard_code="condition-standard",
    reason_code="mapped",
    snapshot_digest=snapshot.digest,
)

result = project_clinical_facts_to_omop(
    (
        OmopFactProjectionInput(
            fact=fact,
            source_key="source_aaaaaaaaaaaaaaaa",
            source_revision=sha256_digest("source-revision-1"),
            mapping=mapping,
            dataset_split="holdout",
        ),
    ),
    vocabulary_snapshot=snapshot,
    etl_version="3.0.0",
    occurred_at="2026-09-21T10:00:00Z",
)

if result.value is not None:
    condition_rows = result.value.table("condition_occurrence")
    lineage = result.value.etl_runs
```

`StoreResult.state` distinguishes success from partial, unknown, conflict,
unsupported, denied, and failure outcomes. A projection can carry a usable
value in a partial state when a mapping needs review or an explicit information
loss was recorded.

## Unmapped concepts stay visible

OpenMed never fabricates a standard concept. An unresolved mapping uses OMOP
concept ID `0`, an explicit non-`mapped` state, and a controlled reason code.
The same decision appears in the domain row, `source_to_concept_map`, and the
projection's mapping outcome. Callers can therefore route unresolved facts to
review without losing the current row.

```python
unmapped = OmopConceptMapping.unmapped(
    source_system="local",
    source_code="synthetic-condition",
    source_concept_id=1001,
    snapshot_digest=snapshot.digest,
    reason_code="standard_concept_not_found",
)
```

Mapping validity defaults to `1970-01-01` through `2099-12-31` when the caller
does not provide a narrower interval. The explicit mapping outcome—not the
OMOP `invalid_reason` validity flag—carries the review reason.

## Incremental correction semantics

`replace_by_source` is the default mode. Each input batch is a complete current
snapshot for every `source_key` present in that batch. Reprocessing the same
batch with the same timestamp is idempotent. A later batch for the same source
replaces its current domain, note, and mapping rows, while the prior ETL run
remains in `etl_runs` and lists superseded fact IDs. Set `ClinicalFact.parent_fact_ids`
to preserve the correction relationship in row provenance.

Use `append` only when source batches are independent and must coexist. A
deterministic row collision, a changed vocabulary snapshot, or the same subject
appearing in multiple dataset splits returns a typed conflict instead of
silently merging state.

## Vocabulary and license boundary

The core package contains no vocabulary data. A restricted snapshot may be
referenced only through the `user_supplied` lane and must not be marked as
bundled. Bundled references are accepted only when the caller declares a
redistributable license from the package allowlist. This is a policy gate, not
legal advice; operators remain responsible for the rights attached to their
vocabulary files and mappings.

## Validation and reproducibility

Use `validate_omop_fact_projection()` for key, reference, mapping, and
provenance checks. Use `assess_omop_fact_round_trip()` to compare input fact
identities with projected or explicitly lost facts. The bundled
`omop_fact_projection.schema.json` validates persisted records, including CDM,
schema, compatibility, ETL, mapping, provenance, and table contracts.

Output order and identifiers derive from canonical inputs, so input iteration
order does not change the serialized projection. Determinism does not make a
mapping clinically correct: validate snapshot provenance, mapping policy,
subgroup behavior, and intended cohort logic before production use.
