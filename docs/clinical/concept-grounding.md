# Concept grounding and interoperability export

OpenMed can ground extracted clinical spans to standard vocabulary concepts and
convert the selected codes into deterministic FHIR R4 resources or in-memory
OMOP CDM v5.4 tables. The output is assistive: it requires qualified review and
must not be used as an autonomous diagnosis, treatment, billing, or coding
decision.

## Ground free vocabularies

The canonical API is available from `openmed.clinical.grounding`:

```python
from openmed.clinical.grounding import VocabLoader, ground

spans = [
    {
        "text": "type 2 diabetes",
        "start": 12,
        "end": 27,
        "label": "condition",
        "source_language": "en",
    }
]

grounded = ground(
    spans,
    systems=["rxnorm", "icd10cm", "loinc", "hpo"],
    loader=VocabLoader(local_only=True),
)
```

`ground()` returns one `GroundedSpan` for every input span, including
abstentions. Each result preserves source offsets, assertion context, source
language, candidate provenance, a `codes` mapping, an optional UMLS `cui`, and
the highest selected `score`. It selects at most one candidate per requested
system after the existing sparse/dense retrieval and context-aware reranking
stage.

`VocabLoader` supports RxNorm, ICD-10-CM, LOINC, HPO, and MeSH. For predictable
offline operation, pre-stage normalized vocabulary artifacts and use
`local_only=True`. A download is attempted only when local-only mode is off and
a source has an expected SHA-256; checksum-free downloads are refused.

## Restricted vocabularies

UMLS and SNOMED CT content is never bundled or downloaded. Activating either
system requires an explicit, caller-owned local normalized CSV, TSV, or JSONL
alias table and an explicit user license key:

```python
from openmed.clinical.grounding import UserKeyVocabularyLoader, ground

umls = UserKeyVocabularyLoader(
    "umls",
    "/licensed/local/umls_aliases.tsv",
    license_key="caller-supplied-proof",
)
grounded = ground(
    [{"text": "example finding", "start": 0, "end": 15}],
    systems=["umls"],
    restricted_loaders={"umls": umls},
)
```

The key is validated and immediately discarded: it is not retained, logged,
serialized, read from environment variables, or sent over a network. The alias
table remains local. Callers remain responsible for UMLS and SNOMED CT license
terms. CPT remains outside the in-process API and no CPT content is shipped.

## Export FHIR R4 and OMOP CDM v5.4

```python
from openmed.clinical.exporters import to_fhir, to_omop

bundle = to_fhir(
    grounded,
    subject_reference="Patient/local-subject",
    document_id="local-document",
)

tables = to_omop(
    grounded,
    document_text="example finding",
    document_id="local-document",
    person_id="local-subject",
    concept_resolver={("UMLS", "C0000000"): 123456},
)
```

`to_fhir()` emits Condition, MedicationStatement, Observation, and Procedure
resources. Iterable input produces a transaction Bundle. Assertion context is
preserved in resource status: refuted and hypothetical findings are never
silently emitted as present findings, and non-patient findings are excluded.
Grounding confidence remains on the source `GroundedSpan`; implementation-only
score fields and unresolved custom extensions are excluded from strict R4
resources. Vocabulary version provenance remains in the standard
`Coding.version` field.

`to_omop()` emits `condition_occurrence`, `drug_exposure`, `measurement`, and
`procedure_occurrence`, with supporting person, visit, note, NOTE_NLP, concept,
and source-to-concept rows. Supply an Athena resolver to populate standard
`*_concept_id` values. An unmapped source remains `concept_id=0` with its source
system, code, and text retained; OpenMed never invents an Athena concept ID.
`achilles_smoke_check()` provides a deterministic offline structural preflight.
A full OHDSI ACHILLES run still requires a deployed CDM database.

CI runs the committed synthetic round trip through the official HL7 FHIR R4
validator and the offline OMOP smoke checks. It also builds a wheel and rejects
restricted UMLS, SNOMED, or CPT vocabulary data paths.

## Run incremental OMOP loads

The OMOP writers use idempotent append mode by default. A rerun with the same
deterministic rows does not create duplicates and does not remove data already
in the target. Use replace-by-note mode when the incoming batch is the complete,
authoritative extraction result for each source note hash in that batch:

```python
from openmed.interop.omop import load_grounded_jsonl, write_omop_sqlite


def refresh_quality_profiles(event):
    quality_queue.enqueue(event.changed_note_hashes)


def refresh_cohort_queries(event):
    cohort_cache.invalidate(event.changed_note_hashes)


tables = load_grounded_jsonl("grounded.jsonl", mode="replace_by_note")
write_omop_sqlite(
    tables,
    "local-cdm.sqlite",
    mode="replace_by_note",
    downstream_consumers=(refresh_quality_profiles, refresh_cohort_queries),
)
```

Replace-by-note removes and reloads NOTE, NOTE_NLP, domain, and
SOURCE_TO_CONCEPT_MAP rows for the incoming hashes. It preserves rows belonging
to omitted note hashes and shared PERSON, VISIT_OCCURRENCE, and CONCEPT rows. An
incoming note with fewer spans therefore removes stale derived rows without
disturbing unrelated notes. The same behavior is available for DuckDB and
Parquet and from the CLI with `openmed omop load --mode replace-by-note`.

When note text is unchanged but extraction results are regenerated, its
content-derived hash remains stable. If a caller supplies `source_note_hash`, it
must be a stable, privacy-safe digest for that logical source note; do not use a
plaintext document or patient identifier. Every downstream callback runs only
after its writer succeeds and receives an `OmopLoadEvent` containing the mode,
changed hashes, and incoming row counts. It receives no note text, lexical
variants, document identifiers, or patient identifiers. Callback failures occur
after persistence and should be retried independently by the integration.

## Deploy and validate the Postgres table subset

`emit_postgres_ddl()` returns one deterministic PostgreSQL script covering every
table owned by the grounded-note loader. The script contains schema only: it
does not bundle vocabulary rows or require a Java or OHDSI runtime. Apply it
with the PostgreSQL deployment mechanism used by your environment before
loading rows:

```python
from pathlib import Path

from openmed.interop.omop import emit_postgres_ddl

Path("openmed-omop-v5.4.sql").write_text(emit_postgres_ddl(), encoding="utf-8")
```

After loading, `validate_omop_database_report(connection)` checks concept
references, NOTE_NLP offsets, and reciprocal NOTE_NLP-to-domain reachability.
Its serializable form contains only a total plus counts by table and reason, so
it is suitable for logs and deployment diagnostics without exposing note text,
lexical variants, document identifiers, patient identifiers, or row keys:

```python
from openmed.interop.omop import validate_omop_database_report

report = validate_omop_database_report(connection)
assert report.is_valid, report.to_dict()
```

## Linking evaluation

`evaluate_medmentions_st21pv()` accepts a caller-created JSONL projection with
`mention`/`text` and `cui`/`expected_cui`, plus a callback backed by the
caller's licensed local UMLS index. It emits a `BenchmarkReport` containing
top-1 accuracy, top-k accuracy, abstention rate, the required `0.55` floor, and
the `0.70` target. The report contains aggregate metrics and license metadata,
not mention text or controlled terminology rows.

```python
from openmed.eval.medmentions_linking import evaluate_medmentions_st21pv

report = evaluate_medmentions_st21pv(
    "/licensed/local/medmentions-st21pv.jsonl",
    provider=local_umls_candidates,
)
assert report.metrics["passed"]
```

No MedMentions or UMLS corpus is committed, and the synthetic test score is not
presented as a MedMentions result. Run the evaluator in the licensed environment
to create the authoritative report. The documented path from the sparse
matcher baseline toward the target is to union a local SapBERT-class dense
index with sparse candidates, apply the context-aware reranker, and calibrate
on a held-out split. NCBI Disease and BC5CDR remain available through the public
biomedical NER evaluation adapter.
