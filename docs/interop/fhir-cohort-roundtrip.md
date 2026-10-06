# FHIR Journey and cohort-definition round trips

OpenMed can project immutable Journey facts and resolution events to a
deterministic FHIR R4 collection Bundle, then reconstruct the exact Journey
records. The same interoperability layer wraps the existing cohort DSL in a
versioned exchange envelope with source-snapshot and criterion custody.

Both paths are local and offline. They do not contact a terminology server,
download a vocabulary, run a hosted validator, or make a clinical decision.
Every unsupported surface is returned as an explicit loss with a typed
`StoreState`; partial conversion never becomes an apparent success.

## Export Journey facts to FHIR R4

```python
from openmed.interop.fhir import export_journey_to_fhir

result = export_journey_to_fhir(
    facts,
    source_snapshot=dataset_snapshot,
    resolutions=resolution_events,
    events=journey_page.events,
)

if result.ok:
    bundle = result.value.bundle
else:
    # PARTIAL and UNSUPPORTED results may retain a value for loss review.
    losses = () if result.value is None else result.value.losses
```

Supported clinical fact types map to standard R4 resources:

| Journey fact type | FHIR R4 resource |
| --- | --- |
| `condition`, `diagnosis`, `problem` | `Condition` |
| `observation`, `laboratory`, `measurement`, `vital`, `social_determinant` | `Observation` |
| `medication`, `drug` | `MedicationStatement` |
| `procedure` | `Procedure` |

Standard fields provide ordinary FHIR usability. Openly identified custom
extensions carry the canonical clinical fact and, when supplied, the complete
materialized Journey event with its evidence paths, conflicts, resolutions,
canonical records, and review state. Each extension records its schema
version, compatibility policy, and canonical hash. That is what makes an
OpenMed round trip exact even when the source contract has more detail than a
standard R4 field. It is not encryption: the Bundle can contain clinical
values and must be protected as clinical data. Applications must not place
Bundle payloads in logs, metrics, traces, or exception messages.

The Bundle also includes:

- deterministic logical IDs and `urn:uuid` full URLs;
- one opaque Patient carrier per Journey subject;
- evidence, fact, derivation, and canonical-hash identifiers;
- FHIR `Provenance` resources for resolution events; and
- the exact `DatasetSnapshot` as an explicit Bundle extension.

Before success, OpenMed runs its offline R4 structural validator and Bundle
reference-integrity checker. A snapshot that does not cover every input fact,
a subject appearing in multiple declared dataset splits, a dangling
resolution target, duplicate identity, or incompatible snapshot fails closed.

## Import an OpenMed FHIR Bundle

```python
from openmed.interop.fhir import import_journey_from_fhir

result = import_journey_from_fhir(
    bundle,
    expected_snapshot=dataset_snapshot,
    strict=True,
)
```

Import verifies the extension digest, resource type, deterministic logical ID,
canonical record hash, source snapshot, and Bundle references. A supported FHIR
resource without the canonical extension is not guessed into a Journey fact;
it is recorded as `canonical_payload_missing`. Unknown resources appear as
`resource_type_unsupported`. With `strict=True`, any loss returns
`StoreState.UNSUPPORTED`; otherwise a usable subset returns
`StoreState.PARTIAL` with the complete loss report.

## Exchange cohort definitions

The canonical format is the existing `PhenotypeDefinition` schema. It already
supports named concept sets, descendant intent, absolute and relative temporal
windows, occurrence bounds, assertion filters, and recursive `and`, `or`, and
`not` expressions. The exchange envelope adds custody without changing the
language:

```python
from openmed.structured.cohort import (
    CohortSourceSnapshot,
    export_cohort_definition,
    import_cohort_definition,
)

snapshot = CohortSourceSnapshot(
    snapshot_id="snapshot_example00000001",
    digest="sha256:" + "a" * 64,
    schema_version="omop-5.4-local",
    license_tags=("synthetic",),
)
exported = export_cohort_definition(definition, source_snapshot=snapshot)
restored = import_cohort_definition(exported.value.to_json_bytes())
```

The envelope is byte-stable and records the definition digest, source snapshot,
criterion-to-concept-set evidence mapping, schema version, compatibility
policy, and explicit losses. It stores concept identifiers and caller-provided
snapshot references, never vocabulary distribution content. Setting
`bundled_vocabulary=True` is rejected.

## Optional service adapter

`CohortDefinitionServiceBridge` connects an explicitly supplied adapter without
adding a core dependency. The adapter may be an executable using JSON over
stdin/stdout or an application-injected callable:

```python
from openmed.interop.bridges import CohortDefinitionServiceBridge

bridge = CohortDefinitionServiceBridge(
    command=("/opt/local/bin/cohort-definition-adapter",),
)
result = bridge.export(
    exported.value,
    target_format="open-service.v1",
    strict=True,
)
```

Executable adapters run without a shell, receive a minimal environment, have
stderr discarded, and must return bounded JSON whose direction and source
digest match the request. The bridge returns `success`, `partial`, `unknown`,
`conflict`, `unsupported`, `denied`, or `failure` exactly as declared; a
loss-bearing response cannot remain `success`.

## Schemas and evidence

Persisted export contracts are validated by
`fhir_journey_roundtrip.schema.json`; cohort envelopes use
`cohort_definition_exchange.schema.json`. Synthetic unit and property tests
cover exact fact/status/evidence preservation, unsupported fields, tamper
detection, referential integrity, split isolation, and vocabulary-license
gates. The integration test resolves the same fabricated cohort before and
after byte-stable envelope round-trip and requires identical membership and
criterion evidence.

These checks are interoperability evidence, not clinical validation,
regulatory certification, or authorization for autonomous patient action.

## Offline OHDSI Circe mapping

`export_circe_expression()` and `import_circe_expression()` provide a
pure-Python mapper over the existing exchange envelope. This issue's surface
is explicitly Python-only. It adds no Swift runtime, WebAPI calls, OHDSI cohort
execution, SQL generation, or vocabulary download. The declared wire subset
follows [OHDSI circe-be's cohort-definition classes](https://github.com/OHDSI/circe-be/tree/43c407c9ec514d345c35cb848a0cc16e8dac79e8/src/main/java/org/ohdsi/circe/cohortdefinition)
and uses mapping version `openmed.cohort.circe.v1`.

```python
from openmed.structured.cohort import (
    export_circe_expression,
    import_circe_expression,
)

# The native DSL has no domain or primary-event field. Supply these explicitly.
converted = export_circe_expression(
    exported.value,
    criterion_domains={
        "has-diabetes": "ConditionOccurrence",
        "has-metformin": "DrugExposure",
    },
    primary_criterion_id="has-diabetes",
    strict=False,
)
protected_circe_json = converted.value.expression
restored = import_circe_expression(
    protected_circe_json,
    source_snapshot=snapshot,
    expected_expression_digest=converted.value.target_digest,
    strict=False,
)
report = restored.value.to_report()  # codes, field offsets, counts and digests
candidate = restored.value.exchange  # may be None; always review losses first
```

Supported mapping behavior:

| Circe surface | Native mapping |
| --- | --- |
| Concept sets | Positive concept IDs and `VOCABULARY_ID` pass through; uniform descendant intent is preserved. No names or vocabulary distribution are bundled. |
| Domains | `ConditionOccurrence`, `DrugExposure`, `Measurement`, `Observation`, and `ProcedureOccurrence`; explicit caller domain bindings on export. |
| Primary criteria | OR of primary event predicates; a single primary supplies the relative-window anchor. |
| Additional criteria and inclusion rules | ALL of the primary predicate, additional group and each inclusion-rule group. |
| Occurrence | Circe types 0 (exactly), 1 (at most, including zero) and 2 (at least). Native inclusive count ranges export as an ALL of lower/upper count predicates. |
| Absolute start dates | `OccurrenceStartDate` with `eq`, `gte`, `lte`, or inclusive `bt` ranges. |
| Relative start windows | Inclusive finite day bounds spanning the single primary's start date. |
| Groups | ALL and ANY; AT_LEAST with a positive feasible count expands into exact native AND/OR combinations with unique criterion IDs. AT_MOST(0) maps boolean negation. |

These are **candidate patient predicates**, not an OHDSI execution-equivalence
claim. Circe constrains domains and observation periods; the native resolver
unions its supported event domains and has no observation-period predicate.
Import reports `domain_scope_not_enforced`, `vocabulary_filter_added`,
`assertion_default_added` (native affirmed-only default), and
`observation_period_membership_not_enforced`. Export reports the reverse domain
and vocabulary differences, observation-period membership, native metadata
loss, and every assertion-filter axis, including the native affirmed default.
The source envelope remains available on export; import generates new native
criterion/concept-set IDs and a caller-chosen definition ID rather than copying
source names. Evidence bindings are rebuilt against the imported definition's
digest.

Relative native counts can combine matches across different anchor events;
Circe applies inclusion predicates to an individual index event. Both directions
therefore report `index_event_correlation_changed` when mapping a finite
relative window. Export also reports non-primary anchors and any added primary
requirement. Choosing a primary that is mandatory in the native expression
avoids adding that requirement. Synthetic membership round trips cover
single-anchor relative windows, inclusive count ranges, absolute dates, ANY
composition and negation; they do not establish parity on real OMOP data.

Loss handling is explicit:

- Both APIs default to `strict=True`: any loss returns `StoreState.UNSUPPORTED`
  while retaining a conversion for review. `strict=False` returns `PARTIAL` when
  a native candidate exists. A loss-bearing result never reports success.
- Demographic criteria, era domains/collapse settings, censoring, observation
  windows, custom end strategies, event limits, distinct/count-column options,
  visit restrictions, end-based anchors and unsupported fields each receive
  loss records. Unknown field names are represented by deterministic sorted
  field offsets, because a key can itself contain PHI.
- Exclusions/mapped concepts or mixed vocabulary/descendant intent are reported;
  their concept set is unavailable for native execution. An unrepresentable
  primary or boolean group (including a demographic child) makes the native
  exchange `None`; a sibling subset is never substituted for that group. Loss
  collection continues through the other groups and rules.
- Unrepresentable one-sided/shifted relative windows are reported rather than
  guessed. Other unsupported restrictions may yield a broader or narrower
  reviewable candidate, always with losses and an unsuccessful state.

Inputs are bounded to 1 MiB of UTF-8 JSON, depth 32 and 20,000 JSON nodes.
These bounds apply to mappings as well as strings/bytes, including cyclic
mappings. Threshold expansion and native output are limited to 256 generated
criteria. Exceeding a bound raises `CirceLimitError`; malformed supported
fields, duplicate JSON keys, invalid references and missing per-concept
vocabulary IDs raise `CirceInputError`. Both errors contain controlled codes
only. A supplied expression digest mismatch returns `StoreState.CONFLICT`.

A conversion binds the canonical Circe JSON digest and the complete existing
exchange-envelope digest in direction order, and exposes the native definition
digest in `to_report()`. Import losses are persisted in the envelope too.
Pre-existing export-envelope losses appear as `source_conversion_loss` with
source-loss offsets; their original paths remain protected in that envelope.
`expression` returns a fresh copy, so editing it cannot mutate the conversion's
custody. Keep expressions and envelopes protected as clinical/configuration
data: only `to_report()` is intended for value-free diagnostics; neither target
payload belongs in logs or audit artifacts. The mapper does not grant cohort
execution or clinical-action authority.
