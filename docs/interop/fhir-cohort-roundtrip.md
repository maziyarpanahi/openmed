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
