# V3 five-source golden Journey

OpenMed ships one deterministic, synthetic patient Journey that exercises the
v3 critical path through real library contracts. It provides a reviewable
release gate for cross-stage changes without requiring network access, a model
download, restricted clinical data, or a hosted service.

The five sources are a plain-text note, a FHIR R4 resource, an HL7 v2 message,
a delimited table, and DICOM-SR-derived evidence. The scenario covers duplicate
facts, negation, uncertainty, a partial date, a unit correction, ambiguous
identity, a resolved conflict, consent withdrawal, OMOP projection loss,
cohort membership, and a metadata-only governed dataset.

## Run the example

```bash
.venv/bin/python examples/v3_golden_journey.py
```

The example executes the local ingestion and Journey stores, identity resolver,
OMOP projector, saved cohort, dataset builder, and Journey resource policy. Its
output is a compact receipt derived from those components; it is not a set of
simulated logs.

## Check golden drift

```bash
.venv/bin/python scripts/regenerate_v3_golden_journey.py
```

Check mode is the default and never changes the committed fixture. A mismatch
prints stable JSON Lines whose `path` values use JSON Pointer escaping.

Golden output can only be changed by explicitly requesting write mode:

```bash
.venv/bin/python scripts/regenerate_v3_golden_journey.py --write
```

Review the semantic diff, the scenario provenance, and every changed digest
before committing an update. The input fixture declares `synthetic: true`; the
generated report repeats that classification and does not copy source payloads
into ordinary output.

## Frozen contract

The fixture pins:

- exact synthetic source bytes and source order;
- schema compatibility, component, model, policy, and vocabulary versions;
- evidence coordinates, artifacts, facts, conflicts, resolutions, and Journey
  events;
- correction and conversion lineage;
- OMOP rows, mapping decisions, and explicit information loss;
- cohort membership and a governed dataset manifest;
- success, empty, partial, unknown, conflict, unsupported, denied, and failure
  states returned through the public resource contract.

The report schema is bundled as `golden_journey.schema.json`. A same-major
compatibility policy applies. Non-success states remain typed and cannot be
reinterpreted as successful output.

This scenario is conformance evidence, not clinical validation. It must not
trigger diagnosis, treatment, enrollment, outreach, ordering, or another
patient-care action.
