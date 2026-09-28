# Longitudinal Journey contracts

OpenMed v3 uses six immutable records as the boundary between ingestion,
clinical extraction, reconciliation, applications, and storage:

- `ClinicalArtifact` identifies source or derived bytes without embedding them.
- `EvidenceLocator` points to exact text, structured, document, DICOM, or table
  coordinates in an artifact.
- `ClinicalFact` records one typed value with evidence and a derivation digest.
- `ConflictSet` groups facts that cannot all be treated as current truth.
- `ResolutionEvent` appends a policy or human review decision without rewriting
  the underlying facts.
- `DatasetSnapshot` pins the inputs, schemas, files, components, licenses, and
  splits of a governed dataset build.

These records are assistive data contracts. They do not establish clinical
truth, clinical validation, or authorization for autonomous patient-care
actions.

## Deterministic identifiers and digests

Identifiers are opaque and carry only a controlled record-kind prefix. Use
`new_opaque_id()` for an unlinkable random identifier or
`derived_opaque_id()` when identical canonical inputs must produce the same
identifier.

```python
from openmed.clinical import (
    ClinicalArtifact,
    derived_opaque_id,
    new_opaque_id,
    sha256_digest,
)

source_id = new_opaque_id("source")
artifact = ClinicalArtifact(
    artifact_id=derived_opaque_id("artifact", source_id, "synthetic-note-1"),
    artifact_type="clinical_note",
    media_type="text/plain",
    content_hash=sha256_digest(b"Synthetic fixture content"),
    byte_size=25,
    source_id=source_id,
    recorded_at="2026-01-02T03:04:05Z",
)

assert artifact.to_json() == ClinicalArtifact.from_json(artifact.to_json()).to_json()
```

`canonical_json()` sorts object keys, rejects non-finite numbers, and emits a
compact byte-stable representation. `canonical_digest()` hashes that exact
representation. `compute_derivation_hash()` additionally binds a component,
component version, sorted input identifiers, and JSON configuration.

## Evidence coordinate types

An `EvidenceLocator` accepts one of seven explicit coordinate shapes:

| Type | Required coordinates |
| --- | --- |
| `text_span` | zero-based half-open `start` and `end` |
| `json_pointer` | RFC 6901-style `pointer` |
| `message_field` | bounded structured `path` such as `PID.3.1` |
| `document_path` | indexed element path with optional one-based section |
| `page_box` | one-based `page`, `[x0, y0, x1, y1]`, and coordinate space |
| `dicom_element` | study, series, instance UIDs and canonical tag |
| `table_cell` | one-based row and column with an optional sheet |

Invalid spans, malformed pointer escapes, inverted or out-of-bounds boxes,
malformed DICOM identifiers, and zero-based table cells fail before a record is
created.

`document_path` was added in locator schema `1.1.0`; a producer using that
coordinate must declare `schema_version="1.1.0"` or a later same-major version.

## Forward-compatible reads

The current contract version is `1.0.0`. Readers accept newer minor and patch
versions with the same major version. Unknown additive top-level fields are
stored immutably and written back unchanged:

```python
payload = artifact.to_dict()
payload["schema_version"] = "1.2.0"
payload["future_review_state"] = {"code": "synthetic-state"}

restored = ClinicalArtifact.from_dict(payload)
assert restored.to_dict()["future_review_state"] == {"code": "synthetic-state"}
```

An unsupported major version raises `JourneySchemaVersionError`. This is a
fail-closed boundary: OpenMed does not guess how to reinterpret a breaking
schema.

## Derivation graph validation

Use `validate_contract_graph()` before committing a group of records. It
checks:

- globally unique opaque identifiers;
- artifact, evidence, fact, conflict, resolution, and snapshot references;
- conflict membership for selected or rejected facts;
- same-conflict resolution supersession; and
- cycles in artifact, fact, resolution, and snapshot derivations.

The returned diagnostics contain counts only. Validation errors identify the
record category but never echo identifiers or clinical values.

## JSON Schemas

`load_journey_schema(name)` and `load_all_journey_schemas()` expose bundled
Draft 2020-12 schemas for all six records. Schemas allow additive top-level
fields within major version 1 but keep every currently defined field and nested
evidence shape strict.

The schemas ship in `openmed/core/schemas/json/` and require no network access.
