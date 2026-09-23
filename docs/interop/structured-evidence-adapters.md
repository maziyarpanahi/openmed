# Structured source evidence adapters

OpenMed can turn text, existing document extraction results, FHIR R4 JSON,
HL7 v2 messages, CDA R2 XML, CSV, and XLSX sources into one immutable
`ClinicalArtifact` plus exact, modality-aware `EvidenceLocator` records. The
adapters preserve source-byte identity, parser provenance, coordinate
convention, and normalization metadata without putting source values in the
result.

Every adapter runs locally after optional dependencies are installed. None of
the adapters makes a network request or requires a hosted parser, terminology
service, or telemetry endpoint.

## Result contract

All adapters implement `StructuredEvidenceAdapter` and return a typed
`StoreResult`. Success contains a `StructuredEvidenceResult` with:

- a deterministic result identifier;
- one immutable `ClinicalArtifact`;
- zero or more `EvidenceLocator` records bound to that artifact;
- the exact source-byte SHA-256 digest and byte count;
- source format and version;
- parser identifier and version;
- an explicit coordinate convention;
- a value-free normalization transform;
- schema version `1.0.0` and compatibility policy `same_major`.

The result never contains field or cell values. `resolve()` is a separate
operation over source content still held by the caller. Its `StoreResult.value`
is excluded from representation, but applications must still avoid logging,
tracing, caching, or persisting resolved values without the appropriate
identified-data controls. Every locator carries the source digest and refuses
to resolve against different bytes.

```python
from openmed.interop.ingest import (
    EvidenceAdapterContext,
    FHIRR4EvidenceAdapter,
)

context = EvidenceAdapterContext(
    source_id="source_0123456789abcdef",
    subject_id="patient_0123456789abcdef",
    encounter_id="encounter_0123456789abcdef",
    recorded_at="2026-01-02T03:04:05Z",
)
source = b'{"resourceType":"Observation","status":"final"}'

adapter = FHIRR4EvidenceAdapter(source_version="4.0.1")
adapted = adapter.adapt(source, context)
assert adapted.ok
assert adapted.value is not None
assert adapted.value.source_digest.startswith("sha256:")

status_locator = next(
    item
    for item in adapted.value.locators
    if item.location == {"pointer": "/status"}
)
status = adapter.resolve(source, status_locator)
assert status.value == "final"
```

The source, subject, and encounter identifiers in `EvidenceAdapterContext` are
opaque keys. Do not put a medical record number, name, account number, or other
patient value in them.

## Coordinate conventions

| Source | Locator | Convention |
| --- | --- | --- |
| UTF-8 text | `text_span` | zero-based, half-open Unicode code-point offsets |
| Existing document result | `text_span` or `page_box` | normalized spans plus explicit source page geometry |
| FHIR R4 JSON | `json_pointer` | RFC 6901 pointer into the parsed resource |
| HL7 v2 | `message_field` | segment, field, component, and segment occurrence |
| CDA R2 | `document_path` | local-name XML path with one-based sibling and section indexes |
| CSV | `table_cell` | one-based logical row and column |
| XLSX | `table_cell` | opaque sheet order plus one-based row and column |

The Journey `EvidenceLocator` contract now includes `document_path` for CDA and
other indexed document trees. Its path addresses element text, tail text, or an
attribute without embedding the value:

```text
/ClinicalDocument[1]/component[1]/section[1]/text[1]/text()
```

Paths are bounded and syntactically validated. An optional `section` field is a
one-based structural coordinate, not a section title.
`document_path` locators declare Journey locator schema `1.1.0`; all previously
defined coordinate types retain `1.0.0` semantics.

## Text and existing document results

`TextEvidenceAdapter` accepts `str` or UTF-8 bytes and performs no Unicode or
newline normalization. A non-empty input gets one full-source `text_span`.
Offsets count Python Unicode code points, not UTF-8 bytes. The result separately
records the exact byte digest and byte length.

`ExistingDocumentEvidenceAdapter` accepts an `ExtractedDocument` together with
the original immutable source bytes:

```python
from openmed.interop.ingest import (
    ExistingDocumentEvidenceAdapter,
    ExistingDocumentInput,
)

source = ExistingDocumentInput(
    document=extracted_document,
    source_bytes=original_bytes,
    bbox_coordinate_space="points",
)
result = ExistingDocumentEvidenceAdapter().adapt(source, context)
```

Normalized spans remain available in each locator's transform. Spans with a
bounding box become `page_box` coordinates. The caller must state whether those
boxes are normalized, pixels, or points; the adapter refuses to guess. Missing,
overlapping, inverted, or out-of-range spans produce a typed conflict
quarantine.

The adapter records a digest of the normalized text, never the text itself.
Per-span metadata from the existing document is intentionally not copied into
the public result because it may contain source paths or application-defined
values.

## FHIR R4 JSON pointers

`FHIRR4EvidenceAdapter` emits one pointer for every scalar in a resource,
including primitive-extension fields and array members. It rejects duplicate
JSON keys, non-finite numbers, a missing string `resourceType`, unsafe dynamic
property names, malformed JSON, and unsupported versions.

FHIR resources do not reliably declare the FHIR release in each JSON payload,
so the caller supplies `source_version`. The adapter currently accepts `4.0`
and `4.0.1`. Any other version is returned as `unsupported` with a value-free
quarantine; it is not interpreted as R4.

Parsing does not validate profiles, terminology, references, or clinical
semantics. Use the existing FHIR validation surfaces separately when those
checks are required.

## HL7 v2 fields and components

`HL7V2EvidenceAdapter` reads separators from `MSH-1` and `MSH-2`, preserves the
original segment framing, and creates paths such as:

```text
PID.3.2[1]
```

This means the second component of PID field 3 in the first PID segment. The
path stays stable when a message uses non-default field or component
delimiters. The adapter accepts versions 2.3 through 2.8.2 from `MSH-12`.

The current coordinate contract does not include repetition or subcomponent
indexes. A populated field containing either dimension is quarantined as an
ambiguous conversion instead of collapsing several values into one coordinate.
This is a deliberate abstention boundary; a later schema can add those
dimensions without changing the meaning of existing paths.

## CDA R2 section paths

`CDAEvidenceAdapter` accepts CDA R2 XML bytes or text. It requires a
`ClinicalDocument` in the CDA namespace, rejects DTD and entity declarations,
and emits indexed paths for non-whitespace element text, tail text, and
attributes. Locators within a section carry its one-based structural index.

Paths use local XML names and one-based sibling indexes. The adapter permits
the CDA, empty, and XML Schema Instance namespaces and rejects other element or
attribute namespaces when it cannot prove an unambiguous coordinate. It does
not run schema, template, or terminology validation.

## CSV and XLSX cells

`DelimitedTableEvidenceAdapter` uses Python's strict CSV parser over UTF-8 or
UTF-8-with-BOM input. Logical row and column indexes survive CRLF/LF variation,
Unicode, escaped quotes, ragged rows, and quoted embedded newlines. Every cell,
including an empty cell, receives a coordinate. A custom one-character
delimiter may be supplied; quote and newline characters are rejected as
delimiters.

`XLSXEvidenceAdapter` loads a workbook locally in read-only, formula-preserving
mode with external workbook links disabled. Non-empty cells receive coordinates
such as `sheet_0002`, row 3, column 4. Real worksheet titles are not exposed,
because a title can itself contain identifying data. Formulas remain formulas;
the adapter does not use cached calculation results or execute workbook code.

XLSX support is optional and lazy. Install the multimodal extra when
`openpyxl` is unavailable:

```bash
pip install "openmed[multimodal]"
```

## Quarantine and bounded parsing

An unsafe, malformed, ambiguous, partial, or unsupported source returns a
`StructuredEvidenceQuarantine`. It contains only source digest and byte count,
format/version, parser provenance, controlled classification/reason codes,
counts, and timestamp. It contains no exception text or source value.

| Condition | Typed state | Example code |
| --- | --- | --- |
| input exceeds the byte limit | `denied` | `source_limit_exceeded` |
| field count exceeds the bound | `partial` | `field_limit_exceeded` |
| coordinate cannot be represented exactly | `conflict` | `hl7_coordinate_ambiguous` |
| source version is not supported | `unsupported` | `fhir_version_unsupported` |
| unsafe XML declaration | `denied` | `cda_declaration_unsafe` |
| malformed source | `failure` | `csv_malformed` |

The default bounds are 64 MiB per source and 100,000 fields. Callers can lower
them per adapter. A quarantine is never converted to a successful artifact and
must not be promoted merely because a downstream parser can produce partial
values.

## Schemas and local verification

Adapter result and quarantine records have bundled JSON Schemas:

```python
from openmed.interop.ingest import load_all_evidence_adapter_schemas

schemas = load_all_evidence_adapter_schemas()
assert schemas["result"]["schema_version"] == 1
```

The synthetic suite includes a cross-format golden journey plus unit and
property tests for exact coordinate resolution, idempotency, Unicode, newlines,
custom delimiters, quoted cells, sheet privacy, unsafe XML, malformed input,
ambiguous coordinates, unsupported versions, parser limits, schema round-trip,
and value-free quarantine:

```bash
pytest \
  tests/unit/interop/ingest/test_evidence_adapters.py \
  tests/unit/clinical/test_journey_contracts.py -q
```

These adapters prove coordinate and serialization behavior over synthetic
sources. They do not prove clinical completeness, format conformance, identity
correctness, or suitability for autonomous care actions.
