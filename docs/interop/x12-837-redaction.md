# X12 837 Claim Redaction

OpenMed can parse professional and institutional X12 837 claim text and redact
structured subscriber, patient, and provider identifiers without changing the
interchange envelope or segment layout. Processing is local and does not call
external services.

```python
from openmed.interop.x12_837 import redact_x12_837

result = redact_x12_837("synthetic-claim.837")
redacted_edi = result.deidentified_text
```

The path must contain an 837 transaction framed by `ISA`/`IEA`, `GS`/`GE`, and
`ST`/`SE`. The parser derives the element, repetition, component, and segment
separators from the fixed-width ISA header. Re-serialization retains those
separators, segment order, envelope/control values, and whitespace between
segments.

## Redacted loops

The redactor uses the `NM101` entity identifier to recognize subscriber (`IL`),
patient (`QC`), and common provider loops. Within those loops it replaces:

- name components and identifiers in `NM1`;
- additional names in `N2`;
- street, city, state, and postal elements in `N3`/`N4`;
- date of birth in `DMG`; and
- loop-local identifiers in `REF`.

Qualifiers such as `NM102`, `NM108`, `DMG01`, and `REF01` remain intact. Claim,
service, hierarchy, and envelope segments are not modified. Composite and
repetition separators inside a redacted element are preserved:

```python
result = redact_x12_837(message_text, replacement="MASKED")
```

Replacement tokens may not contain any separator declared by the interchange.

## Raw-text provenance

Every parsed element records a half-open character span into the original EDI
text. Redaction results carry an element-level offset map between the raw input
and serialized output:

```python
redaction = result.redactions[0]
output_span = result.offset_map.source_to_output(*redaction.source_span)
assert result.offset_map.output_to_source(*output_span) == redaction.source_span
```

This mapping is deliberately element-granular because replacements can change
length. Each redaction record includes the source/output spans, segment and
element coordinates, entity type, replacement, and a SHA-256 digest of the
original value. It does not copy the raw PHI value into the audit record.

## Scope and safety

This adapter is a structural de-identification helper, not a full X12 validator.
It does not implement companion-guide rules, claim adjudication, 835 remittance,
or arbitrary transaction sets. Validate the resulting interchange with the
receiver's normal X12 tooling before production exchange.

All examples and tests use deliberately synthetic names, identifiers,
addresses, dates, providers, and claim values; they contain no real claims.
