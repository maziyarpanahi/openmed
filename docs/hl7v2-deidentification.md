# HL7 v2 De-identification

OpenMed can redact common PHI-bearing fields in HL7 v2.x pipe-delimited
messages while preserving segment order, delimiters, repetitions, components,
and subcomponents.

The helper is local and mechanical. It does not run an MLLP listener, validate
full conformance profiles, or call network services.

```python
from openmed.interop.hl7v2 import redact_hl7v2

redacted = redact_hl7v2("synthetic_oru.hl7", date_shift_days=31)
```

## Supported Scope

The default field map is intended for common ADT, ORU, and ORM message flows.
It parses the delimiter set from `MSH-1` and `MSH-2`, then applies rules keyed
by segment and field position.

Structured fields use these actions:

- `clear`: remove the field value.
- `hash`: replace each leaf value with a deterministic hash token while
  preserving HL7 delimiters.
- `surrogate`: generate label-aware fake values while preserving repetitions,
  components, and subcomponents.
- `date-shift`: shift every configured date by one consistent offset within
  the message.

Free-text fields use OpenMed PII de-identification:

- `OBX-5` when `OBX-2` is `TX` or `FT`.
- `NTE-3`.

## Default Field Map

The default map includes common direct identifiers in these segments:

| Segment | Fields |
| --- | --- |
| `PID` | `3`, `5`, `7`, `11`, `13`, `19` |
| `PD1` | `3` |
| `NK1` | `2`, `4`, `5`, `13` |
| `GT1` | `3`, `5`, `6`, `12`, `13` |
| `IN1` | `16`, `18`, `19`, `36` |
| `IN2` | `1`, `2` |
| `OBX` | `5` for `TX` and `FT` values |
| `NTE` | `3` |

Unknown segments pass through unchanged unless you configure a rule for one of
their fields.

## Extending Rules

Pass a replacement field map keyed by either tuples or strings:

```python
from openmed.interop.hl7v2 import DEFAULT_FIELD_MAP, HL7FieldRule, redact_hl7v2

field_map = {
    **DEFAULT_FIELD_MAP,
    "ZNT-2": HL7FieldRule("redact_text"),
    ("ZID", 4): HL7FieldRule("hash", label="ID_NUM"),
}

redacted = redact_hl7v2(message_text, field_map=field_map)
```

For offline tests, pass a `deidentifier` callable. The callable should return
either a string or an object with `deidentified_text`.

For downstream clinical NLP or review, use the [HL7 v2 narrative
extractor](./interop/hl7v2-narrative-extraction.md) to render de-identified
flat or sectioned text with final-text offsets back to segment fields.
