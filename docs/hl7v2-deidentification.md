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
| `PID` | `3`, `5`, `6`, `7`, `9`, `11`, `13`, `14`, `19`, `20`, `21` |
| `MRG` | `1`–`7` |
| `PV1` | `19` |
| `PD1` | `3` |
| `NK1` | `2`, `4`, `5`, `13` |
| `GT1` | `3`, `5`, `6`, `12`, `13` |
| `IN1` | `16`, `18`, `19`, `36` |
| `IN2` | `1`, `2` |
| `OBX` | `5` for `TX` and `FT` values |
| `NTE` | `3` |

Unknown segments pass through unchanged unless you configure a rule for one of
their fields.

The defaults surrogate maternal names and aliases (`PID-6`, `PID-9`) and prior
names (`MRG-7`); hash business phones, licenses and maternal identifiers
(`PID-14`, `PID-20`, `PID-21`), prior identifiers/accounts (`MRG-1`–`MRG-6`),
and visit identifiers (`PV1-19`). XAD addresses in `PID-11`, `NK1-4`, `GT1-5`
and `IN1-19` use component-specific street, secondary address, city, state,
postal-code and country surrogates. Postal-code punctuation and width, empty
components, repeats, address-type and representation codes are retained.

To make pass-through visible without logging message content, supply an empty
`coverage_report={}` mapping to `redact_hl7v2`. The populated
`unmapped_fields` rows contain only segment indexes/codes, field positions and
lengths; `unmapped_field_count` is the total. The inventory includes text and
coded fields, including unknown Z segments and OBX types not covered by the
default text rule. It does **not** classify unmapped values as safe. Review
local profiles and add rules before releasing a message.

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
