# Synthetic HL7 v2 messages with PHI truth

`generate_hl7v2_messages` renders seeded ADT, ORU and ORM messages from
synthetic identities and returns a truth manifest for every PHI value it
places. `evaluate_hl7v2_leakage` runs `redact_hl7v2()` over those messages
and reports which truth values survive, by field position and label, as
counts only. Everything runs offline; no real patient data is read or
bundled.

```python
from openmed.interop.hl7v2 import HL7V2Encoding
from openmed.training.synthetic import (
    HL7V2SynthesisConfig,
    evaluate_hl7v2_leakage,
    generate_hl7v2_messages,
)

config = HL7V2SynthesisConfig(
    encoding=HL7V2Encoding(
        field="#", component="$", repetition="*", escape="?", subcomponent="%"
    ),
    max_repetitions=3,
    escape_probability=0.5,
)
messages = generate_hl7v2_messages(50, seed=3884, config=config)

first = messages[0]
for truth in first.truth[:3]:
    assert first.text[truth.start : truth.end] == truth.raw

report = evaluate_hl7v2_leakage(
    messages,
    deidentifier=lambda text, **kwargs: "[MASKED]",
    date_shift_days=29,
)
print(report.residual_by_field)
```

## Messages

Each message uses `\r` segment separators, ends with a trailing separator,
and parses with `parse_hl7v2()`. Configured message types rotate in order.
The generator varies:

- **Message shape:** ADT (`A01`, `A04`, `A08`, and `A40` with an MRG
  segment), ORU `R01` with OBX results and NTE notes, and ORM `O01` with
  NTE notes. ORU and ORM messages carry ORC and OBR segments.
- **Encoding characters:** any five distinct, non-alphanumeric,
  non-whitespace characters for the field separator and MSH-2.
- **Repetitions:** patient identifiers, names, addresses, home phones and
  NTE comments repeat up to `max_repetitions` times.
- **Subcomponents:** PID-3 assigning authorities and, for about half of the
  addresses, PID-11 street, street name and dwelling number.
- **Escape sequences:** organization names and free-text connectors contain
  an encoding character with probability `escape_probability`; generated
  values that already contain one are escaped too.
- **Timestamps:** year, month, day, minute, second or fractional-second
  precision with an optional `+HHMM`/`-HHMM` offset.
- **Free text and Z-segments:** OBX values of type `NM`, `ST`, `TX` and
  `FT`, NTE comments with embedded names, phones and dates, and a `ZPD`
  segment with an email, preferred name and employer.

PHI is deliberately placed both inside and outside `DEFAULT_FIELD_MAP`,
including PID-4, PID-18, NK1-6, PV1-7, PV1-44, ORC-12, OBR-7, OBR-16, OBX-5
with value type `ST`, OBX-14 and the `ZPD` fields.

## Truth manifest

`message.manifest()` returns schema version 1 with the message digest
(`text_sha256`) and one record per PHI value:

| Field | Meaning |
| --- | --- |
| `segment_index` | Zero-based index into `HL7Message.segments`, as in the redaction `coverage_report` |
| `segment`, `field` | Segment name and one-based HL7 field position (`MSH-1` is the field separator) |
| `repetition`, `component`, `subcomponent` | One-based positions within the field |
| `start`, `end` | Half-open offsets into the raw message text, including `\r` separators |
| `raw` | Escaped value exactly as written; `text[start:end] == raw` |
| `value` | Decoded value after the `F`, `S`, `R`, `E` and `T` escapes are resolved |
| `label` | Canonical OpenMed label, such as `LAST_NAME`, `DATE_OF_BIRTH` or `ID_NUM` |
| `kind` | Controlled HL7 role, such as `mother_maiden_name` or `attending_provider_npi` |
| `provider` | Surrogate provider that produced the value |
| `whole_leaf` | `false` when the value is embedded in OBX or NTE free text |
| `in_default_map` | Whether a `DEFAULT_FIELD_MAP` rule applies, including OBX value-type gating |

Offsets index the Python string, which equals UTF-8 byte offsets for the
ASCII values the English providers produce. Plain values have
`raw == value`; escaped values differ only by their escape sequences.

## Determinism and providers

Identical seeds and settings produce identical message bytes, manifests and
`manifest_sha256` digests. PHI values come from the OpenMed `Anonymizer` and
the `clinical_ids` generators (`generate_ssn`, `generate_npi` and
`generate_luhn_identifier`), listed in `HL7V2_SURROGATE_PROVIDERS`.
Timestamp dates come from the date surrogate providers; their time,
fractional-second and offset suffixes come from the seeded generator.
Non-PHI scaffolding, such as local test codes and the MSH timestamp, is
fixed or seeded and is not part of the truth manifest.

## Leakage harness

`evaluate_hl7v2_leakage` forwards `field_map`, `deidentifier`,
`deidentify_kwargs`, `date_shift_days` and `seed` to `redact_hl7v2()`. A
truth value is residual when its raw or decoded value is still present, as a
whole token not adjacent to other letters or digits, in the redacted
subcomponent at the same position. Pass a `deidentifier` for offline runs;
the default loads the OpenMed PII pipeline for OBX and NTE free text.

The report contains positions, canonical labels, `in_default_map` and counts
only. It never stores, renders or raises with truth values or message text,
and `HL7V2LeakageError` carries only a controlled code and a message index.

## Bounds and errors

Invalid or unbounded settings raise `HL7V2SynthesisError` with a stable
`code` and a message that does not echo the rejected input:

| Code | Cause |
| --- | --- |
| `unknown_message_type` | A message type other than `ADT`, `ORU` or `ORM` |
| `unknown_trigger_event` | An ADT event other than `A01`, `A04`, `A08` or `A40` |
| `repetitions_out_of_bounds` | `max_repetitions` outside 1 to 5, or not an integer |
| `segments_out_of_bounds` | `max_obx_segments` outside 0 to 20, or `max_nte_segments` outside 0 to 10 |
| `count_out_of_bounds` | A message count outside 1 to 1000 |
| `invalid_encoding` | Encoding characters that repeat, are alphanumeric or are whitespace |
| `invalid_probability` | `escape_probability` outside 0 to 1 |
| `invalid_timestamp_precision`, `invalid_timezone_offset` | Unknown precisions or malformed offsets |
| `invalid_seed`, `invalid_config`, `invalid_flag` | Non-integer seeds or wrongly typed settings |
| `provider_format` | A date surrogate provider returned a value without a date |

The generator owns generation, truth and the harness only. It does not
change redaction rules and is not an HL7 conformance validator.
