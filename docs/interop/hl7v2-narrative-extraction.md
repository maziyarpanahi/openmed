# HL7 v2 Narrative Extraction

OpenMed can turn common ADT, ORU, and ORM pipe messages into readable,
de-identified text for clinical NLP or human review. The result includes exact
character ranges that point back to the originating segment occurrence and
field position.

The extractor builds on the existing [HL7 v2 parser and structured
redactor](../hl7v2-deidentification.md). It does not implement a second parser
or attempt full HL7 conformance validation.

## Quick Start

```python
from openmed.interop.hl7v2_narrative import extract_hl7v2_narrative

result = extract_hl7v2_narrative("synthetic_oru.hl7")

print(result.text)
for span in result.spans_for("OBX", 5):
    print(span.source.path, result.text_for(span))
```

`result.text` is the de-identified narrative. A source path such as
`OBX[2]-5` means field 5 of the second OBX segment. `segment_index` on the
source record is the zero-based position of that segment in the complete
message.

Provenance records contain only offsets, fixed labels, and HL7 coordinates.
They do not retain the original field value.

## Narrative Modes

Flat mode is the default. It emits compact sentence-like text suitable for
downstream NLP:

```python
flat = extract_hl7v2_narrative(message, mode="flat")
```

Sectioned mode emits stable Markdown headings and one field per line for
review surfaces:

```python
sectioned = extract_hl7v2_narrative(message, mode="sectioned")
print(sectioned.text)
```

Typical sections are `Message`, `Patient`, `Encounter`, `Orders`,
`Observations`, and `Notes`. Empty sections are omitted. Both modes preserve
message order within a section and return section-level offsets in
`result.sections`.

## Privacy Pipeline

Narrative extraction has two local privacy layers:

1. `openmed.interop.hl7v2.redact_hl7v2` handles configured structured fields,
   including patient identifiers, names, dates, addresses, and phone numbers.
2. The complete rendered narrative passes once through
   `openmed.core.pii.deidentify` with `method="mask"` by default. This covers
   free text and any other rendered value before it is returned.

The extractor remaps field and section ranges after the second pass, so all
offsets index the final de-identified text even when placeholders change its
length.

Use `deidentify_kwargs` to configure the final text pipeline. For deterministic
offline tests, pass a callable that returns a string or an object or mapping
with `deidentified_text`:

```python
result = extract_hl7v2_narrative(
    message,
    deidentify_kwargs={"policy": "hipaa_safe_harbor"},
)
```

`date_shift_days`, `lang`, `locale`, `seed`, and an optional structured
`field_map` are forwarded to the existing HL7 redaction layer. The default
30-day shift is stable across runs; dates in the complete narrative still pass
through the final privacy pipeline.

## Provenance Lookup

```python
offset = result.text.index("7.1")
for span in result.provenance_at(offset):
    print(span.source.segment)          # OBX
    print(span.source.field_position)   # 5
    print(span.source.path)             # OBX[2]-5
```

`result.spans` and its explicit alias `result.field_mappings` are tuples of
`HL7V2FieldSpan`. Each span provides:

- `start` and `end`: offsets in the final narrative, with an exclusive end;
- `section` and `label`: the fixed narrative context;
- `source.segment`: the three-character segment name;
- `source.segment_index`: the zero-based message position;
- `source.segment_occurrence`: the one-based occurrence for that segment name;
- `source.field_position`: the one-based HL7 field number.

## Supported Rendering Scope

The renderer covers the common context needed from ADT, ORU, and ORM flows:

| Segment | Rendered fields |
| --- | --- |
| `MSH` | Message type and HL7 version |
| `EVN` | Event type and recorded time |
| `PID` | Patient ID, name, date of birth, and administrative sex |
| `PV1` | Patient class, location, attending provider, admit/discharge time |
| `ORC` | Order control and placer/filler order identifiers |
| `OBR` | Requested test and observation time |
| `OBX` | Observation, result, units, reference range, flags, status, time |
| `NTE` | Note text |

Unknown segments are ignored by the narrative renderer. They remain available
to the underlying parser and can still be covered by a custom structured field
map. Mapping the message into FHIR resources is outside this helper's scope.
