---
name: parsing-hl7v2-messages
description: "Decodes pipe-delimited HL7 v2.x messages (ADT, ORU, MDM, ORM) into structured segments/fields/components and surfaces OBX-5 and NTE-3 free-text narrative for OpenMed. Use before OpenMed processing when ingesting HL7 v2 feeds from an interface engine, lab/results system, or ADT stream and you need the embedded clinical note text de-identified and analyzed. Flatten OBX/NTE text then call openmed.deidentify and openmed.analyze_text; segment-aware redaction is available via openmed.interop.hl7v2. Trigger keywords: HL7, HL7 v2, ADT, ORU, OBX, MSH, PID, pipe-delimited, interface engine, Mirth, lab results."
license: Apache-2.0
metadata:
  project: OpenMed
  category: data-ingestion
  pairs: before
  version: "1.0"
---

# Parsing HL7 v2 Messages for OpenMed

HL7 v2.x is the workhorse of hospital interfacing — ADT (admit/discharge/
transfer), ORU (observation results), MDM (document management), and ORM
(orders) messages flow continuously between EHR, lab, radiology, and ancillary
systems. The clinical *narrative* you want for NLP is buried in **OBX-5**
(observation value) and **NTE-3** (notes/comments) fields, wrapped in a
pipe-and-caret encoding. This skill decodes that envelope and hands the free
text to OpenMed.

## When to use

- You receive HL7 v2 messages from an interface engine (Mirth/NextGen Connect,
  Rhapsody, Cloverleaf) and want to mine embedded note/result text.
- A lab feed (ORU^R01) carries impression/comment narrative in OBX/NTE.
- An MDM^T02 transcription message carries a full report in OBX-5.
- You need a de-identified, structured feed into `openmed.analyze_text`.

## HL7 v2 structure in one minute

A message is **segments** separated by `\r` (carriage return). Each segment is
3-letter-named, then **fields** split by `|`, **components** by `^`,
**repetitions** by `~`, **sub-components** by `&`, with `\` as escape. The
encoding characters are declared in **MSH-1** (the field separator) and
**MSH-2** (`^~\&`). Field positions are **one-based**, and MSH is special:
MSH-1 *is* the separator, so MSH-2 is the first real field.

```
MSH|^~\&|LAB|HOSP|EHR|HOSP|20240302101500||ORU^R01|MSG0001|P|2.5
PID|1||MRN12345^^^HOSP^MR||DOE^JANE^Q||19700115|F|||1 FAKE ST^^SPRINGFIELD^IL^62704
OBR|1||ORD9|CBC^Complete Blood Count
OBX|1|TX|IMPRESSION||Mild leukocytosis; clinically correlate.||||||F
NTE|1||Patient reports fatigue x1 week. Dr. Smith notified.
```

## Quick start

Parse the envelope and pull narrative from OBX-5 / NTE-3, then hand off:

```python
import openmed
from openmed.interop.hl7v2 import parse_hl7v2

raw = open("results.hl7", encoding="utf-8").read()
msg = parse_hl7v2(raw)               # -> HL7Message (segments preserved)

narrative_chunks = []
for seg in msg.segments:
    if seg.name == "OBX":
        # OBX-2 is the value type; OBX-5 is the observation value.
        value_type = seg.get_field(2)
        if value_type in {"TX", "FT", "CE", "ST"}:
            narrative_chunks.append(seg.get_field(5) or "")
    elif seg.name == "NTE":
        narrative_chunks.append(seg.get_field(3) or "")

# Decode component delimiters into plain text before NLP.
flat = "\n".join(c.replace("^", " ").replace("&", " ") for c in narrative_chunks if c)

# Hand the narrative to OpenMed.
deid = openmed.deidentify(flat, method="replace", policy="hipaa_safe_harbor")
result = openmed.analyze_text(deid.text, output_format="dict")
```

`HL7Segment.get_field(position)` uses one-based HL7 positions and returns
`None` for absent fields. `HL7Message.segment_names()` lists segments in order.

## Whole-message segment-aware de-identification

When you need to redact the *entire* message (structured PID/NK1/GT1 fields
*and* OBX/NTE free text) while preserving HL7 framing, use the bundled
redactor instead of hand-rolling it:

```python
from openmed.interop.hl7v2 import redact_hl7v2

safe = redact_hl7v2("results.hl7")   # path or message text
# PID-3 hashed, PID-5 name surrogated, PID-7 DOB date-shifted, OBX-5/NTE-3
# free text masked via openmed.deidentify — delimiters and segment order kept.
```

`redact_hl7v2` applies `DEFAULT_FIELD_MAP` (PID, PD1, NK1, GT1, IN1/IN2, OBX,
NTE). Extend or override it with `field_map={("ZPS", 4): {"action": "hash"}}`
for site-specific Z-segments, and pass `date_shift_days=` for a fixed,
interval-preserving shift.

## Workflow

1. **Frame-split safely.** Real feeds use `\r`, `\r\n`, or MLLP framing
   (`\x0b`…`\x1c\r`). `parse_hl7v2` auto-detects the segment separator; strip
   MLLP control bytes before parsing.
2. **Read encoding from MSH** — never assume `|^~\&`. The adapter derives the
   delimiter set from MSH-1/MSH-2 (`HL7V2Encoding.from_msh_segment`).
3. **Locate narrative.** OBX-5 (gated by OBX-2 value type), NTE-3, and
   report-bearing segments. Concatenate repetitions (`~`) and components (`^`).
4. **De-identify**, then **analyze** with OpenMed.
5. **Rejoin** results to the patient/encounter via PID-3 (patient id) and
   PV1-19 (visit number) — but redact those identifiers in anything you persist.

## Hand-off to / from OpenMed

- **To OpenMed:** flattened OBX-5/NTE-3 text → `openmed.deidentify` →
  `openmed.analyze_text`.
- **Adapter:** `openmed.interop.hl7v2` provides `parse_hl7v2`,
  `redact_hl7v2`, `HL7Message`, `HL7Segment`, `HL7V2Encoding`, `HL7FieldRule`,
  and `DEFAULT_FIELD_MAP` for segment-aware de-id that preserves message
  framing. It is parse-and-redact only — not a conformance validator.
- **Re-link by id, not by PHI:** carry PID-3/PV1-19 as keys, but store hashed
  or surrogate values (the default `redact_hl7v2` hashes PID-3).

## Edge cases & gotchas

- **MLLP wrapper.** Messages off a TCP MLLP listener are framed with `\x0b`
  (start) and `\x1c\r` (end). Strip these before `parse_hl7v2`.
- **Escape sequences.** `\F\`, `\S\`, `\T\`, `\R\`, `\E\` encode literal
  delimiters, and `\.br\` is a line break inside OBX text. Unescape before NLP.
- **Repeating OBX.** A single result can span many OBX segments (one line each);
  reassemble in order before summarizing.
- **Value types matter.** Only treat OBX-5 as narrative when OBX-2 is a text
  type (`TX`, `FT`, `ST`, `CE`); numeric (`NM`) and coded-only values are not
  free text. The default redactor restricts free-text redaction to `FT`/`TX`.
- **Z-segments.** Site-defined `Z*` segments often carry extra PHI; add explicit
  `field_map` rules — they are not in the default map.
- **Versions vary.** v2.3 through v2.8 differ in field cardinality; resolve
  positions against MSH-12 (version id), don't hardcode across versions.

## Standards & references

- HL7 v2 product overview: https://www.hl7.org/implement/standards/product_brief.cfm?product_id=185
- HL7 v2.5.1 message structures (ADT, ORU, MDM) — base reference:
  https://hl7-definition.caristix.com/v2/
- MLLP (Minimal Lower Layer Protocol) transport:
  https://www.hl7.org/documentcenter/public/wg/inm/mllp_transport_specification.PDF
- OBX/NTE segment definitions: https://hl7-definition.caristix.com/v2/HL7v2.5.1/Segments/OBX
