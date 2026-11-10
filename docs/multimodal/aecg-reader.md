# Bounded HL7 aECG reader

The Python `openmed.multimodal.aecg.read_aecg` function and OpenMedKit
`OpenMedAECG.read` read caller-supplied, in-memory UTF-8 XML offline. They return
signals for review, without diagnostic interpretation, file access, network
requests, model execution or autonomous clinical action. No dependency is added.

Every record and report carries this notice:

> Non-diagnostic waveform for review only. Explicit reviewer confirmation is
> required before consequential use. No autonomous clinical decisions.

Call `record.confirm_review(reviewer_confirmed=True)` in Python or
`record.confirmReview(reviewerConfirmed: true)` in Swift at a consequential
handoff. False or missing confirmation is rejected with `review_required`.
Confirmation preserves the notice; it does not establish clinical validity.

## Supported subset and sample decoding

The namespace must be `urn:hl7-org:v3`. Each root `component/series` must use the
HL7 `RHYTHM` code. Its optional `derivation/derivedSeries` uses
`REPRESENTATIVE_BEAT`. Each series has exactly one `component/sequenceSet` with
one time sequence and at least one distinct lead sequence.

Time uses `TIME_ABSOLUTE` / `GLIST_TS` or `TIME_RELATIVE` / `GLIST_PQ`, with a
positive finite increment in seconds. The head and all absolute acquisition
and demographic timestamps are withheld. Returned timing consists only of the
sample interval; no absolute or inter-series alignment is claimed.

Lead sequences use the original MDC codes `MDC_ECG_LEAD_I`, `II`, `III`, `AVR`,
`AVL`, `AVF`, and `V1`–`V6` (each with the `MDC_ECG_LEAD_` prefix) and
`SLIST_PQ`. Lead codes require code system `2.16.840.1.113883.6.24`; series and
time codes require `2.16.840.1.113883.5.4`. No alias or lead normalization is
performed. Alternate XML namespace prefixes are supported.

Samples are computed as `origin + scale * digit`. Origin and nonzero scale must
be finite; digits must be whitespace-separated signed 32-bit integers. Origin
and scale must use the same `uV`, `mV` or `V` unit. That source unit is retained;
voltage conversion belongs to the separate normalization contract. All leads
within a series must have equal, nonempty sample counts. Different series may
have different lengths.

The shared synthetic fixture uses origin `1 uV`, scale `2.5 uV` and digits
`-2 0 4`, producing **`(-4.0, 1.0, 11.0) uV`** on lead I. Its derived series
uses digits `0 2`, producing **`(1.0, 6.0) uV`** independently.

`rhythm` and `derived` are separate collections. The returned derived kind is
`derived_representative_beat`, never `rhythm`. Representative beats may be
median or average beats; the reader does not infer an algorithm from this code.
The supported hierarchy follows the [FDA reference reader](https://github.com/FDA/aecg-python/blob/main/src/aecg/io.py).
This is a bounded subset reader, not an HL7 schema validator or regulatory
qualification claim. Other layouts, encodings and lead codes fail closed.

## Privacy and reports

Subject, trial-subject, assignment and demographic-person elements are counted,
and annotation / annotation-set elements are counted. Presence flags and counts
are exposed; identifiers, names, addresses, demographics, free-text annotations,
source IDs, private paths and absolute timestamps are never copied into records,
reports or error messages. All other header metadata is also withheld.

Reports contain only these counts, the SHA-256 digest of the original bytes,
series kinds, lead/sample counts, the notice and `review_required: true`.
They omit samples and time values. Waveform samples themselves can remain
sensitive; parsing is not anonymization. This reader does not redact or export
XML; annotation redaction and clinical validation packs remain separate work.

```python
from openmed.multimodal.aecg import read_aecg

# xml_bytes is supplied by the caller; the reader never opens a path or URL.
record = read_aecg(xml_bytes)
report = record.report()
reviewed = record.confirm_review(reviewer_confirmed=True)
```

```swift
let record = try OpenMedAECG.read(xmlData)
let report = record.report()
let reviewed = try record.confirmReview(reviewerConfirmed: true)
```

## Bounds and controlled rejections

Python `AecgLimits` and Swift `OpenMedAECGLimits` have matching positive defaults:

| Bound | Default |
| --- | ---: |
| Input bytes | 16 MiB |
| XML elements | 100,000 |
| XML depth | 64 |
| Rhythm plus derived series | 16 |
| Leads per series | 64 |
| Samples per lead | 1,000,000 |
| Samples across all leads and series | 2,000,000 |

Byte bounds are checked before UTF-8 decoding. Tree and series bounds are checked
during parsing; sample bounds are checked before each decoded sample is appended.
Exceeding a limit returns no partial record. Callers can lower or raise bounds
explicitly to suit their memory envelope.

DOCTYPE and ENTITY declarations are rejected before XML parsing. External DTDs,
entities, HL7 `reference` elements and XInclude are unsupported. Schema-location
attributes are ignored and never retrieved. XML parsers never resolve external
resources. Only UTF-8 bytes (with an optional UTF-8 BOM) are accepted.

| Reason code | Meaning |
| --- | --- |
| `invalid_input` | Python input is not bytes |
| `invalid_limits` | Bounds are not positive integers |
| `byte_limit`, `element_limit`, `depth_limit`, `series_limit`, `lead_limit`, `sample_limit` | Resource envelope exceeded |
| `unsafe_xml` | DOCTYPE or ENTITY declaration |
| `external_reference` | Unsupported reference or XInclude |
| `malformed_xml` | Invalid XML; parser details are withheld |
| `unsupported_encoding` | Unsupported XML, sequence or unit encoding |
| `unsupported_layout` | Missing, duplicated, misplaced or unsupported series structure |
| `unsupported_lead` | Unknown MDC lead or code system |
| `invalid_numeric` | Invalid, out-of-range or nonfinite numeric value |
| `inconsistent_lengths` | Leads in one series have different sample counts |
| `review_required` | Consequential handoff lacks explicit confirmation |

Python raises `AecgError` with `reason_code`; Swift throws `OpenMedAECGError`
with `reasonCode`. Neither returns submitted values in diagnostics. Synthetic
unit and offline integration tests cover scaled samples, separate derived beats,
privacy sentinels, hostile XML, resource limits and reviewer confirmation.
