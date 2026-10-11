# DICOM-SR provenance mapping

`openmed.multimodal.dicom_sr_provenance` links an opaque finding identifier to
the deterministic content-tree path emitted by the DICOM-SR extractor. The
result carries only structural evidence references:

- `finding_id`: a caller-supplied stable identifier;
- `item_path`: the 1-based dotted path used by `walk_sr_content_tree`;
- `template_id`: the item's template, or the nearest declared ancestor
  template; and
- `source_start` / `source_end`: half-open offsets into the extracted text when
  a matching `SourceSpan` or explicit finding offsets are available.

Concept names, rendered values, units, report text, and arbitrary finding
metadata are not copied into provenance records. Input paths and source spans
are validated strictly. Duplicate paths, conflicting path aliases, duplicate
source spans, and offsets matching multiple paths raise
`AmbiguousDicomSrItemPathError` instead of choosing a possibly incorrect
evidence link.

## Example

```python
from openmed.multimodal import extract_dicom_sr
from openmed.multimodal.dicom_sr_provenance import (
    build_dicom_sr_provenance,
    serialize_dicom_sr_provenance,
)

document = extract_dicom_sr("already-deidentified-report.dcm", deidentify_headers=False)
findings = [
    {"finding_id": "finding-001", "item_path": "1.3.1.3"},
]

records = build_dicom_sr_provenance(findings, document=document)
print(serialize_dicom_sr_provenance(records))
```

The mapper is deterministic and local-only. It is an evidence-linking aid, not
a clinical interpretation or a substitute for qualified review.
The example assumes an already de-identified report. For original reports,
use the cleaning policy below before linking findings to retained content.

Explicit offsets must fit the declared item span, and source spans must fit a
supplied document. Conflicting identifier, template, path and offset aliases
are rejected. Collections are limited to 4,096 entries and item paths to 64
levels; reference strings are limited to 4,096 characters. Rendering revalidates
typed records, and public errors discard raw exception context. Finding and
template identifiers are caller-owned opaque references, not anonymized values;
never place report text or patient identifiers in those fields.

## DICOM carrier removal and pixel coverage

Both `deidentify_dicom_headers` and `redact_dicom_pixels` remove overlay groups
`60xx`, retired curve groups `50xx`, and Icon Image Sequence `(0088,0200)`,
including carriers nested inside other sequences. Carrier audit entries list
only the tag, keyword, VR, structural location and action; they never copy
carrier bytes or stringify thumbnails or documents. Supported retired overlay
planes in unused image bits are removed from stored pixels as well. A plane
that overlaps meaningful image values is refused with
`embedded_overlay_not_cleanable` rather than silently damaging the image.

Header processing does not run OCR. `DicomHeaderDeidResult.pixel_status` and its
audit report distinguish `pixels_not_present`, `pixels_declared_clean` (the
input explicitly declares `BurnedInAnnotation=NO`), and `pixels_not_cleaned`.
A header-only image with unknown or positive burned-in annotation retains its
pixels, receives `PatientIdentityRemoved=NO`, and reports a header-only method.
Set `DicomHeaderDeidPolicy(fail_on_unclean_pixels=True)` to refuse that output
with `DicomDeidentificationError.reason_code == "pixels_not_cleaned"`.
Refusal preserves both the source and any pre-existing destination file.

The pixel pass reports `pixels_cleaned` and sets `BurnedInAnnotation=NO` only
when residual verification actually ran and passed. Skipping verification or
accepting residual findings reports `pixels_not_cleaned` and retains an unsafe
burned-in annotation marker. Pixel processing invalidates inherited identity
and method-code claims because it has not cleaned the headers. The combined
`redact_document` dispatcher runs both stages. These markers describe the
executed stages; OCR quality and clinical review remain the caller's concern.
The combined dispatcher stages both passes in memory before its first write,
so a header-stage refusal preserves the source and existing destination too.
Unprocessed nested pixels keep the overall outcome at `pixels_not_cleaned`.

`DicomPixelRedactionPolicy(overlay_mode="burn")` burns supported main-image
overlays into the image before OCR, then removes their separate carrier tags.
Overlay origins and frame offsets are applied to the actual image frames;
malformed planes fail with `overlay_burn_failed`. Nested overlay burning is
refused with `nested_overlay_burn_unsupported`; the default `remove` mode cleans
nested carriers. Floating-point image pixel redaction is refused with
`pixel_data_unsupported` rather than being reported as a successful empty image.
External Pixel Data Provider URLs are refused in both paths with
`external_pixel_data_unsupported`; these operations never fetch them.

### Encapsulated documents

Encapsulated Document `(0042,0011)` is refused by default with
`encapsulated_document_requires_redaction` in both DICOM paths. To process a
supported embedded document, explicitly enable `redact_encapsulated_documents`
and supply a local callable detector via `document_models`. The existing
registered document handler receives an in-memory stream and a
`document_policy` with `return_bytes=True`; output-file options are removed.
The handler must return non-empty replacement bytes in
`metadata["redacted_document_bytes"]` or the PDF handler's
`metadata["redacted_pdf_bytes"]`. Extraction text and an unchanged payload are
refused. No original embedded payload is written to an intermediate file.

PDF and XML MIME types route to their registered `.pdf` and `.xml` handlers.
Unsupported MIME types yield `encapsulated_document_type_unsupported`.
Absent detectors, handler failures, or invalid byte results yield
`encapsulated_document_redaction_failed` without propagating payloads or handler
exception text. The stock CDA extractor does not supply a byte-output
redaction contract, so XML needs a registered handler that does. Successful
replacement also updates Encapsulated Document Length `(0042,0015)`.

```python
from openmed.multimodal import DicomHeaderDeidPolicy, deidentify_dicom_headers

result = deidentify_dicom_headers(
    "synthetic.dcm",
    policy=DicomHeaderDeidPolicy(
        output_path="synthetic-redacted.dcm",
        fail_on_unclean_pixels=True,
    ),
)
print(result.pixel_status.value)
```

The carrier actions follow the recursive removal and replacement requirements
in [DICOM PS3.15 Annex E](https://dicom.nema.org/medical/dicom/current/output/chtml/part15/chapter_E.html).
Carrier sanitation is separate from the complete free-text Basic Profile
coverage and from qualification of burned-in OCR or pixel decoders.

## Basic Profile actions and retained content

Header processing uses a pinned **DICOM PS3.15 2026d** catalog: all 657 rows of
[Table E.1-1](https://dicom.nema.org/medical/dicom/2026d/output/chtml/part15/chapter_E.html),
including nested sequences, repeating groups and private attributes. Catalog
metadata contains tags, keywords and action codes only. There are no runtime
standard downloads or bundled terminology dictionaries.

The default removes optional identifying attributes (`X`), clears zero-length
attributes (`Z`), replaces required non-empty values with VR-compatible dummies
(`D`), and deterministically remaps instance UIDs (`U`). Composite actions use
the first listed alternative; OpenMed does not infer IOD-specific attribute
types. Dummy sequences contain one item, and default SR `ContentSequence`
contains a dummy comment. Applying these rules does not validate an IOD or
establish clinical utility. Review resulting objects for the intended workflow.

Unlisted public attributes with unknown tags or `UN` values are removed.
Known sequences are recursively processed; numeric/image attributes and code
strings are kept, instance UIDs are remapped, and unlisted names, dates and
free text are cleared. Unlisted code strings remain subject to copied-identifier
scrubbing. Unlisted attributes are reported using tags and controlled actions,
without values, value hashes, keyword or VR. File metadata is rebuilt from SOP
class, final SOP instance UID, transfer syntax and OpenMed implementation
identity; input AE titles, addresses and private information are discarded.
Preambles are zeroed, and group `0004` attributes are removed from image/report
objects. This path is not a DICOMDIR rebuilding service.

All six `DicomHeaderDeidPolicy` option flags default to `False`:

| Flag | DCM method code | Behavior |
| --- | --- | --- |
| `clean_descriptors` | `113105` | Retain descriptor fields after detector cleaning |
| `clean_structured_content` | `113104` | Apply SR concept/value-type rules and clean retained content |
| `retain_longitudinal_temporal_information` | `113107` | Shift whole dates consistently and keep time-of-day values |
| `retain_device_identity` | `113109` | Keep device identity fields specified by the table |
| `retain_patient_characteristics` | `113108` | Keep characteristics specified by the table |
| `retain_uids` | `113110` | Keep instance UIDs and their references |

Explicit legacy date-shift parameters (`date_shift_days`, patient key, shift
range or secret) request the modified-dates option and now declare its code.
With no options or shift request, temporal identity is removed, the reported
shift is zero, and the only method code is Basic Profile `113100`. Enabled
options appear in both De-identification Method and its Code Sequence. Objects
with unclean pixels retain `PatientIdentityRemoved=NO` and receive no method
code sequence claiming whole-instance profile completion.

Cleaning options require an explicit local `detector` (or a detector exposed by
`document_models`). It receives each retained textual value and returns entity
spans, either as a list or through the existing `entities`, `pii_entities`, or
`spans` result seam. Each span must have valid integer `start`/`end` offsets;
intersecting spans are merged and replaced with `[REMOVED]`. A missing detector
raises `profile_detector_required`; failed or malformed results raise
`profile_detector_failed`. Detector exceptions are suppressed, and refusal
preserves the source and existing destination. Retention flags require exact
booleans; strings such as `"false"` raise `invalid_profile_option`.
Known tags with mismatched explicit VRs raise `profile_vr_mismatch`. Cleaning
binary descriptors requires a format-specific implementation; this text
detector path refuses them with `profile_binary_cleaning_unsupported` before
opening the output. Empty required UIDs raise `profile_uid_invalid`. Ambiguous
or multivalued SR concept-code alternatives raise `profile_concept_invalid`;
known identity concepts with the wrong value type are removed, and alternate
scalar `LongCodeValue`/`URNCodeValue` representations apply the same rules.

```python
from openmed.multimodal import DicomHeaderDeidPolicy, extract_dicom_sr


def extract_reviewable_sr(path, local_detector):
    return extract_dicom_sr(
        path,
        policy=DicomHeaderDeidPolicy(
            clean_structured_content=True,
            clean_descriptors=True,
            detector=local_detector,
        ),
    )
```

Structured cleaning follows the concept-code and value-type rules in
[Table E.3.4-1](https://dicom.nema.org/medical/dicom/2026d/output/chtml/part15/sect_E.3.4.html).
Patient characteristics encoded as numeric items require their retention
option too. Retired SNOMED aliases and UMLS/nonstandard non-container concepts
are conservatively removed; no restricted terminology crosswalk is bundled.
The header policy and detector also reach both registered DICOM handlers.
Existing `walk_sr_content_tree` behavior for already de-identified datasets is
unchanged. Cleaning quality depends on the supplied detector; catalog coverage
and method codes do not certify freedom from re-identification.
