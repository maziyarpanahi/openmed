# Multimodal asset limit profiles

A limit profile is one pre-decode admission policy for multimodal assets. It is
evaluated over the privacy-safe asset manifest before any decoder, tensor, or
model is loaded, so a page count, pixel dimension, frame count, or audio
duration that would exhaust memory is refused up front instead of during
inference.

## Overview

`evaluate_asset_limits(profile, manifest, modality)` accepts a `LimitProfile`,
a canonical `AssetManifest` instance or a metadata mapping, and one of the four
manifest modalities (`image`, `pdf`, `dicom`, `audio`). It returns a list of
`LimitFinding` values; an empty list means every rule the modality can express
was evaluated and passed.

The privacy and safety boundary is strict:

- Evaluation uses manifest metadata only. Files are never opened, decoded, or
  downsampled.
- A rule that cannot be evaluated is reported as `insufficient_metadata`, never
  assumed to pass. Missing evidence does not become acceptance.
- Raster geometry is never inferred. A PDF manifest carries no width or height
  under the PDF profile, so a PDF's two pixel rules are always unevaluable and
  are reported as such even when bytes and pages pass. Preflight must abstain
  on that basis rather than accept.
- Findings carry a field name, a reason code, the ceiling, and the observed
  number (or `null`). They never carry identifiers, paths, digests, or content.

Mapping callers remain responsible for running the structural asset-manifest
validator first. Values outside the manifest contract (booleans, zero,
negatives, non-finite numbers, values beyond the manifest bounds) raise
`AssetLimitError` with a content-free message.

## Profiles

Ceilings are inclusive: a value equal to the ceiling passes, a value above it
fails. Profiles are immutable; `profile.with_limits(...)` returns a re-validated
copy for callers that need their own policy.

| Limit | `MOBILE_V1` | `DESKTOP_V1` |
| --- | ---: | ---: |
| `max_byte_size` | 64 MiB (`64 * 1024**2`) | 256 MiB (`256 * 1024**2`) |
| `max_pages` | 10 | 100 |
| `max_pixels` (per image, page, or frame) | 10,000,000 | 40,000,000 |
| `max_total_pixels` | 25,000,000 | 100,000,000 |
| `max_frames` | 128 | 1,024 |
| `max_duration_seconds` | 300 | 1,800 |

These are explicit initial admission-policy choices, not measured device memory
budgets, and not a guarantee that decoding or inference will fit on any
hardware. The desktop page and pixel ceilings match the redacted-PDF renderer's
defaults (`render_pdf.py`) so the tree carries one policy; the other values are
unbenchmarked starting points. Existing renderer behaviour is unchanged by this
module.

## Rules by modality

| Rule | `image` | `pdf` | `dicom` | `audio` |
| --- | :-: | :-: | :-: | :-: |
| `byte_size` | ✓ | ✓ | ✓ | ✓ |
| `pages` | — | ✓ | — | — |
| `pixels` | width × height | unevaluable | width × height (per frame) | — |
| `total_pixels` | width × height | unevaluable | width × height × frames | — |
| `frames` | — | — | ✓ | — |
| `duration_seconds` | — | — | — | ✓ |

A rule marked — is not applicable to the modality and produces no finding. A
rule that is applicable but whose inputs are absent produces
`insufficient_metadata`. Pixel products use checked arithmetic: every factor
must be a bounded positive integer within the manifest count bound, and the
product is bounded by `MAX_PIXEL_PRODUCT` (the cube of that bound).

## Findings

Findings are returned in a fixed, documented order: `byte_size`, `pages`,
`pixels`, `total_pixels`, `frames`, `duration_seconds`. Each carries:

- `field_name`: one of the six limit fields above.
- `reason_code`: `limit_exceeded` (the observed value is above the ceiling) or
  `insufficient_metadata` (the rule could not be evaluated; `observed` is
  `null`).
- `limit`: the inclusive ceiling that applied.
- `observed`: the number the manifest yielded, or `null`.

Findings cannot be constructed with arbitrary field names or reason codes, and
they are separate from the manifest-profile `ValidationFinding` and from the
coarse abstention reason codes. The preflight report carries the coarse
`RESOURCE_LIMIT` abstention alongside these detailed findings for exceeded or
unevaluable limits.


## ZIP document admission

The Python DOCX, PPTX, XLSX, EPUB and ODT adapters now admit the complete ZIP
container before reading selected content or handing it to an optional library.
This is separate from manifest preflight: it checks the actual local archive,
including unused parts. Existing text, character offsets and XLSX cell-coordinate
reports are preserved for admitted packages. OpenMedKit currently has no matching
ZIP document ingesters; this change adds no Swift format capability.

Default ceilings come from `openmed.interop.archive_safety.ArchiveSafetyPolicy`:
10,000 entries, 128 MiB per uncompressed member, 512 MiB total uncompressed bytes,
100x expansion per member and 4,096 characters per member path. The directory
scan also limits central-directory metadata to 16 MiB before `ZipFile` allocates
it. Multi-disk and ZIP64 central directories are unsupported (ordinary ZIP64
local headers within these limits are accepted). Only stored and raw-DEFLATE
members are admitted. Encrypted entries, links, duplicate/unsafe paths and any
quarantine or rejection from the existing policy fail closed.

Metadata is checked before any member decompression. Actual member output is
then streamed and discarded in chunks of at most 64 KiB, stopping at the declared
size plus one byte, with CRC and DEFLATE termination verification. This detects
forged sizes even when both headers and a prefix CRC agree. EPUB/ODT selected
members use this same bounded reader. OOXML libraries receive the same open file
only after every member passes admission; DOCX/PPTX write-back reloads are also
admitted. Callers must not modify a source during processing. This bounds ZIP
expansion, not XML parsing or optional-library memory, and is not decoder sandboxing.

Callers can tighten limits with the keyword `archive_policy` on `extract_docx`,
`extract_pptx`, `extract_epub`, `extract_odt` and `redact_xlsx`. The existing policy
rejects attempts to raise its safety ceilings. Dispatcher calls use the defaults.

```python
from openmed.interop.archive_safety import ArchiveSafetyPolicy
from openmed.multimodal import extract_odt
from openmed.multimodal.exceptions import ArchiveDocumentError

limits = ArchiveSafetyPolicy(max_member_uncompressed_bytes=8 * 1024**2)
try:
    document = extract_odt(source_path, archive_policy=limits)
except ArchiveDocumentError as error:
    # Controlled code and finding count only; no paths, member names or content.
    diagnostic = {"reason_code": error.reason_code, "count": error.count}
```

`ArchiveDocumentError` subclasses `UnsupportedDocumentError`. Archive-policy
reason codes are preserved (`entry_limit`, `member_size_limit`, `total_size_limit`,
`expansion_ratio`, `path_traversal`, and other structural codes). Additional codes
are `archive_invalid`, `archive_metadata_limit`, `archive_zip64_directory`,
`archive_encrypted`, `archive_compression`, `archive_size_mismatch`, `archive_crc`,
`archive_member_missing` and `archive_mimetype`. If several findings exist, the
first code in sorted policy order and its count are reported. Exception messages
contain only that code and count, and suppress upstream ZIP exception messages.
XML entity handling remains a separate format-specific boundary.

Offline synthetic allocation tests in
`tests/unit/multimodal/test_archive_admission.py` assert a **4 MiB peak Python
allocation ceiling** using `tracemalloc` around admission of high-ratio,
oversized-member, excess-entry and forged-size fixtures for all five formats.
Fixture construction occurs outside the measurement. A streamed 100 MiB
high-ratio ODT is also rejected under this ceiling with default limits. These
are fixture-specific Python allocation gates, not process RSS or a universal
memory guarantee for every archive or downstream decoder. Integration tests
compare actual optional parsers with an unchecked negative control, preserving
text/spans and XLSX values/coordinate reports. No external media or model assets
are used.
