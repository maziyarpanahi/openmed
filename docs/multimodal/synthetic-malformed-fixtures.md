# Synthetic malformed header fixtures

The multimodal preflight suite already refuses malformed document and
medical-image headers before any decode. This contribution adds reusable,
deterministic corruption fixtures and focused regression tests for those
boundaries; it does **not** add PDF or DICOM parsing support, relax any refusal,
or change runtime behavior.

```python
from openmed.multimodal.pdf_geometry import read_pdf_geometry
from tests.fixtures.multimodal.malformed import case_by_name

case = case_by_name("pdf-header-cyclic")
report = read_pdf_geometry(case.payload)
assert report.reason_codes == ("pdf_page_tree_invalid",)
```

`tests/fixtures/multimodal/malformed.py` exposes one table,
`MALFORMED_HEADER_CASES`, with five corruption classes for each modality:
truncated, oversized, cyclic, inconsistent, and unsupported. Each entry records
the payload, the boundary that refuses it, and the exact reason that boundary
reports, so tests and documentation cannot drift from implemented behavior.

## Pre-decode boundaries

PDF fixtures are consumed by `openmed.multimodal.pdf_geometry.read_pdf_geometry`,
the bounded, dependency-free header and page-tree reader. Corrupt input does not
raise: the reader returns a report whose `reason_codes` are stable members of
`PDF_REASON_CODES`.

| Fixture | Corruption | Expected reason | Status |
| --- | --- | --- | --- |
| `pdf-header-truncated` | truncated | `pdf_catalog_missing` | rejected |
| `pdf-header-oversized` | oversized | `pdf_object_limit` | rejected |
| `pdf-header-cyclic` | cyclic | `pdf_page_tree_invalid` | rejected |
| `pdf-header-inconsistent` | inconsistent | `page_count_mismatch` | review |
| `pdf-header-unsupported` | unsupported | `pdf_encrypted` | rejected |

`pdf-header-oversized` is a twelve-object file probed with an explicit
`max_objects=4` budget, so the fixture exercises the reader's bounded object
allowance without carrying a large payload. `pdf-header-inconsistent` is the one
case that is not rejected: a declared `/Count 3` over a single page-tree leaf is
reported as `page_count_mismatch` on a `review` verdict, which is the reader's
existing precedence rule.

## DICOM boundary limitation

This repository has **no byte-level DICOM structure preflight**. The only raw
byte boundary is the media-type check in
`openmed.multimodal.media_type.detect_media_type`, which inspects at most
`MAX_MEDIA_TYPE_PREFIX_BYTES` (132) leading bytes and matches the `DICM` magic at
offset 128, followed by `openmed.multimodal.preflight.preflight_asset`, which
reports `unknown` or `mismatch` under its `media_type` check. Every DICOM decode
path takes a path and calls `pydicom.dcmread(..., force=True)`, so a truncated or
badly sized data set may never raise there.

The DICOM fixtures therefore document the bounded magic boundary and the
preflight media-type verdict only. They never reach a decoder and make no claim
about group length, transfer syntax, or pixel data.

| Fixture | Corruption | Expected reason (`media_type`) | Status |
| --- | --- | --- | --- |
| `dicom-header-truncated` | truncated | `unknown` | abstain |
| `dicom-header-oversized` | oversized | `unknown` | abstain |
| `dicom-header-cyclic` | cyclic | `unknown` | abstain |
| `dicom-header-inconsistent` | inconsistent | `mismatch` | abstain |
| `dicom-header-unsupported` | unsupported | `unknown` | abstain |

Each DICOM fixture also records the manifest declaration it is compared against
(a `frames`/`width`/`height` DICOM manifest, or an `image/bmp` manifest for the
inconsistent case), so the preflight report contains exactly one finding: the
media-type verdict that classifies the corruption.

## Determinism and privacy

Every fixture is assembled from constants only — no randomness, no timestamps,
no third-party dependency. Payloads are at most 4096 bytes and, apart from the
PDF structural dictionaries, contain no text, no patient identifiers, no
clinical pixels, and no encoded image data. `PAYLOAD_SHA256` pins each payload's
digest, and the tests fail if a fixture's bytes, size, or recorded reason code
drift.

## Verification

```bash
uv run --frozen --extra dev pytest tests/unit/multimodal/test_malformed_headers.py -q
```

The tests run entirely offline and assert, per fixture, the boundary's status,
the exact reason code, the media type reported by the bounded sniffer, the
presence of a valid `DICM` magic only where one is expected, and the absence of
PHI markers. Existing PDF geometry, media-type, and preflight tests are retained
unmodified. No network calls, decoders, model weights, or credentials are needed.
