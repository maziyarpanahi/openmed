# DICOM transfer-syntax preflight

A compressed or retired DICOM transfer syntax changes what a pipeline must load
before it can touch pixel data. The File Meta Information group declares that
choice in the first bytes of the file, so it can be classified before an imaging
library or a pixel-data codec is imported.

`openmed.multimodal.dicom_transfer_syntax` reads only that bounded header. It
returns a routing outcome for the declared syntax and never decodes pixels,
installs or exercises a codec, validates clinical image content, or fetches
anything over the network. No patient tag, non-transfer-syntax UID, file path,
or pixel byte appears in a report.

```python
from openmed.multimodal import read_dicom_transfer_syntax

report = read_dicom_transfer_syntax(header_bytes)
print(report.outcome.value, report.reason)
# optional_decoder transfer_syntax_jpeg_baseline
```

## Outcomes

| Outcome | Meaning |
| --- | --- |
| `native` | Uncompressed little-endian syntaxes a reader can handle without extra codecs. |
| `optional_decoder` | Compressed syntaxes that need an installed pixel-data codec before decoding. |
| `review` | Retired syntaxes or syntaxes that need an extra parsing decision before decoding. |
| `unsupported` | Missing, malformed, private, or unknown declarations. |

## Transfer-syntax catalog

The catalog is closed: a declared UID either matches an entry below or is
reported as `transfer_syntax_unknown` / `transfer_syntax_private`.

| UID | Name | Outcome | Codec required | Retired |
| --- | --- | --- | --- | --- |
| `1.2.840.10008.1.2` | Implicit VR Little Endian | `native` | no | no |
| `1.2.840.10008.1.2.1` | Explicit VR Little Endian | `native` | no | no |
| `1.2.840.10008.1.2.1.99` | Deflated Explicit VR Little Endian | `review` | no | no |
| `1.2.840.10008.1.2.2` | Explicit VR Big Endian | `review` | no | yes |
| `1.2.840.10008.1.2.4.50` | JPEG Baseline (Process 1) | `optional_decoder` | yes | no |
| `1.2.840.10008.1.2.4.51` | JPEG Extended (Process 2 and 4) | `optional_decoder` | yes | no |
| `1.2.840.10008.1.2.4.57` | JPEG Lossless, Non-Hierarchical (Process 14) | `optional_decoder` | yes | no |
| `1.2.840.10008.1.2.4.70` | JPEG Lossless, Non-Hierarchical, First-Order Prediction | `optional_decoder` | yes | no |
| `1.2.840.10008.1.2.4.80` | JPEG-LS Lossless | `optional_decoder` | yes | no |
| `1.2.840.10008.1.2.4.81` | JPEG-LS Lossy (Near-Lossless) | `optional_decoder` | yes | no |
| `1.2.840.10008.1.2.4.90` | JPEG 2000 Image Compression (Lossless Only) | `optional_decoder` | yes | no |
| `1.2.840.10008.1.2.4.91` | JPEG 2000 Image Compression | `optional_decoder` | yes | no |
| `1.2.840.10008.1.2.4.92` | JPEG 2000 Part 2 Multi-component (Lossless Only) | `optional_decoder` | yes | no |
| `1.2.840.10008.1.2.4.93` | JPEG 2000 Part 2 Multi-component | `optional_decoder` | yes | no |
| `1.2.840.10008.1.2.5` | RLE Lossless | `optional_decoder` | yes | no |

## Reading only the file-meta group

The reader accepts `bytes`, `bytearray`, or `memoryview` and stops at the end of
group `0002` (or at the first element that belongs to another group). It checks
the 128-byte preamble, the `DICM` magic, then parses explicit-VR little-endian
elements until it has `(0002,0010) TransferSyntaxUID`. A declared group length
that disagrees with the parsed group end is reported instead of trusted.

Bounds are explicit keyword-only arguments and default to
`DEFAULT_MAX_FILE_META_BYTES` (64 KiB), `DEFAULT_MAX_FILE_META_ELEMENTS` (64),
and `DEFAULT_MAX_TRANSFER_SYNTAX_UID_BYTES` (128). Arguments that cannot be used
raise `TransferSyntaxError` with a stable category such as `max_bytes_invalid`
or `source_invalid`; unusable *content* is never raised.

## Reason codes

| Group | Reason codes |
| --- | --- |
| File-meta structure | `file_meta_missing`, `file_meta_truncated`, `file_meta_malformed`, `file_meta_group_length_mismatch`, `file_meta_element_limit` |
| Declaration | `transfer_syntax_missing`, `transfer_syntax_malformed`, `transfer_syntax_unknown`, `transfer_syntax_private` |
| Catalog results | `transfer_syntax_implicit_little_endian`, `transfer_syntax_explicit_little_endian`, `transfer_syntax_deflated`, `transfer_syntax_big_endian_retired`, `transfer_syntax_jpeg_baseline`, `transfer_syntax_jpeg_extended`, `transfer_syntax_jpeg_lossless`, `transfer_syntax_jpeg_lossless_sv1`, `transfer_syntax_jpeg_ls_lossless`, `transfer_syntax_jpeg_ls_near_lossless`, `transfer_syntax_jpeg2000_lossless`, `transfer_syntax_jpeg2000_lossy`, `transfer_syntax_jpeg2000_multi_lossless`, `transfer_syntax_jpeg2000_multi`, `transfer_syntax_rle_lossless` |

`TRANSFER_SYNTAX_REASON_CODES` lists all of them and
`TRANSFER_SYNTAX_OUTCOMES` the four outcome values.

## Determinism and privacy

Reports are frozen dataclasses. `to_dict()` returns a fixed key order and
`to_json()` is compact JSON with sorted keys, so byte-stable output can be
diffed across runs. A report carries the declared transfer-syntax UID, the
catalog name, the outcome, the reason codes, and two counts — the bytes consumed
by the parsed file-meta group and the number of elements read. It never carries
pixels, paths, patient tags, or any other UID, and the reader imports no imaging
library.

Buffer limits are enforced on the bytes handed in: an oversized buffer is read
up to the bound and the rest ignored, and a truncated group is reported as
`file_meta_truncated` rather than parsed optimistically.

## Not in scope

Decoding pixels, installing or probing codecs, validating clinical image
content, and retrieving files are out of scope. This module is a standalone
preflight: it does not change `openmed.multimodal.preflight_asset` or the media
type detection catalog.

## Verification

```bash
uv run --frozen --extra dev pytest tests/unit/multimodal/test_dicom_transfer_syntax.py -q
```
