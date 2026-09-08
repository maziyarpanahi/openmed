# Multimodal preflight report

`preflight_asset(manifest, source)` is the single pre-decode entry point for
multimodal assets. It runs the committed preflight contracts in a fixed order
and folds their results into one accept-or-abstain report, so every image,
PDF, DICOM, waveform, and audio provider applies the same safety behaviour
before a decoder, tensor, or model is loaded.

```python
from openmed.multimodal.preflight import PreflightStatus, preflight_asset

manifest = {
    "asset_id": "study-0001",
    "media_type": "image/png",
    "sha256": "<sha256 of the file>",
    "byte_size": 48213,
    "width": 1024,
    "height": 768,
}

with open("synthetic-scan.png", "rb") as stream:
    report = preflight_asset(manifest, stream)

if report.status is PreflightStatus.ACCEPT:
    ...  # hand the validated manifest and digest to the decoder
else:
    print(report.to_json())
```

## Checks, in order

| Check | Contract | What fails closed |
| --- | --- | --- |
| `manifest` | `AssetManifest` | A malformed manifest (`malformed_manifest`). Preflight ends here; the source is never touched and no submitted value is reflected. |
| `media_type` | `detect_media_type` | The type detected from at most 132 leading bytes disagrees with the declared type (`mismatch`), or cannot be detected at all (`unknown`). An undetectable prefix is never treated as a match. |
| `metadata` | `validate_manifest_metadata` | The declared modality's manifest profile rejects the metadata fields (the profile's own reason codes), or the media type has no profile (`unsupported_modality`, currently `application/dicom+json`). |
| `limits` | `evaluate_asset_limits` | A resource ceiling is exceeded (`limit_exceeded`) or cannot be evaluated (`insufficient_metadata`). |
| `digest` | `digest_asset` | The hashed byte count disagrees with the declared size (`byte_count_mismatch`), the digest disagrees (`sha256_mismatch`), or hashing was skipped (`not_evaluated`). |

The modality is derived from the declared media type: `application/pdf`,
`application/dicom`, any `image/` type, and any `audio/` type map to the
`PDF_V1`, `DICOM_V1`, `IMAGE_V1`, and `AUDIO_V1` manifest profiles. The limit
profile defaults to `DESKTOP_V1`; pass `limit_profile=` to apply `MOBILE_V1`
or a caller-owned profile.

The digest pass reads at most one byte past the declared size, and it runs
only when the byte-size ceiling was evaluated and passed. Otherwise it is
reported as `not_evaluated`, so preflight never reads more than the limit
profile's byte ceiling. Streams are read from their current position; seekable
streams are restored on success or failure, non-seekable streams are hashed
through a replay of the already-read prefix, and caller-owned streams are
never closed. Bytes-like values are hashed in memory.

## Decision

The report accepts only when every check passed: the manifest is valid, the
media type matches, the profile reports no findings, every applicable limit
was evaluated and passed, and the digest matches. Anything else abstains.

An abstaining report carries an `AbstentionRecord` for the `preflight` stage.
The coarse reason follows the earliest failing check: `resource_limit` when
that check is `limits`, and `unsupported_media` for every other check, because
the manifest, media-type, metadata, and digest checks all establish whether the
asset is a supported, coherent input. The detailed findings carry the precise
cause.

Missing evidence never becomes acceptance. A rule the profile cannot evaluate
is a finding, not a pass. In particular, a PDF manifest carries no raster
geometry under the current manifest contract, so a PDF whose bytes, pages,
media type, and digest all pass still abstains with `resource_limit` and two
`insufficient_metadata` findings for `pixels` and `total_pixels`. Preflight
does not weaken that gate; supplying page geometry is a separate contract.

## Findings

Findings are returned in the documented check order, and within a check in the
order the underlying contract defines. Each carries:

- `check`: one of `manifest`, `media_type`, `metadata`, `limits`, `digest`.
- `reason_code`: allowlisted for that check. Metadata findings reuse the
  manifest-profile reason codes and limit findings reuse the limit reason
  codes, so no code can be invented here.
- `field_name`: the manifest field for metadata findings, the limit field for
  limit findings, `byte_size` or `sha256` for digest findings, otherwise
  `null`.
- `limit` and `observed`: the inclusive ceiling and the observed number for
  limit findings; the declared size and the number of bytes hashed for a
  byte-count mismatch (`observed` is `null` when reading stopped at the
  declared size); otherwise `null`.

## Report shape

`to_dict()` and `to_json()` produce a fixed key order with no insignificant
whitespace, so equal inputs produce byte-identical output:

```json
{"schema_version":1,"status":"abstain","abstention":{"schema_version":1,"stage":"preflight","reason":"resource_limit"},"manifest":{"version":1,"asset_id":"study-0002","media_type":"application/pdf","sha256":"<sha256>","byte_size":72,"pages":2},"modality":"pdf","metadata_profile":{"modality":"pdf","version":"1.0"},"limit_profile":{"name":"desktop","version":"1.0"},"media_type":{"detected":"application/pdf","status":"match"},"digest":{"sha256":"<sha256>","byte_count":72},"findings":[{"check":"limits","reason_code":"insufficient_metadata","field_name":"pixels","limit":40000000,"observed":null},{"check":"limits","reason_code":"insufficient_metadata","field_name":"total_pixels","limit":100000000,"observed":null}]}
```

The report is composed only of contracts that already enforce the privacy
boundary: the validated manifest, the profile and limit-profile identities,
the detected media type and comparison status, the observed digest, the
abstention record, and the findings. It contains no filenames, paths, source
bytes, OCR text, transcripts, DICOM values, or credentials, and a malformed
manifest leaves every optional field `null`. Source failures raise
`PreflightError` with a stable category (`preflight_source_read_error`,
`preflight_source_position_error`, `preflight_source_restore_error`,
`preflight_source_contract_error`) and no underlying detail attached.

Preflight does not decode media, run a model, downsample inputs, choose
clinical thresholds, or certify that accepted media is benign.
