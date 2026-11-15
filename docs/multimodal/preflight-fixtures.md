# Synthetic malformed-asset preflight fixtures

The version-1 bundled negative pack contains **13** minimal synthetic cases for
PDF, PNG, JPEG, TIFF, DICOM and WAV. Each case declares stable expected finding
codes from the existing `preflight_asset` or `validate_asset_batch` contract.
Headers are at most 132 bytes: they contain signatures, truncations and zero
preambles, with no image pixels, audio samples, DICOM dataset, patient text or
credentials. These cases are not fuzzing, malware or real clinical media.

```python
from openmed.multimodal.preflight_fixtures import load_preflight_fixtures

for case in load_preflight_fixtures():
    assert case.actual_findings() == case.expected_findings
```

Cases cover each format's truncated signature, a declared-type mismatch, zero
dimensions, an oversized frame count, a bad digest, duplicate asset IDs/content,
unknown media and a pixel-resource ceiling. The PDF truncation also retains the
existing resource contract's insufficient-geometry finding. Expected codes are
complete code sets, sorted without copying asset values or source bytes.

The loader checks a strict schema and JSON fields, duplicate JSON keys, unique
case IDs, closed categories and expected codes, numeric metadata, hex encoding,
at most 64 cases and a 64 KiB input ceiling. Cases and manifest mappings are
immutable. Optional payloads must be synthetic JSON bytes or text; filenames,
transports and unknown payload fields are not accepted. The default pack ships
under `openmed/multimodal/fixtures/malformed_asset_preflight.json`.

The single-asset preflight detects signatures, validates declared metadata,
checks limits and hashes bytes; it does **not** deeply parse every header or
prove a file is decodable. These fixtures exercise that existing boundary and
do not claim complete malformed-file coverage, PHI absence in arbitrary input,
clinical validity or model qualification.
