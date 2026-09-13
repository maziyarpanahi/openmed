# Content-free multimodal provider results

`ProviderResultEnvelope` records the outcome of a VLM, OCR, DICOM, waveform,
or ASR provider call without copying its clinical output into logs or summary
artifacts. The envelope contains stable identifiers, SHA-256 digests, timing,
and bounded aggregate counts only.

```python
from openmed.multimodal.provider_result import (
    ProviderResultEnvelope,
    ProviderResultOutcome,
)

result = ProviderResultEnvelope(
    provider_id="doctr-1",
    model_id="ocr-base-v2",
    input_digest="1" * 64,
    output_digest="2" * 64,
    outcome=ProviderResultOutcome.SUCCESS,
    duration_ms=12.5,
    count_metadata={"page_count": 2, "token_count": 37},
)

encoded = result.to_json()
restored = ProviderResultEnvelope.from_json(encoded)
assert restored == result
```

The JSON representation is deterministic. Mapping order does not affect it,
and parsing produces an immutable envelope with immutable count metadata.

## Outcomes and required fields

| Outcome | `output_digest` | `abstention_code` |
| --- | --- | --- |
| `success` | Required | Must be absent |
| `abstention` | Must be absent | Required |
| `provider_unavailable` | Must be absent | Must be absent |
| `validation_failure` | Must be absent | Must be absent |

An abstention code is selected from a closed vocabulary:

- `unsupported_media`
- `malformed_media`
- `resource_limit`
- `low_quality`
- `phi_uncertainty`
- `speaker_uncertainty`
- `temporal_instability`

Provider unavailability has its own terminal outcome and therefore is not an
abstention code in this envelope.

## Digests, identifiers, and timing

`input_digest` is always required. `output_digest` is present only after a
successful provider call. Each is exactly 64 lowercase hexadecimal characters
and represents a SHA-256 digest computed outside this module. The envelope does
not read, hash, invoke, or store provider inputs and outputs.

Provider and model identifiers are lowercase ASCII identifiers containing at
most 128 letters, digits, dots, underscores, or hyphens. Paths, URLs, arbitrary
messages, credential-like identifiers, prompts, and binary values are rejected.

`duration_ms` is a finite number from zero through 86,400,000 milliseconds.
Negative values, booleans, non-numeric values, NaN, and infinities are rejected.

## Count metadata

The optional `count_metadata` mapping accepts only non-negative integers up to
2<sup>63</sup> - 1. Its closed field vocabulary is:

- `input_bytes`
- `input_items`
- `output_items`
- `page_count`
- `frame_count`
- `sample_count`
- `segment_count`
- `token_count`
- `detection_count`

Unknown counter names, booleans, floats, strings, nested data, and out-of-range
integers fail validation.

## Privacy and validation boundary

The schema has no fields for generated text, OCR text, transcripts, prompts,
DICOM values, pixel data, waveform samples, paths, URLs, credentials, arbitrary
messages, or provider exceptions. Unknown fields and duplicate JSON keys are
rejected. JSON input is limited to 64 KiB.

Validation errors use fixed descriptions of the failed contract and never
include a submitted key or value. Do not attach provider output or raw exception
text beside this envelope when logging it.

This module defines and validates metadata only. It does not invoke a provider,
define a clinical output schema, convert provider-specific output, persist a
result, verify digest content, or attest to output quality.

Duration bounds are checked before float conversion, including oversized Python
integers. Unknown enum values and malformed JSON produce value-free failures
without retaining the original exception or submitted content in a cause chain.
