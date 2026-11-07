# Multimodal Abstention Reasons

Image, document, waveform, and audio pipelines should explain why processing
stopped without copying source content into logs or audit artifacts. Use
`AbstentionRecord` for that boundary. Its serialized form contains only a
schema version, a pipeline stage, and a stable reason code.

```python
from openmed.multimodal.abstention import (
    AbstentionReason,
    AbstentionRecord,
    AbstentionStage,
)

record = AbstentionRecord(
    stage=AbstentionStage.INFERENCE,
    reason=AbstentionReason.PHI_UNCERTAINTY,
)
print(record.to_json())
```

```json
{"schema_version":1,"stage":"inference","reason":"phi_uncertainty"}
```

There is intentionally no message or context field. Do not add OCR text,
transcripts, pixel data, DICOM values, paths, URLs, prompts, or provider error
text beside this record. Keep operational detail in a separate private system
whose retention and access policy is appropriate for PHI.

## Stages and allowed reasons

| Stage | Allowed reason codes |
| --- | --- |
| `preflight` | `unsupported_media`, `resource_limit`, `provider_unavailable` |
| `decode` | `malformed_media`, `resource_limit`, `low_quality` |
| `inference` | `resource_limit`, `low_quality`, `phi_uncertainty`, `speaker_uncertainty`, `temporal_instability`, `provider_unavailable` |
| `post_process` | `resource_limit`, `low_quality`, `phi_uncertainty`, `speaker_uncertainty`, `temporal_instability` |

Use the earliest stage that can make the decision. For example, reject an
unsupported container during preflight and malformed bytes during decode.
Low-quality output belongs to decode, inference, or post-processing depending
on where the measurable quality gate runs.

## Strict parsing

`AbstentionRecord.from_json()` requires exactly these fields:

- `schema_version`: currently `1`
- `stage`: one documented stage
- `reason`: one reason allowed for that stage

Unknown fields, unknown values, malformed JSON, unsupported schema versions,
and invalid stage-reason pairs fail closed. Validation errors name the failed
part of the contract but never echo the submitted value.
