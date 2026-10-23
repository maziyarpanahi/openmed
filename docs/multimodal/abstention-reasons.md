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

- `schema_version`: `1` for legacy codes, `2` for the six codes below
- `stage`: one documented stage
- `reason`: one reason allowed for that stage

Unknown fields, unknown values, malformed JSON, unsupported schema versions,
and invalid stage-reason pairs fail closed. Validation errors name the failed
part of the contract but never echo the submitted value.


## Consent, capture, language and policy vocabulary (v2)

The following codes are shared with `ProviderAbstentionCode`. They add no fields,
free-text explanation, consent identity, language tag or captured content.

| Code | Valid stages | Meaning |
| --- | --- | --- |
| `consent_unavailable` | `preflight` | Required consent is missing or cannot be established before processing starts. |
| `consent_withdrawn` | `preflight`, `decode`, `inference`, `post_process` | Previously established consent has been withdrawn; stop at whichever gate observes withdrawal. |
| `capture_interrupted` | `decode`, `inference`, `post_process` | Capture stopped before completion; detected during ingestion, streaming inference or finalization. |
| `language_unsupported` | `preflight`, `inference`, `post_process` | A declared or identified language has no supported local processing path. |
| `language_uncertain` | `inference`, `post_process` | Measured language evidence is missing, conflicting or below the caller's confidence gate. |
| `policy_refused` | `preflight`, `inference`, `post_process` | A policy gate refuses the requested processing or release. |

The earlier stage table describes legacy codes; this table adds the v2 pairs.
Decode validates media, rather than choosing language routes or evaluating
processing/release policy. Consent unavailability gates entry; withdrawal can
stop any stage. Capture interruption requires capture to have started.

Omitting `schema_version` when constructing a record selects the version from
the reason. Legacy codes still produce exactly the same v1 bytes. New codes
produce v2 bytes. An explicit version must match the reason: v1 with a new code,
v2 with a legacy code, booleans, null and unknown versions are rejected by the
wire parser. `ABSTENTION_SCHEMA_VERSION` names the latest supported version (2),
not the version of every record.

```python
record = AbstentionRecord(
    stage=AbstentionStage.PREFLIGHT,
    reason=AbstentionReason.CONSENT_UNAVAILABLE,
)
assert record.to_json() == (
    '{"schema_version":2,"stage":"preflight","reason":"consent_unavailable"}'
)
```

This vocabulary does not establish consent, route languages, invoke providers,
or authorize a clinical decision. Session, CLI, REST and Swift adapters adopt
these codes in their own issues. Consequential adapter outputs still require
explicit reviewer confirmation and a non-diagnostic notice.

## JSON Schema drift expectations

The multimodal JSON Schema exporter is tracked separately in
[#3051](https://github.com/maziyarpanahi/openmed/issues/3051), with an implementation
pending in [PR #3616](https://github.com/maziyarpanahi/openmed/pull/3616). There is
no exporter in this baseline; this vocabulary change does not duplicate it.

Exporter integration must preserve the closed v1 enum and stage pairs, add a
separate v2 branch restricted to the six new codes and their pairs above, and
keep `additionalProperties: false`. The provider-result branch must bind
`openmed.multimodal.provider_result.v2` to outcome `abstention` and a new code;
all other existing records remain `openmed.multimodal.provider_result.v1`.
Version-specific identifiers and pinned schema/catalog digests must be updated
when those definitions land. Drift tests must compare every stage/code/version
combination with the Python loaders, reject unknown/free-text fields, and retain
the synthetic v1 golden bytes in `tests/fixtures/multimodal/abstention_v1.json`.
Do not widen a published v1 enum or mechanically substitute the latest version
constant for every record's version. Containers carrying these reasons need
their own version-aware adoption before emitting new codes.
