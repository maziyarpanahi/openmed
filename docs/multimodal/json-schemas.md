# Multimodal preflight JSON Schemas

The multimodal preflight artifacts are metadata-only records: asset manifests,
asset batches, abstention records, processing summaries, and provider result
envelopes. Each one has a Python loader that fails closed on malformed input, and
`openmed.multimodal.schemas` exports the same contract as a self-contained
[JSON Schema draft 2020-12](https://json-schema.org/draft/2020-12/schema)
document for tools that never import OpenMed.

```python
import json

from jsonschema import Draft202012Validator

from openmed.multimodal.schemas import (
    export_multimodal_schema,
    export_multimodal_schemas_json,
)

schema = export_multimodal_schema("provider_result")
Draft202012Validator.check_schema(schema)
Draft202012Validator(schema).validate(
    {
        "schema_version": "openmed.multimodal.provider_result.v1",
        "provider_id": "doctr-1",
        "model_id": "ocr-base-v2",
        "input_digest": "1" * 64,
        "output_digest": "2" * 64,
        "outcome": "success",
        "abstention_code": None,
        "duration_ms": 12.5,
        "count_metadata": {"page_count": 2},
    }
)

catalog = json.loads(export_multimodal_schemas_json())
assert set(catalog) == {
    "asset_manifest",
    "asset_batch",
    "abstention_record",
    "processing_summary",
    "provider_result",
}
```

## Schema catalog

| Name | `$id` | Python contract |
| --- | --- | --- |
| `asset_manifest` | `https://openmed.ai/schemas/multimodal/asset-manifest-v1.schema.json` | `AssetManifest` |
| `asset_batch` | `https://openmed.ai/schemas/multimodal/asset-batch-v1.schema.json` | `AssetBatch` |
| `abstention_record` | `https://openmed.ai/schemas/multimodal/abstention-record-v1.schema.json` | `AbstentionRecord` |
| `processing_summary` | `https://openmed.ai/schemas/multimodal/processing-summary-v1.schema.json` | `ProcessingSummary` |
| `provider_result` | `https://openmed.ai/schemas/multimodal/provider-result-v1.schema.json` | `ProviderResultEnvelope` |

The exported identifiers are stable and versioned. A contract change that alters
any of these documents must introduce a new version suffix rather than mutate a
published identifier.

## Determinism and offline resolution

Every export builds a fresh document, so callers may keep or mutate one without
affecting a later call. Rendering is byte-stable: keys are sorted, separators are
compact, non-finite numbers are refused, and no trailing newline is added.

All `$ref` values are fragment-local (`#/$defs/...`) and every referenced
definition is contained in the same document. Validators therefore resolve the
whole contract without network access or a shared schema store. Passing a registry
whose retrieval hook raises on any URI is a useful assertion that no external
schema is required.

The same fields, enumerations, digest patterns, bounds, and version constants are
shared with the Python loaders. `tests/unit/multimodal/test_schemas.py` binds the
two representations together, so a loader change that is not reflected in the
exported schema fails the drift tests, and pinned digests make an unintended
rendering change visible.

## Bounds and enumerations

- Manifest and batch identifiers are lowercase-safe opaque identifiers bounded to
  128 characters; paths, URLs, and home-relative strings are rejected.
- SHA-256 digests are exactly 64 lowercase hexadecimal characters.
- Media types are lowercase `type/subtype` values limited to the supported
  document, image, and audio families.
- `byte_size` is an integer from 1 through 2<sup>63</sup> - 1; page, frame, and
  dimension counts are integers from 1 through 2<sup>31</sup> - 1; durations are
  finite positive numbers bounded by 2<sup>31</sup> - 1 seconds.
- Batch aggregates are bounded by the same limits and batches hold between 1 and
  10,000 manifests.
- Abstention stages and reasons are closed vocabularies, and each reason is
  validated against the stage it is reported for.
- Processing summary totals are non-negative; group counts are at least one.
- Provider results bind the outcome to the optional fields: `success` requires
  `output_digest`, `abstention` requires `abstention_code`, and the remaining
  outcomes must omit both. Provider and model identifiers use a closed lowercase
  vocabulary of at most 128 characters with credential-like segments rejected.

Every object in the exported documents is closed with
`"additionalProperties": false` except the bounded free-form `count_metadata`
mapping, which is closed by `propertyNames` instead.

## Known limits

A JSON Schema cannot express every rule that the Python loaders enforce:

- JSON `integer` accepts a value such as `1.0`, while the loaders require an
  actual Python `int`.
- NaN is not representable in JSON text and compares false against every numeric
  bound, so it can pass the numeric keywords. The loaders reject it and the
  renderers set `allow_nan=False`.
- Cross-record batch rules - distinct asset identifiers, distinct digests, and
  aggregates that match the listed manifests - are enforced by `AssetBatch`,
  not by the schema.
- Processing summary totals have no artificial upper bound because one run may
  aggregate several bounded artifacts.

Validating a document against the schema is therefore a first-pass check, not a
replacement for the loaders, which remain the authority for accepting input.

## Privacy and scope

The exported documents describe metadata only. They contain no fields for
generated text, OCR text, transcripts, prompts, DICOM values, pixel data,
waveform samples, paths, URLs, credentials, or arbitrary messages. Exporting a
schema is a local, deterministic operation: it does not read the filesystem,
contact a provider, persist anything, or attest to output quality.

See [Privacy-safe asset manifests](asset-manifests.md),
[Privacy-safe asset batches](asset-batches.md),
[Multimodal abstention reasons](abstention-reasons.md),
[Processing summaries](processing-summaries.md), and
[Content-free provider results](provider-results.md) for the corresponding
loaders and their validation boundaries.
