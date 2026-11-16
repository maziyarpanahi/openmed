# Portable SDK configuration

Python and the OpenMedKit ports for Swift, Kotlin and the web spell the same
settings differently today. The portable SDK configuration is one small JSON
document that every surface reads with the same key names, types, bounds and
defaults. Version 1 has six keys and rejects any other key.

| Key | Type and bounds | Default | Python source |
|---|---|---|---|
| `confidence_threshold` | number in `[0, 1]` | `0.5` | per-call `confidence_threshold` argument of `extract_pii` (default `0.5`) and `deidentify` (default `0.7`); not an `OpenMedConfig` field |
| `local_only` | boolean | `true` | `OpenMedConfig.local_only` (Python default `false`; `OPENMED_OFFLINE` turns it on) |
| `model_id` | repository id or `null` | `null` (kit default model) | `OpenMedConfig.pii_model` |
| `model_revision` | revision or `null`; needs a non-null `model_id` | `null` | `OpenMedConfig.pii_model_revision` |
| `onnx_variant` | `auto`, `int8`, `fp16`, `fp32` | `auto` | `OpenMedConfig.onnx_variant` |
| `policy_profile` | one of the portable policy profiles | `hipaa_safe_harbor` | per-call `policy` argument of `deidentify`; not an `OpenMedConfig` field |

The schema is `openmed/core/schemas/json/sdk_config.schema.json`
(`$id` `https://openmed.life/schemas/sdk_config.schema.json`). Types and enums
match `openmed/core/config.schema.json`. The portable document is stricter in
three places:

- `local_only` defaults to `true`, so a kit that reads an empty document stays
  offline. The Python `OpenMedConfig` default stays `false`.
- `model_id` must look like a Hub repository id
  (`namespace/name` or `name`, ASCII letters, digits, `.`, `_` and `-`, no
  `..`). Filesystem paths and URLs are refused because they are not portable.
- `model_revision` uses the same characters without `/`, and only makes sense
  with a `model_id`; on its own it fails with `missing_dependency`.

`policy_profile` accepts the canonical names in
`openmed.core.portable_policy.PORTABLE_PROFILES`: `australia_privacy_act`,
`canada_pipeda`, `clinical_minimal_redaction`, `gdpr_art9_health`,
`gdpr_pseudonymization`, `hipaa_expert_review_assist`, `hipaa_safe_harbor`,
`research_limited_dataset` and `strict_no_leak`. Aliases and Python-only
profiles are refused.

`onnx_variant` is ignored by runtimes that do not load ONNX weights, such as
the Swift kit's MLX and Core ML backends.

## Spelling in each kit today

The kit loaders that read this document are separate work (#5387 Swift,
#5388 Kotlin, #5389 web). Until they land, the same intent is spelled like
this:

| Key | Swift (OpenMedKit) | Kotlin (OpenMedKit) | TypeScript (`openmedkit-web`) |
|---|---|---|---|
| `confidence_threshold` | `confidenceThreshold:` per call, default `0.5` | `confidenceThreshold` per call, default `0.5f` | `ExtractPiiOptions.threshold`, default `0` |
| `local_only` | `allowNetworkAccess:` per initializer, inverted, default `false` | `ArtifactAcquisitionRequest.networkConsent`, default `DENIED` | `offlineOnly`, `LoadModelOptions.localFilesOnly`, `allowRemoteModels` |
| `model_id` | `OpenMedModelStore.downloadMLXModel(repoID:)` | `ArtifactRef.repoId` | `ExtractPiiOptions.model` |
| `model_revision` | `revision:`, default `"main"` | `ArtifactRef.revision` | `LoadModelOptions.revision` |
| `onnx_variant` | not used (MLX and Core ML) | `OpenMedKit.fromDirectory(variant =)`, default `"int8"` | `LoadOnnxModelOptions.variant` |
| `policy_profile` | `deidentify(_:policy:)`, required | `deidentify(policy =)`, default `hipaa_safe_harbor` | `loadPortablePolicy(name)`, default `hipaa_safe_harbor` |

The defaults that differ today (the web threshold of `0`, the Kotlin variant
of `int8`) are what the portable defaults replace once each loader lands.

## Web package

The `openmed` web package reads the document with `parseSdkConfig(value)`,
which takes a parsed JSON value and returns a frozen configuration with all
six keys. `validateSdkConfig(value)` returns the ordered issues without
throwing; `parseSdkConfig` throws `SdkConfigError`, whose `issues` are the
same `{ code, field }` list and whose message names codes and known fields
only. The existing option types are unchanged; two helpers map onto them:

```ts
import {
  deidentify,
  loadPortablePolicy,
  parseSdkConfig,
  toExtractPiiOptions,
} from "openmed";

const config = parseSdkConfig(JSON.parse(configText));
const result = await deidentify(note, toExtractPiiOptions(config));
const policy = loadPortablePolicy(config.policy_profile);
```

| Key | Web option |
|---|---|
| `confidence_threshold` | `ExtractPiiOptions.threshold` |
| `local_only` | `LoadModelOptions.localFilesOnly: true` and `allowRemoteModels: false`; `false` sets the opposite |
| `model_id` | `ExtractPiiOptions.model`; `null` keeps `DEFAULT_MODEL_ID` |
| `model_revision` | `LoadModelOptions.revision`; `null` keeps the loader's revision (`DEFAULT_MODEL_REVISION` for the default model) |
| `onnx_variant` | `LoadOnnxModelOptions.variant`; `auto` keeps the loader default (`int8`) |
| `policy_profile` | not an extraction option; pass it to `loadPortablePolicy()` |

`toLoadModelOptions(config)` returns the loader options and
`toExtractPiiOptions(config)` returns `threshold`, `model` and those
`loaderOptions`. Both revalidate their input. `local_only: true` never
downloads from the Hub; a Hub id still resolves against the local model path
and the Transformers.js cache. The package's stricter `offlineOnly` mode, which
refuses Hub ids outright, stays a separate opt-in.
`tests/web/test_sdk_config.spec.ts` runs every shared vector.

## Validation and error codes

Validation fails closed. Every problem is an issue with a `code` and a
`field`. `field` is one of the six keys, or `null` for document-level issues.
Issues never contain an input value or a caller-supplied key name.

| Code | Field | Meaning |
|---|---|---|
| `config_not_object` | `null` | The document is not a JSON object |
| `unknown_field` | `null` | At least one key is not one of the six keys |
| `wrong_type` | key | Wrong JSON type (`true` is not a number; `null` only for `model_id` and `model_revision`) |
| `out_of_range` | key | `confidence_threshold` outside `[0, 1]` |
| `not_allowed_value` | key | `onnx_variant` or `policy_profile` outside its enum |
| `invalid_format` | key | `model_id` or `model_revision` does not match its pattern |
| `missing_dependency` | `model_revision` | A revision is set while `model_id` is absent or `null` |

Issues come in a fixed order so every runtime can return the same list:
`config_not_object` alone when the document is not an object; otherwise at
most one `unknown_field`, then at most one issue per key in alphabetical key
order.

## Python API

```python
from openmed.core.config import OpenMedConfig
from openmed.core.sdk_config import resolve_sdk_config, validate_sdk_config

validate_sdk_config({"local_only": "yes"})
# (SdkConfigIssue(code='wrong_type', field='local_only'),)

resolve_sdk_config({"model_id": "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-mlx"})
# all six keys, absent ones filled with the defaults above

OpenMedConfig.from_profile("low_resource").to_sdk_config(policy_profile="strict_no_leak")
```

`OpenMedConfig.to_sdk_config(confidence_threshold=0.5,
policy_profile="hipaa_safe_harbor")` returns the validated subset of a config.
It raises `SdkConfigError` when a value is outside the portable subset, for
example a `pii_model` that is a local path.

## Shared vectors

`tests/fixtures/parity/sdk_config_v1.json` is the cross-runtime contract. It
is synthetic and holds the field list, the defaults, the closed issue-code
list, valid documents with their resolved form, and invalid documents (unknown
keys, wrong types, out-of-range values, bad formats, a missing dependency)
with the exact ordered issues each must produce. Every port must pass every
vector. `tests/unit/core/test_sdk_config.py` checks the Python reference and
the JSON Schema against them.

Adding a key or changing a default is a new version: it bumps
`x-sdk-config-version` in the schema and ships a new `sdk_config_v2.json`
vector file. The version 1 vectors are never edited.
