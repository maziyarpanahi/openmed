# Clinical task router and model packs

OpenMed model packs route bounded clinical NLP tasks to pinned local
specialists. A route is a metadata and integrity decision: the router does not
download a model, contact a provider, import an optional runtime, or perform
inference.

The public contracts are in `openmed.clinical.model_packs`. Their initial
schema version is `1.0.0`, with a `same_major` compatibility policy. Bundled
JSON Schemas describe model packs and path-free route records.

## Why the router is separate

Span extraction, classification, pair scoring, relation extraction, assertion,
temporality, and token classification have different output shapes. A single
implicit model selection rule makes those shapes difficult to reproduce and
can silently substitute a generative model for a bounded decision.

The task router instead requires a model-pack entry to declare:

- one task and output schema;
- an immutable artifact revision and SHA-256 digest;
- an explicit local alias;
- license, runtime, model kind, and quantization mode;
- language and domain coverage;
- seeded calibration and independent holdout evidence when calibration is
  claimed;
- measured quantization delta and its maximum tolerance for quantized
  artifacts; and
- an optional fallback alias that points to another entry in the same pack.

This contract can describe GLiNER2.5-class zero-shot span encoders,
DeBERTa-style and ModernBERT-style classifiers, existing OpenMed token
classifiers, and deterministic rule implementations without coupling callers
to a single model family.

`MODEL_FAMILY_CAPABILITIES` publishes those family classes and their bounded
task/runtime shapes. It is a capability catalog, not an artifact catalog: an
entry still needs its own pinned revision, digest, license, calibration, and
local alias.

## Offline and local-only boundary

`artifact_id` is metadata. It is never treated as a filesystem path or remote
location. Every usable entry must have a `LocalArtifactBinding` supplied by the
caller:

```python
from pathlib import Path

from openmed.clinical.model_packs import (
    ClinicalTaskRequest,
    ClinicalTaskRouter,
    LocalArtifactBinding,
    load_model_pack,
)

pack = load_model_pack("model-pack.json")
router = ClinicalTaskRouter(
    pack,
    bindings=(
        LocalArtifactBinding(
            alias="span.clinical",
            runtime="torch",
            path=Path("/opt/openmed/models/span/model.safetensors"),
        ),
    ),
    available_runtimes=("torch",),
)

result = router.route(
    ClinicalTaskRequest(
        task="span_extraction",
        output_schema="clinical_component",
        output_schema_version="1.0.0",
        language="en",
        domain="clinical",
    )
)
```

Constructing or using the router does not make a network request. Downloading
an artifact is a separate, explicit operator action. A registry identifier in
a model pack cannot bypass the local binding requirement.

## Selection order

For a request without an explicit alias, the router filters entries by task,
output-schema major version, language, and domain. It then applies this stable
ordering:

1. exact language before multilingual coverage;
2. exact domain before `general` coverage;
3. the caller's runtime preference;
4. bounded specialists before deterministic implementations and experimental
   generative entries;
5. numeric priority and then alias.

An explicit `requested_alias` must satisfy the same task, schema, language, and
domain constraints. It never falls through to an unrelated entry.

Generative entries must be marked experimental in the pack, named through
`requested_alias`, and accompanied by `allow_experimental_generative=True` on
the request. They are not selected for bounded tasks by default, even when
assigned a numerically higher priority.

## Verification order

The selected entry passes gates before a route is returned:

1. the declared license must be in the router's allowlist;
2. a quantized artifact's measured delta must not exceed its declared maximum;
3. the runtime must be explicitly available;
4. the local alias must be bound and the binding runtime must match;
5. the local file or directory must exist, must not be a symbolic link, and
   must match the pinned artifact digest.

File digests cover exact bytes. Directory digests cover sorted relative paths
and exact file bytes, so renaming a file changes the digest. Empty directories
and symbolic links are rejected.

License, failed quantization qualification, and integrity failures are
fail-closed. They do not invoke a fallback. This prevents a changed or
unapproved artifact from being hidden by an apparently successful route.

## Explicit fallback

A fallback is followed only when the primary entry reports an unavailable
runtime or an unbound local alias. The fallback must be named in the primary
entry, must exist in the same pack, and must implement the same task. Cycles
and self-references are rejected when the manifest is constructed.

Deterministic implementations use a `builtin` binding with an explicit ID and
digest. `builtin` is always an available runtime, but the ID and digest still
have to match the manifest. Successful fallback routes preserve
`fallback_from` in their provenance.

## Typed outcomes

The router uses the Journey `StoreResult` envelope. Non-success states never
become apparent success:

| State | Example code | Meaning |
| --- | --- | --- |
| `unsupported` | `task_not_supported` | No compatible task/schema/language/domain entry exists. |
| `unsupported` | `runtime_unavailable` | The selected optional runtime is absent and no fallback succeeds. |
| `unknown` | `alias_unbound` | The pack is known but the operator supplied no local binding. |
| `conflict` | `artifact_digest_mismatch` | Local bytes differ from the pinned digest. |
| `denied` | `license_denied` | License policy refuses the entry. |
| `denied` | `quantization_delta_exceeded` | The recorded quantized holdout delta is outside tolerance. |
| `denied` | `generative_profile_required` | A bounded request did not explicitly allow an experimental generative entry. |
| `failure` | `artifact_verification_failed` | Local artifact verification could not complete. |

The serialized `ModelRoute` contains pack, artifact, schema, runtime, and
fallback provenance. It deliberately excludes the local filesystem path. The
in-process `local_reference` remains available to the loader but is marked
non-repr and has no field in the public route schema.

## Calibration and quantization

Calibrated entries record the method, threshold, calibration dataset digest,
independent holdout digest, and random seed. `method="none"` cannot carry
calibration claims.

Quantized entries record the metric, measured delta, maximum accepted delta,
and evaluation digest. A quantized entry outside its declared tolerance is
denied before its artifact is opened. Full-precision entries cannot attach
quantized-delta claims.

These are provenance and admission gates, not clinical-performance claims.
Each task/language/domain combination still needs its own evaluation and
review policy.

## Manifest updates and replay

`ModelPackManifest.digest` hashes canonical JSON. Repeating the same request
against the same manifest, local bindings, and runtime capabilities therefore
resolves the same pinned artifact. A revision, digest, policy, calibration,
fallback, or priority change produces a different pack digest rather than
overwriting prior provenance.

Persist both `pack_digest` and the path-free route record with downstream stage
manifests. Retain older packs while their derived facts remain queryable.

## Privacy and safety

- Do not place source text, prompts, credentials, patient identifiers, or
  filesystem paths in a model pack.
- Do not serialize `local_reference` into audit events.
- Keep restricted assets user supplied and outside package data.
- Treat a successful route as artifact qualification, not as clinical
  validation or authorization for patient-care action.
- Propagate partial, unknown, conflict, unsupported, denied, and failure states
  to the caller without converting them into model output.

The committed golden fixture is synthetic. It proves pack/schema compatibility
and deterministic fallback metadata without bundling model weights or clinical
data.
