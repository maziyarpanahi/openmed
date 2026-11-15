# Decision calibration profiles in bundles

Contract version 1 (OpenMed v4.0). A decision model and the thresholds
qualified for it must ship together. Otherwise a deployment can load new
weights with old thresholds, or a quantized variant with thresholds qualified
on full precision. The `org.openmed.decision_calibration` extension of the
[signed portable bundle manifest](portable-bundle.md) carries the calibration
profiles inside the signed manifest. The bundle signature therefore covers
every threshold, and every profile is bound to the model files it was
qualified on.

The extension carries thresholds and digests only. Calibration data, examples,
and evaluation inputs never ship in a bundle. Decision results stay assistive
review artifacts, and nothing in this contract triggers a clinical action.

## Extension value

The value of `extensions["org.openmed.decision_calibration"]` is:

```json
{
  "schema_version": 1,
  "profiles": [
    {
      "calibration_id": "openmed.synthetic.triage.v1",
      "version": "1.0.0",
      "method": "deterministic_margin",
      "parameters": {},
      "thresholds": {
        "minimum_confidence_ppm": 700000,
        "minimum_margin_ppm": 150000,
        "multi_label_threshold_ppm": 700000
      },
      "scope": {"kind": "options", "digest": "sha256:<hex>"},
      "model": {
        "id": "OpenMed/synthetic-tiny-decision",
        "revision": "<40-hex commit>",
        "runtimes": [{"id": "onnx-fp32", "digest": "sha256:<hex>"}]
      },
      "decision_schema": {"min": "1.0.0", "max_exclusive": "2.0.0"},
      "provenance": {
        "dataset_digest": "sha256:<hex>",
        "in_domain_digest": "sha256:<hex>",
        "out_of_domain_digest": "sha256:<hex>",
        "evaluation": {"path": "evidence/decision-calibration.json", "sha256": "sha256:<hex>"}
      },
      "profile_digest": "sha256:<hex>"
    }
  ]
}
```

The JSON Schema is
`openmed/core/schemas/json/portable_bundle_decision_calibration.schema.json`.
Like the manifest, the extension uses canonical JSON, so it holds no
non-integer numbers. Every probability is an integer in parts per million
(`700000` is `0.7`).

| Field | Meaning |
|---|---|
| `calibration_id`, `version` | The profile identity. `calibration_id` is the identifier a decision request names. It matches `DecisionCalibrationProfile.calibration_id` in `openmed.structured.decision`. |
| `method`, `parameters` | One of `deterministic_margin` (no parameters), `temperature_scaling` (`temperature_ppm`, 1 to 100,000,000), `platt_scaling` (`slope_ppm` non-zero and `intercept_ppm`, each within ±100,000,000), or `isotonic_regression` (2 to 64 `points` of `input_ppm`/`output_ppm`, with strictly increasing inputs and non-decreasing outputs). Any other method fails closed. |
| `thresholds` | The abstention bands of the decision contract. A result abstains with `low_confidence` below `minimum_confidence_ppm`, with `ambiguous_scores` when the top-two margin is below `minimum_margin_ppm`, and with `no_label_above_threshold` when no multi-label score reaches `multi_label_threshold_ppm`. `minimum_margin_ppm` cannot exceed `minimum_confidence_ppm`. |
| `scope` | What the profile was calibrated for. `labels` is the bundle label map, and its digest must equal the label file's inventory digest. `options` is an ordered option list. `questions` is an ordered question set. |
| `model` | The model the thresholds were qualified on: the manifest `model.id` and `model.revision`, plus one runtime digest for each qualified runtime variant (1 to 32). |
| `decision_schema` | The decision schema versions the profile is compatible with, as `min <= version < max_exclusive`. |
| `provenance` | Digests of the frozen calibration data and its in-domain and out-of-domain slices. Optionally, a bundled evaluation report (`path`, `sha256`). |
| `profile_digest` | The profile's self digest, which deployment qualification records can pin. |

A bundle may carry 1 to 16 profiles. One `calibration_id` may appear in several
profiles, for example with stricter thresholds for an `int8` variant. Each
`(calibration_id, runtime)` pair must still be unique, so a loaded runtime
always selects exactly one profile.

## Digests

All digests are `sha256:` plus the lowercase hex SHA-256 of an RFC 8785
canonical encoding:

| Digest | Input |
|---|---|
| `profile_digest` | The profile object without its `profile_digest` member |
| Runtime digest | `{"entrypoint", "files": [{"path", "sha256"}], "format", "id", "precision"}` of the runtime, with its files in declared order and each `sha256` from the manifest inventory |
| `options` scope | The ordered option array, for example `["no","unsure","yes"]` |
| `questions` scope | The ordered array of question objects (`DecisionQuestion.to_dict()`) |
| `labels` scope | The label file's `sha256` in the manifest inventory |

Re-exported weights, a changed precision, or a different entrypoint all change
the runtime digest. A bundle rebuilt with new weights but stale calibration
therefore fails.

## Verification

`org.openmed.decision_calibration` is a built-in extension
(`BUILTIN_EXTENSIONS` in `openmed.core.portable_bundle`). A reader that lists it
in `understood_extensions` validates its content during bundle verification,
right after the `unknown_critical_extension` check. A malformed or
inconsistent extension fails the bundle with `extension_invalid`. Producers
mark the extension critical by default. A reader that does not implement it
then refuses the bundle with `unknown_critical_extension` instead of running a
decision model without its qualified thresholds. A bundle that lists the
extension as critical but does not carry it fails with `manifest_malformed`.

Readers that ignore a non-critical extension accept the bundle unchanged.

## Reading and resolving

The reader functions take the manifest bytes and the verifier result for those
bytes. The result must be `ok` and carry the same content identity. Resolving
also requires `files_verified`, because the caller is about to load model files.
Every reader reports the first failing check in this order:

| Step | Reason code |
|---|---|
| Verified bundle with this content identity (and verified files when resolving) | `calibration_bundle_unverified` |
| Extension present | `calibration_extension_missing` |
| Extension object with an integer `schema_version` | `calibration_malformed` |
| `schema_version` is 1 | `calibration_version_unsupported` |
| Exact members, 1 to 16 profiles, every profile's shape, unique `(calibration_id, runtime)` pairs | `calibration_malformed` |
| For each profile in order: the self digest | `calibration_digest_mismatch` |
| Model id and revision equal the manifest model | `calibration_model_mismatch` |
| Every bound runtime is declared and its digest matches | `calibration_runtime_mismatch` |
| A `labels` scope matches the label map digest | `calibration_scope_mismatch` |
| The evaluation report is a declared file with that digest | `calibration_evidence_mismatch` |

Resolving then selects a profile for what the caller loaded:

| Step | Reason code |
|---|---|
| A profile with the requested `calibration_id` | `calibration_profile_missing` |
| That profile binds the loaded runtime | `calibration_runtime_unbound` |
| The loaded model id, revision, and runtime digest match, when given | `calibration_model_mismatch` |
| The caller's decision schema version is in range | `calibration_schema_incompatible` |
| The caller's scope kind and digest match, when given | `calibration_scope_mismatch` |
| The pinned profile digest matches, when given | `calibration_pin_mismatch` |

A malformed request raises `calibration_request_invalid`. Results and errors
hold only reason codes, identifiers, and digests.

| Surface | Entry points |
|---|---|
| Python | `openmed.core.portable_bundle_calibration.read_decision_calibration`, `resolve_decision_calibration`, `DecisionCalibrationRequest` |
| Swift (OpenMedKit) | `PortableBundleCalibration.read(manifest:verification:)`, `PortableBundleCalibration.resolve(manifest:verification:request:)` |
| Kotlin (OpenMedKit Android) | `com.openmed.openmedkit.bundle.PortableBundleCalibration.read(...)`, `.resolve(...)` |
| TypeScript (`openmed` npm) | `readDecisionCalibration(...)`, `resolveDecisionCalibration(...)` |

The Swift, Kotlin, and TypeScript readers parse and validate profiles. They do
not run decision models; on-device decision runtimes consume these profiles
separately.

## Producing profiles

`build_calibration_profile` reads the model identity, runtime digests, label
map digest, and evaluation report digest from the manifest, then computes the
self digest. `with_decision_calibration` adds the extension, marks it critical,
and validates the result:

```python
from openmed.core.portable_bundle import encode_portable_manifest, sign_portable_manifest
from openmed.core.portable_bundle_calibration import (
    DECISION_CALIBRATION_EXTENSION,
    build_calibration_profile,
    options_scope_digest,
    with_decision_calibration,
)

profile = build_calibration_profile(
    manifest,
    calibration_id="org.example.triage.v1",
    version="1.0.0",
    runtime_ids=["onnx-fp32", "coreml-fp16"],
    scope_kind="options",
    scope_digest=options_scope_digest(["no", "unsure", "yes"]),
    decision_schema_min="1.0.0",
    decision_schema_max_exclusive="2.0.0",
    dataset_digest=report["fixture_digest"],
    in_domain_digest=in_domain_digest,
    out_of_domain_digest=out_of_domain_digest,
    minimum_confidence_ppm=700000,
    minimum_margin_ppm=150000,
    multi_label_threshold_ppm=700000,
    evaluation_path="evidence/decision-calibration.json",
)
manifest = with_decision_calibration(manifest, [profile])
data = encode_portable_manifest(manifest, understood_extensions=[DECISION_CALIBRATION_EXTENSION])
envelope = sign_portable_manifest(data, [("org.example.release-2026", my_ed25519_seed)])
```

`openmed.structured.decision_bundle.bundle_profile_from_decision` builds the
same entry from a qualified `DecisionCalibrationProfile`. It refuses a
threshold that is not exact in parts per million, so a qualified threshold
never changes when it is signed into a bundle.

## Using a profile in Python

`resolve_bundle_decision_profile` checks the request's calibration id,
contract version, and options or questions, and the backend's model id and
revision. It returns a `DecisionCalibrationProfile` for `decide`:

```python
from openmed.core.portable_bundle import VerificationContext, verify_portable_bundle_dir
from openmed.core.portable_bundle_calibration import DECISION_CALIBRATION_EXTENSION
from openmed.structured.decision import decide
from openmed.structured.decision_bundle import resolve_bundle_decision_profile

context = VerificationContext(target=target, understood_extensions={DECISION_CALIBRATION_EXTENSION})
verification = verify_portable_bundle_dir(bundle_dir, trust_policy=policy, context=context)
profile = resolve_bundle_decision_profile(
    manifest_bytes,
    verification,
    request=request,
    runtime_id="onnx-int8",
    backend=backend.identity,
    profile_digest=qualified_profile_digest,
)
result = decide(request, backend=backend, calibration_profiles={profile.calibration_id: profile})
```

Any mismatch raises `DecisionCalibrationError` with the reason code. There is
no fallback to a default profile.

## Golden vectors

`tests/fixtures/portable_bundle_calibration/vectors.json` extends the bundle
corpus with a synthetic decision-model bundle that has four profiles, covering
every method and scope kind. It holds signed verification cases (valid,
critical but not understood, non-critical, malformed, stale weights, and
thresholds edited after signing), read cases for every structural and
cross-reference failure, and resolve cases for every selection failure and
malformed request. Python, Swift, Kotlin, and TypeScript tests all consume
it. Verification cases are signed with the TEST-ONLY keys of the bundle
corpus. To regenerate or check it:

```bash
python tests/fixtures/portable_bundle_calibration/generate_vectors.py
python tests/fixtures/portable_bundle_calibration/generate_vectors.py --check
```

## Versioning

A breaking change to the extension value, its digests, or the check order
increments `schema_version`. Readers reject versions they do not implement
with `calibration_version_unsupported`, which verification reports as
`extension_invalid`.
