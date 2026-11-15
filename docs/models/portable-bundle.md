# Signed portable model bundles

Contract version 1 (OpenMed v4.0). A portable bundle is one directory that the
Python package, OpenMedKit for Swift, OpenMedKit for Android, and the
`openmed` web package all read the same way. It replaces the per-format
`openmed-coreml.json`, `openmed-onnx.json`, `openmed-mlx.json` and catalog
fragments as the thing a kit trusts before it uses a model. The converter
manifests stay in the bundle as signed evidence.

A bundle holds the model files plus two documents:

| File | Contents |
|---|---|
| `openmed-bundle.json` | The manifest. Its bytes must be the canonical JSON encoding of the manifest value. |
| `openmed-bundle.sig.json` | The signature envelope: the content identity plus one or more Ed25519 signatures. |

Verification is offline and value-free. No reader opens a network
connection, runs bundled code, follows references outside the bundle root, or
reports anything except reason codes, digests, counts, and indexes. Signing
keys and trust roots always come from the caller. Verification is separate
from activation: a later cache or activation step must refuse any bundle
whose result is not `ok`.

## Schemas

The JSON Schemas ship in `openmed/core/schemas/json/`:

| Schema | Document |
|---|---|
| `portable_bundle.schema.json` | Manifest (`openmed.portable_bundle`, version 1) |
| `portable_bundle_envelope.schema.json` | Signature envelope (`openmed.portable_bundle.signatures`, version 1) |
| `portable_bundle_trust_policy.schema.json` | Caller trust policy (`openmed.portable_bundle.trust_policy`, version 1) |
| `portable_bundle_verification.schema.json` | Verifier result (`openmed.portable_bundle.verification`, version 1) |

The schemas describe each document's shape. The verifiers also check rules a
schema cannot express: sorted unique paths, declared references, acyclic
fallbacks, and the format tables below.

## Manifest

Every top-level member is required. Optional sections are written as `null`.

| Member | Covers |
|---|---|
| `schema`, `schema_version` | `openmed.portable_bundle`, `1` |
| `bundle` | `id`, `version` (`MAJOR.MINOR.PATCH`), `sequence` (anti-rollback counter), `issued_at`, `expires_at` |
| `model` | `id` (repository id or `local:sha256:<hex>`), `revision` (40-hex commit or `null`), `family`, `architecture`, `task`, `languages` |
| `tokenizer` | `kind`, `contract_version`, `files` |
| `labels` | `path`, `count`, `scheme` (the label map digest is that file's inventory digest) |
| `offsets` | `unit` (`unicode_scalar`, `utf16_code_unit`, `utf8_byte`), `contract_version` |
| `decoder` | `kind`, `contract_version` |
| `segmenter` | `id`, `files` |
| `policy` | `min_version`, `max_version_exclusive` (supported privacy-policy versions), `path` |
| `calibration` | `profile_version`, `path` (slot for calibration profiles) |
| `compatibility` | `sdks`: `name`, `min`, `max_exclusive` for each reading SDK |
| `runtimes` | Runtime variants: `id`, `format`, `precision`, `entrypoint`, `files`, `platforms` (`platform`, `min_os`, `max_os_exclusive`), `min_memory_bytes`, `min_storage_bytes`, `fallback` |
| `licenses` | `id` (lowercase SPDX identifier) and the license `files` |
| `provenance` | `source_digest`, `export_record` (path of an export-provenance record) |
| `evidence` | `kind` and `path` for each evidence file (parity reports, converter manifests) |
| `files` | Every bundled file: `path`, `media_type`, `size_bytes`, `sha256`, `role`, `executable` |
| `extensions`, `critical_extensions` | Namespaced extensions and the subset a reader must implement |

Paths are relative POSIX paths built from segments that match
`[A-Za-z0-9_][A-Za-z0-9._+-]{0,127}`. That rules out absolute paths, `.` and
`..`, backslashes, drive letters, hidden files, and empty segments. A path can
have at most 16 segments and 512 characters. `files` must be sorted by UTF-8
bytes, with no exact or case-folded duplicates. A path must not also be used
as a directory of another path. Every path named anywhere else in the
manifest must appear in `files`. `executable` must be `false`: version 1
bundles carry no executable content.

### Runtime formats

| Format | Platforms | Precisions |
|---|---|---|
| `coreml` | ios, macos, watchos, visionos | fp32, fp16, int8, int4 |
| `mlx` | ios, macos | fp32, fp16, bf16, int8, int4, int2 |
| `onnx` | android, ios, macos, linux, windows, web | fp32, fp16, int8, int4 |
| `ort` | android, ios | fp32, fp16, int8 |
| `transformersjs` | web | fp32, fp16, int8, int4 |
| `pytorch` | linux, macos, windows | fp32, fp16, bf16 |

A manifest that declares any other combination fails with
`runtime_combination_unsupported`. Each `fallback` names another runtime of
the same bundle. The fallback graph must be acyclic.

### Extensions

Every member of a version 1 manifest is critical, so an unknown member fails
with `unknown_critical_field`. Vendor data goes in `extensions` under a
dotted name such as `org.example.note`. An extension listed in
`critical_extensions` must be implemented by the reader, or verification
fails with `unknown_critical_extension`. Readers keep the manifest bytes
unchanged, so unknown non-critical extensions round-trip byte for byte.

## Canonical form and content identity

The manifest and the envelope must both be the
[RFC 8785](https://www.rfc-editor.org/rfc/rfc8785) (JCS) encoding of their
value. The profile is restricted to:

- `null`, booleans, strings, arrays, objects, and integers in the IEEE 754
  safe range (`-(2^53 - 1)` to `2^53 - 1`). Non-integer numbers are rejected.
- Object members sorted by UTF-16 code units, with no whitespace, no
  duplicate members, and ECMAScript `JSON.stringify` string escapes.
- Valid UTF-8 with no unpaired surrogates.

Each reader parses the bytes, encodes them again, and requires the result to
match exactly. Reordered members, extra whitespace, escaped ASCII,
non-integer numbers, and duplicate members all fail with
`manifest_not_canonical`. Different JSON parsers therefore cannot read the
same signed bytes differently.

The content identity is `sha256:` followed by the lowercase hex SHA-256 of the
manifest bytes.

## Signature envelope

```json
{"content_id":"sha256:<hex>","schema":"openmed.portable_bundle.signatures","schema_version":1,"signatures":[{"alg":"Ed25519","kid":"<key id>","sig":"<unpadded base64url>"}]}
```

Each signature is an Ed25519 signature over the
[DSSE v1](https://github.com/secure-systems-lab/dsse/blob/master/protocol.md)
pre-authentication encoding:

```
"DSSEv1" SP len(type) SP type SP len(body) SP body
type = application/vnd.openmed.portable-bundle.v1+json
body = exact manifest bytes
```

Lengths are ASCII decimal byte counts. Signatures must be canonical: unpadded
base64url that encodes exactly 64 bytes, with the scalar `S` below the group
order. An envelope may carry up to 16 signatures, so one bundle can be signed
by an outgoing key and an incoming key during rotation.

## Trust policy, rotation, and revocation

Trust roots are supplied by the caller and never bundled. Each key is an
RFC 8037 OKP JWK with a validity window and a status:

```json
{"crv":"Ed25519","kid":"<key id>","kty":"OKP","not_after":"2027-07-01T00:00:00Z","not_before":"2026-01-01T00:00:00Z","status":"active","x":"<unpadded base64url public key>"}
```

- **Rotation:** give the old and new keys overlapping windows, and sign
  bundles with both during the overlap. Once the old key's `not_after` passes,
  its signatures fail with `key_expired`. Bundles that the new key also signed
  still verify.
- **Threshold:** `threshold` distinct trusted keys must each have a valid
  signature. A key that signs twice counts once.
- **Revocation:** set a key's `status` to `revoked` (`key_revoked`), or list a
  bundle's content identity in `revoked_content_ids` (`bundle_revoked`).
- **Expiry:** a bundle is valid while `issued_at <= now < expires_at`. A key
  is valid while `not_before <= now < not_after`. Timestamps use
  `YYYY-MM-DDTHH:MM:SSZ` and are compared as text, which matches time order
  for this fixed-width format.
- **Anti-rollback:** the caller stores the highest accepted `sequence` for
  each bundle id and passes it as a sequence floor. A lower sequence fails
  with `rollback_rejected`.

OpenMed does not ship a production trust root or signing key. The
maintainer's release key and the hosted signing workflow are configured
outside this contract.

## Verification order and reason codes

Every reader runs these checks in this order and reports the first failure.

| Step | Reason codes |
|---|---|
| Size limits | `manifest_too_large`, `envelope_too_large` |
| Envelope | `envelope_malformed`, `envelope_version_unsupported`, `content_id_mismatch`, `bundle_revoked` |
| Signatures | `signature_missing`, `signature_algorithm_unsupported`, `untrusted_key`, `key_revoked`, `key_not_yet_valid`, `key_expired`, `signature_invalid`, `signature_verifier_unavailable`, `signature_threshold_not_met` |
| Canonical form | `manifest_not_canonical` |
| Schema | `schema_unsupported`, `schema_version_unsupported`, `unknown_critical_field`, `manifest_malformed` |
| Structure | `path_unsafe`, `executable_file_forbidden`, `runtime_combination_unsupported`, `duplicate_path`, `path_conflict`, `file_order_invalid`, `file_reference_undeclared`, `fallback_invalid`, `unknown_critical_extension`, `size_limit_exceeded` |
| Validity and rollback | `bundle_not_yet_valid`, `bundle_expired`, `rollback_rejected` |
| Compatibility | `sdk_incompatible`, `policy_incompatible`, `platform_incompatible` |
| File tree | `file_missing`, `file_size_mismatch`, `file_digest_mismatch`, `file_undeclared` |

For each signature, the order is algorithm, key lookup, revocation, validity
window, then the cryptographic check. If fewer than `threshold` signatures
verify, the reason is the first signature failure, or
`signature_threshold_not_met` when none failed. Within the file tree, each
declared file is checked for missing, size, and digest in manifest order;
undeclared extra files are checked last.

Malformed caller inputs raise errors rather than returning a result:
`trust_policy_invalid`, `context_invalid`, and (when producing bundles)
`signing_key_invalid`.

The result holds `ok`, `reason`, `content_id`, `valid_signatures`,
`eligible_runtimes` (indexes into `runtimes`), `files_checked`, and
`files_verified`. Verification without a file tree checks only the manifest
and signatures, and reports `files_verified: false`. Only a result with both
`ok` and `files_verified` true is eligible for activation.

Runtime selection: a runtime is eligible when its `format` is one the reader
can load, one of its `platforms` matches the target platform and OS range,
and its memory and storage requirements fit the target. The target SDK must
appear in `compatibility.sdks`, and its version must be in range. When the
target states a policy version, it must fall in the `policy` range.

## Readers

| Surface | Entry point |
|---|---|
| Python | `openmed.core.portable_bundle.verify_portable_bundle`, `verify_portable_bundle_dir` |
| Swift (OpenMedKit) | `PortableBundle.verify(manifest:envelope:trustPolicy:context:files:)`, `PortableBundle.verifyDirectory(at:trustPolicy:context:)`, using CryptoKit |
| Kotlin (OpenMedKit Android) | `com.openmed.openmedkit.bundle.PortableBundle.verify(...)` |
| TypeScript (`openmed` npm) | `verifyPortableBundle(...)`, using WebCrypto Ed25519 |

Android API 26 and JDK 11 ship no Ed25519 provider. The Kotlin reader
therefore includes a verify-only RFC 8032 implementation built on
`java.math.BigInteger` and SHA-512. It handles only public inputs and never
signs. Apps can pass their own `Ed25519SignatureVerifier`, for example one
backed by Tink, BouncyCastle, or the JDK 15+ provider. Browsers without
WebCrypto Ed25519 fail with `signature_verifier_unavailable`. So does Python
without the optional `cryptography` package (`openmed[integrity]`).

```python
from openmed.core.portable_bundle import (
    TrustPolicy,
    VerificationContext,
    VerificationTarget,
    verify_portable_bundle_dir,
)

policy = TrustPolicy.from_mapping(my_trust_policy_document)
context = VerificationContext(
    target=VerificationTarget(
        platform="macos",
        sdk_name="openmed-python",
        sdk_version="4.0.0",
        runtimes=("coreml", "onnx"),
        os_version="15.1",
    ),
    sequence_floors={"my.bundle.id": last_accepted_sequence},
)
result = verify_portable_bundle_dir("path/to/bundle", trust_policy=policy, context=context)
if not result.eligible_for_activation:
    raise RuntimeError(result.reason)
```

## Producing bundles from existing exports

`openmed.core.portable_bundle_adapters` maps the existing metadata into a
manifest draft:

- `manifest_from_onnx`: maps `openmed-onnx.json`. Artifact formats `onnx`,
  `onnx-int8`, `onnx-android`, `webgpu`, `ort-android`, and `transformersjs`
  have portable runtimes.
- `manifest_from_mlx`: maps `openmed-mlx.json`. An `mlx-fp` artifact needs an
  explicit `precision`, because the MLX manifest does not record its dtype.
- `manifest_from_coreml`: maps `openmed-coreml.json` variants.
- `model_section_from_catalog`: maps one `models.jsonl` row into the `model`
  section and license.

Each adapter measures every file itself, reuses the converters' segmenter
validation, and records every field it does not project as an
`AdapterLoss`. The loss reasons are:

- `preserved_in_converter_manifest`: the converter manifest stays in the
  bundle as signed evidence.
- `catalog_only`: the field stays in the catalog.
- `defaulted`: the adapter supplied a documented default, such as OpenMedKit's
  minimum OS versions.
- `unknown`: the source does not record the field.
- `artifact_format_not_portable`: the artifact was skipped. This happens only
  with `skip_unsupported=True`.

Where an adapter would otherwise have to guess, it raises
`PortableBundleAdapterError` instead. Its reason codes include
`source_format_unsupported`, `precision_ambiguous`, `license_unmapped`,
`metadata_conflict`, and `source_identity_not_portable`. Local filesystem
paths never reach a manifest.

Export-provenance records with `path`, `size_bytes`, and `sha256` rows, and a
`source` block with `revision` and `content_digest`, can be passed as
`export_provenance`. Their rows are cross-checked against the measured tree,
and their source identity fills `model.revision` and
`provenance.source_digest`.

```python
from openmed.core.portable_bundle import sign_portable_manifest, write_portable_bundle
from openmed.core.portable_bundle_adapters import BundleRelease, manifest_from_onnx

draft = manifest_from_onnx(
    "exports/my-model-onnx",
    release=BundleRelease(
        bundle_id="org.example.my-model",
        version="1.0.0",
        sequence=1,
        issued_at="2026-11-01T00:00:00Z",
        expires_at="2027-11-01T00:00:00Z",
    ),
    catalog_entry=catalog_row,
)
manifest = draft.encode()
envelope = sign_portable_manifest(manifest, [("org.example.release-2026", my_ed25519_seed)])
write_portable_bundle("exports/my-model-onnx", manifest, envelope)
```

## Golden vectors

`tests/fixtures/portable_bundle/vectors.json` holds the shared conformance
corpus: the canonical golden manifest, canonicalization vectors, RFC 8032
signature vectors, trust policies, synthetic file trees, and 75 accepted and
rejected cases with the exact expected result. Python, Swift, Kotlin, and
TypeScript tests all consume it.

The vectors are signed with TEST-ONLY keys derived from public labels in
`tests/fixtures/portable_bundle/generate_vectors.py`. Never add them to a
production trust policy. To regenerate or check the corpus:

```bash
python tests/fixtures/portable_bundle/generate_vectors.py
python tests/fixtures/portable_bundle/generate_vectors.py --check
```

## Relationship to existing integrity checks

- `.openmed-integrity.json` (`openmed.model_integrity.v1`) still guards
  Python cache snapshots. A portable bundle is the cross-kit contract that
  covers the same files and also binds roles, media types, runtimes,
  compatibility, and licenses into the signed identity.
- Signed offline language bundles (`openmed.offline_language_bundle.v1`) use
  the same pattern: a canonical signed manifest, a key id, Ed25519 signatures,
  fail-closed unknown critical fields, and value-free reason codes.

## Versioning

A breaking change to the manifest, envelope, canonical form, or verification
order increments `schema_version` and the DSSE payload type. Readers reject
versions they do not implement with `schema_version_unsupported` or
`envelope_version_unsupported`. New optional data goes in namespaced
extensions until the next schema version.
