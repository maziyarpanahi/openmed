# Skill bundle integrity verification

OpenMed's skill bundle verifier performs a deterministic, pre-install integrity
check for portable skill bundles. A skill bundle is a directory containing a
`manifest.json` that declares the bundle identifier, entry points, per-file
SHA-256 digests, and an optional signature over the canonical manifest. One
signature scheme is supported: HMAC-SHA256 (shared-secret model). Verification
is purely local: it performs no network calls
and never logs raw file contents, full hashes, manifest JSON, signature bytes,
or signature keys.

## Manifest format

The bundle directory must contain a `manifest.json` file. The schema is:

| Field | Contract |
|---|---|
| `manifest_version` | Schema version string. Must be in the verifier's accepted set. |
| `bundle_id` | Stable, non-empty bundle identifier. |
| `entry_points` | List of relative paths of executable entry points in the bundle. |
| `files` | Mapping of relative file path to declared SHA-256 hex digest. |
| `signature_scheme` | Signature scheme; `"none"` or `"hmac-sha256"`. |
| `signature` | Hex-encoded signature tag; empty when scheme is `"none"`. |

A minimal manifest:

```json
{
  "manifest_version": "1.0",
  "bundle_id": "com.example.my-skill",
  "entry_points": ["main.py"],
  "files": {
    "main.py": "<sha256-hex-digest>",
    "utils.py": "<sha256-hex-digest>"
  },
  "signature_scheme": "none",
  "signature": ""
}
```

## Verification checks

The verifier runs the following checks in order. The first failure short-circuits
the result with a stable failure category:

1. Manifest is valid JSON and passes schema validation.
2. Manifest version is in the accepted set (default `{"1.0"}`).
3. Every declared file exists in the bundle directory.
4. Every declared file's computed SHA-256 matches the declared digest.
5. Every declared entry point file exists in the bundle directory.
6. Every entry point is listed in the manifest `files` map.
7. When `signature_scheme` is `hmac-sha256`, the supplied key produces a
   matching HMAC-SHA256 over the canonical manifest.

## Failure categories

All failures are represented in the result and never raised, except filesystem
errors reading `manifest.json` itself. The nine stable `REASON_*` categories:

| Category | Description |
|---|---|
| `manifest_malformed` | `manifest.json` is not valid JSON or fails schema validation. |
| `manifest_version_unsupported` | Manifest version is not in the accepted set. |
| `file_missing` | A declared file does not exist in the bundle directory. |
| `hash_mismatch` | A file's computed SHA-256 does not match the declared digest. |
| `entry_point_missing` | A declared entry point file does not exist. |
| `entry_point_not_declared` | An entry point is not listed in the manifest files map. |
| `signature_required` | Signature scheme is `hmac-sha256` but no key was supplied. |
| `signature_invalid` | The supplied HMAC-SHA256 signature does not match. |
| `signature_scheme_unsupported` | The declared signature scheme is not recognized. |

## Usage example

Verify an unsigned bundle:

```python
from openmed.skills.bundle_verify import verify_bundle

result = verify_bundle("/path/to/bundle")
if not result.valid:
    print(f"Rejected: {result.failure_category}")
```

Verify a signed bundle by supplying the HMAC-SHA256 key:

```python
result = verify_bundle("/path/to/bundle", signature_key=b"my-secret-key")
```

For repeated verification with a custom accepted-versions set, use the verifier
class directly:

```python
from openmed.skills.bundle_verify import BundleVerifier

verifier = BundleVerifier(supported_versions=frozenset({"1.0"}))
result = verifier.verify("/path/to/bundle", signature_key=b"my-secret-key")
```

## Privacy and logging

The verifier performs no network calls. It logs only operational telemetry:
bundle IDs, file counts, hash prefixes (first 12 characters), and category
labels. Full hashes, file contents, manifest JSON, signature bytes, and
signature keys are never logged. `BundleFileResult.to_dict()` and
`BundleVerificationResult.to_dict()` expose hash prefixes only, never full
digests or raw contents.

## API reference

Public surface in `openmed.skills.bundle_verify`:

| Symbol | Description |
|---|---|
| `verify_bundle(bundle_dir, *, signature_key=None, supported_versions=...)` | Convenience function wrapping `BundleVerifier.verify`. |
| `BundleVerifier(*, supported_versions=...)` | Verifier class with `.verify(bundle_dir, *, signature_key=None)`. |
| `SkillBundleManifest` | Frozen dataclass for manifest metadata; `from_mapping()` and `to_dict()`. |
| `BundleVerificationResult` | Frozen dataclass with `.valid`, `.failure_category`, `.to_dict()`. |
| `BundleFileResult` | Per-file result; `.to_dict()` exposes hash prefixes only. |
