# Clinical SLM artifact manifest

OpenMed clinical small language model (SLM) packages use a local manifest as a
load-time safety boundary. It binds one model revision to the files needed by
the runtime, their SHA-256 digests and sizes, quantization metadata, explicit
component licenses, and the supported task identifiers.

The manifest is metadata-only. It must not contain model bytes, prompt or
clinical text, credentials, URLs, or patient-derived identifiers. Package
paths are relative to the package root and are checked for traversal and
symlinks. A package is incomplete unless it declares at least one `weights`,
`tokenizer`, `templates`, and `quantization` artifact, and every declared
artifact has a digest. The `quantization` artifact is the content-addressed
runtime configuration; `quantization` metadata describes the selected scheme.

## Constructing a manifest

Manifest objects are frozen and normalize their component, license, and task
collections into deterministic tuples. A manifest digest is derived from all
metadata except the digest itself. The persisted JSON form includes that
digest:

```python
import hashlib
from pathlib import Path

from openmed.models.clinical_slm_manifest import (
    ClinicalSLMArtifact,
    ClinicalSLMArtifactManifest,
    ClinicalSLMQuantization,
)

manifest = ClinicalSLMArtifactManifest(
    model_id="OpenMed/Synthetic-Clinical-SLM",
    revision="0123456789abcdef0123456789abcdef01234567",
    components=(
        ClinicalSLMArtifact(
            component="weights",
            path="weights/model.safetensors",
            sha256=hashlib.sha256(b"weights").hexdigest(),
            size_bytes=7,
        ),
        ClinicalSLMArtifact(
            component="tokenizer",
            path="tokenizer/tokenizer.json",
            sha256=hashlib.sha256(b"tokenizer").hexdigest(),
            size_bytes=9,
        ),
        ClinicalSLMArtifact(
            component="templates",
            path="templates/templates.json",
            sha256=hashlib.sha256(b"templates").hexdigest(),
            size_bytes=9,
        ),
        ClinicalSLMArtifact(
            component="quantization",
            path="quantization/quantization.json",
            sha256=hashlib.sha256(b"int4").hexdigest(),
            size_bytes=4,
        ),
    ),
    quantization=ClinicalSLMQuantization(scheme="int4", bits=4),
    licenses={
        "weights": "Apache-2.0",
        "tokenizer": "MIT",
        "templates": "BSD-3-Clause",
        "quantization": "Apache-2.0",
    },
    supported_tasks=("clinical-ner", "clinical-summarization"),
)

package_root = Path("/opt/openmed/clinical-slm")
package_root.joinpath("clinical-slm-manifest.json").write_text(
    manifest.to_json() + "\n",
    encoding="utf-8",
)
```

Use real local file digests when generating a deployment package. The example
uses synthetic bytes only; it is not a model or a clinical decision artifact.
The accepted license vocabulary is explicit and permissive. Missing,
`unknown`, `other`, proprietary, or otherwise unrecognized license values are
rejected. Moving revisions such as `main`, `master`, `latest`, and branch or
tag references are also rejected; use an immutable commit or content digest.

## Verifying before model loading

Verification is local and deterministic. It performs no Hugging Face lookup,
download, optional runtime import, or network call. It reads the manifest and
declared files, rejects missing, non-regular, symlinked, resized, or changed
files, and compares every declared SHA-256 digest:

```python
from pathlib import Path

from openmed.models.clinical_slm_manifest import (
    load_clinical_slm_manifest,
    verify_clinical_slm_package,
)

package_root = Path("/opt/openmed/clinical-slm")
manifest = load_clinical_slm_manifest(package_root)
verification = verify_clinical_slm_package(package_root, manifest)

# Construct a local runtime only after verification succeeds.
assert verification.verified
```

`load_clinical_slm_manifest` requires the persisted `manifest_digest` by
default. `verify_clinical_slm_package` returns aggregate counts and the
manifest digest; errors contain stable reason codes and no paths, digests,
model identifiers, or file contents. Callers should additionally pin the
trusted manifest digest when distributing a package. The manifest is a
load-time integrity and provenance gate, not a compliance certification,
medical device, or autonomous clinical decision guarantee. Human review is
required by the manifest and cannot be disabled through its metadata.

Collection inputs are limited to 4,096 entries and serialized manifests to
8 MiB. Typed records are revalidated at the verification boundary; conflicting
aliases, component roles, and quantization bit widths are rejected. Errors
discard upstream exception context, which may otherwise retain input values.

Disk verification currently requires POSIX directory-descriptor and no-follow
file-open support (Linux/macOS). On other platforms it fails closed; metadata
construction and validation remain available. Directory descriptors prevent
following a component or parent symlink swapped during traversal. File identity,
size, timestamps, and bytes read are checked across each read. Keep the package
immutable through verification **and subsequent runtime loading**: this gate
does not lock files against another process or verify a later runtime's reads.
A self-digest detects inconsistency, not authenticity; pin the expected manifest
digest through a trusted distribution channel. Hashes are not anonymization.

## Verifying a package from the command line

Air-gapped operators can gate a package without writing Python:

```console
openmed models slm-verify /opt/openmed/clinical-slm \
    --task bounded_summarization --memory-budget 8000000000 --json
```

`--task` and `--memory-budget` are required so every run states the capability
and the device budget it checked; repeat `--task` to require several
capabilities, and add `--headroom-bytes` to reserve memory that must stay free
after loading. The command reads package metadata and declared component bytes
only. It never imports an inference runtime, constructs a tokenizer or model,
or opens a network connection, so it runs on a host with no runtime
dependencies installed.

`--json` prints a machine-readable report with an `ok` envelope; without it the
same verdict is printed as one line. A completed run exits `0` when the verdict
is `pass` and `1` when it is `fail`, and still prints the full report either
way. Usage errors (a missing or malformed option) exit `2`.

```json
{
  "command": "models slm-verify",
  "ok": true,
  "data": {
    "verdict": "pass",
    "manifest": {"component_count": 4, "manifest_digest": "sha256:6db7c4a9…"},
    "verification": {"verified": true, "reason_codes": []},
    "capabilities": {"decision": "supported", "reason_codes": []},
    "memory": {"decision": "accepted", "reason_codes": []},
    "reason_codes": [],
    "network": {"mandatory": false}
  }
}
```

Each sub-report keeps the schema of the corresponding Python API, so the same
reason codes appear in both. Failures are reported as `verdict: "fail"` with
every contributing reason code collected in `reason_codes`; the command also
reports reasons that a single-library check would hide, such as an unsupported
capability next to a memory rejection. Capability and context metadata come from
`clinical-slm-capabilities.json` when a package ships one, and otherwise from
the artifact manifest, in which case the probe reports missing context limits.

A run that cannot produce a report fails closed with a stable error code in the
`error` envelope instead of a verdict:

| `error.code` | Meaning |
| --- | --- |
| `unsupported_platform` | The platform cannot hash components (no POSIX no-follow directory descriptors), so verification is refused before any check. |
| `slm_manifest_invalid` | The manifest was rejected while loading or validating; the message carries the manifest reason code, such as `unknown_license`, `manifest_missing`, or `manifest_digest_required`. |
| `slm_package_invalid` | A declared component could not be read or hashed, or the memory artifact record is unusable. |
| `slm_unknown_task` | A requested capability or task is outside the closed vocabulary (`bounded_summarization`, `nli`, and their documented aliases). |
| `slm_capability_metadata_invalid` | Capability metadata could not be normalized into the closed vocabulary. |
| `slm_memory_profile_invalid` | The runtime profile implied by the memory options was rejected. |

Reported codes replace values: no package path, model identifier, component
file name, license string, or file content appears in the report, in an error
message, or in the one-line summary, so the output can be attached to a review
ticket or an audit log without redaction. The verdict covers package integrity,
declared capability coverage, and the explicit memory budget only. It is not a
clinical validation, and human review remains required before any patient-facing
use.

