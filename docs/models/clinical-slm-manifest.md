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
