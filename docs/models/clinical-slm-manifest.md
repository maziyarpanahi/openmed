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

## Provisioning the registered MLX summarizer

The registered `mlx`, `maple` and `maple-preview` aliases now require an
explicitly provisioned package and a trusted manifest digest. They select the
same pinned Maple model and revision. OpenMed does not ship a reviewed package
digest for this artifact: an unprovisioned alias raises
`LocalSummarizerPackageError` with `code="package_unpinned"`. A cached Hub
snapshot or its immutable revision alone cannot satisfy package admission.
The deterministic `extractive` backend and caller-owned local callables keep
their existing behavior.

Provisioning belongs to the deploying application:

1. Prepare a dedicated package directory from the reviewed, pinned model
   revision. Copy files into regular files; do not pass a default Hugging Face
   snapshot containing symlinks into a blob store. Keep the package read-only
   to the runtime account, and immutable throughout verification and loading.
2. Declare every file consumed by the MLX runtime in the manifest, including
   all weights, tokenizer assets, indices and runtime configuration. Include
   `config.json` as a `quantization` component and `templates.json` as a
   `templates` component. The latter is the JSON serialization of
   `build_maple_task_messages("summarize", "{source}")`. Include other files
   only as declared, licensed components. The manifest file is the only
   automatically allowed file; undeclared files, extra directories and all
   symlinks are refused.
3. Bind the manifest's `model_id` and `revision` to
   `resolve_summarizer_model(alias)`. Declare `clinical-summarization`, explicit
   `context_limits`, and `required_runtime_features=["mlx"]`. Quantization
   must match the content-addressed model configuration. Obtain and review
   the canonical manifest digest through a trusted distribution channel;
   reading a self-digest from an untrusted package does not establish trust.
4. Register the package and trusted pin at application startup, before using
   the summarizer. Aliases for the same model and revision share this binding.
   Registration is process-local and performs no file access or network call.

For a caller's reviewed deployment metadata:

```python
from openmed.core.model_registry import register_summarizer_package
from openmed.clinical.summarize import summarize_deidentified

register_summarizer_package(
    "mlx",
    package_root=provisioned_package_directory,
    manifest_digest=trusted_deployment_manifest_digest,
)
result = summarize_deidentified(safe_note, model="mlx")
```

The manifest's additional runtime metadata is digest-bound:

```python
context_limits = {
    "max_context_tokens": 8192,
    "max_input_tokens": 6144,
    "max_output_tokens": 2048,
}
required_runtime_features = ("mlx",)
```

These fields are optional for generic legacy manifests. Omission preserves
their original JSON and digest. Registered summarizer admission requires both
fields and uses their verified values for the capability probe, input budget
and output reservation; it does not assert task support on the model's behalf.
The manifest's context cannot exceed the model configuration's native context,
and OpenMed still caps runtime context at 8,192 tokens and generation at 2,048
tokens. Input is bounded before construction and again after tokenization.

Before `_load_model()`, admission calls `verify_clinical_slm_package()` with
the independently pinned `expected_manifest_digest` and
`reject_undeclared_files=True`. This verifies every declared component and the
complete package inventory, with bounded no-follow traversal. Configuration
and template reads additionally check their declared digest on the bytes
actually read. POSIX `O_NOFOLLOW`, directory descriptors and no-follow stat
and directory enumeration APIs are required; other platforms fail closed with
`platform_unsupported` before reading package contents or constructing a model.
There is no cache lookup, download, remote inference or automatic fallback.

Package refusal messages contain only the controlled `code`, without paths,
artifact contents, credentials or upstream exception context. Common codes
include `package_unpinned`, `manifest_missing`, `manifest_digest_mismatch`,
`component_digest_mismatch`, `component_missing_on_disk`,
`undeclared_component`, `unsafe_component_path`, `model_identity_mismatch`,
`task_unsupported`, `context_metadata_missing`, `runtime_metadata_missing`,
`configuration_mismatch`, `template_mismatch` and `platform_unsupported`.
Existing optional-runtime and inference errors retain their separate contracts.
Use `clear_summarizer_package(alias)` to remove a binding without modifying
package files. This admission gate is integrity evidence, not clinical
validation, and it does not lock files against external changes after checking.
