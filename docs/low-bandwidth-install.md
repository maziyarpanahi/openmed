# Low-bandwidth, mirror, and proxy installation

OpenMed can use an institutional Python package index, a Hugging Face mirror,
and an HTTP proxy without application changes. Keep one Hugging Face cache,
warm it while bandwidth is inexpensive, and then enable OpenMed's enforced
offline mode.

## Keep downloads in one cache

Choose the cache before downloading a model. The same `HF_HOME` must be visible
to the online prefetch step and the offline runtime.

```bash
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_HOME/hub"
```

`HF_HUB_CACHE` can override the final Hub cache directory. Run `openmed doctor`
after setting the environment to confirm the active location.

## Install through a pip mirror

Set `PIP_INDEX_URL` to the `/simple` endpoint supplied by your institution or
regional mirror. The fallback below is the public PyPI index, so the block is
also runnable on a machine without a mirror configuration.

```bash
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"
python -m pip install \
  --index-url "$PIP_INDEX_URL" \
  --retries 20 \
  --timeout 120 \
  "openmed[hf]"
```

The command-line `--index-url` wins for that invocation. Setting
`PIP_INDEX_URL` applies the same mirror to later pip commands. Do not add
`--trusted-host` to bypass TLS verification; ask the network administrator for
the institution's CA certificate instead.

## Select a Hugging Face mirror

`prefetch_model()` delegates downloads to `huggingface_hub`, which reads
`HF_ENDPOINT`. OpenMed does not replace or auto-detect that value. Set the
mirror before starting Python or the `hf` CLI:

```bash
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
```

Use the exact HTTPS endpoint approved by the institution. The public Hub is the
fallback above. A mirror must implement the compatible Hub repository-info and
file-download routes; a generic caching web proxy is not automatically a Hub
mirror. Hub clients can send an implicitly discovered Hugging Face token to a
custom endpoint. For an anonymous mirror, set
`HF_HUB_DISABLE_IMPLICIT_TOKEN=1`; for an authenticated mirror, use only a
mirror-scoped credential. Never send a broader Hub credential to an untrusted
mirror.

## Configure an HTTP proxy

Python, pip, and `huggingface_hub` honor the conventional proxy variables. Both
uppercase and lowercase spellings are common; keep the values consistent if a
host application sets both. Lowercase values take precedence when both forms
are present. `ALL_PROXY` is also honored when one proxy should handle every
scheme.

```bash
export HTTP_PROXY="http://proxy.example.org:8080"
export HTTPS_PROXY="http://proxy.example.org:8080"
export NO_PROXY="localhost,127.0.0.1,.example.org"
```

Replace the example host and port with the values supplied by the network
administrator. Keep mirror and internal hosts in `NO_PROXY` only when they are
directly reachable. Avoid embedding proxy passwords in scripts or support
logs; `openmed doctor` redacts URL credentials when it prints proxy settings.

## Make model downloads retryable

The Hugging Face cache records completed blobs and incomplete transfers. Rerun
the same command after an interruption: cached files are reused and partial
downloads resume when the server supports HTTP range requests. A single worker
is slower but more reliable on fragile links.

```bash
export MODEL_ID="OpenMed/OpenMed-NER-DiseaseDetect-TinyMed-135M"
hf download \
  "$MODEL_ID" \
  --cache-dir "$HF_HOME/hub" \
  --max-workers 1 \
  --no-force-download
```

OpenMed's Python helper warms the same cache and resolves registry aliases:

```bash
export OPENMED_MODEL_ALIAS="disease_detection_tiny"
python - <<'PY'
import os

from openmed import prefetch_model

print(prefetch_model(os.environ["OPENMED_MODEL_ALIAS"]))
PY
```

Do not set `OPENMED_OFFLINE=1` until the prefetch succeeds. Offline mode rejects
a cache miss instead of falling back to the network.

## Metered-connection checklist

1. Inspect the manifest estimate before downloading. This command reads local
   metadata unless `--remote` is explicitly requested:

    ```bash
    openmed models size disease_detection_tiny
    ```

2. On inexpensive Wi-Fi or a scheduled network window, disable stale offline
   flags and prefetch the model. Rerunning this block is safe:

    ```bash
    unset OPENMED_OFFLINE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
    export OPENMED_MODEL_ALIAS="disease_detection_tiny"
    python - <<'PY'
    import os

    from openmed import prefetch_model

    print(prefetch_model(os.environ["OPENMED_MODEL_ALIAS"]))
    PY
    ```

3. Disconnect from the network, enable enforced offline mode, and verify the
   cached snapshot plus diagnostics before starting the application:

    ```bash
    export OPENMED_OFFLINE=1
    export OPENMED_MODEL_ALIAS="disease_detection_tiny"
    python - <<'PY'
    import os

    from openmed import prefetch_model

    print(prefetch_model(os.environ["OPENMED_MODEL_ALIAS"]))
    PY
    openmed doctor
    ```

After the prefetch, the size check, cache verification, and doctor command are
local-only. Keep `OPENMED_OFFLINE=1` set when launching the real workload so a
missing file fails clearly instead of consuming metered data.

## Diagnose the active environment

```bash
openmed doctor
openmed doctor --json
```

The report includes the active Hub endpoint, `HTTP_PROXY`, `HTTPS_PROXY`,
`ALL_PROXY`, `NO_PROXY`, Hugging Face cache, and `OPENMED_OFFLINE` state. It
performs no network request and never prints a Hugging Face token. URL
credentials, query strings, and fragments are omitted from diagnostic output.

If an online prefetch still fails, compare those fields with the institution's
network configuration. For an offline cache miss, temporarily disable offline
mode only during an approved network window, rerun the same prefetch command,
and restore `OPENMED_OFFLINE=1`.

## Install an offline clinical language bundle

Sites without a dependable connection can move a whole clinical language as
one signed directory, for example on removable media, instead of fetching the
model, lexicons, policy, and evidence separately.
`openmed.clinical.languages.offline_bundle` packages, verifies, and installs
these bundles locally; no step opens a network connection.

Each bundle carries one or more components of every kind:

| Kind | Contents |
|---|---|
| `model` | Model files the runtime loads. |
| `lexicon` | Public lexicons. |
| `policy` | Policy metadata, such as the de-identification profile. |
| `self_test` | Synthetic JSONL cases that the installer runs offline. |
| `evidence` | Evaluation evidence for the packaged model. |

`offline-bundle.json` declares the ISO 639-1 language, scripts, capability
identifiers, minimum RAM and storage, and, for each component, its files, SPDX
license, provenance identifier, and data classification. `manifest.json` uses
the [skill bundle format](skills/bundle-integrity.md): it pins every file,
including the descriptor, to a SHA-256 digest and must be signed with
HMAC-SHA256 or Ed25519. Unsigned bundles are rejected.

Self-test cases are synthetic and list the spans the packaged model must find:

```json
{"case_id": "sw-001", "synthetic": true, "text": "Mgonjwa Zawadi Kifaru alipimwa leo.", "expected": [{"start": 8, "end": 21, "label": "NAME"}]}
```

Package on the connected build machine. Equal inputs and keys produce
byte-identical descriptors and manifests:

```python
from pathlib import Path

from openmed.clinical.languages.offline_bundle import (
    OfflineBundleComponent,
    OfflineBundleDescriptor,
    ResourceRequirements,
    package_offline_bundle,
)


def component(component_id, kind, files, license, data_classification="public"):
    return OfflineBundleComponent(
        component_id=component_id,
        kind=kind,
        files=files,
        license=license,
        source=f"example:sw/{component_id}",
        data_classification=data_classification,
    )


descriptor = OfflineBundleDescriptor(
    bundle_id="org.example.lang.sw.deid",
    bundle_version="1.0.0",
    language="sw",
    scripts=("Latin",),
    capabilities=("pii_detection",),
    resources=ResourceRequirements(
        min_ram_bytes=2 * 1024**3,
        min_storage_bytes=600 * 1024**2,
    ),
    components=(
        component("model", "model", ("model/model.onnx",), "Apache-2.0"),
        component("names", "lexicon", ("lexicons/names.tsv",), "CC0-1.0"),
        component("policy", "policy", ("policy/policy.json",), "Apache-2.0"),
        component(
            "self-tests",
            "self_test",
            ("selftests/cases.jsonl",),
            "CC0-1.0",
            data_classification="synthetic",
        ),
        component("evidence", "evidence", ("evidence/eval.json",), "CC-BY-4.0"),
    ),
)
build = Path("build/sw")
packaged = package_offline_bundle(
    descriptor,
    {path: build / path for path in descriptor.payload_files},
    "dist/sw-bundle",
    ed25519_private_key=private_key_bytes,  # raw 32 bytes from your key store
)
print(packaged.to_dict())
```

Ed25519 signing and verification need the `openmed[integrity]` extra; pass
`signature_key=` instead for an HMAC-SHA256 shared secret. Packaging refuses
licenses outside the permissive allowlist, restricted dataset or terminology
markers such as MIMIC, i2b2, n2c2, UMLS, SNOMED, or CPT, credential-like files
such as `*.pem` or `credentials.json`, hidden or unsafe paths, non-synthetic
self-tests, and a storage requirement smaller than the bundle itself.

Install at the offline site. Pass host facts explicitly, or probe them with
`HostResources.detect()`, and supply a runner that loads the packaged model
from the staged bundle root and returns `(start, end, label)` spans:

```python
from openmed.clinical.languages.offline_bundle import (
    HostResources,
    install_offline_bundle,
)

installation = install_offline_bundle(
    "/media/usb/sw-bundle",
    "/opt/openmed/bundles",
    host=HostResources.detect("/opt/openmed/bundles"),
    signature_public_key=public_key_bytes,  # raw 32-byte Ed25519 public key
    self_test_runner=run_packaged_model,  # (bundle_root, text) -> spans
)
print(installation.reason or "installed", installation.to_dict()["install_relpath"])
```

Verification runs these checks in order. Integrity failures stop early;
policy failures are all reported together in `failures`:

| Check | Failure codes |
|---|---|
| `integrity` | `signature_required`, `signature_invalid`, `hash_mismatch`, `file_undeclared`, `unsigned_bundle`, `manifest_fields_unsupported`, `symlink_present` |
| `descriptor` | `descriptor_malformed`, `bundle_id_mismatch` |
| `components` | `component_files_mismatch`, `component_missing` |
| `licenses` | `license_not_permissive` |
| `restricted_sources` | `restricted_source` |
| `credentials` | `credential_like_file` |
| `self_tests` | `self_test_malformed`, `self_test_not_synthetic` |
| `storage` | `storage_requirement_understated`, `insufficient_storage` |
| `ram` | `insufficient_ram` |

`install_offline_bundle()` runs verification, copying, and self-tests inside
OpenMed's socket guard, which also sets the Hugging Face offline flags for the
process. It copies the bundle into a staging directory, re-checks every copied
digest, and runs each self-test against the staged copy. A case passes when the
runner returns every expected span exactly; extra spans are allowed. Any
outbound connection attempt fails the install with `network_attempted`. Only
then is the copy renamed to `<install_root>/<bundle_id>/<bundle_version>`. An
existing installation is never replaced (`install_target_exists`). Pass
`require_self_tests=False` only when the self-tests will run later through
`run_offline_self_tests()`.

`to_dict()` reports contain reason codes, counts, bundle-relative paths,
digests, license records, capabilities, and resource requirements. They never
contain self-test text, key material, or absolute paths. The
`OfflineBundleVerification` result is the input for activating the language;
installing a bundle does not register it with the
[language-pack registry](language-packs.md).

A bundle is a deployment artifact. Passing verification and self-tests shows
that the files are intact and the packaged model runs offline. It is not a
compliance certification or evidence of clinical accuracy, and it never
triggers a clinical decision.
