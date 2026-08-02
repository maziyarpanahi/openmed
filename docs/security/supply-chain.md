# Supply Chain Controls

OpenMed commits `uv.lock` so Python dependency resolution is reproducible
across local development and CI. Pull requests and pushes run a dedicated
lockfile drift gate:

```bash
uv lock --check
```

The gate verifies that `uv.lock` still matches `pyproject.toml` without
installing the full development environment. If a dependency or optional extra
changes, regenerate the lockfile locally before pushing:

```bash
uv lock
```

Commit the updated `uv.lock` with the dependency change. The CI failure message
uses the same remediation: `Run 'uv lock' and commit the updated uv.lock.`
Do not edit the lockfile by hand.

## Model artifact integrity

OpenMed verifies registry model artifacts before their first model or pipeline
construction. The first online load pins the current repository revision to the
catalog row's `reproducibility_hash`, downloads only runtime files from that
revision, verifies content-addressed Hub metadata, and records an offline
`.openmed-integrity.json` artifact set under the configured OpenMed cache. Each
artifact entry has a streamed SHA-256 digest. Later loads verify that local set
without contacting the network.

The catalog `reproducibility_hash` is a repository/provenance digest; it is not
reused as an artifact-byte digest. Keeping the two values separate avoids
comparing unrelated hashes while still binding the verified artifact set to the
catalog revision.

Re-check one local model directory or all prepared caches with:

```bash
openmed models verify /path/to/local/model
openmed models verify OpenMed/model-id
openmed models verify --all
```

`models verify` never performs network access. It prints the expected and
actual artifact-set hashes and exits `0` when every selected cache passes. A
missing artifact, malformed integrity record, or digest mismatch prints `FAIL`
and exits `1`.

Default mode verifies hashes when an integrity manifest is present and emits a
prominent warning when a registry model has not yet been prepared. Set
`OPENMED_MODEL_VERIFY_STRICT=1` to make a missing registry hash or integrity
manifest fail closed. Emergency opt-out is available with
`OPENMED_SKIP_MODEL_VERIFY=1`; it logs a prominent warning and should only be
used while investigating or recovering a trusted cache.

### Failure runbook

If verification fails:

1. Do not construct a PHI-processing pipeline from that cache.
2. Record the model id, artifact path, expected SHA-256, and actual SHA-256. Do
   not attach model inputs, patient data, or other PHI.
3. Remove the named cached artifact and redownload the pinned revision:

   ```bash
   hf download OpenMed/model-id --cache-dir ~/.cache/openmed --force-download
   ```

   Replace `~/.cache/openmed` when `cache_dir` is customized.
4. Load the model once to rebuild its verified cache record, then run
   `openmed models verify OpenMed/model-id` again.
5. If the mismatch recurs or the signed catalog is rejected, follow the
   [security disclosure policy](disclosure-policy.md). Do not bypass the check
   in production.

### Optional signed catalog

If `models.jsonl.sigstore.json` exists beside `models.jsonl`, catalog loading
verifies the detached Sigstore bundle fully offline before parsing any row. The
manifest digest, signature, pinned signer identity, and pinned public-key hash
must all match. Install the optional verifier dependency with
`pip install "openmed[integrity]"`. A present but invalid bundle fails closed;
an absent bundle leaves the unsigned catalog behavior unchanged.

## Software bill of materials

CI and every tagged release also generate a CycloneDX SBOM (`sbom.cdx.json`)
inventorying the dependency tree. See the
[Software Bill of Materials](sbom.md) guide to regenerate or consume it.

Container image releases also publish `image-sbom.cdx.json`, which inventories
the built Docker image including OS packages and Python components. See the
[Container Image SBOM](../supply-chain/sbom.md) guide for release artifacts,
image labels, and digest verification.

## PyPI provenance

Tagged library releases build and check the wheel and source distribution
before uploading to PyPI, and attempt to generate repository-level SLSA
provenance evidence in the same release path. The current PyPI upload uses the
project-scoped `PYPI_API_TOKEN` GitHub secret until the PyPI trusted publisher
is configured for this workflow. After the upload, the release path signs each
distribution with Sigstore keyless signing and attaches the provenance bundle,
digest manifest, and signature bundles to the tagged GitHub release, so
consumers can verify a release without the GitHub attestation API. See the
[PyPI Publishing](../release/trusted-publishing.md) guide for the upload
contract and migration checklist.

## Container image signatures

Published service images are signed by digest with Sigstore keyless signing.
The signing workflow also attaches signed CycloneDX SBOM and SLSA provenance
attestations, then verifies the signature and both attestations before the run
can pass. See [Container Image Signing](../supply-chain/image-signing.md) for
verification commands and the cluster admission policy.

## SLSA build provenance

Tagged releases generate SLSA provenance for the wheel, source distribution, and
GHCR image digest before publishing can complete. The release workflow verifies
the distribution attestations before the PyPI upload job runs, using the same
builder workflow that produced the wheel and source distribution. The container
workflow verifies the pushed manifest-list attestation before the image job can
pass. See [SLSA Build Provenance](../supply-chain/provenance.md) for consumer
verification commands.
