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

## Software bill of materials

CI and every tagged release also generate a CycloneDX SBOM (`sbom.cdx.json`)
inventorying the dependency tree. See the
[Software Bill of Materials](sbom.md) guide to regenerate or consume it.

Container image releases also publish `image-sbom.cdx.json`, which inventories
the built Docker image including OS packages and Python components. See the
[Container Image SBOM](../supply-chain/sbom.md) guide for release artifacts,
image labels, and digest verification.

## PyPI provenance

Tagged library releases build and verify SLSA provenance before uploading the
wheel and source distribution to PyPI. The current PyPI upload uses the
project-scoped `PYPI_API_TOKEN` GitHub secret until the PyPI trusted publisher
is configured for this workflow. See the
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
