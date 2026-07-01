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

## PyPI provenance

Tagged library releases use PyPI Trusted Publishing instead of a stored PyPI
API token. The publish workflow uploads distributions with Sigstore
attestations, giving each wheel and source distribution signed provenance from
the release workflow identity. See the
[PyPI Trusted Publishing](../release/trusted-publishing.md) guide for the PyPI
configuration and token-retirement checklist.

## SLSA build provenance

Tagged releases generate SLSA provenance for the wheel, source distribution, and
GHCR image digest before publishing can complete. The release workflow verifies
the distribution attestations before the PyPI upload job runs, using the same
builder workflow that produced the wheel and source distribution. The container
workflow verifies the pushed manifest-list attestation before the image job can
pass. See [SLSA Build Provenance](../supply-chain/provenance.md) for consumer
verification commands.
