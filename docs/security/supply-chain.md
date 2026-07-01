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
