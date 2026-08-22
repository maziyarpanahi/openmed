# Deterministic SBOM evidence

OpenMed can produce a machine-readable [CycloneDX](https://cyclonedx.org/)
software bill of materials from the repository's local dependency manifests.
The evidence bundle identifies the runtime dependency closure, package artifact
hashes recorded by `uv.lock`, the source revision, and hashes of both input
manifests.

## Generate it locally

```bash
python scripts/licenses/sbom.py --output sbom.cdx.json
```

The generator reads `pyproject.toml` and `uv.lock` only. It does not resolve
packages, inspect an installed environment, contact a package index, or require
network access. When `--source-revision` is omitted, it reads the local Git
`HEAD`; a revision can be supplied explicitly for an exported source tree:

```bash
python scripts/licenses/sbom.py \
  --source-revision <commit-sha> \
  --output sbom.cdx.json
```

The default output is `sbom.cdx.json` at the repository root. It is a generated
artifact and is not committed (see `.gitignore`). The JSON has no generated
timestamp or random serial number, so repeated runs over the same revision and
manifests are byte-identical.

## Evidence and privacy boundaries

The `metadata.properties` section records:

- `openmed:source-revision`
- `openmed:pyproject-sha256`
- `openmed:uv-lock-sha256`
- `openmed:manifest-sha256`

Package PURLs, versions, and artifact hashes come from safe fields in the local
manifests. Explicit lock-file license values and the repository's reviewed
runtime license defaults populate component licenses. Lock-file download URLs,
credentials, local source paths, build paths, timestamps, and environment
details are intentionally omitted. If a package has no license value in the
local evidence, the component is marked `NOASSERTION`; the generator never
guesses a license from a network service.

This artifact supports dependency inventory and reproducibility checks. It is
not a compliance certification or a clinical decision guarantee.

## Environment SBOMs

The existing `make sbom` target and CI jobs also publish an environment SBOM
using `scripts/security/generate_sbom.py`. That workflow is useful for the
installed runtime profile and may resolve an environment with package tooling;
use the generator above when the evidence must be reproducible from checked-in
manifests without network access.

## Existing release SBOMs

Container releases publish a separate [image SBOM](../supply-chain/sbom.md)
that covers operating-system packages and image contents. The deterministic
evidence bundle described here is the source-manifest view for the Python
package and does not replace the image SBOM.
