# Container Image SBOM

OpenMed publishes a CycloneDX JSON software bill of materials for the service
container image. This image SBOM is separate from the Python package SBOM: it is
generated from the built Docker image, so it captures operating-system packages,
Python wheels, and system libraries present in the runtime image.

## CI generation

The `Image SBOM` workflow builds `Dockerfile`, scans the local image with Syft,
and writes `image-sbom.cdx.json`. The workflow fails if the SBOM is missing,
empty, malformed JSON, not a CycloneDX document, or does not include both OS
package components and Python package components.

Each run uploads:

- `image-sbom.cdx.json`
- `image-sbom.cdx.json.sha256`

Tagged release runs also attach both files to the GitHub release.

## Image label

For tagged releases, the workflow publishes the release image to
`ghcr.io/maziyarpanahi/openmed` and labels it with the SBOM digest:

```bash
docker inspect ghcr.io/maziyarpanahi/openmed:vX.Y.Z \
  --format '{{ index .Config.Labels "org.opencontainers.image.sbom.digest" }}'
```

The label value is `sha256:<digest>` for the attached
`image-sbom.cdx.json` file.

## Consume and verify

Download the image SBOM from a release:

```bash
gh release download vX.Y.Z \
  --repo maziyarpanahi/openmed \
  --pattern image-sbom.cdx.json \
  --pattern image-sbom.cdx.json.sha256
```

Verify the digest:

```bash
sha256sum -c image-sbom.cdx.json.sha256
```

Inspect or scan the CycloneDX SBOM with a compatible tool:

```bash
grype sbom:image-sbom.cdx.json
trivy sbom image-sbom.cdx.json
```

The image SBOM does not replace the package-level SBOM. Use the package SBOM to
audit the Python distribution and the image SBOM to audit the shipped container
runtime.
