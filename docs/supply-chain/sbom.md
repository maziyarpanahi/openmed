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

## Published image

The `Image SBOM` workflow does not publish or retag container images. Release
images are published exclusively by the `Container Multi-Arch` workflow so the
`vX.Y.Z` and `latest` tags retain both `linux/amd64` and `linux/arm64` manifest
entries. The `Container Image Signing` workflow attaches a CycloneDX SBOM
attestation to that immutable multi-architecture manifest digest.

Use `image-sbom.cdx.json.sha256` to verify the standalone release asset as
shown below. For OCI attestation verification, follow the
[Container Image Signing](image-signing.md) guide.

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
