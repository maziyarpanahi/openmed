# Container Image SBOM

OpenMed publishes a CycloneDX JSON software bill of materials for the service
container image. This image SBOM is separate from the Python package SBOM: it is
generated from the built Docker image, so it captures operating-system packages,
Python wheels, and system libraries present in the runtime image.

## CI generation

For pull requests and branch builds, the `Image SBOM` workflow builds
`Dockerfile` and scans the local image with Syft. Both service Dockerfiles use
a dated Debian snapshot so their exact runtime package versions remain
available after live mirrors replace them. APT still verifies the signed
Release metadata and package hashes; only the snapshot metadata's expiration
check is disabled. Review the snapshot and package pins together when updating
runtime dependencies.

For release tags, the workflow waits for the published multi-architecture
image, resolves its immutable manifest digest, and verifies the publisher's
SLSA provenance against the exact tag, commit, repository, and
`container-multiarch.yml` workflow. It then pulls and scans the `linux/amd64`
image. The workflow writes `image-sbom.cdx.json` and fails if the SBOM is missing,
empty, malformed JSON, not a CycloneDX document, or does not include both OS
package components and Python package components.

Each run uploads:

- `image-sbom.cdx.json`
- `image-sbom.cdx.json.sha256`

Tagged release runs also attach both files and `image-sbom-source.json` to the
GitHub release. The source record binds the SBOM digest to the image manifest
digest, platform, release tag, and source commit. Assets are uploaded only after
SBOM validation succeeds.

## Recover an existing release

If a tag's image SBOM job failed, run the repaired workflow from `master` with
the existing tag:

```bash
gh workflow run sbom-image.yml --ref master -f release_tag=vX.Y.Z
```

Recovery verifies and scans the already-published image by digest. It does not
rebuild or republish the released image, move the tag, or upload Python/npm
packages. Unknown tags, unverified images, and recovery dispatches from another
branch fail before release assets can be uploaded. A manual run without
`release_tag` retains the normal local-build behavior.

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
