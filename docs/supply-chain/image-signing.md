# Container Image Signing

OpenMed signs each published service image with Sigstore keyless signing after
the multi-architecture container workflow has pushed the manifest list to GHCR.
The signing workflow resolves the immutable image digest, signs that digest with
the GitHub Actions OIDC identity, attaches a CycloneDX image SBOM, attaches SLSA
provenance, and verifies all three records before the job can pass.

The signed image is:

```text
ghcr.io/maziyarpanahi/openmed@sha256:<digest>
```

Tags such as `latest`, `vX.Y.Z`, and `sha-<commit>` are convenience pointers.
Verify the digest you deploy, not just a mutable tag.

## Verify Locally

Install `cosign`, then resolve the digest for the tag you plan to deploy:

```bash
image=ghcr.io/maziyarpanahi/openmed:latest
digest="$(docker buildx imagetools inspect "$image" --format '{{ .Manifest.Digest }}')"
image_ref="ghcr.io/maziyarpanahi/openmed@$digest"
```

Verify the keyless signature against the project workflow identity:

```bash
cosign verify "$image_ref" \
  --certificate-identity-regexp '^https://github.com/maziyarpanahi/openmed/\.github/workflows/image-signing\.yml@refs/(heads/master|tags/v.*)$' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```

Verify the attached CycloneDX SBOM and SLSA provenance attestations:

```bash
cosign verify-attestation "$image_ref" \
  --type https://cyclonedx.org/bom \
  --certificate-identity-regexp '^https://github.com/maziyarpanahi/openmed/\.github/workflows/image-signing\.yml@refs/(heads/master|tags/v.*)$' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com

cosign verify-attestation "$image_ref" \
  --type https://slsa.dev/provenance/v1 \
  --certificate-identity-regexp '^https://github.com/maziyarpanahi/openmed/\.github/workflows/image-signing\.yml@refs/(heads/master|tags/v.*)$' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```

These commands fail closed: a missing signature, wrong issuer, wrong workflow
identity, or missing attestation returns a non-zero exit code.

## Admission Policy

`deploy/policy/image-verification.yaml` contains a Sigstore Policy Controller
`ClusterImagePolicy` for OpenMed images. Apply it after installing the controller:

```bash
kubectl apply -f deploy/policy/image-verification.yaml
```

The policy is in enforce mode. Pods that reference `ghcr.io/maziyarpanahi/openmed`
images are admitted only when the image has:

- a keyless Sigstore signature from the OpenMed signing workflow,
- a signed CycloneDX SBOM attestation, and
- a signed SLSA provenance attestation tied to the OpenMed container workflow.

Keep image references digest-pinned in production manifests so admission verifies
the exact artifact you reviewed:

```yaml
image: ghcr.io/maziyarpanahi/openmed@sha256:<digest>
```
