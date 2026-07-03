"""Tests for container image signing workflow and policy metadata."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github" / "workflows" / "image-signing.yml"
POLICY = ROOT / "deploy" / "policy" / "image-verification.yaml"
DOCS = ROOT / "docs" / "supply-chain" / "image-signing.md"
SUPPLY_CHAIN_DOCS = ROOT / "docs" / "security" / "supply-chain.md"
MULTIARCH_DOCS = ROOT / "docs" / "deploy" / "multi-arch.md"
MKDOCS = ROOT / "mkdocs.yml"


def test_image_signing_workflow_signs_and_verifies_digest():
    content = WORKFLOW.read_text(encoding="utf-8")

    assert "workflows:\n      - Container Multi-Arch" in content
    assert "id-token: write" in content
    assert "packages: write" in content
    assert "sigstore/cosign-installer@v4.1.2" in content
    assert "anchore/sbom-action/download-syft@v0.24.0" in content
    assert "docker buildx imagetools inspect" in content
    assert 'cosign sign --yes "$IMAGE_DIGEST_REF"' in content
    assert 'cosign verify "$IMAGE_DIGEST_REF"' in content
    assert '--certificate-identity-regexp "$SIGNER_IDENTITY_RE"' in content
    assert '--certificate-oidc-issuer "$OIDC_ISSUER"' in content


def test_image_signing_workflow_attests_sbom_and_provenance():
    content = WORKFLOW.read_text(encoding="utf-8")

    assert 'syft "${{ steps.image.outputs.image_digest_ref }}"' in content
    assert "--type https://cyclonedx.org/bom" in content
    assert "--type https://slsa.dev/provenance/v1" in content
    assert 'cosign verify-attestation "$IMAGE_DIGEST_REF"' in content
    assert "sbom-attestation.jsonl" in content
    assert "provenance-attestation.jsonl" in content
    assert ".github/workflows/container-multiarch.yml" in content
    assert ".github/workflows/image-signing.yml" in content


def test_image_verification_policy_enforces_keyless_signature_and_attestations():
    content = POLICY.read_text(encoding="utf-8")

    assert "kind: ClusterImagePolicy" in content
    assert "mode: enforce" in content
    assert "glob: ghcr.io/maziyarpanahi/openmed*" in content
    assert "issuer: https://token.actions.githubusercontent.com" in content
    assert (
        "subjectRegExp: ^https://github.com/maziyarpanahi/openmed/\\.github/"
        "workflows/image-signing\\.yml@refs/(heads/master|tags/v.*)$"
    ) in content
    assert "predicateType: https://cyclonedx.org/bom" in content
    assert "predicateType: https://slsa.dev/provenance/v1" in content
    assert 'bomFormat: "CycloneDX"' in content
    assert 'publishingWorkflow: ".github/workflows/container-multiarch.yml"' in content


def test_image_signing_docs_include_verification_and_admission_policy():
    content = DOCS.read_text(encoding="utf-8")

    assert 'cosign verify "$image_ref"' in content
    assert 'cosign verify-attestation "$image_ref"' in content
    assert "--type https://cyclonedx.org/bom" in content
    assert "--type https://slsa.dev/provenance/v1" in content
    assert "kubectl apply -f deploy/policy/image-verification.yaml" in content
    assert "ghcr.io/maziyarpanahi/openmed@sha256:<digest>" in content
    assert "image: ghcr.io/maziyarpanahi/openmed@sha256:<digest>" in content


def test_image_signing_docs_are_linked_from_existing_docs_and_nav():
    assert "supply-chain/image-signing.md" in MKDOCS.read_text(encoding="utf-8")
    assert "Container Image Signing" in SUPPLY_CHAIN_DOCS.read_text(encoding="utf-8")
    assert "Container Image Signing" in MULTIARCH_DOCS.read_text(encoding="utf-8")
