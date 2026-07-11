"""Release provenance workflow regression tests."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = ROOT / ".github" / "workflows"
PUBLISH_WORKFLOW = WORKFLOWS / "publish.yml"
PROVENANCE_WORKFLOW = WORKFLOWS / "provenance.yml"
CONTAINER_WORKFLOW = WORKFLOWS / "container-multiarch.yml"
PROVENANCE_DOCS = ROOT / "docs" / "supply-chain" / "provenance.md"
MKDOCS = ROOT / "mkdocs.yml"


def _load_workflow(path: Path) -> dict[str, object]:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_reusable_provenance_workflow_attests_and_verifies_distributions():
    workflow = _load_workflow(PROVENANCE_WORKFLOW)
    content = PROVENANCE_WORKFLOW.read_text(encoding="utf-8")
    jobs = workflow["jobs"]
    job = jobs["python-distributions"]

    assert workflow["on"]["workflow_call"]["inputs"]["distribution-artifact-name"]
    assert job["name"] == "Build, attest, and verify Python distributions"
    assert job["permissions"] == {
        "contents": "read",
        "id-token": "write",
        "attestations": "write",
    }
    assert "actions/checkout@v7" in content
    assert "actions/setup-python@v6" in content
    assert "python -m build" in content
    assert "twine check dist/*" in content
    assert "actions/attest@v4" in content
    assert "continue-on-error: true" in content
    assert "subject-checksums: release-artifact-digests.txt" in content
    assert "release-artifact-digests.txt" in content
    assert 'gh attestation verify "$artifact"' in content
    assert "steps.attest-distributions.outcome == 'success'" in content
    assert "--predicate-type https://slsa.dev/provenance/v1" in content
    assert ".github/workflows/provenance.yml" in content
    assert '--source-digest "$GITHUB_SHA"' in content
    assert '--source-ref "$GITHUB_REF"' in content
    assert "actions/upload-artifact@v7" in content


def test_publish_workflow_blocks_pypi_upload_on_provenance_verification():
    workflow = _load_workflow(PUBLISH_WORKFLOW)
    jobs = workflow["jobs"]
    provenance = jobs["provenance"]
    publish = jobs["publish"]

    assert provenance["uses"] == "./.github/workflows/provenance.yml"
    assert provenance["permissions"] == {
        "contents": "read",
        "id-token": "write",
        "attestations": "write",
    }
    assert publish["needs"] == "provenance"
    assert "pypa/gh-action-pypi-publish@v1.14.0" in PUBLISH_WORKFLOW.read_text(
        encoding="utf-8"
    )


def test_container_workflow_attests_pushed_manifest_digest():
    workflow = _load_workflow(CONTAINER_WORKFLOW)
    content = CONTAINER_WORKFLOW.read_text(encoding="utf-8")

    assert workflow["permissions"]["contents"] == "read"
    assert workflow["permissions"]["id-token"] == "write"
    assert workflow["permissions"]["attestations"] == "write"
    assert workflow["permissions"]["packages"] == "write"
    assert "id: push" in content
    assert "subject-name: ${{ env.IMAGE_NAME }}" in content
    assert "subject-digest: ${{ steps.push.outputs.digest }}" in content
    assert "push-to-registry: true" in content
    assert "create-storage-record: false" in content
    assert 'gh attestation verify "oci://${image_ref}"' in content
    assert ".github/workflows/container-multiarch.yml" in content
    assert '--source-digest "$GITHUB_SHA"' in content
    assert '--source-ref "$GITHUB_REF"' in content


def test_provenance_docs_are_published_and_cover_offline_verification():
    docs = PROVENANCE_DOCS.read_text(encoding="utf-8")
    nav = MKDOCS.read_text(encoding="utf-8")

    assert "supply-chain/provenance.md" in nav
    assert "gh attestation download" in docs
    assert "gh attestation trusted-root > trusted_root.jsonl" in docs
    assert '--bundle "$BUNDLE"' in docs
    assert "--custom-trusted-root trusted_root.jsonl" in docs
    assert '--source-digest "$COMMIT"' in docs
    assert "sha256sum --check artifact.sha256" in docs
    assert "oci://ghcr.io/maziyarpanahi/openmed@$DIGEST" in docs
