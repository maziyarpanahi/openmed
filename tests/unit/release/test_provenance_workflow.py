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


def _steps_by_name(job: dict[str, object]) -> dict[str, dict[str, object]]:
    return {step["name"]: step for step in job["steps"] if "name" in step}


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
    assert "actions/setup-python@v7" in content
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
    npm_verify = jobs["npm-verify"]
    npm_publish = jobs["npm-publish"]

    assert provenance["uses"] == "./.github/workflows/provenance.yml"
    assert provenance["permissions"] == {
        "contents": "read",
        "id-token": "write",
        "attestations": "write",
    }
    assert publish["needs"] == ["provenance", "npm-verify"]
    assert npm_publish["needs"] == ["provenance", "npm-verify"]
    assert npm_publish["permissions"]["id-token"] == "write"
    assert "npm test" in str(npm_verify)
    assert "npm run build" in str(npm_publish)
    assert "npm publish --ignore-scripts --access public --provenance" in str(
        npm_publish
    )
    assert "pypa/gh-action-pypi-publish@v1.14.1" in PUBLISH_WORKFLOW.read_text(
        encoding="utf-8"
    )


def test_publish_workflow_never_gates_pypi_upload_on_evidence():
    workflow = _load_workflow(PUBLISH_WORKFLOW)
    jobs = workflow["jobs"]
    evidence = jobs["evidence"]

    # Publishing must not become dependent on GitHub OIDC or Sigstore
    # availability, so no registry upload may wait on the evidence job.
    assert "evidence" not in jobs["publish"]["needs"]
    assert "evidence" not in jobs["npm-publish"]["needs"]

    # The evidence job runs strictly after the PyPI upload it collects evidence for.
    assert evidence["needs"] == ["provenance", "publish"]
    assert evidence["permissions"] == {"contents": "write", "id-token": "write"}


def test_evidence_job_signs_best_effort_but_fails_on_unverifiable_evidence():
    workflow = _load_workflow(PUBLISH_WORKFLOW)
    content = PUBLISH_WORKFLOW.read_text(encoding="utf-8")
    steps = _steps_by_name(workflow["jobs"]["evidence"])

    # Producing evidence is best effort.
    assert steps["Download provenance evidence"]["continue-on-error"] == "true"
    assert steps["Install Sigstore"]["continue-on-error"] == "true"
    assert steps["Sign distributions with Sigstore"]["continue-on-error"] == "true"

    # Evidence that was produced must verify and attach, or the job fails.
    assert "continue-on-error" not in steps["Verify Sigstore bundles"]
    assert (
        "continue-on-error"
        not in steps["Attach release evidence to the tagged GitHub release"]
    )

    assert "sigstore==4.4.0" in content
    assert "python -m sigstore sign" in content
    assert "python -m sigstore verify github" in content
    assert "--offline" in content
    assert (
        'identity="https://github.com/${GITHUB_REPOSITORY}/${SIGNER_WORKFLOW}@${GITHUB_REF}"'
        in content
    )
    assert "SIGNER_WORKFLOW: .github/workflows/publish.yml" in content
    assert '--repository "$GITHUB_REPOSITORY"' in content
    assert '--sha "$GITHUB_SHA"' in content
    assert '--ref "$GITHUB_REF"' in content


def test_evidence_job_attaches_provenance_and_signatures_to_the_release():
    content = PUBLISH_WORKFLOW.read_text(encoding="utf-8")

    assert (
        "release-provenance/provenance-bundles/python-distributions.intoto.json"
        in content
    )
    assert "release-provenance/release-artifact-digests.txt" in content
    assert "assets+=(sigstore-bundles/*.sigstore.json)" in content
    assert 'gh release upload "$GITHUB_REF_NAME" "${assets[@]}" --clobber' in content

    # A partially successful signing run leaves unverified bundles on disk, so
    # signatures are only attached when the whole signing step succeeded.
    assert "SIGN_OUTCOME: ${{ steps.sign-distributions.outcome }}" in content
    assert 'if [ "$SIGN_OUTCOME" = "success" ]; then' in content

    # A successful artifact download must contain both required provenance
    # assets; partial or empty evidence must fail before release attachment.
    assert "PROVENANCE_OUTCOME: ${{ steps.provenance-evidence.outcome }}" in content
    assert 'if [ "$PROVENANCE_OUTCOME" = "success" ]; then' in content
    assert 'if [ ! -s "$candidate" ]; then' in content
    assert "Expected provenance evidence is missing or empty" in content

    # The job needs an explicit repo: it does not check the repository out.
    assert "GH_REPO: ${{ github.repository }}" in content


def test_provenance_docs_cover_sigstore_verification_from_release_assets():
    docs = PROVENANCE_DOCS.read_text(encoding="utf-8")

    assert "gh release download" in docs
    assert "python -m sigstore verify github" in docs
    assert "--cert-identity" in docs
    assert "python-distributions.intoto.json" in docs
    assert "sha256sum --check --ignore-missing release-artifact-digests.txt" in docs


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
