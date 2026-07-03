"""Release workflow regression tests."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = ROOT / ".github" / "workflows"
PUBLISH_WORKFLOW = ROOT / ".github" / "workflows" / "publish.yml"
IMAGE_SBOM_WORKFLOW = ROOT / ".github" / "workflows" / "sbom-image.yml"
ABOUT_FILE = ROOT / "openmed" / "__about__.py"


def test_publish_workflow_reads_version_without_importing_openmed_package():
    """The publish job installs build tools before runtime deps.

    Reading the release version must not import ``openmed`` because the package
    root imports runtime modules that depend on installed project dependencies.
    """

    workflow = PUBLISH_WORKFLOW.read_text(encoding="utf-8")

    assert "from openmed import __version__" not in workflow
    assert "openmed/__about__.py" in workflow


def test_about_version_is_parseable_without_runtime_dependencies():
    content = ABOUT_FILE.read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"([^"]+)"', content)

    assert match is not None
    assert re.fullmatch(r"\d+\.\d+\.\d+", match.group(1))


def test_only_publish_workflow_uses_trusted_publishing_action():
    publishing_workflows = [
        workflow
        for workflow in WORKFLOWS_DIR.glob("*.yml")
        if "pypa/gh-action-pypi-publish" in workflow.read_text(encoding="utf-8")
    ]

    assert publishing_workflows == [PUBLISH_WORKFLOW]

    for workflow in WORKFLOWS_DIR.glob("*.yml"):
        content = workflow.read_text(encoding="utf-8")
        assert "hatch publish" not in content
        assert "PYPI_API_TOKEN" not in content
        assert "HATCH_INDEX_AUTH" not in content


def test_publish_workflow_keeps_release_gates():
    workflow = PUBLISH_WORKFLOW.read_text(encoding="utf-8")

    assert "fetch-depth: 0" in workflow
    assert "tags:\n      - 'v*'" in workflow
    assert "workflow_dispatch:" not in workflow
    assert "pull_request:" not in workflow
    assert "python scripts/release/check_repo_policy.py" in workflow
    assert "Compute release metadata" in workflow
    assert "python scripts/release/changelog.py" in workflow
    assert "steps.release_metadata.outputs.next_version" in workflow
    assert "Verify version matches tag" in workflow
    assert "twine check dist/*" in workflow
    assert "name: pypi" in workflow
    assert "id-token: write" in workflow
    assert "pypa/gh-action-pypi-publish@v1.14.0" in workflow
    assert "attestations: true" in workflow
    assert "HATCH_INDEX_AUTH: ${{ secrets.PYPI_API_TOKEN }}" not in workflow


def test_image_sbom_workflow_builds_and_validates_cyclonedx_image_sbom():
    workflow = IMAGE_SBOM_WORKFLOW.read_text(encoding="utf-8")

    assert "name: Image SBOM" in workflow
    assert "docker/build-push-action" in workflow
    assert "anchore/sbom-action" in workflow
    assert "format: cyclonedx-json" in workflow
    assert "Validate image SBOM" in workflow
    assert 'bom.get("bomFormat") != "CycloneDX"' in workflow
    assert "image SBOM is empty" in workflow
    assert "image SBOM is malformed JSON" in workflow
    assert "pkg:deb/" in workflow
    assert "pkg:pypi/" in workflow
    assert "if-no-files-found: error" in workflow


def test_image_sbom_release_path_attaches_artifact_and_labels_image():
    workflow = IMAGE_SBOM_WORKFLOW.read_text(encoding="utf-8")

    assert "gh release upload" in workflow
    assert "image-sbom.cdx.json" in workflow
    assert "image-sbom.cdx.json.sha256" in workflow
    assert "docker/login-action" in workflow
    assert "push: true" in workflow
    assert (
        "org.opencontainers.image.sbom.digest=${{ steps.sbom_digest.outputs.digest }}"
    ) in workflow
