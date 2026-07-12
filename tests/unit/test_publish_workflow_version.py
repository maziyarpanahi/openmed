"""Release workflow regression tests."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = ROOT / ".github" / "workflows"
PUBLISH_WORKFLOW = ROOT / ".github" / "workflows" / "publish.yml"
PROVENANCE_WORKFLOW = ROOT / ".github" / "workflows" / "provenance.yml"
IMAGE_SBOM_WORKFLOW = ROOT / ".github" / "workflows" / "sbom-image.yml"
ANDROID_PUBLISH_WORKFLOW = ROOT / ".github" / "workflows" / "android-publish.yml"
ANDROID_BUILD = ROOT / "android" / "openmedkit" / "build.gradle.kts"
ANDROID_README = ROOT / "android" / "README.md"
JITPACK_CONFIG = ROOT / "jitpack.yml"
ABOUT_FILE = ROOT / "openmed" / "__about__.py"


def _load_workflow(path: Path) -> dict[str, object]:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_publish_workflow_reads_version_without_importing_openmed_package():
    """The publish job installs build tools before runtime deps.

    Reading the release version must not import ``openmed`` because the package
    root imports runtime modules that depend on installed project dependencies.
    """

    publish_workflow = PUBLISH_WORKFLOW.read_text(encoding="utf-8")
    provenance_workflow = PROVENANCE_WORKFLOW.read_text(encoding="utf-8")

    assert "from openmed import __version__" not in publish_workflow
    assert "from openmed import __version__" not in provenance_workflow
    assert "openmed/__about__.py" in provenance_workflow


def test_about_version_is_parseable_without_runtime_dependencies():
    content = ABOUT_FILE.read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"([^"]+)"', content)

    assert match is not None
    assert re.fullmatch(r"\d+\.\d+\.\d+", match.group(1))


def test_only_publish_workflow_uses_pypi_publish_action():
    publishing_workflows = [
        workflow
        for workflow in WORKFLOWS_DIR.glob("*.yml")
        if "pypa/gh-action-pypi-publish" in workflow.read_text(encoding="utf-8")
    ]

    assert publishing_workflows == [PUBLISH_WORKFLOW]

    for workflow in WORKFLOWS_DIR.glob("*.yml"):
        content = workflow.read_text(encoding="utf-8")
        assert "hatch publish" not in content
        assert "HATCH_INDEX_AUTH" not in content

    publish_workflow = PUBLISH_WORKFLOW.read_text(encoding="utf-8")
    assert "PYPI_API_TOKEN" in publish_workflow


def test_publish_workflow_keeps_release_gates():
    publish_workflow = PUBLISH_WORKFLOW.read_text(encoding="utf-8")
    provenance_workflow = PROVENANCE_WORKFLOW.read_text(encoding="utf-8")
    workflow = _load_workflow(PUBLISH_WORKFLOW)
    publish_job = workflow["jobs"]["publish"]
    publish_step = next(
        step
        for step in publish_job["steps"]
        if step.get("uses", "").startswith("pypa/gh-action-pypi-publish@")
    )

    assert "tags:\n      - 'v*'" in publish_workflow
    assert "workflow_dispatch:" not in publish_workflow
    assert "pull_request:" not in publish_workflow
    assert "uses: ./.github/workflows/provenance.yml" in publish_workflow
    assert "needs: provenance" in publish_workflow
    assert "pypa/gh-action-pypi-publish@v1.14.0" in publish_workflow
    assert "HATCH_INDEX_AUTH: ${{ secrets.PYPI_API_TOKEN }}" not in publish_workflow

    assert publish_job["environment"]["name"] == "pypi"
    assert publish_job["environment"]["url"] == "https://pypi.org/p/openmed"
    assert publish_job["permissions"] == {"contents": "read"}
    assert "id-token" not in publish_job["permissions"]
    assert publish_step["with"]["password"] == "${{ secrets.PYPI_API_TOKEN }}"
    assert publish_step["with"]["attestations"] == "false"

    assert "fetch-depth: 0" in provenance_workflow
    assert "id-token: write" in provenance_workflow
    assert "python scripts/release/check_repo_policy.py" in provenance_workflow
    assert "Compute release metadata" in provenance_workflow
    assert "python scripts/release/changelog.py" in provenance_workflow
    assert "steps.release_metadata.outputs.next_version" in provenance_workflow
    assert "Verify version matches tag" in provenance_workflow
    assert "twine check dist/*" in provenance_workflow


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


def test_android_publish_skips_unchanged_artifacts_and_runs_its_own_tests():
    workflow = ANDROID_PUBLISH_WORKFLOW.read_text(encoding="utf-8")

    assert "fetch-depth: 0" in workflow
    assert "Detect Android artifact changes" in workflow
    assert 'git describe --tags --abbrev=0 "${GITHUB_SHA}^"' in workflow
    assert "android/ models.jsonl scripts/android/build_android_catalog.py" in workflow
    assert "':(exclude,glob)android/**/*.md'" in workflow
    assert 'echo "publish_android=false"' in workflow
    assert workflow.count("if: needs.guard.outputs.publish_android == 'true'") == 2
    assert ":openmedkit:assembleDebug" in workflow
    assert ":openmedkit:testDebugUnitTest" in workflow
    assert "check-runs" not in workflow
    assert "Android AAR size and cold-start gate" not in workflow


def test_jitpack_builds_the_android_release_from_github_tags():
    config = _load_workflow(JITPACK_CONFIG)
    install_command = config["install"][0]
    android_build = ANDROID_BUILD.read_text(encoding="utf-8")
    android_readme = ANDROID_README.read_text(encoding="utf-8")

    assert config["jdk"] == ["openjdk11"]
    assert "cd android" in install_command
    assert ":openmedkit:publishReleasePublicationToMavenLocal" in install_command
    assert '-PopenmedAndroidVersion="$VERSION"' in install_command
    assert '-PopenmedAndroidGroup="$GROUP"' in install_command
    assert '-PopenmedAndroidArtifact="$ARTIFACT"' in install_command

    assert 'gradleProperty("openmedAndroidGroup")' in android_build
    assert 'gradleProperty("openmedAndroidArtifact")' in android_build
    assert "groupId = publicationGroup" in android_build
    assert "artifactId = publicationArtifact" in android_build
    assert 'it.name.startsWith("publish")' not in android_build

    coordinates = "com.github.maziyarpanahi:openmed:v1.8.2"
    assert coordinates in android_readme
    assert coordinates in (ROOT / "README.md").read_text(encoding="utf-8")
