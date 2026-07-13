"""Release workflow regression tests."""

from __future__ import annotations

import json
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
WEB_PACKAGE = ROOT / "js" / "openmedkit-web" / "package.json"
WEB_PACKAGE_README = ROOT / "js" / "openmedkit-web" / "README.md"


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


def test_npm_package_tracks_openmed_release_version():
    content = ABOUT_FILE.read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"([^"]+)"', content)
    package = json.loads(WEB_PACKAGE.read_text(encoding="utf-8"))

    assert match is not None
    assert package["name"] == "openmed"
    assert package["version"] == match.group(1)
    assert package["publishConfig"] == {
        "access": "public",
        "provenance": True,
    }


def test_npm_package_readme_uses_public_unpinned_package_name():
    readme = WEB_PACKAGE_README.read_text(encoding="utf-8")

    assert "npm install openmed" in readme
    assert 'from "openmed"' in readme
    assert "npm install openmed@" not in readme
    assert "@openmed/openmedkit-web" not in readme


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
    assert "pypa/gh-action-pypi-publish@v1.14.0" in publish_workflow
    assert "HATCH_INDEX_AUTH: ${{ secrets.PYPI_API_TOKEN }}" not in publish_workflow

    assert publish_job["environment"]["name"] == "pypi"
    assert publish_job["environment"]["url"] == "https://pypi.org/p/openmed"
    assert publish_job["permissions"] == {"contents": "read"}
    assert "id-token" not in publish_job["permissions"]
    assert publish_step["with"]["password"] == "${{ secrets.PYPI_API_TOKEN }}"
    assert publish_step["with"]["attestations"] == "false"
    assert publish_job["needs"] == ["provenance", "npm-verify"]

    assert "fetch-depth: 0" in provenance_workflow
    assert "id-token: write" in provenance_workflow
    assert "python scripts/release/check_repo_policy.py" in provenance_workflow
    assert "Compute release metadata" in provenance_workflow
    assert "python scripts/release/changelog.py" in provenance_workflow
    assert "steps.release_metadata.outputs.next_version" in provenance_workflow
    assert "Verify version matches tag" in provenance_workflow
    assert "twine check dist/*" in provenance_workflow


def test_publish_workflow_verifies_and_publishes_npm_package():
    workflow = _load_workflow(PUBLISH_WORKFLOW)
    content = PUBLISH_WORKFLOW.read_text(encoding="utf-8")
    npm_verify = workflow["jobs"]["npm-verify"]
    npm_publish = workflow["jobs"]["npm-publish"]
    sbom = workflow["jobs"]["sbom"]
    publish_step = next(
        step
        for step in npm_publish["steps"]
        if step.get("name") == "Publish npm package with provenance"
    )

    assert npm_verify["permissions"] == {"contents": "read"}
    assert "actions/setup-node@v6" in content
    assert "node-version: '24'" in content
    assert "package-manager-cache: false" in content
    assert "npm audit --audit-level=low" in content
    assert "npm test" in content
    assert "npm pack --dry-run" in content
    assert "NPM_VERSION" in content

    assert npm_publish["needs"] == ["provenance", "npm-verify"]
    assert npm_publish["environment"] == {
        "name": "npm",
        "url": "https://www.npmjs.com/package/openmed",
    }
    assert npm_publish["permissions"] == {
        "contents": "read",
        "id-token": "write",
    }
    assert publish_step["run"] == (
        "npm publish --ignore-scripts --access public --provenance"
    )
    assert publish_step["env"]["NODE_AUTH_TOKEN"] == ("${{ secrets.NPM_ACCESS_TOKEN }}")
    assert sbom["needs"] == ["publish", "npm-publish"]


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


def test_jitpack_builds_the_latest_android_release_from_github():
    config = _load_workflow(JITPACK_CONFIG)
    install_command = config["install"][0]
    before_install = config["before_install"]
    android_build = ANDROID_BUILD.read_text(encoding="utf-8")
    android_readme = ANDROID_README.read_text(encoding="utf-8")

    assert config["jdk"] == ["openjdk11"]
    assert "https://astral.sh/uv/0.11.16/install.sh" in before_install[0]
    assert "uv python install 3.12" in before_install[1]
    assert "cd android" in install_command
    assert "uv python find 3.12" in install_command
    assert ":openmedkit:publishReleasePublicationToMavenLocal" in install_command
    assert '-PopenmedAndroidVersion="$VERSION"' in install_command
    assert '-PopenmedAndroidGroup="$GROUP"' in install_command
    assert '-PopenmedAndroidArtifact="$ARTIFACT"' in install_command
    assert '-PopenmedPython="$PYTHON_BIN"' in install_command

    assert 'gradleProperty("openmedAndroidGroup")' in android_build
    assert 'gradleProperty("openmedAndroidArtifact")' in android_build
    assert "groupId = publicationGroup" in android_build
    assert "artifactId = publicationArtifact" in android_build
    assert 'it.name.startsWith("publish")' not in android_build

    coordinates = "com.github.maziyarpanahi:openmed:master-SNAPSHOT"
    assert coordinates in android_readme
    assert coordinates in (ROOT / "README.md").read_text(encoding="utf-8")
    assert re.search(r"com\.github\.maziyarpanahi:openmed:v\d", android_readme) is None


def test_readme_install_guidance_tracks_latest_openmed_release():
    readmes = set(ROOT.glob("README*.md"))
    for directory in ("android", "deploy", "examples", "js", "openmed", "swift"):
        readmes.update((ROOT / directory).glob("**/README.md"))
    readmes = sorted(
        path
        for path in readmes
        if not any(part.startswith(".") for part in path.relative_to(ROOT).parts)
        and "node_modules" not in path.parts
    )
    assert readmes

    violations = []
    for path in readmes:
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(ROOT)
        if re.search(r'from: "\d+\.\d+', text):
            violations.append(f"{relative}: pinned Swift package")
        if re.search(r"com\.github\.maziyarpanahi:openmed:v\d", text):
            violations.append(f"{relative}: pinned Android package")
        if re.search(r"openmed:[0-9]+\.[0-9]+", text):
            violations.append(f"{relative}: pinned Docker tag")
        if re.search(r"npm install openmed@\d", text):
            violations.append(f"{relative}: pinned npm package")
        if "@openmed/openmedkit-web" in text:
            violations.append(f"{relative}: unpublished npm package name")
        for line in text.splitlines():
            if 'pip install "openmed' in line and "pip install --upgrade" not in line:
                violations.append(f"{relative}: install does not use --upgrade")

    assert violations == []


def test_localized_readmes_advertise_current_model_count():
    readmes = sorted(ROOT.glob("README*.md"))

    assert len(readmes) >= 14
    assert all("2%2C000+" in path.read_text(encoding="utf-8") for path in readmes)
