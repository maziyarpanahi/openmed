"""Release workflow regression tests."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = ROOT / ".github" / "workflows"
PUBLISH_WORKFLOW = ROOT / ".github" / "workflows" / "publish.yml"
PROVENANCE_WORKFLOW = ROOT / ".github" / "workflows" / "provenance.yml"
ABOUT_FILE = ROOT / "openmed" / "__about__.py"


def test_publish_workflow_reads_version_without_importing_openmed_package():
    """The publish job installs build tools before runtime deps.

    Reading the release version must not import ``openmed`` because the package
    root imports runtime modules that depend on installed project dependencies.
    """

    workflow = PROVENANCE_WORKFLOW.read_text(encoding="utf-8")

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
    publish_workflow = PUBLISH_WORKFLOW.read_text(encoding="utf-8")
    provenance_workflow = PROVENANCE_WORKFLOW.read_text(encoding="utf-8")

    assert "tags:\n      - 'v*'" in publish_workflow
    assert "workflow_dispatch:" not in publish_workflow
    assert "pull_request:" not in publish_workflow
    assert "uses: ./.github/workflows/provenance.yml" in publish_workflow
    assert "needs: provenance" in publish_workflow
    assert "name: pypi" in publish_workflow
    assert "id-token: write" in publish_workflow
    assert "pypa/gh-action-pypi-publish@v1.14.0" in publish_workflow
    assert "attestations: true" in publish_workflow
    assert "HATCH_INDEX_AUTH: ${{ secrets.PYPI_API_TOKEN }}" not in publish_workflow

    assert "fetch-depth: 0" in provenance_workflow
    assert "python scripts/release/check_repo_policy.py" in provenance_workflow
    assert "Compute release metadata" in provenance_workflow
    assert "python scripts/release/changelog.py" in provenance_workflow
    assert "steps.release_metadata.outputs.next_version" in provenance_workflow
    assert "Verify version matches tag" in provenance_workflow
    assert "twine check dist/*" in provenance_workflow
