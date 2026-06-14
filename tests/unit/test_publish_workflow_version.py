"""Release workflow regression tests."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = ROOT / ".github" / "workflows"
PUBLISH_WORKFLOW = ROOT / ".github" / "workflows" / "publish.yml"
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


def test_only_publish_workflow_runs_hatch_publish():
    publishing_workflows = [
        workflow
        for workflow in WORKFLOWS_DIR.glob("*.yml")
        if "hatch publish" in workflow.read_text(encoding="utf-8")
    ]

    assert publishing_workflows == [PUBLISH_WORKFLOW]


def test_publish_workflow_keeps_release_gates():
    workflow = PUBLISH_WORKFLOW.read_text(encoding="utf-8")

    assert "tags:\n      - 'v*'" in workflow
    assert "workflow_dispatch:" in workflow
    assert "python scripts/release/check_repo_policy.py" in workflow
    assert "Verify version matches tag" in workflow
    assert "twine check dist/*" in workflow
    assert "name: pypi" in workflow
    assert "HATCH_INDEX_AUTH: ${{ secrets.PYPI_API_TOKEN }}" in workflow
