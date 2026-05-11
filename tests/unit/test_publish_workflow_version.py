"""Release workflow regression tests."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
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
