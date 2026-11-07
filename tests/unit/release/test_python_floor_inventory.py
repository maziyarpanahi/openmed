"""Tests for python_floor_inventory.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "release" / "python_floor_inventory.py"


def test_inventory_covers_guard_marker_workflow_and_docs(tmp_path: Path) -> None:
    """Synthetic tree covering the required kinds."""
    # Python guard
    py = tmp_path / "src" / "mod.py"
    py.parent.mkdir()
    py.write_text(
        "import sys
"
        "if sys.version_info < (3, 11):
"
        "    pass
"
    )

    # pyproject marker
    toml = tmp_path / "pyproject.toml"
    toml.write_text(
        '[project]
'
        'requires-python = ">=3.11"
'
        'dependencies = [
'
        '  "foo; python_version < '3.12'",
'
        ']
'
    )

    # workflow
    wf_dir = tmp_path / ".github" / "workflows"
    wf_dir.mkdir(parents=True)
    wf = wf_dir / "ci.yml"
    wf.write_text(
        "jobs:
"
        "  test:
"
        "    steps:
"
        "      - uses: actions/setup-python@v5
"
        "        with:
"
        "          python-version: '3.11'
"
    )

    # docs
    docs = tmp_path / "docs" / "getting-started.md"
    docs.parent.mkdir()
    docs.write_text("Requires Python 3.11+
")

    # run
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--floor", "3.12", "--json", "--root", str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    kinds = {item["kind"] for item in data}
    assert "version_info_guard" in kinds
    assert "python_version_marker" in kinds or "requires_python" in kinds
    assert "workflow_python_version" in kinds
    assert "docs_mention" in kinds
    # no file contents
    for item in data:
        assert set(item.keys()) == {"path", "line_number", "kind"}
        assert "content" not in item


def test_empty_when_all_above_floor(tmp_path: Path) -> None:
    py = tmp_path / "ok.py"
    py.write_text("if sys.version_info >= (3, 12):
    pass
")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--floor", "3.12", "--json", "--root", str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    assert data == []
