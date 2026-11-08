"""The floor inventory reports only relevant source locations."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "release"
    / "python_floor_inventory.py"
)


def _run(root: Path, *options: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), *options],
        capture_output=True,
        text=True,
        check=True,
    )


def test_inventory_covers_guard_marker_workflow_and_docs(tmp_path: Path) -> None:
    files = {
        "guard.py": "if sys.version_info[:2] < (3, 11):  # PRIVATE_SENTINEL\n    pass\n",
        "pyproject.toml": (
            '[project]\nrequires-python = ">=3.11"\n'
            "dependencies = [\"demo; python_version >= '3.12'\"]\n"
        ),
        ".github/workflows/ci.yml": 'jobs:\n  test:\n    python: ["3.11", "3.12"]\n',
        "README.md": "Supports Python 3.11+. The API version is 3.2.\n",
        ".venv/README.md": "This is not project documentation for Python 3.11.\n",
    }
    for name, value in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")

    result = _run(tmp_path, "--floor", "3.12", "--json")
    records = json.loads(result.stdout)
    assert [(x["path"], x["line_number"], x["kind"]) for x in records] == [
        (".github/workflows/ci.yml", 3, "workflow_python_version"),
        ("README.md", 1, "docs_mention"),
        ("guard.py", 1, "version_info_guard"),
        ("pyproject.toml", 2, "requires_python"),
        ("pyproject.toml", 3, "python_version_marker"),
    ]
    assert all(set(record) == {"path", "line_number", "kind"} for record in records)
    assert "PRIVATE_SENTINEL" not in _run(tmp_path).stdout
    assert "PRIVATE_SENTINEL" not in result.stdout


def test_keeps_version_checks_that_still_matter(tmp_path: Path) -> None:
    (tmp_path / "guards.py").write_text(
        "if sys.version_info >= (3, 12, 5) or sys.version_info > (3, 12):\n    pass\n",
        encoding="utf-8",
    )
    assert json.loads(_run(tmp_path, "--json").stdout) == []
