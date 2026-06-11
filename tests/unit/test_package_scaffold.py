"""Smoke tests for the section 4.2 package scaffold."""

from importlib import import_module
from pathlib import Path


def test_section_4_2_packages_import_cleanly():
    """All new top-level package shells should be importable."""
    package_names = [
        "openmed.clinical",
        "openmed.eval",
        "openmed.multimodal",
        "openmed.structured",
        "openmed.risk",
        "openmed.interop",
        "openmed.interop.bridges",
    ]

    for package_name in package_names:
        import_module(package_name)

    repo_root = Path(__file__).resolve().parents[2]
    assert not (repo_root / "openmed" / "evals").exists()
