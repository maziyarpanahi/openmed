"""Source-of-truth guards for the troubleshooting guide."""

from __future__ import annotations

import re
from pathlib import Path

try:  # Python 3.11+
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as _toml  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "docs" / "troubleshooting.md"
FAQ = ROOT / "docs" / "faq.md"
MKDOCS = ROOT / "mkdocs.yml"
PYPROJECT = ROOT / "pyproject.toml"

REQUIRED_SECTIONS = (
    "## Install / Extras",
    "## Model Download & Offline",
    "## Performance / Cold-start & Memory",
    "## Device (CPU / GPU / MLX)",
)
REQUIRED_EXTRAS = {"cli", "coreml", "hf", "mlx", "service"}


def _documented_extras(text: str) -> set[str]:
    references: set[str] = set()
    for match in re.findall(r"openmed\[([^\]]+)\]", text):
        references.update(extra.strip() for extra in match.split(","))
    return references


def test_troubleshooting_guide_is_published_and_cross_linked() -> None:
    guide = GUIDE.read_text(encoding="utf-8")
    faq = FAQ.read_text(encoding="utf-8")
    nav = MKDOCS.read_text(encoding="utf-8")

    assert all(section in guide for section in REQUIRED_SECTIONS)
    assert "Troubleshooting & Common Errors: troubleshooting.md" in nav
    assert "[Troubleshooting & Common Errors](troubleshooting.md)" in faq
    assert "[FAQ](faq.md)" in guide


def test_documented_install_extras_are_declared_in_pyproject() -> None:
    guide = GUIDE.read_text(encoding="utf-8")
    with PYPROJECT.open("rb") as handle:
        declared = set(_toml.load(handle)["project"]["optional-dependencies"])

    documented = _documented_extras(guide)
    assert REQUIRED_EXTRAS <= documented
    assert documented <= declared
