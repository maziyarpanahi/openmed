#!/usr/bin/env python3
"""Generate a CycloneDX 1.6 SBOM for OpenMed's runtime environment.

Produces ``sbom.cdx.json`` describing ``openmed`` (the root component) plus the
runtime dependencies installed in the target virtual environment. Companion to
the other supply-chain helpers under ``scripts/security/``.

OpenMed's version is dynamic (hatch reads it from ``openmed/__about__.py``), so
``cyclonedx-py --pyproject`` cannot resolve it from the PEP 621 metadata; we
stamp the real version and PURL onto the root component after generation.

Usage::

    uv sync --frozen
    uv run --no-project --with 'cyclonedx-bom>=4.6,<7' \
        python scripts/security/generate_sbom.py

The target environment defaults to ``.venv`` and can be overridden with the
``SBOM_PYTHON`` environment variable (a path to a venv or interpreter).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTFILE = ROOT / "sbom.cdx.json"
ABOUT = ROOT / "openmed" / "__about__.py"
PYPROJECT = ROOT / "pyproject.toml"
SPEC_VERSION = "1.6"


def read_version() -> str:
    """Parse ``__version__`` from ``openmed/__about__.py`` without importing it."""
    match = re.search(r'__version__\s*=\s*"([^"]+)"', ABOUT.read_text(encoding="utf-8"))
    if not match:
        raise SystemExit(f"could not parse __version__ from {ABOUT}")
    return match.group(1)


def target_environment() -> str:
    """Resolve the environment cyclonedx-py should introspect."""
    override = os.environ.get("SBOM_PYTHON")
    if override:
        return override
    venv = ROOT / ".venv"
    if not venv.exists():
        raise SystemExit(
            f"runtime environment {venv} not found; run 'uv sync --frozen' first "
            "or set SBOM_PYTHON"
        )
    return str(venv)


def main() -> int:
    version = read_version()
    subprocess.run(
        [
            "cyclonedx-py",
            "environment",
            target_environment(),
            "--pyproject",
            str(PYPROJECT),
            "--mc-type",
            "library",
            "--sv",
            SPEC_VERSION,
            "--of",
            "JSON",
            "-o",
            str(OUTFILE),
        ],
        check=True,
    )

    bom = json.loads(OUTFILE.read_text(encoding="utf-8"))
    component = bom.setdefault("metadata", {}).setdefault("component", {})
    component["version"] = version
    component["purl"] = f"pkg:pypi/openmed@{version}"
    component.setdefault("bom-ref", f"openmed@{version}")
    OUTFILE.write_text(json.dumps(bom, indent=2) + "\n", encoding="utf-8")

    print(
        f"wrote {OUTFILE.relative_to(ROOT)}: openmed {version} + "
        f"{len(bom.get('components', []))} runtime dependencies "
        f"(CycloneDX {bom.get('specVersion')})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
