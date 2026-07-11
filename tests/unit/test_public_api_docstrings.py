"""Docstring-coverage gate for OpenMed's public API surface.

These tests enforce that every public export (the names in ``openmed.__all__``)
carries a docstring, so the mkdocstrings-generated API reference never ships an
empty entry. The measurement is performed statically by
``scripts/check_public_api_docstrings.py`` (stdlib-only ``ast`` parsing) so it
does not need the heavy runtime dependencies to import.

The floor is pinned to the current coverage and is intended to ratchet upward,
never downward: adding an undocumented public export fails the gate.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = REPO_ROOT / "scripts" / "check_public_api_docstrings.py"


def _load_checker():
    """Import the stdlib docstring checker from ``scripts/`` by path."""
    module_name = "openmed_public_api_docstring_checker"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, CHECKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before executing so dataclass introspection can resolve the
    # module's namespace during class creation.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()

# The coverage floor is set to the current measured coverage (100%). Keep this
# at or below the live value; raise it as coverage improves, never lower it to
# accommodate a regression.
MIN_COVERAGE = 100.0


def test_public_api_docstring_coverage_meets_floor():
    """Every public function/class export must carry a docstring."""
    report = checker.measure_public_api_docstrings(min_coverage=MIN_COVERAGE)
    if not report.passed:
        offenders = "\n".join(
            f"  - {s.name} ({s.module or 'openmed/__init__.py'}): {s.reason}"
            for s in report.missing
        )
        pytest.fail(
            "Public API docstring coverage "
            f"{report.coverage:.1f}% is below the {MIN_COVERAGE:.1f}% floor.\n"
            "Add a docstring to each undocumented public export:\n"
            f"{offenders}"
        )


def test_public_api_has_scored_symbols():
    """Sanity check: the public surface is non-trivial and being measured."""
    report = checker.measure_public_api_docstrings()
    # Guards against the resolver silently scoring nothing (e.g. if __all__ or
    # the import graph fails to parse), which would make the gate vacuous.
    assert len(report.scored) >= 50


def test_key_dataclasses_are_documented():
    """The headline PII dataclasses must resolve and be documented."""
    report = checker.measure_public_api_docstrings()
    by_name = {s.name: s for s in report.symbols}
    for name in ("PIIEntity", "DeidentificationResult", "PIIPattern"):
        assert name in by_name, f"{name} missing from public API report"
        symbol = by_name[name]
        assert symbol.kind == "class"
        assert symbol.documented, f"{name} is missing a docstring"


def test_live_public_api_exports_resolve_and_are_documented():
    """Every live public export must resolve and expose a docstring."""
    import openmed

    assert openmed.__all__, "openmed.__all__ must define the public API"
    for name in openmed.__all__:
        exported = getattr(openmed, name)
        assert inspect.getdoc(exported), f"openmed.{name} is missing a docstring"


def test_cli_gate_passes_at_floor():
    """The CLI entry point exits 0 when coverage meets the floor."""
    exit_code = checker.main(["--min-coverage", str(MIN_COVERAGE)])
    assert exit_code == 0
