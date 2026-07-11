"""Docstring and export-inventory gates for OpenMed's public API surface.

The stdlib-only static checker resolves every name in the source
``openmed.__all__`` without importing the package. Runtime parity tests import
``openmed`` separately to prove that the live export order matches the static
inventory. Exported functions/classes require meaningful docstrings; data
exports are checked against an explicit name/module/type inventory instead of
passing through generic built-in instance documentation.

Function/class coverage is fixed at 100%: adding an undocumented public
function or class fails the gate.
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

# Every exported function/class must remain documented.
MIN_COVERAGE = 100.0

EXPECTED_DATA_EXPORTS = {
    "__version__": ("openmed", str),
    "PII_PATTERNS": ("openmed.core.pii_entity_merger", list),
    "SUPPORTED_LANGUAGES": ("openmed.core.pii_i18n", set),
    "DEFAULT_PII_MODELS": ("openmed.core.pii_i18n", dict),
    "LANGUAGE_PII_PATTERNS": ("openmed.core.pii_i18n", dict),
    "CANONICAL_LABELS": ("openmed.core.labels", frozenset),
    "LANG_TO_LOCALE": ("openmed.core.anonymizer.locales", dict),
    "ENCRYPTION_SCHEME": ("openmed.core.surrogate_vault", str),
}


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


def test_live_public_api_matches_static_inventory_and_callable_docs():
    """Live exports must match static order and document functions/classes."""
    import openmed

    assert openmed.__all__, "openmed.__all__ must define the public API"
    assert len(openmed.__all__) == len(set(openmed.__all__))

    report = checker.measure_public_api_docstrings()
    assert [symbol.name for symbol in report.symbols] == openmed.__all__

    for symbol in report.scored:
        exported = getattr(openmed, symbol.name)
        docstring = inspect.getdoc(exported)
        assert docstring, f"openmed.{symbol.name} is missing a docstring"
        assert len(docstring.strip()) >= checker.MIN_DOCSTRING_CHARS


def test_data_exports_match_explicit_inventory():
    """Data exports must resolve by exact name, module, and runtime type."""
    import openmed

    report = checker.measure_public_api_docstrings()
    by_name = {symbol.name: symbol for symbol in report.data}

    assert set(by_name) == set(EXPECTED_DATA_EXPORTS)
    for name, (expected_module, expected_type) in EXPECTED_DATA_EXPORTS.items():
        symbol = by_name[name]
        assert symbol.module == expected_module
        assert type(getattr(openmed, name)) is expected_type
        assert symbol.reason == "data export (inventory only)"


def test_cli_gate_passes_at_floor(capsys: pytest.CaptureFixture[str]):
    """The CLI passes and prints the exact data-export inventory."""
    exit_code = checker.main(["--min-coverage", str(MIN_COVERAGE)])
    assert exit_code == 0
    output = capsys.readouterr().out
    for name in EXPECTED_DATA_EXPORTS:
        assert f"  - {name} (" in output
