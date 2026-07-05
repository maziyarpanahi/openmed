"""Static docstring-coverage checker for OpenMed's public API surface.

Motivation
----------
``pre-commit`` runs pydocstyle for docstring *style*, but nothing enforces
docstring *coverage* on the public API. Because ``mkdocstrings`` autogenerates
the API reference from docstrings, an undocumented public export (for example a
new name added to ``openmed.__all__``) would silently ship an empty reference
entry. This checker closes that gap with a focused, stdlib-only pass over the
exported names.

Design
------
The checker is deliberately **stdlib-only** (``ast`` plus path arithmetic) so it
runs in CI without importing the runtime package or pulling in optional heavy
dependencies (``transformers``, backend runtimes, and so on). It:

1. Parses ``openmed/__init__.py`` to read ``__all__`` and the ``from .module
   import name`` bindings that back each exported name.
2. Resolves each binding to its defining source file purely from the filesystem
   layout of the ``openmed`` package (no imports are executed).
3. Parses that source file with ``ast`` and inspects the ``class``/``def`` node
   for a non-empty docstring via :func:`ast.get_docstring`.

Only *docstringable* symbols (functions and classes) are scored. Module-level
data constants (dicts, sets, strings, and similar) cannot carry a meaningful
docstring and are reported separately rather than counted against coverage.

Usage
-----
Run as a CLI to print a coverage report::

    python scripts/check_public_api_docstrings.py --min-coverage 100

Or import :func:`measure_public_api_docstrings` from a test to gate coverage.
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "openmed"
PACKAGE_INIT = PACKAGE_ROOT / "__init__.py"

# The default floor is the current measured coverage. It is intentionally set so
# the gate passes today and can be ratcheted upward as coverage improves, rather
# than breaking the build on day one.
DEFAULT_MIN_COVERAGE = 100.0

# Minimum docstring length (stripped) to count as documented. Guards against
# placeholder one-word docstrings sneaking past the check.
MIN_DOCSTRING_CHARS = 10


@dataclass(frozen=True)
class SymbolReport:
    """Coverage outcome for a single exported public symbol.

    Attributes:
        name: The exported name as it appears in ``openmed.__all__``.
        module: Dotted module the name is imported from, or ``None`` when it is
            defined directly in ``openmed/__init__.py``.
        kind: One of ``"function"``, ``"class"``, or ``"data"``.
        documented: Whether the symbol carries a sufficiently long docstring.
            Always ``True`` for ``"data"`` symbols, which are not scored.
        reason: Human-readable explanation when ``documented`` is ``False`` or
            when the symbol could not be resolved.
    """

    name: str
    module: Optional[str]
    kind: str
    documented: bool
    reason: str = ""


@dataclass
class CoverageReport:
    """Aggregate docstring-coverage result over the public API surface.

    Attributes:
        symbols: Per-symbol outcomes in export order.
        min_coverage: The coverage floor the run was gated against.
    """

    symbols: List[SymbolReport] = field(default_factory=list)
    min_coverage: float = DEFAULT_MIN_COVERAGE

    @property
    def scored(self) -> List[SymbolReport]:
        """Return only symbols that count toward coverage (functions/classes)."""
        return [s for s in self.symbols if s.kind != "data"]

    @property
    def documented(self) -> List[SymbolReport]:
        """Return scored symbols that carry a sufficient docstring."""
        return [s for s in self.scored if s.documented]

    @property
    def missing(self) -> List[SymbolReport]:
        """Return scored symbols that are missing a sufficient docstring."""
        return [s for s in self.scored if not s.documented]

    @property
    def coverage(self) -> float:
        """Return docstring coverage as a percentage of scored symbols."""
        if not self.scored:
            return 100.0
        return 100.0 * len(self.documented) / len(self.scored)

    @property
    def passed(self) -> bool:
        """Return whether measured coverage meets the configured floor."""
        return self.coverage >= self.min_coverage


def _module_to_source(module: str) -> Optional[Path]:
    """Map a dotted ``openmed`` submodule to its source file on disk.

    Returns ``None`` when no matching ``.py`` file exists. Resolution is purely
    path-based so no package code is imported.
    """
    if module == "openmed":
        return PACKAGE_INIT
    if not module.startswith("openmed."):
        return None
    relative = module[len("openmed.") :].replace(".", "/")
    candidate = PACKAGE_ROOT / f"{relative}.py"
    if candidate.exists():
        return candidate
    package_init = PACKAGE_ROOT / relative / "__init__.py"
    if package_init.exists():
        return package_init
    return None


def _module_package(module: str) -> str:
    """Return the package a dotted module lives in (its parent)."""
    return module.rsplit(".", 1)[0] if "." in module else module


def _resolve_relative_module(
    module: Optional[str], level: int, current_package: str
) -> str:
    """Resolve a relative ``from . import`` target to an absolute module.

    Args:
        module: The module portion of the import (may be ``None`` for bare
            relative imports such as ``from . import x``).
        level: Number of leading dots in the relative import.
        current_package: The package the importing module belongs to.

    Returns:
        The absolute dotted module name rooted at ``openmed``.
    """
    if level == 0:
        # Absolute import; already fully qualified.
        return module or ""
    # A single dot resolves within ``current_package``; each extra dot ascends
    # one package level.
    base = current_package
    for _ in range(level - 1):
        base = _module_package(base)
    if module:
        return f"{base}.{module}"
    return base


def _collect_all_names(tree: ast.Module) -> List[str]:
    """Extract the string entries of ``__all__`` from a parsed module."""
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        targets = [t for t in node.targets if isinstance(t, ast.Name)]
        if not any(t.id == "__all__" for t in targets):
            continue
        if isinstance(node.value, (ast.List, ast.Tuple, ast.Set)):
            names: List[str] = []
            for element in node.value.elts:
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    names.append(element.value)
            return names
    return []


def _collect_import_bindings(tree: ast.Module, current_package: str) -> Dict[str, str]:
    """Map each imported name to the dotted module it is imported from.

    Only ``from ... import name`` statements are considered, which matches how
    ``openmed/__init__.py`` and the package ``__init__`` files re-export the
    public surface. ``current_package`` anchors relative imports.
    """
    bindings: Dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        module = _resolve_relative_module(node.module, node.level, current_package)
        if not module:
            continue
        for alias in node.names:
            bound = alias.asname or alias.name
            bindings[bound] = module
    return bindings


def _index_definitions(tree: ast.Module) -> Dict[str, ast.AST]:
    """Index top-level ``def``/``class`` definitions by name in a module."""
    definitions: Dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions[node.name] = node
    return definitions


def _assigned_names(tree: ast.Module) -> Set[str]:
    """Return names bound by top-level assignments (module data constants)."""
    names: Set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


_SOURCE_CACHE: Dict[Path, ast.Module] = {}


def _parse_source(path: Path) -> ast.Module:
    """Parse and cache the AST for a source file."""
    cached = _SOURCE_CACHE.get(path)
    if cached is None:
        cached = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        _SOURCE_CACHE[path] = cached
    return cached


def _has_docstring(node: ast.AST) -> bool:
    """Return whether an AST node has a sufficiently long docstring."""
    if not isinstance(
        node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
    ):
        return False
    doc = ast.get_docstring(node)
    return bool(doc) and len(doc.strip()) >= MIN_DOCSTRING_CHARS


def _report_for_node(name: str, module: Optional[str], node: ast.AST) -> SymbolReport:
    """Build a :class:`SymbolReport` for a resolved definition node."""
    kind = "class" if isinstance(node, ast.ClassDef) else "function"
    documented = _has_docstring(node)
    reason = "" if documented else "missing or too-short docstring"
    return SymbolReport(name, module, kind, documented, reason)


# Cap on re-export hops to follow before giving up (guards against cycles).
_MAX_REEXPORT_HOPS = 12


def _resolve_symbol(name: str, module: str) -> SymbolReport:
    """Follow re-export chains to find where ``name`` is actually defined.

    Package ``__init__`` files frequently re-export names from deeper submodules
    (sometimes several hops away). This walks ``from .x import name`` chains
    until it reaches the real ``def``/``class`` or a data assignment.
    """
    visited: Set[str] = set()
    current_module = module
    for _ in range(_MAX_REEXPORT_HOPS):
        if current_module in visited:
            break
        visited.add(current_module)

        source = _module_to_source(current_module)
        if source is None:
            return SymbolReport(
                name,
                current_module,
                "class",
                False,
                f"source for {current_module} not found",
            )

        tree = _parse_source(source)
        definitions = _index_definitions(tree)
        if name in definitions:
            return _report_for_node(name, current_module, definitions[name])

        if name in _assigned_names(tree):
            # Module-level data constant (dict/set/str/...); not docstringable.
            return SymbolReport(name, current_module, "data", True)

        # A package ``__init__.py`` belongs to the package itself; a plain
        # module belongs to its parent package. Relative imports anchor here.
        if source.name == "__init__.py":
            current_package = current_module
        else:
            current_package = _module_package(current_module)
        bindings = _collect_import_bindings(tree, current_package)
        next_module = bindings.get(name)
        if next_module is None or next_module == current_module:
            break
        current_module = next_module

    return SymbolReport(
        name, module, "class", False, f"definition for {name} not found in {module}"
    )


def _evaluate_symbol(
    name: str, bindings: Dict[str, str], init_tree: ast.Module
) -> SymbolReport:
    """Resolve one exported name and report its docstring status."""
    if name == "__version__":
        # Dunder version string; not part of the documented API surface.
        return SymbolReport(name, "openmed", "data", True)

    init_defs = _index_definitions(init_tree)
    # Names defined directly in openmed/__init__.py (e.g. analyze_text).
    if name in init_defs:
        return _report_for_node(name, "openmed", init_defs[name])

    module = bindings.get(name)
    if module is None:
        return SymbolReport(
            name, None, "class", False, "not imported in openmed/__init__.py"
        )

    return _resolve_symbol(name, module)


def measure_public_api_docstrings(
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> CoverageReport:
    """Measure docstring coverage over the public API surface.

    Parses ``openmed/__init__.py`` and the modules backing every exported name,
    then reports docstring coverage for the docstringable symbols (functions and
    classes). Data constants are listed but not scored.

    Args:
        min_coverage: The coverage floor (percentage) to gate against.

    Returns:
        A :class:`CoverageReport` describing per-symbol status and the aggregate
        coverage percentage.
    """
    init_tree = _parse_source(PACKAGE_INIT)
    names = _collect_all_names(init_tree)
    bindings = _collect_import_bindings(init_tree, "openmed")

    report = CoverageReport(min_coverage=min_coverage)
    for name in names:
        report.symbols.append(_evaluate_symbol(name, bindings, init_tree))
    return report


def format_report(report: CoverageReport) -> str:
    """Render a human-readable summary of a coverage report."""
    lines: List[str] = []
    lines.append("OpenMed public API docstring coverage")
    lines.append("=" * 40)
    lines.append(f"Scored symbols (functions/classes): {len(report.scored)}")
    lines.append(f"Documented: {len(report.documented)}")
    lines.append(f"Missing:    {len(report.missing)}")
    data_symbols = [s for s in report.symbols if s.kind == "data"]
    lines.append(f"Data constants (not scored): {len(data_symbols)}")
    lines.append(f"Coverage: {report.coverage:.1f}% (floor {report.min_coverage:.1f}%)")
    if report.missing:
        lines.append("")
        lines.append("Undocumented public exports:")
        for symbol in report.missing:
            location = symbol.module or "openmed/__init__.py"
            lines.append(f"  - {symbol.name} ({location}): {symbol.reason}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: print the report and gate on the coverage floor."""
    parser = argparse.ArgumentParser(
        description="Check docstring coverage of the OpenMed public API surface."
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=DEFAULT_MIN_COVERAGE,
        help=(
            "Minimum acceptable coverage percentage. Defaults to the current "
            f"floor ({DEFAULT_MIN_COVERAGE:.0f}%%)."
        ),
    )
    args = parser.parse_args(argv)

    report = measure_public_api_docstrings(min_coverage=args.min_coverage)
    print(format_report(report))
    if not report.passed:
        print(
            f"\nFAIL: coverage {report.coverage:.1f}% is below the "
            f"{args.min_coverage:.1f}% floor.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
