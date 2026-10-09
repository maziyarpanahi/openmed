#!/usr/bin/env python3
"""Import-free effect inventory and built-in surface dependency guard.

Output contains controlled classifications, Python module/function names and
fixed failure codes only. It never includes endpoints, SQL, payloads or paths.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib.util import resolve_name
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "scripts" / "security" / "effect_paths.json"
BUILTIN_SURFACES = (
    "openmed.mcp.server",
    "openmed.mcp.tool_registry",
    "openmed.service.app",
    "openmed.cli.main",
    "openmed.service.journey_workflows",
    "openmed.agent.workflows",
)
LEGACY_MODULES = frozenset({"openmed.interop.fhir_server", "openmed.interop.openmrs"})
LEGACY_METHODS = frozenset(
    {
        "put_resource",
        "fetch_and_deidentify",
        "write_back",
        "write_rest_resource",
        "write_fhir_resource",
    }
)
LEGACY_CLASSES = frozenset({"FHIRServerClient", "OpenMRSClient", "OpenMRSAdapter"})
CLASSIFICATIONS = frozenset({"governed", "legacy_explicit_opt_in", "operational"})
_HTTP_WRITES = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_SYMBOL = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\Z", re.ASCII)
_ERROR_CODES = frozenset(
    {
        "effect_symbol_invalid",
        "effect_source_invalid",
        "effect_manifest_invalid",
        "effect_path_unclassified",
        "effect_inventory_stale",
        "effect_classification_invalid",
        "legacy_classification_invalid",
        "effect_surface_unverified",
        "legacy_writer_reachable",
        "effect_inventory_failed",
    }
)


class EffectPathError(ValueError):
    """Fixed-code refusal; source text and paths are never attached."""

    def __init__(self, code: str):
        super().__init__(code if code in _ERROR_CODES else "effect_inventory_failed")


@dataclass(frozen=True, order=True)
class EffectPath:
    """One statically detected candidate effect owned by a Python function."""

    module: str
    function: str

    def __post_init__(self) -> None:
        if not _SYMBOL.fullmatch(self.module) or not _SYMBOL.fullmatch(self.function):
            raise EffectPathError("effect_symbol_invalid")


def read_sources(root: Path = ROOT) -> dict[str, ast.Module]:
    """Parse package sources without importing or executing application code."""
    sources = {}
    failed = False
    try:
        for path in sorted((root / "openmed").rglob("*.py")):
            parts = list(path.relative_to(root).with_suffix("").parts)
            is_package = parts[-1] == "__init__"
            if is_package:
                parts.pop()
            module = ".".join(parts)
            if not _SYMBOL.fullmatch(module):
                raise ValueError
            parsed = ast.parse(path.read_text(encoding="utf-8"))
            setattr(parsed, "_effect_is_package", is_package)
            sources[module] = parsed
    except Exception:
        failed = True
    if failed or not sources:
        raise EffectPathError("effect_source_invalid")
    return sources


def _prefix(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        words = node.value.lstrip().split()
        return words[0].upper() if words else ""
    if isinstance(node, ast.JoinedStr) and node.values:
        return _prefix(node.values[0])
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _prefix(node.left)
    return None


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    if (
        isinstance(node, ast.Call)
        and _call_name(node.func) == "getattr"
        and len(node.args) >= 2
    ):
        value = node.args[1]
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
    return ""


def _is_effect(node: ast.Call) -> bool:
    name = _call_name(node.func)
    first = node.args[0] if node.args else None
    if name in {
        "post",
        "put",
        "patch",
        "delete",
        "executemany",
        "executescript",
        "commit",
        "put_resource",
        "fetch_and_deidentify",
        "write_back",
        "write_rest_resource",
        "write_fhir_resource",
        "write_fhir_omop_sqlite",
        "write_omop_sqlite",
    }:
        return True  # Conservative: local names may also denote operational effects.
    if name in {"request", "_request", "execute", "_execute"}:
        method = _prefix(first)
        if method is None:
            return True  # Dynamic methods/SQL require classification and review.
        # Only literal read prefixes are excluded. CTEs, commented SQL and
        # other unfamiliar prefixes may still execute a mutation.
        if name in {"request", "_request"}:
            return method not in {"GET", "HEAD", "OPTIONS"}
        return method != "SELECT"
    if name in {"Request", "urlopen", "send"}:
        if name == "Request":
            keywords = {x.arg: x.value for x in node.keywords}
            method = _prefix(keywords.get("method"))
            return method in _HTTP_WRITES or (
                method is None
                and ("method" in keywords or "data" in keywords or len(node.args) > 1)
            )
        return True  # A prebuilt request may carry any HTTP method.
    return False


class _Effects(ast.NodeVisitor):
    def __init__(self, module: str):
        self.module = module
        self.stack: list[str] = []
        self.paths: set[EffectPath] = set()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.stack.append(node.name)
        for statement in node.body:
            self.visit(statement)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.stack.append(node.name)
        # Route decorators declare inbound handlers, rather than HTTP writes.
        # Bodies, default expressions and annotations are still scanned.
        self.visit(node.args)
        for statement in node.body:
            self.visit(statement)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        if _is_effect(node):
            self.paths.add(EffectPath(self.module, ".".join(self.stack) or "_module"))
        self.generic_visit(node)


def scan_effect_paths(sources: Mapping[str, ast.Module]) -> tuple[EffectPath, ...]:
    """Return deterministic conservative HTTP/database effect candidates."""
    paths = set()
    for module, source in sorted(sources.items()):
        visitor = _Effects(module)
        visitor.visit(source)
        paths.update(visitor.paths)
    return tuple(sorted(paths))


def load_classifications(path: Path = MANIFEST) -> dict[EffectPath, str]:
    """Read the explicit reviewed inventory; never classify a new path implicitly."""
    result = {}
    failed = False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or set(payload) != {"schema_version", "paths"}
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1
        ):
            raise ValueError
        for row in payload["paths"]:
            if (
                not isinstance(row, dict)
                or set(row) != {"module", "function", "classification"}
                or row["classification"] not in CLASSIFICATIONS
            ):
                raise ValueError
            key = EffectPath(row["module"], row["function"])
            if key in result:
                raise ValueError
            result[key] = row["classification"]
    except Exception:
        failed = True
    if failed:
        raise EffectPathError("effect_manifest_invalid")
    return result


def validate_inventory(
    paths: Iterable[EffectPath], classifications: Mapping[EffectPath, str]
) -> None:
    """Refuse new or removed effect paths until the reviewed manifest is updated."""
    observed = set(paths)
    if observed - classifications.keys():
        raise EffectPathError("effect_path_unclassified")
    if classifications.keys() - observed:
        raise EffectPathError("effect_inventory_stale")
    if any(value not in CLASSIFICATIONS for value in classifications.values()):
        raise EffectPathError("effect_classification_invalid")
    if any(
        path.module in LEGACY_MODULES
        and classifications[path] != "legacy_explicit_opt_in"
        for path in observed
    ):
        raise EffectPathError("legacy_classification_invalid")


def _imports(module: str, source: ast.Module, modules: set[str]) -> set[str]:
    result = set()
    # Resolve relatives against the longest existing package parent. Package
    # modules have children, while file modules use their containing package.
    is_package = getattr(
        source,
        "_effect_is_package",
        any(name.startswith(module + ".") for name in modules),
    )
    package = module if is_package else module.rpartition(".")[0]
    dynamic_loaders = {"import_module", "__import__"}
    for node in ast.walk(source):
        if isinstance(node, ast.ImportFrom) and node.module == "importlib":
            dynamic_loaders.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "import_module"
            )
    for node in ast.walk(source):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parent = package.split(".")[: len(package.split(".")) - node.level + 1]
                base = ".".join(
                    (*parent, *((node.module or "").split(".") if node.module else ()))
                )
            else:
                base = node.module or ""
            result.add(base)
            result.update(
                base + "." + alias.name for alias in node.names if alias.name != "*"
            )
        elif (
            isinstance(node, ast.Call)
            and _call_name(node.func) in dynamic_loaders
            and node.args
        ):
            value = node.args[0]
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                target = value.value
                if target.startswith("."):
                    package_argument = (
                        node.args[1]
                        if len(node.args) > 1
                        else next(
                            (
                                item.value
                                for item in node.keywords
                                if item.arg == "package"
                            ),
                            None,
                        )
                    )
                    if isinstance(package_argument, ast.Constant) and isinstance(
                        package_argument.value, str
                    ):
                        target = resolve_name(target, package_argument.value)
                    elif (
                        isinstance(package_argument, ast.Name)
                        and package_argument.id == "__package__"
                    ):
                        target = resolve_name(target, package)
                result.add(target)
        elif (
            isinstance(node, ast.Call)
            and _call_name(node.func) in {"get_adapter", "get_interop_adapter"}
            and node.args
        ):
            value = node.args[0]
            if isinstance(value, ast.Constant) and value.value in {
                "fhir_server",
                "openmrs",
            }:
                result.add("openmed.interop." + value.value)
    # Importing a child also executes each available package initializer.
    for name in tuple(result):
        parts = name.split(".")
        result.update(
            parent
            for length in range(1, len(parts))
            if (parent := ".".join(parts[:length])) in modules
        )
    return result


def assert_no_legacy_reachability(
    sources: Mapping[str, ast.Module], roots: Iterable[str] = BUILTIN_SURFACES
) -> None:
    """Check all static package dependencies of constructed built-in surfaces.

    Conservative module dependency reachability covers imports inside functions,
    aliases, relative imports and constant dynamic import/adapter requests. It
    does not establish isolation from arbitrary runtime or native plugin code.
    """
    pending = list(roots)
    visited = set()
    modules = set(sources)
    if any(not _SYMBOL.fullmatch(name) or name not in modules for name in pending):
        raise EffectPathError("effect_surface_unverified")
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        visited.add(module)
        for node in ast.walk(sources[module]):
            if (
                isinstance(node, ast.Attribute)
                and node.attr in LEGACY_METHODS | LEGACY_CLASSES
            ):
                raise EffectPathError("legacy_writer_reachable")
            if isinstance(node, ast.Name) and node.id in LEGACY_CLASSES:
                raise EffectPathError("legacy_writer_reachable")
            if (
                isinstance(node, ast.Call)
                and _call_name(node.func) == "getattr"
                and len(node.args) >= 2
            ):
                value = node.args[1]
                if (
                    isinstance(value, ast.Constant)
                    and value.value in LEGACY_METHODS | LEGACY_CLASSES
                ):
                    raise EffectPathError("legacy_writer_reachable")
        imports = _imports(module, sources[module], modules)
        if any(
            name == legacy or name.startswith(legacy + ".")
            for name in imports
            for legacy in LEGACY_MODULES
        ):
            raise EffectPathError("legacy_writer_reachable")
        pending.extend(
            name for name in imports if name in modules and name not in visited
        )


def inventory_document(
    paths: Iterable[EffectPath], classifications: Mapping[EffectPath, str]
) -> dict[str, list[dict[str, str]]]:
    """Project only controlled classifications and module/function names."""
    validate_inventory(paths, classifications)
    result: dict[str, list[dict[str, str]]] = {
        key: [] for key in sorted(CLASSIFICATIONS)
    }
    for path in sorted(paths):
        result[classifications[path]].append(
            {"module": path.module, "function": path.function}
        )
    return result


def main(argv: list[str] | None = None) -> int:
    """Run the lint or print the value-free reviewed inventory."""
    arguments = sys.argv[1:] if argv is None else argv
    if arguments == ["--help"]:
        print("usage: check_effect_paths.py [--check]")
        return 0
    if arguments not in ([], ["--check"]):
        print("effect_arguments_invalid", file=sys.stderr)
        return 2
    failure = None
    try:
        sources = read_sources()
        paths = scan_effect_paths(sources)
        classifications = load_classifications()
        validate_inventory(paths, classifications)
        assert_no_legacy_reachability(sources)
        document = inventory_document(paths, classifications)
    except EffectPathError as exc:
        failure = str(exc)
    except Exception:
        failure = "effect_inventory_failed"
    if failure is not None:
        print(failure)
        return 1
    print(
        "effect_paths_verified" if arguments else json.dumps(document, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
