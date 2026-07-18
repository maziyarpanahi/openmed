#!/usr/bin/env python3
"""Compare the public Python API exposed by two Git references.

The extractor reads source files from Git (or the current worktree) and parses
them with :mod:`ast`.  It deliberately never imports the package being
inspected, which keeps release checks offline-safe and free from import-time
side effects.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence

SCHEMA_VERSION = 1
WORKTREE_REF = "WORKTREE"


class ApiSurfaceError(RuntimeError):
    """Raised when a source tree cannot be read or parsed."""


@dataclass(frozen=True)
class Parameter:
    """One callable parameter relevant to compatibility checks."""

    name: str
    kind: str
    required: bool
    annotation: str | None = None
    default: str | None = None


@dataclass(frozen=True)
class Symbol:
    """Static description of one public symbol."""

    name: str
    module: str
    qualname: str
    kind: str
    signature: str | None = None
    parameters: tuple[Parameter, ...] | None = None
    deprecated: bool = False
    fingerprint: str = ""
    source_target: str | None = None

    def public_dict(self) -> dict[str, object]:
        """Return the stable JSON representation without matching internals."""

        payload = asdict(self)
        payload.pop("fingerprint")
        return payload


@dataclass(frozen=True)
class Change:
    """One added, deprecated, or breaking API change."""

    change: str
    symbol: str
    before: Symbol | None = None
    after: Symbol | None = None
    replacement: str | None = None
    details: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable change record."""

        payload: dict[str, object] = {
            "change": self.change,
            "symbol": self.symbol,
        }
        if self.before is not None:
            payload["before"] = self.before.public_dict()
        if self.after is not None:
            payload["after"] = self.after.public_dict()
        if self.replacement is not None:
            payload["replacement"] = self.replacement
        if self.details:
            payload["details"] = list(self.details)
        return payload


@dataclass(frozen=True)
class ApiSurfaceDiff:
    """Complete public API comparison between two source trees."""

    before_ref: str
    after_ref: str
    package: str
    before_symbol_count: int
    after_symbol_count: int
    added: tuple[Change, ...]
    deprecated: tuple[Change, ...]
    breaking: tuple[Change, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the stable machine-readable diff schema."""

        return {
            "schema_version": SCHEMA_VERSION,
            "before_ref": self.before_ref,
            "after_ref": self.after_ref,
            "package": self.package,
            "summary": {
                "before_symbols": self.before_symbol_count,
                "after_symbols": self.after_symbol_count,
                "added": len(self.added),
                "deprecated": len(self.deprecated),
                "breaking": len(self.breaking),
            },
            "added": [change.to_dict() for change in self.added],
            "deprecated": [change.to_dict() for change in self.deprecated],
            "breaking": [change.to_dict() for change in self.breaking],
        }


def _run_git(repo_root: Path, args: Sequence[str]) -> bytes:
    command = ["git", "-C", str(repo_root), *args]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise ApiSurfaceError(f"{' '.join(command)} failed: {stderr}")
    return result.stdout


def _sources_from_ref(
    repo_root: Path,
    ref: str,
    package: str,
) -> dict[PurePosixPath, str]:
    if ref == WORKTREE_REF:
        package_root = repo_root / package
        if not package_root.is_dir():
            raise ApiSurfaceError(f"package directory not found: {package_root}")
        return {
            PurePosixPath(path.relative_to(repo_root).as_posix()): path.read_text(
                encoding="utf-8"
            )
            for path in sorted(package_root.rglob("*.py"))
            if "__pycache__" not in path.parts
        }

    archive = _run_git(repo_root, ["archive", "--format=tar", ref, package])
    sources: dict[PurePosixPath, str] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
            for member in bundle.getmembers():
                path = PurePosixPath(member.name)
                if not member.isfile() or path.suffix != ".py":
                    continue
                fileobj = bundle.extractfile(member)
                if fileobj is None:
                    continue
                sources[path] = fileobj.read().decode("utf-8")
    except tarfile.TarError as exc:
        raise ApiSurfaceError(
            f"could not read source archive for {ref}: {exc}"
        ) from exc
    return sources


def _module_name(path: PurePosixPath) -> str:
    parts = list(path.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _is_public_module(module: str, package: str) -> bool:
    relative_parts = module.split(".")[len(package.split(".")) :]
    return all(not part.startswith("_") for part in relative_parts)


def _string_sequence(
    expression: ast.expr,
    values: Mapping[str, tuple[str, ...]],
) -> tuple[str, ...] | None:
    if isinstance(expression, (ast.List, ast.Tuple, ast.Set)):
        result: list[str] = []
        for item in expression.elts:
            if isinstance(item, ast.Starred):
                nested = _string_sequence(item.value, values)
                if nested is None:
                    return None
                result.extend(nested)
            elif isinstance(item, ast.Constant) and isinstance(item.value, str):
                result.append(item.value)
            else:
                return None
        return tuple(result)
    if isinstance(expression, ast.Name):
        return values.get(expression.id)
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add):
        left = _string_sequence(expression.left, values)
        right = _string_sequence(expression.right, values)
        if left is None or right is None:
            return None
        return (*left, *right)
    return None


def _static_all(tree: ast.Module) -> tuple[str, ...] | None:
    values: dict[str, tuple[str, ...]] = {}
    exports: tuple[str, ...] | None = None
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if not isinstance(target, ast.Name):
                continue
            value = _string_sequence(statement.value, values)
            if value is not None:
                values[target.id] = value
                if target.id == "__all__":
                    exports = value
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            if statement.value is None:
                continue
            value = _string_sequence(statement.value, values)
            if value is not None:
                values[statement.target.id] = value
                if statement.target.id == "__all__":
                    exports = value
        elif (
            isinstance(statement, ast.AugAssign)
            and isinstance(statement.target, ast.Name)
            and isinstance(statement.op, ast.Add)
        ):
            value = _string_sequence(statement.value, values)
            current = values.get(statement.target.id)
            if current is not None and value is not None:
                combined = (*current, *value)
                values[statement.target.id] = combined
                if statement.target.id == "__all__":
                    exports = combined
        elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            call = statement.value
            if not (
                isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "__all__"
                and len(call.args) == 1
            ):
                continue
            current = values.get("__all__")
            if current is None:
                continue
            added = _string_sequence(call.args[0], values)
            if call.func.attr == "append" and isinstance(call.args[0], ast.Constant):
                value = call.args[0].value
                added = (value,) if isinstance(value, str) else None
            if call.func.attr in {"append", "extend"} and added is not None:
                exports = (*current, *added)
                values["__all__"] = exports
    return exports


def _render_expression(expression: ast.AST | None) -> str | None:
    if expression is None:
        return None
    return ast.unparse(expression).strip()


def _parameters(
    arguments: ast.arguments, *, drop_first: bool = False
) -> tuple[Parameter, ...]:
    positional = [*arguments.posonlyargs, *arguments.args]
    positional_defaults = [None] * (len(positional) - len(arguments.defaults)) + list(
        arguments.defaults
    )
    result: list[Parameter] = []
    for index, (argument, default) in enumerate(zip(positional, positional_defaults)):
        if drop_first and index == 0 and argument.arg in {"self", "cls"}:
            continue
        kind = (
            "positional_only"
            if index < len(arguments.posonlyargs)
            else "positional_or_keyword"
        )
        result.append(
            Parameter(
                name=argument.arg,
                kind=kind,
                required=default is None,
                annotation=_render_expression(argument.annotation),
                default=_render_expression(default),
            )
        )
    if arguments.vararg is not None:
        result.append(
            Parameter(
                name=arguments.vararg.arg,
                kind="var_positional",
                required=False,
                annotation=_render_expression(arguments.vararg.annotation),
            )
        )
    for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults):
        result.append(
            Parameter(
                name=argument.arg,
                kind="keyword_only",
                required=default is None,
                annotation=_render_expression(argument.annotation),
                default=_render_expression(default),
            )
        )
    if arguments.kwarg is not None:
        result.append(
            Parameter(
                name=arguments.kwarg.arg,
                kind="var_keyword",
                required=False,
                annotation=_render_expression(arguments.kwarg.annotation),
            )
        )
    return tuple(result)


def _render_signature(
    parameters: Sequence[Parameter],
    returns: ast.expr | None,
) -> str:
    rendered: list[str] = []
    positional_only_count = sum(
        parameter.kind == "positional_only" for parameter in parameters
    )
    saw_var_positional = False
    inserted_keyword_separator = False
    for index, parameter in enumerate(parameters):
        value = parameter.name
        if parameter.annotation:
            value += f": {parameter.annotation}"
        if parameter.default is not None:
            value += f" = {parameter.default}"
        if parameter.kind == "var_positional":
            value = "*" + value
            saw_var_positional = True
        elif parameter.kind == "var_keyword":
            value = "**" + value
        elif (
            parameter.kind == "keyword_only"
            and not saw_var_positional
            and not inserted_keyword_separator
        ):
            rendered.append("*")
            inserted_keyword_separator = True
        rendered.append(value)
        if positional_only_count and index + 1 == positional_only_count:
            rendered.append("/")
    signature = f"({', '.join(rendered)})"
    rendered_return = _render_expression(returns)
    if rendered_return:
        signature += f" -> {rendered_return}"
    return signature


def _is_deprecated(decorators: Iterable[ast.expr]) -> bool:
    for decorator in decorators:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Name) and target.id == "deprecated":
            return True
        if isinstance(target, ast.Attribute) and target.attr == "deprecated":
            return True
    return False


def _fingerprint(node: ast.AST, kind: str) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        body = list(node.body)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]
        material = ast.dump(
            ast.Module(body=body, type_ignores=[]),
            annotate_fields=True,
            include_attributes=False,
        )
    elif isinstance(node, ast.Assign):
        material = ast.dump(node.value, annotate_fields=True, include_attributes=False)
    elif isinstance(node, ast.AnnAssign):
        material = "annotation=" + ast.dump(
            node.annotation, annotate_fields=True, include_attributes=False
        )
        if node.value is not None:
            material += ":value=" + ast.dump(
                node.value, annotate_fields=True, include_attributes=False
            )
    else:
        material = ast.dump(node, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(f"{kind}:{material}".encode()).hexdigest()


def _function_symbol(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    module: str,
    qualname: str,
    *,
    drop_first: bool = False,
) -> Symbol:
    parameters = _parameters(node.args, drop_first=drop_first)
    kind = "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"
    return Symbol(
        name=f"{module}.{qualname}",
        module=module,
        qualname=qualname,
        kind=kind,
        signature=_render_signature(parameters, node.returns),
        parameters=parameters,
        deprecated=_is_deprecated(node.decorator_list),
        fingerprint=_fingerprint(node, kind),
    )


def _assigned_names(target: ast.expr) -> tuple[str, ...]:
    if isinstance(target, ast.Name):
        return (target.id,)
    if isinstance(target, (ast.Tuple, ast.List)):
        return tuple(name for item in target.elts for name in _assigned_names(item))
    return ()


def _self_attributes(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    attributes: set[str] = set()
    for child in ast.walk(node):
        target: ast.expr | None = None
        if isinstance(child, ast.Assign):
            for candidate in child.targets:
                if isinstance(candidate, ast.Attribute):
                    target = candidate
                    if (
                        isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                        and not target.attr.startswith("_")
                    ):
                        attributes.add(target.attr)
        elif isinstance(child, ast.AnnAssign):
            target = child.target
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and not target.attr.startswith("_")
            ):
                attributes.add(target.attr)
    return attributes


def _class_symbols(node: ast.ClassDef, module: str) -> dict[str, Symbol]:
    constructor = next(
        (
            statement
            for statement in node.body
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
            and statement.name == "__init__"
        ),
        None,
    )
    parameters: tuple[Parameter, ...] | None = None
    signature: str | None = None
    if constructor is not None:
        parameters = _parameters(constructor.args, drop_first=True)
        signature = _render_signature(parameters, constructor.returns)
    class_name = f"{module}.{node.name}"
    symbols = {
        class_name: Symbol(
            name=class_name,
            module=module,
            qualname=node.name,
            kind="class",
            signature=signature,
            parameters=parameters,
            deprecated=_is_deprecated(node.decorator_list),
            fingerprint=_fingerprint(node, "class"),
        )
    }
    instance_attributes: set[str] = set()
    for statement in node.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            instance_attributes.update(_self_attributes(statement))
            if statement.name.startswith("_"):
                continue
            method = _function_symbol(
                statement,
                module,
                f"{node.name}.{statement.name}",
                drop_first=True,
            )
            method_kind = "method" if method.kind == "function" else "async_method"
            symbols[method.name] = replace(method, kind=method_kind)
        elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target]
            )
            for target in targets:
                for name in _assigned_names(target):
                    if name.startswith("_"):
                        continue
                    full_name = f"{class_name}.{name}"
                    symbols[full_name] = Symbol(
                        name=full_name,
                        module=module,
                        qualname=f"{node.name}.{name}",
                        kind="attribute",
                        fingerprint=_fingerprint(statement, "attribute"),
                    )
    for name in sorted(instance_attributes):
        full_name = f"{class_name}.{name}"
        symbols.setdefault(
            full_name,
            Symbol(
                name=full_name,
                module=module,
                qualname=f"{node.name}.{name}",
                kind="attribute",
                fingerprint=hashlib.sha256(
                    f"instance-attribute:{name}".encode()
                ).hexdigest(),
            ),
        )
    return symbols


def _absolute_import(module: str, is_package: bool, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    base = module.split(".") if is_package else module.split(".")[:-1]
    remove = max(0, node.level - 1)
    if remove:
        base = base[:-remove]
    if node.module:
        base.extend(node.module.split("."))
    return ".".join(base)


def _module_symbols(
    path: PurePosixPath, source: str, package: str
) -> dict[str, Symbol]:
    module = _module_name(path)
    if not _is_public_module(module, package):
        return {}
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise ApiSurfaceError(f"could not parse {path}: {exc}") from exc
    exports = _static_all(tree)
    local: dict[str, Symbol] = {}
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            local[statement.name] = _function_symbol(statement, module, statement.name)
        elif isinstance(statement, ast.ClassDef):
            class_symbols = _class_symbols(statement, module)
            local[statement.name] = class_symbols[f"{module}.{statement.name}"]
            for full_name, symbol in class_symbols.items():
                if full_name != f"{module}.{statement.name}":
                    local[symbol.qualname] = symbol
        elif isinstance(statement, ast.ImportFrom):
            target_module = _absolute_import(
                module, path.name == "__init__.py", statement
            )
            for alias in statement.names:
                if alias.name == "*":
                    continue
                name = alias.asname or alias.name
                target = ".".join(part for part in (target_module, alias.name) if part)
                local[name] = Symbol(
                    name=f"{module}.{name}",
                    module=module,
                    qualname=name,
                    kind="import",
                    fingerprint=hashlib.sha256(f"import:{target}".encode()).hexdigest(),
                    source_target=target,
                )
        elif isinstance(statement, ast.Import):
            for alias in statement.names:
                name = alias.asname or alias.name.split(".")[0]
                local[name] = Symbol(
                    name=f"{module}.{name}",
                    module=module,
                    qualname=name,
                    kind="import",
                    fingerprint=hashlib.sha256(
                        f"import:{alias.name}".encode()
                    ).hexdigest(),
                    source_target=alias.name,
                )
        elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target]
            )
            for target in targets:
                for name in _assigned_names(target):
                    if name == "__all__":
                        continue
                    local[name] = Symbol(
                        name=f"{module}.{name}",
                        module=module,
                        qualname=name,
                        kind="data",
                        fingerprint=_fingerprint(statement, "data"),
                    )

    selected: dict[str, Symbol] = {}
    names = (
        exports
        if exports is not None
        else tuple(
            name
            for name, symbol in local.items()
            if "." not in name and symbol.kind != "import" and not name.startswith("_")
        )
    )
    for name in names:
        if name.startswith("_") and name != "__version__":
            continue
        symbol = local.get(name)
        if symbol is None:
            symbol = Symbol(
                name=f"{module}.{name}",
                module=module,
                qualname=name,
                kind="unknown",
                fingerprint=hashlib.sha256(
                    f"unknown:{module}.{name}".encode()
                ).hexdigest(),
            )
        selected[symbol.name] = symbol
        if symbol.kind == "class":
            prefix = f"{name}."
            for local_name, member in local.items():
                if local_name.startswith(prefix):
                    selected[member.name] = member
    return selected


def extract_surface_from_sources(
    sources: Mapping[PurePosixPath, str],
    package: str = "openmed",
) -> dict[str, Symbol]:
    """Extract a public symbol map from Python source text."""

    surface: dict[str, Symbol] = {}
    for path, source in sorted(sources.items(), key=lambda item: str(item[0])):
        if path.suffix != ".py" or path.parts[0] != package:
            continue
        surface.update(_module_symbols(path, source, package))
    return _resolve_imported_symbols(surface)


def _resolve_imported_symbols(surface: Mapping[str, Symbol]) -> dict[str, Symbol]:
    """Copy signatures and class members onto statically resolvable re-exports."""

    resolved = dict(surface)
    for _ in range(8):
        updates: dict[str, Symbol] = {}
        for symbol in tuple(resolved.values()):
            if symbol.kind != "import" or symbol.source_target is None:
                continue
            target = resolved.get(symbol.source_target)
            if target is None or target.kind == "import":
                continue
            enriched = Symbol(
                name=symbol.name,
                module=symbol.module,
                qualname=symbol.qualname,
                kind=target.kind,
                signature=target.signature,
                parameters=target.parameters,
                deprecated=target.deprecated,
                fingerprint=target.fingerprint,
                source_target=symbol.source_target,
            )
            if enriched != symbol:
                updates[enriched.name] = enriched
            if target.kind != "class":
                continue
            target_prefix = f"{target.name}."
            alias_prefix = f"{symbol.name}."
            for member in tuple(resolved.values()):
                if not member.name.startswith(target_prefix):
                    continue
                suffix = member.name[len(target_prefix) :]
                alias_name = alias_prefix + suffix
                updates.setdefault(
                    alias_name,
                    Symbol(
                        name=alias_name,
                        module=symbol.module,
                        qualname=f"{symbol.qualname}.{suffix}",
                        kind=member.kind,
                        signature=member.signature,
                        parameters=member.parameters,
                        deprecated=member.deprecated,
                        fingerprint=member.fingerprint,
                        source_target=member.name,
                    ),
                )
        if not updates:
            break
        resolved.update(updates)
    return resolved


def extract_surface_from_path(
    package_root: Path,
    package: str | None = None,
) -> dict[str, Symbol]:
    """Extract a surface from a package directory without importing it."""

    package_root = package_root.resolve()
    package_name = package or package_root.name
    sources = {
        PurePosixPath(package_name, path.relative_to(package_root).as_posix()): (
            path.read_text(encoding="utf-8")
        )
        for path in sorted(package_root.rglob("*.py"))
        if "__pycache__" not in path.parts
    }
    return extract_surface_from_sources(sources, package_name)


def extract_surface(
    repo_root: Path,
    ref: str,
    package: str = "openmed",
) -> dict[str, Symbol]:
    """Extract a surface from ``ref`` or from the ``WORKTREE`` sentinel."""

    sources = _sources_from_ref(repo_root.resolve(), ref, package)
    return extract_surface_from_sources(sources, package)


def _narrowing_reasons(before: Symbol, after: Symbol) -> tuple[str, ...]:
    if before.kind != after.kind:
        return (f"kind changed from {before.kind} to {after.kind}",)
    if before.parameters is None or after.parameters is None:
        return ()
    reasons: list[str] = []
    old_by_name = {parameter.name: parameter for parameter in before.parameters}
    new_by_name = {parameter.name: parameter for parameter in after.parameters}
    for name, old in old_by_name.items():
        new = new_by_name.get(name)
        if new is None:
            reasons.append(f"parameter {name!r} was removed")
            continue
        if not old.required and new.required:
            reasons.append(f"parameter {name!r} became required")
        allowed_transitions = {
            "positional_only": {"positional_only", "positional_or_keyword"},
            "positional_or_keyword": {"positional_or_keyword"},
            "keyword_only": {"keyword_only", "positional_or_keyword"},
            "var_positional": {"var_positional"},
            "var_keyword": {"var_keyword"},
        }
        if new.kind not in allowed_transitions[old.kind]:
            reasons.append(f"parameter {name!r} changed from {old.kind} to {new.kind}")
    for name, new in new_by_name.items():
        if name not in old_by_name and new.required:
            reasons.append(f"required parameter {name!r} was added")
    old_positional = [
        parameter.name
        for parameter in before.parameters
        if parameter.kind in {"positional_only", "positional_or_keyword"}
    ]
    new_positional = [
        parameter.name
        for parameter in after.parameters
        if parameter.kind in {"positional_only", "positional_or_keyword"}
        and parameter.name in old_by_name
    ]
    retained_old = [name for name in old_positional if name in new_by_name]
    if retained_old != new_positional:
        reasons.append("positional parameter order changed")
    return tuple(dict.fromkeys(reasons))


def _rename_pairs(
    removed: Mapping[str, Symbol],
    added: Mapping[str, Symbol],
) -> list[tuple[Symbol, Symbol]]:
    old_by_fingerprint: dict[str, list[Symbol]] = {}
    new_by_fingerprint: dict[str, list[Symbol]] = {}
    for symbol in removed.values():
        if symbol.fingerprint:
            old_by_fingerprint.setdefault(symbol.fingerprint, []).append(symbol)
    for symbol in added.values():
        if symbol.fingerprint:
            new_by_fingerprint.setdefault(symbol.fingerprint, []).append(symbol)
    pairs: list[tuple[Symbol, Symbol]] = []
    for fingerprint in sorted(old_by_fingerprint.keys() & new_by_fingerprint.keys()):
        before = old_by_fingerprint[fingerprint]
        after = new_by_fingerprint[fingerprint]
        if len(before) == 1 and len(after) == 1:
            pairs.append((before[0], after[0]))
    return pairs


def diff_surfaces(
    before: Mapping[str, Symbol],
    after: Mapping[str, Symbol],
    *,
    before_ref: str = "before",
    after_ref: str = "after",
    package: str = "openmed",
) -> ApiSurfaceDiff:
    """Classify additions, deprecations, removals, renames, and narrowing."""

    before_names = set(before)
    after_names = set(after)
    removed = {name: before[name] for name in before_names - after_names}
    added = {name: after[name] for name in after_names - before_names}
    renamed = _rename_pairs(removed, added)
    renamed_old = {old.name for old, _ in renamed}
    renamed_new = {new.name for _, new in renamed}

    breaking: list[Change] = [
        Change(
            change="renamed",
            symbol=old.name,
            before=old,
            after=new,
            replacement=new.name,
            details=(f"renamed to {new.name}",),
        )
        for old, new in renamed
    ]
    for name in sorted(removed.keys() - renamed_old):
        breaking.append(Change(change="removed", symbol=name, before=removed[name]))

    deprecated: list[Change] = []
    for name in sorted(before_names & after_names):
        old = before[name]
        new = after[name]
        if new.deprecated and not old.deprecated:
            deprecated.append(
                Change(change="deprecated", symbol=name, before=old, after=new)
            )
            continue
        reasons = _narrowing_reasons(old, new)
        if reasons:
            breaking.append(
                Change(
                    change="signature-narrowed",
                    symbol=name,
                    before=old,
                    after=new,
                    details=reasons,
                )
            )

    added_changes: list[Change] = []
    for name in sorted(added.keys() - renamed_new):
        symbol = added[name]
        if symbol.deprecated:
            deprecated.append(Change(change="deprecated", symbol=name, after=symbol))
        else:
            added_changes.append(Change(change="added", symbol=name, after=symbol))

    return ApiSurfaceDiff(
        before_ref=before_ref,
        after_ref=after_ref,
        package=package,
        before_symbol_count=len(before),
        after_symbol_count=len(after),
        added=tuple(sorted(added_changes, key=lambda change: change.symbol)),
        deprecated=tuple(sorted(deprecated, key=lambda change: change.symbol)),
        breaking=tuple(
            sorted(breaking, key=lambda change: (change.symbol, change.change))
        ),
    )


def compare_refs(
    repo_root: Path,
    before_ref: str,
    after_ref: str,
    package: str = "openmed",
) -> ApiSurfaceDiff:
    """Extract and compare ``package`` at two Git references."""

    before = extract_surface(repo_root, before_ref, package)
    after = extract_surface(repo_root, after_ref, package)
    return diff_surfaces(
        before,
        after,
        before_ref=before_ref,
        after_ref=after_ref,
        package=package,
    )


def missing_migration_symbols(
    diff: ApiSurfaceDiff,
    document: str,
) -> tuple[str, ...]:
    """Return breaking symbols not named exactly in a migration document."""

    missing: list[str] = []
    for change in diff.breaking:
        symbol_pattern = re.compile(
            rf"(?<![A-Za-z0-9_.]){re.escape(change.symbol)}(?![A-Za-z0-9_.])"
        )
        if symbol_pattern.search(document) is None:
            missing.append(change.symbol)
    return tuple(missing)


def check_migration_document(
    diff: ApiSurfaceDiff,
    document_path: Path,
) -> tuple[str, ...]:
    """Return breaking symbols missing from ``document_path``."""

    try:
        document = document_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ApiSurfaceError(
            f"could not read migration document {document_path}: {exc}"
        ) from exc
    return missing_migration_symbols(diff, document)


def render_human_summary(diff: ApiSurfaceDiff) -> str:
    """Render a concise, deterministic release-log summary."""

    lines = [
        f"API surface: {diff.before_ref} -> {diff.after_ref} ({diff.package})",
        (
            f"Symbols: {diff.before_symbol_count} -> {diff.after_symbol_count}; "
            f"added={len(diff.added)}, deprecated={len(diff.deprecated)}, "
            f"breaking={len(diff.breaking)}"
        ),
    ]
    for heading, changes, limit in (
        ("Breaking", diff.breaking, None),
        ("Deprecated", diff.deprecated, None),
        ("Added", diff.added, 25),
    ):
        if not changes:
            continue
        lines.append(f"{heading}:")
        visible = changes if limit is None else changes[:limit]
        for change in visible:
            suffix = f" -> {change.replacement}" if change.replacement else ""
            lines.append(f"  - {change.change}: {change.symbol}{suffix}")
        if limit is not None and len(changes) > limit:
            lines.append(f"  - ... {len(changes) - limit} more (see JSON output)")
    return "\n".join(lines)


def _write_json(diff: ApiSurfaceDiff, destination: str) -> None:
    rendered = json.dumps(diff.to_dict(), indent=2, sort_keys=True) + "\n"
    if destination == "-":
        sys.stdout.write(rendered)
        return
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before_ref", help="Baseline Git ref, for example v1.8.0")
    parser.add_argument(
        "after_ref",
        help=f"Candidate Git ref or {WORKTREE_REF} for current files",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Git repository root (default: current directory)",
    )
    parser.add_argument("--package", default="openmed", help="Package directory")
    parser.add_argument(
        "--json",
        dest="json_destination",
        metavar="PATH",
        help="Write the machine-readable diff to PATH (use - for stdout)",
    )
    parser.add_argument(
        "--check",
        type=Path,
        metavar="MIGRATION_DOC",
        help="Fail if a breaking symbol is absent from this migration guide",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the API-surface diff and optional migration completeness gate."""

    args = build_parser().parse_args(argv)
    try:
        diff = compare_refs(
            args.repo_root,
            args.before_ref,
            args.after_ref,
            args.package,
        )
        if args.json_destination:
            _write_json(diff, args.json_destination)
        if args.json_destination != "-":
            print(render_human_summary(diff))
        if args.check:
            document_path = args.check
            if not document_path.is_absolute():
                document_path = args.repo_root / document_path
            missing = check_migration_document(diff, document_path)
            if missing:
                print(
                    "Migration guide is missing breaking API symbols:",
                    file=sys.stderr,
                )
                for symbol in missing:
                    print(f"  - {symbol}", file=sys.stderr)
                return 1
            print(f"Migration guide completeness check passed: {document_path}")
    except ApiSurfaceError as exc:
        print(f"api-surface-diff: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
