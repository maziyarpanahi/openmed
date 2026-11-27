"""Static scan for environment-variable reads in the ``openmed`` package.

The settings schema tests use this module to find every ``OPENMED_*`` name the
package reads from the process environment without importing or executing any
scanned module.  The scan is purely syntactic: it parses each module with
:mod:`ast`, resolves the key expression passed to an environment read, and
reports the variable name with its repository-relative path and line.  It
never reads environment values.

Recognised reads:

* ``os.getenv(KEY)``, ``getenv(KEY)``, ``os.environ.get(KEY)``,
  ``os.environ[KEY]``, ``os.environ.pop(KEY)``, ``os.environ.setdefault(KEY)``
  and ``KEY in os.environ``;
* ``mapping.get(KEY)``, ``mapping[KEY]``, ``KEY in mapping`` and a bound
  ``get = mapping.get`` alias when ``KEY`` resolves to an ``OPENMED_*`` name,
  which covers the injected ``environ: Mapping[str, str] = os.environ`` style
  used across the service;
* calls to the package's own environment helper wrappers: functions whose
  parameter flows into one of the reads above, found automatically, followed
  through further wrappers, and resolved through parameter defaults when a
  caller omits the argument.  A helper that derives a companion name, such as
  ``secret_file_setting(name) -> f"{name}_FILE"``, is followed as well.

Key expressions resolve through string literals, module constants, imported
constants, class attributes, module-level tables (``TABLE[...]`` and loops
over ``TABLE.items()`` or a literal tuple of rows), simple local assignments
and conditional expressions.  Keys built with f-strings, ``+``, ``%`` or
``format`` from an ``OPENMED_`` fragment are reported as dynamic because no
literal name can be checked for them.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "openmed"

ENV_NAME_RE = re.compile(r"^OPENMED_[A-Z0-9_]*[A-Z0-9]$")
_ENV_FRAGMENT = "OPENMED_"
_SINK_METHODS = frozenset({"get", "pop", "setdefault"})
_TABLE_WRAPPERS = frozenset({"MappingProxyType", "frozenset", "tuple", "dict"})
_MAX_DEPTH = 8
# A module that mentions none of these cannot read an ``OPENMED_*`` variable
# by any of the recognised forms, so the scan skips parsing it.
_MAY_READ_RE = re.compile(r"environ|getenv|OPENMED_|ENV")


@dataclass(frozen=True, slots=True, order=True)
class EnvRead:
    """One statically resolved environment read.

    Attributes:
        name: The resolved environment variable name.
        path: Repository-relative POSIX path of the module.
        line: One-based line of the read.
        sink: Short description of the read form, such as ``os.getenv``.
    """

    name: str
    path: str
    line: int
    sink: str

    @property
    def location(self) -> str:
        """Return ``path:line`` for failure messages."""

        return f"{self.path}:{self.line}"


@dataclass(frozen=True, slots=True, order=True)
class DynamicEnvRead:
    """An environment read whose key is built at runtime.

    Attributes:
        path: Repository-relative POSIX path of the module.
        line: One-based line of the read.
        expression: Source of the key expression; code only, never a value.
    """

    path: str
    line: int
    expression: str

    @property
    def location(self) -> str:
        """Return ``path:line`` for failure messages."""

        return f"{self.path}:{self.line}"


@dataclass(slots=True)
class EnvScan:
    """Result of scanning one or more modules."""

    reads: list[EnvRead] = field(default_factory=list)
    dynamic: list[DynamicEnvRead] = field(default_factory=list)

    def names(self, prefix: str = _ENV_FRAGMENT) -> set[str]:
        """Return the distinct resolved names that start with ``prefix``."""

        return {read.name for read in self.reads if read.name.startswith(prefix)}

    def locations(self, name: str) -> list[str]:
        """Return sorted ``path:line`` locations that read ``name``."""

        return sorted({read.location for read in self.reads if read.name == name})


class _Dynamic:
    """Marker for a key that is built at runtime from an ``OPENMED_`` part."""


_DYNAMIC = _Dynamic()


@dataclass(frozen=True, slots=True)
class _Param:
    """A key that is a wrapper parameter, optionally with a derived suffix."""

    name: str
    suffix: str = ""


@dataclass(frozen=True, slots=True)
class _WrapperParam:
    position: int
    default: ast.AST | None
    suffixes: frozenset[str]


@dataclass(slots=True)
class _Entry:
    line: int
    key: ast.AST
    sink: str
    strict: bool
    func: ast.AST | None
    module: _Module
    suffix: str = ""


@dataclass(slots=True)
class _Module:
    dotted: str
    path: Path
    rel: str
    tree: ast.Module
    constants: dict[str, str] = field(default_factory=dict)
    tables: dict[str, frozenset[str]] = field(default_factory=dict)
    class_constants: dict[tuple[str, str], str] = field(default_factory=dict)
    transforms: dict[str, ast.AST] = field(default_factory=dict)
    imported_names: dict[str, tuple[str, str]] = field(default_factory=dict)
    imported_modules: dict[str, str] = field(default_factory=dict)
    os_aliases: set[str] = field(default_factory=set)
    environ_aliases: set[str] = field(default_factory=set)
    getenv_aliases: set[str] = field(default_factory=set)
    sinks: list[_Entry] = field(default_factory=list)
    calls: list[tuple[ast.Call, ast.AST | None]] = field(default_factory=list)
    bindings_nodes: dict[int, list[ast.AST]] = field(default_factory=dict)
    bindings: dict[int, dict[str, object]] = field(default_factory=dict)


def _dotted_name(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _string_value(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _env_strings(nodes: Iterable[ast.AST]) -> set[str]:
    found: set[str] = set()
    for node in nodes:
        text = _string_value(node)
        if text is not None and ENV_NAME_RE.match(text):
            found.add(text)
        elif isinstance(node, (ast.Tuple, ast.List)):
            found.update(_env_strings(node.elts))
    return found


def _table_values(node: ast.AST | None) -> frozenset[str] | None:
    """Return ``OPENMED_*`` values of a literal table, unwrapping proxies."""

    if isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords:
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else None
        if isinstance(func, ast.Name):
            name = func.id
        if name in _TABLE_WRAPPERS:
            return _table_values(node.args[0])
    if isinstance(node, ast.Dict):
        found = _env_strings(value for value in node.values if value is not None)
    elif isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        found = _env_strings(node.elts)
    else:
        return None
    return frozenset(found) if found else None


def _assignment_pairs(node: ast.AST) -> Iterator[tuple[str, ast.AST]]:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                yield target.id, node.value
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        if node.value is not None:
            yield node.target.id, node.value


def _bind(local: dict[str, object], name: str, value: frozenset[str]) -> None:
    """Bind ``name`` flow-insensitively: reuse of a local unions its names."""

    existing = local.get(name)
    local[name] = existing | value if isinstance(existing, frozenset) else value


def _parameters(func: ast.AST) -> dict[str, tuple[int, ast.AST | None]]:
    """Return each parameter's position (``-1`` for keyword-only) and default."""

    args = func.args  # type: ignore[attr-defined]
    ordered = [*args.posonlyargs, *args.args]
    defaults: list[ast.AST | None] = [None] * (len(ordered) - len(args.defaults))
    defaults.extend(args.defaults)
    result = {arg.arg: (index, defaults[index]) for index, arg in enumerate(ordered)}
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        result[arg.arg] = (-1, default)
    return result


def _suffix_node(func: ast.FunctionDef) -> ast.AST | None:
    """Return ``SUFFIX`` for ``def f(name): return f"{name}SUFFIX"`` helpers."""

    body = [
        node
        for node in func.body
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant))
    ]
    params = [*func.args.posonlyargs, *func.args.args]
    if len(body) != 1 or not isinstance(body[0], ast.Return) or len(params) != 1:
        return None
    value = body[0].value
    param = params[0].arg
    if (
        isinstance(value, ast.JoinedStr)
        and len(value.values) == 2
        and isinstance(value.values[0], ast.FormattedValue)
        and isinstance(value.values[0].value, ast.Name)
        and value.values[0].value.id == param
    ):
        return value.values[1]
    return None


class _Collector(ast.NodeVisitor):
    """Single pass that records sinks, calls and local binding statements."""

    def __init__(self, module: _Module) -> None:
        self.module = module
        self.stack: list[ast.AST] = []
        self.get_aliases: dict[int, set[str]] = {}

    def _current(self) -> ast.AST | None:
        return self.stack[-1] if self.stack else None

    def _visit_function(self, node: ast.AST) -> None:
        self.stack.append(node)
        self.module.bindings_nodes.setdefault(id(node), [])
        self.generic_visit(node)
        self.stack.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def _record_binding(self, node: ast.AST) -> None:
        current = self._current()
        if current is not None:
            self.module.bindings_nodes[id(current)].append(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self._record_binding(node)
        current = self._current()
        if (
            current is not None
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr in _SINK_METHODS
        ):
            aliases = self.get_aliases.setdefault(id(current), set())
            aliases.add(node.targets[0].id)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self._record_binding(node)
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self._record_binding(node)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "os":
                self.module.os_aliases.add(alias.asname or "os")
            elif alias.name.startswith("openmed") and alias.asname:
                self.module.imported_modules[alias.asname] = alias.name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        source = node.module or ""
        if node.level:
            parts = self.module.dotted.split(".")
            if self.module.path.name != "__init__.py":
                parts = parts[:-1]
            if node.level > 1:
                parts = parts[: len(parts) - (node.level - 1)]
            base = ".".join(parts)
            source = f"{base}.{source}" if source else base
        for alias in node.names:
            local = alias.asname or alias.name
            if source == "os" and alias.name == "environ":
                self.module.environ_aliases.add(local)
            elif source == "os" and alias.name == "getenv":
                self.module.getenv_aliases.add(local)
            elif source.startswith("openmed"):
                self.module.imported_names[local] = (source, alias.name)

    def _is_environ(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Attribute) and node.attr == "environ":
            return isinstance(node.value, ast.Name) and (
                node.value.id in self.module.os_aliases
            )
        return isinstance(node, ast.Name) and node.id in self.module.environ_aliases

    def _add(self, node: ast.AST, key: ast.AST, sink: str, strict: bool) -> None:
        self.module.sinks.append(
            _Entry(node.lineno, key, sink, strict, self._current(), self.module)
        )

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if node.args:
            if isinstance(func, ast.Attribute):
                if (
                    func.attr == "getenv"
                    and isinstance(func.value, ast.Name)
                    and func.value.id in self.module.os_aliases
                ):
                    self._add(node, node.args[0], "os.getenv", True)
                elif func.attr in _SINK_METHODS:
                    if self._is_environ(func.value):
                        self._add(node, node.args[0], f"os.environ.{func.attr}", True)
                    else:
                        self._add(node, node.args[0], f"mapping.{func.attr}", False)
            elif isinstance(func, ast.Name):
                current = self._current()
                aliases = self.get_aliases.get(id(current), set())
                if func.id in self.module.getenv_aliases:
                    self._add(node, node.args[0], "os.getenv", True)
                elif func.id in aliases:
                    self._add(node, node.args[0], "mapping.get", False)
        self.module.calls.append((node, self._current()))
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.ctx, ast.Load):
            if self._is_environ(node.value):
                self._add(node, node.slice, "os.environ[]", True)
            else:
                self._add(node, node.slice, "mapping[]", False)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        if len(node.ops) == 1 and isinstance(node.ops[0], (ast.In, ast.NotIn)):
            if self._is_environ(node.comparators[0]):
                self._add(node, node.left, "in os.environ", True)
            else:
                self._add(node, node.left, "in mapping", False)
        self.generic_visit(node)


def _load_module(path: Path, root: Path) -> _Module:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    module = _Module(
        dotted=_dotted_name(path, root),
        path=path,
        rel=path.relative_to(root).as_posix(),
        tree=tree,
    )
    for node in tree.body:
        for name, value in _assignment_pairs(node):
            text = _string_value(value)
            if text is not None:
                module.constants[name] = text
                continue
            table = _table_values(value)
            if table is not None:
                module.tables[name] = table
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                for name, value in _assignment_pairs(item):
                    text = _string_value(value)
                    if text is not None:
                        module.class_constants[(node.name, name)] = text
        elif isinstance(node, ast.FunctionDef):
            suffix_node = _suffix_node(node)
            if suffix_node is not None:
                module.transforms[node.name] = suffix_node
    _Collector(module).visit(tree)
    return module


class _Scanner:
    def __init__(self, paths: Iterable[Path], root: Path) -> None:
        self.root = root
        self.modules: dict[str, _Module] = {}
        self.missing: set[str] = set()
        self.wrappers: dict[tuple[str, str], dict[str, _WrapperParam]] = {}
        for path in paths:
            self._add(path.resolve())

    def _add(self, path: Path) -> _Module:
        dotted = _dotted_name(path, self.root)
        module = self.modules.get(dotted)
        if module is None:
            module = _load_module(path, self.root)
            self.modules[dotted] = module
        return module

    def module(self, dotted: str) -> _Module | None:
        """Return a loaded module, parsing an ``openmed`` module on demand.

        Only package ``__init__`` modules, which re-export constants, and
        modules that could hold a read or an ``OPENMED_*`` constant are
        parsed; anything else cannot contribute a name.
        """

        found = self.modules.get(dotted)
        if found is not None or dotted in self.missing:
            return found
        relative = Path(*dotted.split("."))
        for candidate in (
            self.root / relative.with_suffix(".py"),
            self.root / relative / "__init__.py",
        ):
            if not candidate.is_file():
                continue
            if candidate.name == "__init__.py" or _MAY_READ_RE.search(
                candidate.read_text(encoding="utf-8")
            ):
                return self._add(candidate)
            break
        self.missing.add(dotted)
        return None

    # -- constant resolution -------------------------------------------------

    def _imported(self, module: _Module, local: str) -> tuple[_Module, str] | None:
        target = module.imported_names.get(local)
        if target is None:
            return None
        other = self.module(target[0])
        if other is None:
            return None
        return other, target[1]

    def _constant(self, module: _Module, name: str, depth: int = 0) -> str | None:
        if name in module.constants:
            return module.constants[name]
        imported = self._imported(module, name)
        if imported is not None and depth < _MAX_DEPTH:
            return self._constant(*imported, depth + 1)
        return None

    def _table(
        self, module: _Module, name: str, depth: int = 0
    ) -> frozenset[str] | None:
        if name in module.tables:
            return module.tables[name]
        imported = self._imported(module, name)
        if imported is not None and depth < _MAX_DEPTH:
            return self._table(*imported, depth + 1)
        return None

    def _transform(self, module: _Module, call: ast.Call) -> str | None:
        """Return the suffix when ``call`` invokes a name-deriving helper."""

        if len(call.args) != 1 or call.keywords:
            return None
        func = call.func
        if isinstance(func, ast.Name):
            node = module.transforms.get(func.id)
            owner = module
            if node is None:
                imported = self._imported(module, func.id)
                if imported is not None:
                    owner, name = imported
                    node = owner.transforms.get(name)
            if node is not None:
                return self._suffix_text(owner, node)
        return None

    def _suffix_text(self, module: _Module, node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.FormattedValue) and isinstance(node.value, ast.Name):
            return self._constant(module, node.value.id)
        return None

    def _attribute_module(self, module: _Module, node: ast.AST) -> _Module | None:
        if not isinstance(node, ast.Name):
            return None
        dotted = module.imported_modules.get(node.id)
        if dotted is None:
            imported = module.imported_names.get(node.id)
            if imported is not None:
                dotted = f"{imported[0]}.{imported[1]}"
        return self.module(dotted) if dotted is not None else None

    def resolve(
        self, module: _Module, node: ast.AST, local: dict[str, object]
    ) -> object:
        """Resolve ``node`` to names, a wrapper parameter, dynamic, or None."""

        text = _string_value(node)
        if text is not None:
            return frozenset({text})
        if isinstance(node, ast.Name):
            if node.id in local:
                return local[node.id]
            value = self._constant(module, node.id)
            return frozenset({value}) if value is not None else None
        if isinstance(node, ast.Attribute):
            return self._resolve_attribute(module, node)
        if isinstance(node, ast.Subscript):
            return self._container(module, node.value, local)
        if isinstance(node, ast.IfExp):
            left = self.resolve(module, node.body, local)
            right = self.resolve(module, node.orelse, local)
            if isinstance(left, frozenset) and isinstance(right, frozenset):
                return left | right
            return left if isinstance(left, frozenset) else right
        if isinstance(node, ast.Call):
            suffix = self._transform(module, node)
            if suffix is not None:
                inner = self.resolve(module, node.args[0], local)
                if isinstance(inner, frozenset):
                    return frozenset(name + suffix for name in inner)
                if isinstance(inner, _Param):
                    return _Param(inner.name, inner.suffix + suffix)
                return None
        if self._is_dynamic(node):
            return _DYNAMIC
        return None

    def _resolve_attribute(self, module: _Module, node: ast.Attribute) -> object:
        other = self._attribute_module(module, node.value)
        if other is not None:
            value = self._constant(other, node.attr)
            if value is not None:
                return frozenset({value})
        if isinstance(node.value, ast.Name):
            owner = node.value.id
            if owner in {"self", "cls"}:
                matches = {
                    value
                    for (_, attr), value in module.class_constants.items()
                    if attr == node.attr
                }
                if len(matches) == 1:
                    return frozenset(matches)
            value = module.class_constants.get((owner, node.attr))
            if value is not None:
                return frozenset({value})
        return None

    def _container(
        self, module: _Module, node: ast.AST, local: dict[str, object]
    ) -> frozenset[str] | None:
        if isinstance(node, ast.Name):
            value = local.get(node.id)
            if isinstance(value, frozenset):
                return value
            return self._table(module, node.id)
        if isinstance(node, ast.Attribute):
            other = self._attribute_module(module, node.value)
            if other is not None:
                return self._table(other, node.attr)
            return None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"items", "values"}:
                return self._container(module, node.func.value, local)
            return None
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            found: set[str] = set()
            for element in node.elts:
                if isinstance(element, (ast.Tuple, ast.List)):
                    found.update(self._container(module, element, local) or ())
                    continue
                value = self.resolve(module, element, local)
                if isinstance(value, frozenset):
                    found.update(name for name in value if ENV_NAME_RE.match(name))
            return frozenset(found) if found else None
        return None

    @staticmethod
    def _is_dynamic(node: ast.AST) -> bool:
        def has_fragment(part: ast.AST) -> bool:
            return any(
                isinstance(sub, ast.Constant)
                and isinstance(sub.value, str)
                and _ENV_FRAGMENT in sub.value
                for sub in ast.walk(part)
            )

        if isinstance(node, ast.JoinedStr):
            return has_fragment(node)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod)):
            return has_fragment(node)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "format"
        ):
            return has_fragment(node.func.value)
        return False

    # -- local bindings ------------------------------------------------------

    def _bindings(self, module: _Module, func: ast.AST | None) -> dict[str, object]:
        if func is None:
            return {}
        cached = module.bindings.get(id(func))
        if cached is not None:
            return cached
        local: dict[str, object] = {name: _Param(name) for name in _parameters(func)}
        module.bindings[id(func)] = local
        for _ in range(2):
            for node in module.bindings_nodes.get(id(func), ()):
                if isinstance(node, ast.Assign):
                    target = node.targets[0] if len(node.targets) == 1 else None
                    if isinstance(target, ast.Name):
                        value = self.resolve(module, node.value, local)
                        if isinstance(value, frozenset):
                            _bind(local, target.id, value)
                    continue
                self._bind_loop(module, node, local)
        return local

    def _bind_loop(
        self, module: _Module, node: ast.AST, local: dict[str, object]
    ) -> None:
        target = node.target  # type: ignore[attr-defined]
        iterable = node.iter  # type: ignore[attr-defined]
        if isinstance(target, ast.Name):
            table = self._container(module, iterable, local)
            if table:
                _bind(local, target.id, table)
            return
        if not isinstance(target, ast.Tuple):
            return
        names = [elt.id if isinstance(elt, ast.Name) else None for elt in target.elts]
        rows = (
            list(iterable.elts) if isinstance(iterable, (ast.Tuple, ast.List)) else []
        )
        if rows and all(
            isinstance(row, (ast.Tuple, ast.List)) and len(row.elts) == len(names)
            for row in rows
        ):
            for position, name in enumerate(names):
                if name is None:
                    continue
                found: set[str] = set()
                for row in rows:
                    element = row.elts[position]  # type: ignore[attr-defined]
                    value = self.resolve(module, element, local)
                    if isinstance(value, frozenset):
                        found.update(n for n in value if ENV_NAME_RE.match(n))
                if found:
                    _bind(local, name, frozenset(found))
            return
        if (
            isinstance(iterable, ast.Call)
            and isinstance(iterable.func, ast.Attribute)
            and iterable.func.attr == "items"
            and len(names) == 2
            and names[1] is not None
        ):
            table = self._container(module, iterable, local)
            if table:
                _bind(local, names[1], table)

    # -- wrappers ------------------------------------------------------------

    def _call_targets(
        self, module: _Module, call: ast.Call
    ) -> Iterator[tuple[str, str]]:
        func = call.func
        if isinstance(func, ast.Name):
            yield module.dotted, func.id
            imported = module.imported_names.get(func.id)
            if imported is not None:
                self.module(imported[0])
                yield imported
        elif isinstance(func, ast.Attribute):
            other = self._attribute_module(module, func.value)
            if other is not None:
                yield other.dotted, func.attr
            elif isinstance(func.value, ast.Name) and func.value.id in {
                "self",
                "cls",
            }:
                yield module.dotted, func.attr

    def _wrapper_entries(
        self, module: _Module, call: ast.Call, func: ast.AST | None
    ) -> Iterator[_Entry]:
        for target in self._call_targets(module, call):
            params = self.wrappers.get(target)
            if not params:
                continue
            owner = self.modules[target[0]]
            sink = f"{target[1]}()"
            for param, spec in params.items():
                key: ast.AST | None = None
                key_module, key_func = module, func
                if 0 <= spec.position < len(call.args):
                    key = call.args[spec.position]
                for keyword in call.keywords:
                    if keyword.arg == param:
                        key = keyword.value
                if key is None and spec.default is not None:
                    # The caller relies on the wrapper's default name.
                    key, key_module, key_func = spec.default, owner, None
                if key is None:
                    continue
                for suffix in sorted(spec.suffixes):
                    yield _Entry(
                        call.lineno, key, sink, True, key_func, key_module, suffix
                    )
            return

    def _entries(self, module: _Module) -> Iterator[_Entry]:
        yield from module.sinks
        for call, func in module.calls:
            yield from self._wrapper_entries(module, call, func)

    def discover_wrappers(self) -> None:
        """Find helper functions whose parameter flows into an env read."""

        done: set[tuple[int, int, str]] = set()
        changed = True
        while changed:
            loaded = len(self.modules)
            changed = False
            for module in list(self.modules.values()):
                for entry in self._entries(module):
                    func = entry.func
                    marker = (id(entry.key), id(func), entry.suffix)
                    if func is None or marker in done:
                        continue
                    done.add(marker)
                    value = self.resolve(
                        entry.module, entry.key, self._bindings(entry.module, func)
                    )
                    if not isinstance(value, _Param):
                        continue
                    target = (entry.module.dotted, func.name)  # type: ignore[attr-defined]
                    params = self.wrappers.setdefault(target, {})
                    position, default = _parameters(func)[value.name]
                    suffix = value.suffix + entry.suffix
                    spec = params.get(value.name)
                    suffixes = frozenset({suffix})
                    if spec is not None:
                        if suffix in spec.suffixes:
                            continue
                        suffixes |= spec.suffixes
                    params[value.name] = _WrapperParam(position, default, suffixes)
                    changed = True
            changed = changed or len(self.modules) != loaded

    def scan(self, module: _Module, result: EnvScan) -> None:
        seen: set[tuple[str, int]] = set()
        for entry in self._entries(module):
            value = self.resolve(
                entry.module, entry.key, self._bindings(entry.module, entry.func)
            )
            if value is _DYNAMIC:
                if entry.strict:
                    result.dynamic.append(
                        DynamicEnvRead(
                            module.rel, entry.line, ast.unparse(entry.key)[:120]
                        )
                    )
                continue
            if not isinstance(value, frozenset):
                continue
            for base in sorted(value):
                name = base + entry.suffix
                if not ENV_NAME_RE.match(name) or (name, entry.line) in seen:
                    continue
                seen.add((name, entry.line))
                result.reads.append(EnvRead(name, module.rel, entry.line, entry.sink))


def iter_modules(paths: Iterable[Path]) -> Iterator[Path]:
    """Yield Python files under ``paths`` in a stable order."""

    for root in paths:
        if root.is_file():
            yield root
            continue
        yield from sorted(path for path in root.rglob("*.py") if path.is_file())


def scan_env_reads(
    targets: Iterable[Path] | None = None, *, root: Path = REPO_ROOT
) -> EnvScan:
    """Scan ``targets`` for environment reads.

    Modules imported by the targets are parsed on demand to resolve imported
    constants and helper wrappers; only reads inside ``targets`` are reported.
    A target module whose source mentions neither ``environ``, ``getenv``,
    ``OPENMED_`` nor ``ENV`` cannot contain a recognised read and is skipped.

    Args:
        targets: Files or directories to scan.  Defaults to the whole
            ``openmed`` package.
        root: Directory that holds the top-level package; reported paths
            and dotted module names are relative to it.

    Returns:
        The sorted reads and dynamic keys found in ``targets``.
    """

    target_paths = [
        path
        for path in iter_modules(targets or [PACKAGE_ROOT])
        if _MAY_READ_RE.search(path.read_text(encoding="utf-8"))
    ]
    root = root.resolve()
    scanner = _Scanner(target_paths, root)
    scanner.discover_wrappers()
    result = EnvScan()
    for path in target_paths:
        scanner.scan(scanner.modules[_dotted_name(path.resolve(), root)], result)
    result.reads.sort()
    result.dynamic.sort()
    return result


def literal_env_names(
    targets: Iterable[Path],
    prefix: str = _ENV_FRAGMENT,
    *,
    root: Path = REPO_ROOT,
) -> dict[str, list[str]]:
    """Return whole-string ``OPENMED_*`` literals outside docstrings.

    This is a deliberately broad cross-check for :func:`scan_env_reads`: any
    name the read scan cannot resolve still shows up here as a literal.

    Args:
        targets: Files or directories to scan.
        prefix: Only names starting with this prefix are returned.
        root: Directory reported paths are relative to.

    Returns:
        Mapping of each name to its sorted ``path:line`` locations.
    """

    found: dict[str, set[str]] = {}
    for path in iter_modules(targets):
        resolved = path.resolve()
        source = resolved.read_text(encoding="utf-8")
        if prefix not in source:
            continue
        rel = resolved.relative_to(root.resolve()).as_posix()
        tree = ast.parse(source, filename=str(path))
        nodes = list(ast.walk(tree))
        docstrings: set[int] = set()
        for node in nodes:
            body = getattr(node, "body", None)
            if (
                isinstance(
                    node,
                    (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
                )
                and body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
            ):
                docstrings.add(id(body[0].value))
        for node in nodes:
            text = _string_value(node)
            if text is None or id(node) in docstrings:
                continue
            if ENV_NAME_RE.match(text) and text.startswith(prefix):
                found.setdefault(text, set()).add(f"{rel}:{node.lineno}")
    return {name: sorted(locations) for name, locations in found.items()}
