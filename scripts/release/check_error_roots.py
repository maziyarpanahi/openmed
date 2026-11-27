#!/usr/bin/env python3
"""Gate public OpenMed exceptions that are not rooted under ``OpenMedError``.

The scan imports every public module of a package (no path component starting
with an underscore), collects the exception classes those modules expose under
public names, and reports each class by its qualified name
(``module:QualName``) only. Warning categories are not failures and are not
scanned. Modules that cannot be imported because an optional dependency is
missing are listed as skipped; classes they define are neither checked nor
treated as stale.

The committed allowlist names the exceptions that predate the rooted taxonomy.
It may only shrink: a public exception missing from it fails the gate, and so
does an entry whose class is now rooted or gone, so a rerooting change has to
delete the lines it fixed (``--prune`` does that). Every
:class:`~openmed.core.errors.LegacyRootedError` subclass must also keep the
roots directly in front of ``Exception`` in its method resolution order and
report a valid stable code.

``--record`` captures, before a package is rerooted, each class's method
resolution order, stable code, and the ``args``/``str()``/pickling behavior of
a few synthetic constructions. ``tests/unit/core/test_error_roots.py`` replays
those snapshots against the rerooted classes to prove nothing observable
changed. Snapshots contain synthetic tokens, class names and exception type
names only.

Usage::

    python scripts/release/check_error_roots.py
    python scripts/release/check_error_roots.py --prune
    python scripts/release/check_error_roots.py --record openmed.multimodal \\
        --record-output tests/fixtures/error_roots/rerooted/multimodal.json
"""

from __future__ import annotations

import argparse
import builtins
import importlib
import json
import pickle
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ALLOWLIST = (
    ROOT / "tests" / "fixtures" / "error_roots" / "unrooted_allowlist.txt"
)
REPORT_SCHEMA_VERSION = 1
SNAPSHOT_SCHEMA_VERSION = 1

# Synthetic constructor calls replayed against every rerooted class. They hold
# placeholder tokens only, never clinical text.
SAMPLE_CALLS: tuple[tuple[str, ...], ...] = (
    (),
    ("synthetic_reason",),
    ("synthetic_reason", "synthetic_detail"),
)

_STABLE_CODE = re.compile(r"[a-z][a-z0-9_]*")
_BUILTIN_EXCEPTIONS = frozenset(
    value
    for value in vars(builtins).values()
    if isinstance(value, type) and issubclass(value, BaseException)
)


@dataclass(frozen=True)
class ScanReport:
    """Result of scanning one package for public exception classes.

    Attributes:
        package: Scanned top-level package name.
        modules_scanned: Number of public modules the scan tried to import.
        import_failures: Module name to exception type name for modules that
            could not be imported.
        imported_modules: Package modules present after the scan, used to
            decide whether an allowlist entry can be verified.
        unrooted: Qualified names of public exceptions outside OpenMedError.
        legacy_rooted: Qualified name to the code and invariant violations of
            each public ``LegacyRootedError`` subclass.
    """

    package: str
    modules_scanned: int
    import_failures: Mapping[str, str]
    imported_modules: frozenset[str]
    unrooted: tuple[str, ...]
    legacy_rooted: Mapping[str, Mapping[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready report."""

        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "package": self.package,
            "modules_scanned": self.modules_scanned,
            "import_failures": dict(sorted(self.import_failures.items())),
            "imported_modules": sorted(self.imported_modules),
            "unrooted": list(self.unrooted),
            "legacy_rooted": {
                name: dict(entry) for name, entry in sorted(self.legacy_rooted.items())
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScanReport":
        """Rebuild a report written by :meth:`to_dict`."""

        if payload.get("schema_version") != REPORT_SCHEMA_VERSION:
            raise ValueError("unsupported error-root report schema version")
        return cls(
            package=str(payload["package"]),
            modules_scanned=int(payload["modules_scanned"]),
            import_failures=dict(payload["import_failures"]),
            imported_modules=frozenset(payload["imported_modules"]),
            unrooted=tuple(payload["unrooted"]),
            legacy_rooted=dict(payload["legacy_rooted"]),
        )


@dataclass(frozen=True)
class GateResult:
    """Comparison of a scan with the committed allowlist.

    Attributes:
        new: Unrooted public exceptions missing from the allowlist.
        stale: Allowlist entries whose class is rooted or no longer public.
        unverified: Allowlist entries defined in modules that did not import.
        violations: ``LegacyRootedError`` subclasses breaking an invariant,
            as ``name: reason`` strings.
        duplicates: Allowlist entries listed more than once.
        unsorted: Whether the allowlist is out of order.
    """

    new: tuple[str, ...]
    stale: tuple[str, ...]
    unverified: tuple[str, ...]
    violations: tuple[str, ...]
    duplicates: tuple[str, ...] = field(default=())
    unsorted: bool = False

    @property
    def passed(self) -> bool:
        """Whether the gate passes."""

        return (
            not (self.new or self.stale or self.violations or self.duplicates)
            and not self.unsorted
        )

    def describe(self) -> str:
        """Return a reviewer-facing summary listing qualified names only."""

        lines: list[str] = []
        sections = (
            (
                "Public exceptions not rooted under OpenMedError (root them with "
                "LegacyRootedError instead of allowlisting them)",
                self.new,
            ),
            (
                "Stale allowlist entries (delete them; the allowlist only "
                "shrinks; --prune does it)",
                self.stale,
            ),
            ("LegacyRootedError invariant violations", self.violations),
            ("Duplicate allowlist entries", self.duplicates),
        )
        for title, names in sections:
            if names:
                lines.append(f"{title}:")
                lines.extend(f"  - {name}" for name in names)
        if self.unsorted:
            lines.append("The allowlist must be sorted.")
        return "\n".join(lines) if lines else "error-root gate passed"


def qualified_name(error_class: type) -> str:
    """Return the ``module:QualName`` identifier used by the gate."""

    return f"{error_class.__module__}:{error_class.__qualname__}"


def public_modules(package: str, package_dir: Path) -> list[str]:
    """List the public module names of a package from its source tree.

    Args:
        package: Importable package name.
        package_dir: Directory holding the package's ``__init__.py``.

    Returns:
        Sorted module names whose path components below ``package`` do not
        start with an underscore. Directories without ``__init__.py`` are not
        packages and are skipped.
    """

    modules = [package]
    for path in sorted(package_dir.rglob("*.py")):
        relative = path.relative_to(package_dir)
        parts = list(relative.parts[:-1])
        stem = relative.stem
        if any(part.startswith("_") for part in parts):
            continue
        if not all(
            (package_dir.joinpath(*parts[: index + 1]) / "__init__.py").is_file()
            for index in range(len(parts))
        ):
            continue
        if stem == "__init__":
            if parts:
                modules.append(".".join([package, *parts]))
            continue
        if stem.startswith("_"):
            continue
        modules.append(".".join([package, *parts, stem]))
    return sorted(set(modules))


def _is_scanned_exception(value: Any, package: str) -> bool:
    return (
        isinstance(value, type)
        and issubclass(value, BaseException)
        and not issubclass(value, Warning)
        and (value.__module__ == package or value.__module__.startswith(package + "."))
    )


def collect_public_exceptions(
    package: str,
) -> tuple[dict[str, type], dict[str, str], int]:
    """Import a package's public modules and collect its exception classes.

    Args:
        package: Importable package name.

    Returns:
        A ``(classes, import_failures, modules_scanned)`` tuple. ``classes``
        maps qualified names to classes exposed under a public module
        attribute; ``import_failures`` maps module names to the exception type
        that stopped their import.
    """

    root_module = importlib.import_module(package)
    package_dir = Path(next(iter(root_module.__path__)))
    names = public_modules(package, package_dir)
    classes: dict[str, type] = {}
    failures: dict[str, str] = {}
    for module_name in names:
        try:
            module = importlib.import_module(module_name)
        except Exception as error:  # noqa: BLE001 - recorded by type only
            failures[module_name] = type(error).__name__
            continue
        for attribute, value in sorted(vars(module).items()):
            if attribute.startswith("_") or not _is_scanned_exception(value, package):
                continue
            classes.setdefault(qualified_name(value), value)
    return classes, failures, len(names)


def legacy_root_violations(error_class: type) -> list[str]:
    """Return invariant violations for a ``LegacyRootedError`` subclass.

    Args:
        error_class: A subclass of ``LegacyRootedError``.

    Returns:
        Machine-readable reasons; empty when the class is compliant.
    """

    from openmed.core.errors import LegacyRootedError, OpenMedError

    violations: list[str] = []
    mro = error_class.__mro__
    position = mro.index(LegacyRootedError)
    if mro[position + 1 : position + 3] != (OpenMedError, Exception):
        violations.append("roots_not_directly_before_exception")
    code = getattr(error_class, "code", None)
    if not isinstance(code, str) or not _STABLE_CODE.fullmatch(code):
        violations.append("invalid_code")
    return violations


def scan(package: str = "openmed") -> ScanReport:
    """Scan a package and report unrooted and legacy-rooted exceptions.

    Args:
        package: Importable package name.

    Returns:
        The scan report.
    """

    from openmed.core.errors import LegacyRootedError, OpenMedError, legacy_error_code

    classes, failures, scanned = collect_public_exceptions(package)
    unrooted = sorted(
        name
        for name, error_class in classes.items()
        if not issubclass(error_class, OpenMedError)
    )
    legacy: dict[str, dict[str, Any]] = {}
    for name, error_class in classes.items():
        if not issubclass(error_class, LegacyRootedError):
            continue
        try:
            code: Any = legacy_error_code(error_class)
        except TypeError:
            code = None
        legacy[name] = {
            "code": code,
            "violations": legacy_root_violations(error_class),
        }
    imported = frozenset(
        name
        for name in sys.modules
        if name == package or name.startswith(package + ".")
    )
    return ScanReport(
        package=package,
        modules_scanned=scanned,
        import_failures=failures,
        imported_modules=imported,
        unrooted=tuple(unrooted),
        legacy_rooted=legacy,
    )


def read_allowlist(path: Path) -> list[str]:
    """Read allowlist entries, ignoring blank lines and ``#`` comments.

    Args:
        path: Allowlist file.

    Returns:
        Entries in file order.
    """

    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            entries.append(stripped)
    return entries


def evaluate(report: ScanReport, allowlist: Sequence[str]) -> GateResult:
    """Compare a scan with allowlist entries.

    Args:
        report: Scan of the package.
        allowlist: Allowlist entries in file order.

    Returns:
        The gate result.
    """

    allowed = set(allowlist)
    unrooted = set(report.unrooted)
    counts: dict[str, int] = {}
    for entry in allowlist:
        counts[entry] = counts.get(entry, 0) + 1
    duplicates = sorted(entry for entry, count in counts.items() if count > 1)
    stale: list[str] = []
    unverified: list[str] = []
    for entry in sorted(allowed - unrooted):
        module = entry.partition(":")[0]
        if module in report.imported_modules:
            stale.append(entry)
        else:
            unverified.append(entry)
    violations = [
        f"{name}: {reason}"
        for name, entry in sorted(report.legacy_rooted.items())
        for reason in entry["violations"]
    ]
    return GateResult(
        new=tuple(sorted(unrooted - allowed)),
        stale=tuple(stale),
        unverified=tuple(unverified),
        violations=tuple(violations),
        duplicates=tuple(duplicates),
        unsorted=list(allowlist) != sorted(allowlist),
    )


def prune_allowlist(path: Path, stale: Iterable[str]) -> int:
    """Remove stale entries from the allowlist file, keeping its comments.

    Args:
        path: Allowlist file.
        stale: Entries verified as stale by :func:`evaluate`.

    Returns:
        Number of removed lines.
    """

    remove = set(stale)
    kept: list[str] = []
    removed = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() in remove:
            removed += 1
            continue
        kept.append(line)
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return removed


def _mro_names(error_class: type) -> list[str]:
    from openmed.core.errors import LegacyRootedError, OpenMedError

    return [
        qualified_name(klass)
        for klass in error_class.__mro__
        if klass not in (LegacyRootedError, OpenMedError, object)
    ]


def _observe(error_class: type, call: tuple[str, ...]) -> dict[str, Any]:
    try:
        error = error_class(*call)
    except Exception as failure:  # noqa: BLE001 - recorded by type only
        return {"raises": type(failure).__name__}
    observed: dict[str, Any] = {"args": repr(error.args)}
    try:
        observed["str"] = str(error)
    except Exception as failure:  # noqa: BLE001
        observed["str_raises"] = type(failure).__name__
    try:
        restored = pickle.loads(pickle.dumps(error))
    except Exception as failure:  # noqa: BLE001
        observed["pickle_raises"] = type(failure).__name__
    else:
        observed["pickle_type"] = qualified_name(type(restored))
        observed["pickle_args"] = repr(restored.args)
        try:
            observed["pickle_str"] = str(restored)
        except Exception as failure:  # noqa: BLE001
            observed["pickle_str_raises"] = type(failure).__name__
    return observed


def behavior_snapshot(error_class: type) -> dict[str, Any]:
    """Capture the observable behavior a rerooting must preserve.

    The method resolution order is recorded without the two OpenMed roots, so
    the snapshot of a class is identical before and after it is rerooted when
    nothing else changed.

    Args:
        error_class: Exception class to observe.

    Returns:
        A JSON-ready snapshot with the MRO, stable code and the results of the
        synthetic :data:`SAMPLE_CALLS`.
    """

    from openmed.core.errors import legacy_error_code

    samples = []
    for index, call in enumerate(SAMPLE_CALLS):
        first = _observe(error_class, call)
        second = _observe(error_class, call)
        if first != second:
            outcome = "raises" if "raises" in first else "constructs"
            first = {"nondeterministic": outcome}
        samples.append({"call": index, **first})
    return {
        "mro": _mro_names(error_class),
        "code": legacy_error_code(error_class),
        "samples": samples,
    }


def load_class(name: str) -> type:
    """Import a class from its ``module:QualName`` identifier.

    Args:
        name: Qualified name as reported by the gate.

    Returns:
        The class object.

    Raises:
        TypeError: If the name does not resolve to a class.
    """

    module_name, _, qualname = name.partition(":")
    value: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        value = getattr(value, part)
    if not isinstance(value, type):
        raise TypeError(f"{name} is not a class")
    return value


def replay_snapshot(name: str, recorded: Mapping[str, Any]) -> list[str]:
    """Compare a rerooted class with the snapshot recorded before rerooting.

    Args:
        name: Qualified class name.
        recorded: Snapshot entry produced by :func:`record_snapshots`.

    Returns:
        Machine-readable mismatches; empty when the class is rooted, meets the
        root invariants and behaves exactly as recorded.
    """

    from openmed.core.errors import LegacyRootedError

    error_class = load_class(name)
    if not issubclass(error_class, LegacyRootedError):
        return ["not_rooted"]
    mismatches = legacy_root_violations(error_class)
    current = behavior_snapshot(error_class)
    for key in ("mro", "code", "samples"):
        if current[key] != recorded.get(key):
            mismatches.append(f"{key}_changed")
    if getattr(error_class, "code", None) != recorded.get("code"):
        mismatches.append("class_code_changed")
    return mismatches


def record_snapshots(
    prefixes: Sequence[str], package: str = "openmed"
) -> dict[str, Any]:
    """Record behavior snapshots of unrooted exceptions under module prefixes.

    Run this on the tree *before* rerooting, commit the output, then reroot.

    Args:
        prefixes: Module prefixes such as ``"openmed.multimodal"``.
        package: Package to scan.

    Returns:
        A JSON-ready snapshot document keyed by qualified class name.
    """

    from openmed.core.errors import OpenMedError

    classes, _, _ = collect_public_exceptions(package)
    selected = {}
    for name, error_class in sorted(classes.items()):
        module = name.partition(":")[0]
        if issubclass(error_class, OpenMedError):
            continue
        if any(
            module == prefix or module.startswith(prefix + ".") for prefix in prefixes
        ):
            selected[name] = behavior_snapshot(error_class)
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "sample_calls": [list(call) for call in SAMPLE_CALLS],
        "classes": selected,
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--package", default="openmed")
    parser.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    parser.add_argument(
        "--json-output", type=Path, help="Write the scan report as JSON here."
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Delete verified-stale entries from the allowlist.",
    )
    parser.add_argument(
        "--record",
        action="append",
        default=[],
        metavar="MODULE_PREFIX",
        help="Record behavior snapshots for unrooted classes under this prefix.",
    )
    parser.add_argument("--record-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the gate from the command line.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status: ``0`` when the gate passes, ``1`` otherwise.
    """

    args = _parse_args(argv)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if args.record:
        if args.record_output is None:
            raise SystemExit("--record requires --record-output")
        document = record_snapshots(args.record, package=args.package)
        args.record_output.parent.mkdir(parents=True, exist_ok=True)
        args.record_output.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"recorded {len(document['classes'])} exception classes")
        return 0

    report = scan(args.package)
    if args.json_output is not None:
        args.json_output.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    result = evaluate(report, read_allowlist(args.allowlist))
    if args.prune and result.stale:
        removed = prune_allowlist(args.allowlist, result.stale)
        print(f"removed {removed} stale allowlist entries")
        result = evaluate(report, read_allowlist(args.allowlist))
    print(result.describe())
    if report.import_failures:
        print(f"skipped {len(report.import_failures)} modules that did not import")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
