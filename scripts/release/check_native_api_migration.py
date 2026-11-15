#!/usr/bin/env python3
"""Require a migration entry when an OpenMedKit native API baseline breaks.

OpenMedKit commits two kinds of public API baseline:

* Swift: ``swift/OpenMedKit/api/<platform>/<Module>.json``, written by
  ``scripts/release/swift_api_baseline.py`` from ``swift-api-digester`` dumps.
* Kotlin: ``android/openmedkit/api/openmedkit.api``, written by the Kotlin
  binary-compatibility validator (``./gradlew :openmedkit:apiDump``).

The platform checks (``swift_api_baseline.py --check`` and ``apiCheck``) compare
the code with its baseline. This gate compares the committed baselines of
two Git references. A declaration that disappears or whose recorded signature
changes is a breaking change. Every breaking symbol must be named in a
``docs/migration/*.md`` file that the same change adds or modifies, mirroring
the Python ``api_surface_diff.py --check`` migration gate. Additions never need
a migration entry.

The script is stdlib-only, reads files through Git, and prints only symbol
names, counts and repository-relative paths.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
WORKTREE_REF = "WORKTREE"
SWIFT_BASELINE_DIR = PurePosixPath("swift/OpenMedKit/api")
KOTLIN_BASELINE_DIRS = (PurePosixPath("android/openmedkit/api"),)
MIGRATION_DIR = PurePosixPath("docs/migration")
SWIFT_SCHEMA_VERSION = 1

PLATFORMS = ("swift", "kotlin")

# BCV member lines look like ``public final fun name (desc)ret`` or
# ``public static final field NAME Ltype;``.
_KOTLIN_MEMBER = re.compile(
    r"^(?P<prefix>.*?\b(?:fun|field))\s+(?P<name>\S+)(?P<rest>.*)$"
)
# BCV writes every classifier, interfaces included, as ``... class <name>``.
_KOTLIN_CLASS = re.compile(
    r"^(?P<header>.*?\bclass\s+(?P<name>\S+))"
    r"(?:\s*:\s*(?P<supertypes>.+?))?\s*\{$"
)


class NativeApiError(RuntimeError):
    """Raised when a baseline or Git reference cannot be read."""


@dataclass(frozen=True)
class BaselineChange:
    """One symbol whose recorded declarations differ between two baselines."""

    platform: str
    baseline: str
    symbol: str
    change: str  # "removed" | "changed" | "added"
    before: tuple[str, ...] = ()
    after: tuple[str, ...] = ()

    @property
    def breaking(self) -> bool:
        """Return whether the change removes or alters a recorded declaration."""

        return self.change in {"removed", "changed"}


SymbolTable = dict[str, frozenset[str]]


def _add(table: dict[str, set[str]], symbol: str, line: str) -> None:
    table.setdefault(symbol, set()).add(line)


def _freeze(table: Mapping[str, set[str]]) -> SymbolTable:
    return {symbol: frozenset(lines) for symbol, lines in table.items()}


def parse_swift_baseline(text: str) -> SymbolTable:
    """Parse a normalized Swift baseline document into ``symbol -> lines``.

    Args:
        text: JSON text written by ``swift_api_baseline.py``.

    Returns:
        Mapping from module-qualified symbol names to recorded declarations.

    Raises:
        NativeApiError: If the document is not a supported Swift baseline.
    """

    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise NativeApiError(f"invalid Swift API baseline JSON: {exc.msg}") from exc
    if not isinstance(document, dict):
        raise NativeApiError("Swift API baseline must be a JSON object")
    if document.get("schema_version") != SWIFT_SCHEMA_VERSION:
        raise NativeApiError("unsupported Swift API baseline schema_version")
    symbols = document.get("symbols")
    if not isinstance(symbols, dict):
        raise NativeApiError("Swift API baseline is missing a symbols object")
    table: dict[str, set[str]] = {}
    for symbol, lines in symbols.items():
        if not isinstance(symbol, str) or not isinstance(lines, list):
            raise NativeApiError("Swift API baseline symbols must map to lists")
        for line in lines:
            if not isinstance(line, str):
                raise NativeApiError("Swift API baseline declarations must be text")
            _add(table, symbol, line)
    return _freeze(table)


def _kotlin_symbol(jvm_name: str) -> str:
    return jvm_name.replace("/", ".").replace("$", ".")


def _kotlin_member_name(name: str) -> str:
    # ``foo$default`` carries Kotlin default arguments for ``foo``.
    return name.removesuffix("$default")


def parse_kotlin_api_dump(text: str) -> SymbolTable:
    """Parse a binary-compatibility-validator ``.api`` dump.

    Class headers are keyed by the dotted class name, with each supertype
    recorded as its own line so that adding a supertype is an addition. Members
    are keyed by ``<class>.<member>``; synthetic ``$default`` bridges share the
    key of the function they belong to.

    Args:
        text: Contents of an ``.api`` file.

    Returns:
        Mapping from dotted symbol names to recorded declaration lines.

    Raises:
        NativeApiError: If a member appears outside a class block.
    """

    table: dict[str, set[str]] = {}
    current: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "}":
            current = None
            continue
        if raw_line[:1] not in {"\t", " "}:
            class_match = _KOTLIN_CLASS.match(line)
            if class_match is None or current is not None:
                raise NativeApiError("Kotlin API dump has an unrecognized class header")
            current = _kotlin_symbol(class_match.group("name"))
            _add(table, current, class_match.group("header").strip())
            supertypes = class_match.group("supertypes")
            if supertypes:
                for supertype in supertypes.split(","):
                    _add(table, current, f"supertype {supertype.strip()}")
            continue
        if current is None:
            raise NativeApiError("Kotlin API dump has a member outside a class")
        member_match = _KOTLIN_MEMBER.match(line)
        name = (
            _kotlin_member_name(member_match.group("name"))
            if member_match is not None
            else line
        )
        _add(table, f"{current}.{name}", line)
    return _freeze(table)


def diff_tables(
    before: SymbolTable,
    after: SymbolTable,
    *,
    platform: str,
    baseline: str,
) -> list[BaselineChange]:
    """Classify the symbol differences between two parsed baselines.

    A symbol is ``removed`` when it has no declarations left, ``changed`` when
    at least one recorded declaration no longer appears, and ``added`` when it
    only gains declarations (a new symbol, overload, conformance or supertype).

    Args:
        before: Baseline at the base reference.
        after: Baseline at the head reference.
        platform: ``swift`` or ``kotlin``.
        baseline: Repository-relative baseline path used in reports.

    Returns:
        Changes sorted by symbol name.
    """

    changes: list[BaselineChange] = []
    for symbol in sorted(set(before) | set(after)):
        old = before.get(symbol, frozenset())
        new = after.get(symbol, frozenset())
        if old == new:
            continue
        missing = old - new
        if not new:
            kind = "removed"
        elif missing:
            kind = "changed"
        else:
            kind = "added"
        changes.append(
            BaselineChange(
                platform=platform,
                baseline=baseline,
                symbol=symbol,
                change=kind,
                before=tuple(sorted(missing)),
                after=tuple(sorted(new - old)),
            )
        )
    return changes


def _run_git(repo_root: Path, args: Sequence[str]) -> str:
    command = ["git", "-C", str(repo_root), *args]
    result = subprocess.run(command, capture_output=True, check=False, text=True)
    if result.returncode != 0:
        raise NativeApiError(f"git {' '.join(args[:2])} failed")
    return result.stdout


def _ref_files(repo_root: Path, ref: str, directory: PurePosixPath) -> list[str]:
    """List files below ``directory`` at ``ref``, recursively."""

    if ref == WORKTREE_REF:
        base = repo_root / directory
        if not base.is_dir():
            return []
        return sorted(
            path.relative_to(repo_root).as_posix()
            for path in base.rglob("*")
            if path.is_file()
        )
    output = _run_git(
        repo_root, ["ls-tree", "-r", "--name-only", ref, "--", f"{directory}/"]
    )
    return sorted(line for line in output.splitlines() if line)


def _read(repo_root: Path, ref: str, path: str) -> str:
    if ref == WORKTREE_REF:
        try:
            return (repo_root / path).read_text(encoding="utf-8")
        except OSError as exc:
            raise NativeApiError(f"could not read {path}") from exc
    return _run_git(repo_root, ["show", f"{ref}:{path}"])


def _baseline_paths(repo_root: Path, ref: str, platform: str) -> set[str]:
    if platform == "swift":
        directories: Iterable[PurePosixPath] = (SWIFT_BASELINE_DIR,)
        suffix = ".json"
    else:
        directories = KOTLIN_BASELINE_DIRS
        suffix = ".api"
    return {
        path
        for directory in directories
        for path in _ref_files(repo_root, ref, directory)
        if path.endswith(suffix)
    }


def parse_baseline(platform: str, text: str) -> SymbolTable:
    """Parse ``text`` with the parser for ``platform``."""

    if platform == "swift":
        return parse_swift_baseline(text)
    if platform == "kotlin":
        return parse_kotlin_api_dump(text)
    raise NativeApiError(f"unknown platform {platform!r}")


def compare_refs(
    repo_root: Path,
    base_ref: str,
    head_ref: str,
    platforms: Sequence[str] = PLATFORMS,
) -> list[BaselineChange]:
    """Compare every committed native API baseline between two references.

    Args:
        repo_root: Repository root.
        base_ref: Git reference holding the previous baselines.
        head_ref: Git reference, or ``WORKTREE``, holding the new baselines.
        platforms: Platforms to compare.

    Returns:
        Every added, changed and removed symbol across the baselines.
    """

    changes: list[BaselineChange] = []
    for platform in platforms:
        before_paths = _baseline_paths(repo_root, base_ref, platform)
        after_paths = _baseline_paths(repo_root, head_ref, platform)
        for path in sorted(set(before_paths) | set(after_paths)):
            before = (
                parse_baseline(platform, _read(repo_root, base_ref, path))
                if path in before_paths
                else {}
            )
            after = (
                parse_baseline(platform, _read(repo_root, head_ref, path))
                if path in after_paths
                else {}
            )
            changes.extend(diff_tables(before, after, platform=platform, baseline=path))
    return changes


def _merge_base(repo_root: Path, base_ref: str, head_ref: str) -> str:
    target = "HEAD" if head_ref == WORKTREE_REF else head_ref
    try:
        output = _run_git(repo_root, ["merge-base", base_ref, target]).strip()
    except NativeApiError:
        return base_ref
    return output or base_ref


def changed_migration_documents(
    repo_root: Path, base_ref: str, head_ref: str
) -> dict[str, str]:
    """Return ``path -> text`` for migration guides added or modified.

    Args:
        repo_root: Repository root.
        base_ref: Base Git reference.
        head_ref: Head Git reference, or ``WORKTREE``.

    Returns:
        Markdown migration documents changed between the two references.
    """

    pathspec = f"{MIGRATION_DIR}/"
    if head_ref == WORKTREE_REF:
        output = _run_git(
            repo_root,
            ["diff", "--name-only", "--diff-filter=AMR", base_ref, "--", pathspec],
        )
        untracked = _run_git(
            repo_root, ["ls-files", "--others", "--exclude-standard", "--", pathspec]
        )
        paths = {*output.splitlines(), *untracked.splitlines()}
    else:
        output = _run_git(
            repo_root,
            [
                "diff",
                "--name-only",
                "--diff-filter=AMR",
                base_ref,
                head_ref,
                "--",
                pathspec,
            ],
        )
        paths = set(output.splitlines())
    return {
        path: _read(repo_root, head_ref, path)
        for path in sorted(paths)
        if path.endswith(".md")
    }


def symbol_pattern(symbol: str) -> re.Pattern[str]:
    """Return the boundary-aware pattern used to find ``symbol`` in prose."""

    # A trailing dot only disqualifies the match when it continues the name,
    # so ``OpenMedKit.OpenMed.`` at the end of a sentence still counts.
    return re.compile(rf"(?<![A-Za-z0-9_.$]){re.escape(symbol)}(?!\.?[A-Za-z0-9_$])")


def missing_migration_symbols(
    changes: Sequence[BaselineChange], documents: Mapping[str, str]
) -> tuple[str, ...]:
    """Return breaking symbols that no changed migration document names.

    A symbol counts as documented when its full dotted name appears with
    identifier boundaries, optionally followed by a Swift argument-label list,
    for example ``OpenMedKit.OpenMed.analyzeText(_:confidenceThreshold:)``.
    """

    corpus = "\n".join(documents.values())
    missing = {
        change.symbol
        for change in changes
        if change.breaking and symbol_pattern(change.symbol).search(corpus) is None
    }
    return tuple(sorted(missing))


def render_report(changes: Sequence[BaselineChange]) -> str:
    """Render a deterministic summary of baseline changes."""

    breaking = [change for change in changes if change.breaking]
    added = [change for change in changes if not change.breaking]
    lines = [
        f"Native API baselines: {len(breaking)} breaking symbol(s), "
        f"{len(added)} added symbol(s)."
    ]
    for change in breaking:
        lines.append(f"  - {change.change}: {change.symbol} ({change.baseline})")
        for declaration in change.before:
            lines.append(f"      was: {declaration}")
        for declaration in change.after:
            lines.append(f"      now: {declaration}")
    return "\n".join(lines)


def run_gate(
    repo_root: Path,
    base_ref: str,
    head_ref: str,
    platforms: Sequence[str] = PLATFORMS,
) -> int:
    """Run the migration gate and return a process exit code."""

    merge_base = _merge_base(repo_root, base_ref, head_ref)
    changes = compare_refs(repo_root, merge_base, head_ref, platforms)
    print(render_report(changes), flush=True)
    breaking = [change for change in changes if change.breaking]
    if not breaking:
        print("No public native declaration was removed or changed.")
        return 0
    documents = changed_migration_documents(repo_root, merge_base, head_ref)
    if not documents:
        print(
            f"\n{len(breaking)} public native declaration(s) were removed or "
            f"changed, but no {MIGRATION_DIR}/*.md file was added or modified.",
            file=sys.stderr,
        )
        print(
            "Add a migration entry that names each symbol listed above.",
            file=sys.stderr,
        )
        return 1
    missing = missing_migration_symbols(breaking, documents)
    if missing:
        print(
            "\nThe changed migration guide(s) do not name these breaking symbols:",
            file=sys.stderr,
        )
        for symbol in missing:
            print(f"  - {symbol}", file=sys.stderr)
        print("Checked: " + ", ".join(sorted(documents)), file=sys.stderr)
        return 1
    print(
        "Migration entry names every breaking symbol: " + ", ".join(sorted(documents))
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--base",
        required=True,
        help="Base Git ref, for example origin/master or HEAD^1 on a PR merge",
    )
    parser.add_argument(
        "--head",
        default="HEAD",
        help=f"Head Git ref, or {WORKTREE_REF} for uncommitted files (default HEAD)",
    )
    parser.add_argument(
        "--platform",
        choices=(*PLATFORMS, "all"),
        default="all",
        help="Baselines to compare (default all)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=ROOT,
        help="Repository root (default: this checkout)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the native API migration gate from the command line."""

    args = build_parser().parse_args(argv)
    platforms = PLATFORMS if args.platform == "all" else (args.platform,)
    try:
        return run_gate(args.repo_root, args.base, args.head, platforms)
    except NativeApiError as exc:
        print(f"check-native-api-migration: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
