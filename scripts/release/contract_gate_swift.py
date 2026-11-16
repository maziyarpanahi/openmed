#!/usr/bin/env python3
"""Run the OpenMedKit Swift contract-gate consumers and check ``swift.json``.

The Swift reporter is an XCTest helper,
``swift/OpenMedKit/Tests/OpenMedKitTests/ContractGateReporter.swift``. Swift
consumer test classes subclass ``ContractGateTestCase`` and read their fixture
through ``ContractGateReporter.read``; when ``OPENMED_CONTRACT_GATE_RESULTS_DIR``
is set, the reporter writes ``swift.json`` into that directory. This script
derives the XCTest filter from the Swift consumer files that
``tests/fixtures/parity/contract_gate.json`` declares, runs ``swift test`` once
with the results directory set, and checks the written file against the
contract-gate result schema. ``docs/contract-gate.md`` documents the reporter.

Usage::

    python scripts/release/contract_gate_swift.py --results-dir results
    python scripts/release/contract_gate_swift.py --print-filter

``swift test`` output is discarded so the gate never echoes fixture values; run
the consumer classes with ``swift test --filter`` directly to see assertion
details. The script makes no network call beyond what ``swift test`` needs to
resolve the package.

Exit codes: 0 once a schema-valid ``swift.json`` is written, whatever the case
outcomes; 1 with ``--check`` when a case did not pass; 2 when no consumer class
is declared, the run wrote no valid result, or on a usage error.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any


def _load_gate() -> ModuleType:
    """Load the sibling ``contract_gate.py`` without changing ``sys.path``."""
    loaded = sys.modules.get("contract_gate")
    if loaded is not None:
        return loaded
    spec = importlib.util.spec_from_file_location(
        "contract_gate", Path(__file__).resolve().with_name("contract_gate.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate()

SWIFT_RUNTIME = "swift"
RESULT_NAME = f"{SWIFT_RUNTIME}.json"
SWIFT_TESTS_ROOT = PurePosixPath("swift/OpenMedKit/Tests")
_CONSUMER_CLASS_RE = re.compile(
    r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*ContractGateTestCase\b"
)

SwiftRunner = Callable[[Sequence[str], Path, Mapping[str, str]], int]


def consumer_classes(
    root: Path = gate.ROOT, manifest: str = gate.DEFAULT_MANIFEST
) -> list[tuple[str, str]]:
    """List the XCTest classes that consume a required Swift fixture.

    A class counts when it is declared in a Swift consumer file the manifest
    lists for a ``required`` Swift cell and subclasses ``ContractGateTestCase``.

    Args:
        root: Repository root.
        manifest: Repository-relative manifest path.

    Returns:
        Sorted ``(test target, class name)`` pairs. Consumer files that are
        missing are left out; the reporter records them as
        ``consumer_missing``.

    Raises:
        ContractGateError: ``manifest_unreadable`` or a structural manifest
            code, or ``consumer_path_invalid`` for a consumer outside
            ``swift/OpenMedKit/Tests/<target>/``.
    """
    parsed = gate.load_manifest(root, manifest)
    classes: set[tuple[str, str]] = set()
    for entry in parsed.fixtures:
        cell = entry.cells.get(SWIFT_RUNTIME)
        if cell is None or cell.status != gate.STATUS_REQUIRED:
            continue
        for consumer in cell.consumers:
            path = PurePosixPath(consumer)
            if not path.is_relative_to(SWIFT_TESTS_ROOT) or path.suffix != ".swift":
                raise gate.ContractGateError("consumer_path_invalid", consumer)
            parts = path.relative_to(SWIFT_TESTS_ROOT).parts
            if len(parts) < 2:
                raise gate.ContractGateError("consumer_path_invalid", consumer)
            target = root / consumer
            if not target.is_file():
                continue
            source = target.read_text(encoding="utf-8")
            classes.update(
                (parts[0], name) for name in _CONSUMER_CLASS_RE.findall(source)
            )
    return sorted(classes)


def xctest_filter(classes: Sequence[tuple[str, str]]) -> str:
    """Build the ``swift test --filter`` expression for consumer classes.

    Args:
        classes: ``(test target, class name)`` pairs.

    Returns:
        A regular expression matching every test of those classes only.
    """
    by_target: dict[str, list[str]] = {}
    for target, name in classes:
        by_target.setdefault(target, []).append(name)
    alternatives = [
        f"{re.escape(target)}\\.({'|'.join(sorted(names))})"
        for target, names in sorted(by_target.items())
    ]
    return f"^({'|'.join(alternatives)})/"


def swift_test_command(
    filter_expression: str,
    *,
    package_path: Path,
    scratch_path: Path | None = None,
    skip_build: bool = False,
) -> list[str]:
    """Return the ``swift test`` command for the consumer classes.

    Args:
        filter_expression: The ``--filter`` regular expression.
        package_path: Swift package directory.
        scratch_path: Optional SwiftPM scratch directory.
        skip_build: Pass ``--skip-build`` to reuse an earlier build.

    Returns:
        The command as an argument list.
    """
    command = ["swift", "test", "--package-path", str(package_path)]
    if scratch_path is not None:
        command += ["--scratch-path", str(scratch_path)]
    if skip_build:
        command.append("--skip-build")
    return [*command, "--filter", filter_expression]


def run_swift_test(command: Sequence[str], cwd: Path, env: Mapping[str, str]) -> int:
    """Run ``swift test`` with its console output discarded.

    Args:
        command: The ``swift test`` argument list.
        cwd: Working directory.
        env: Process environment, including the results directory.

    Returns:
        The ``swift test`` exit code.
    """
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(env),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode


def run_swift(
    root: Path,
    manifest: str,
    results_dir: Path,
    *,
    package_path: Path | None = None,
    scratch_path: Path | None = None,
    skip_build: bool = False,
    runner: SwiftRunner | None = None,
) -> tuple[dict[str, Any], int]:
    """Run the Swift consumers and return the checked ``swift.json``.

    Args:
        root: Repository root.
        manifest: Repository-relative manifest path.
        results_dir: Directory the reporter writes ``swift.json`` into.
        package_path: Swift package directory. Defaults to the repository root.
        scratch_path: Optional SwiftPM scratch directory.
        skip_build: Reuse an earlier ``swift build --build-tests``.
        runner: Callable ``(command, cwd, env) -> exit code``. Defaults to a
            ``swift test`` subprocess.

    Returns:
        The result document and the ``swift test`` exit code.

    Raises:
        ContractGateError: ``consumer_not_adopted`` when no consumer class
            subclasses ``ContractGateTestCase``, ``result_missing`` or
            ``result_unreadable`` when the run wrote no result, and
            ``invalid_result`` when it fails the schema or names another
            runtime.
    """
    classes = consumer_classes(root, manifest)
    if not classes:
        raise gate.ContractGateError("consumer_not_adopted")
    directory = results_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / RESULT_NAME
    output.unlink(missing_ok=True)
    env = {**os.environ, gate.RESULTS_DIR_ENV: str(directory)}
    command = swift_test_command(
        xctest_filter(classes),
        package_path=package_path or root,
        scratch_path=scratch_path,
        skip_build=skip_build,
    )
    status = (runner or run_swift_test)(command, root, env)
    if not output.is_file():
        raise gate.ContractGateError("result_missing")
    try:
        document = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise gate.ContractGateError("result_unreadable") from error
    findings = gate.validate_result(document, gate.load_result_schema(root, manifest))
    if findings or document.get("runtime") != SWIFT_RUNTIME:
        raise gate.ContractGateError("invalid_result")
    return document, status


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Swift contract-gate command line.

    Args:
        argv: Arguments without the program name. Defaults to ``sys.argv``.

    Returns:
        The process exit code.
    """
    args = _build_parser().parse_args(argv)
    root = Path(args.root).resolve() if args.root else gate.ROOT
    try:
        if args.print_filter:
            print(xctest_filter(consumer_classes(root, args.manifest)))
            return 0
        directory = args.results_dir or os.environ.get(gate.RESULTS_DIR_ENV)
        if not directory:
            print(
                "contract_gate run-swift: pass --results-dir or set "
                f"{gate.RESULTS_DIR_ENV}",
                file=sys.stderr,
            )
            return 2
        result, status = run_swift(
            root,
            args.manifest,
            Path(directory),
            package_path=Path(args.package_path) if args.package_path else None,
            scratch_path=Path(args.scratch_path) if args.scratch_path else None,
            skip_build=args.skip_build,
        )
    except gate.ContractGateError as error:
        print(
            f"contract_gate run-swift: {error.code} path={error.field_path}",
            file=sys.stderr,
        )
        return 2
    incomplete = 0
    for item in result["fixtures"]:
        print(
            "contract_gate run-swift: "
            f"fixture={item['fixture_id']} passed={len(item['passed'])} "
            f"failed={len(item['failed'])} skipped={len(item['skipped'])}"
        )
        if item["failed"] or item["skipped"] or item.get("errors"):
            incomplete += 1
    print(f"contract_gate run-swift: swift_test_exit={status} wrote {RESULT_NAME}")
    return 1 if args.check and incomplete else 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="contract_gate_swift.py",
        description="Run the Swift contract-gate consumers and write swift.json.",
    )
    parser.add_argument(
        "--manifest",
        default=gate.DEFAULT_MANIFEST,
        help="Repository-relative manifest path.",
    )
    parser.add_argument("--root", help=argparse.SUPPRESS)
    parser.add_argument(
        "--results-dir",
        help=f"Directory for swift.json (default: ${gate.RESULTS_DIR_ENV}).",
    )
    parser.add_argument(
        "--package-path", help="Swift package directory (default: repository root)."
    )
    parser.add_argument("--scratch-path", help="SwiftPM scratch directory.")
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Reuse an earlier swift build --build-tests.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 when any case did not pass.",
    )
    parser.add_argument(
        "--print-filter",
        action="store_true",
        help="Print the swift test --filter expression and exit.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
