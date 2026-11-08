#!/usr/bin/env python3
"""Build OpenMedKit with complete Swift concurrency checking and fail on diagnostics.

The package keeps the Swift 5 language mode for consumers. This lane compiles
every first-party OpenMedKit target with ``-strict-concurrency=complete`` (the
Swift 6 data-race checks reported as warnings) and fails when any warning or
error points into ``swift/OpenMedKit/Sources`` or ``swift/OpenMedKit/Tests``.
Diagnostics in resolved dependencies are not first-party and are ignored.

Platforms:

* ``macos`` builds the consumer-facing root package, tests included, with
  ``swift build --build-tests -Xswiftc -strict-concurrency=complete``.
* ``ios`` builds the library and extension targets of the drift-checked nested
  package for the iOS Simulator with ``SWIFT_STRICT_CONCURRENCY=complete``.

Each build uses a fresh scratch or DerivedData directory by default, because an
incremental build only re-reports diagnostics for files it recompiles. Pass
``--scratch-path`` to reuse a directory locally when that caveat is acceptable.

The raw compiler log goes to ``<log dir>/<platform>.log`` (``--log-dir``, or a
temporary directory removed afterwards). The report lists repository-relative
locations and compiler messages only, never absolute paths; a failed build with
no first-party diagnostic prints the log tail with those paths made relative.

Usage::

    python3 scripts/release/check_swift_strict_concurrency.py --platform macos
    python3 scripts/release/check_swift_strict_concurrency.py --platform ios
    python3 scripts/release/check_swift_strict_concurrency.py --from-log build.log

Exit codes: 0 when first-party code builds with zero diagnostics, 1 when a
first-party diagnostic is reported, 2 when a build fails without one or the
toolchain is unavailable.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import IO, Any

ROOT = Path(__file__).resolve().parents[2]
NESTED_PACKAGE = PurePosixPath("swift/OpenMedKit")
FIRST_PARTY_PREFIXES = (
    NESTED_PACKAGE / "Sources",
    NESTED_PACKAGE / "Tests",
)
PLATFORMS = ("macos", "ios")
IOS_SCHEME = "OpenMedKit-Package"
IOS_DESTINATION = "generic/platform=iOS Simulator"
FAILURE_TAIL_LINES = 40

_DIAGNOSTIC_RE = re.compile(
    r"^(?P<path>/[^:\n]+\.swift):(?P<line>\d+):(?P<column>\d+): "
    r"(?P<severity>warning|error): (?P<message>.*)$"
)

Runner = Callable[..., subprocess.CompletedProcess[Any]]


@dataclass(frozen=True, order=True)
class Diagnostic:
    """One first-party compiler diagnostic.

    Attributes:
        path: Repository-relative POSIX path of the Swift file.
        line: One-based line number.
        column: One-based column number.
        severity: ``warning`` or ``error``.
        message: Compiler message with repository paths made relative.
    """

    path: str
    line: int
    column: int
    severity: str
    message: str

    def render(self) -> str:
        """Return the diagnostic as ``path:line:column: severity: message``."""
        return f"{self.path}:{self.line}:{self.column}: {self.severity}: {self.message}"


def _relative_to_root(path: str, roots: Sequence[str]) -> PurePosixPath | None:
    resolved = os.path.realpath(path)
    for root in roots:
        if resolved == root or resolved.startswith(root + os.sep):
            return PurePosixPath(Path(resolved).relative_to(root).as_posix())
    return None


def _root_spellings(root: Path) -> list[str]:
    spellings = {str(root), os.path.realpath(root)}
    return sorted(spellings, key=len, reverse=True)


def parse_diagnostics(lines: Iterable[str], *, root: Path = ROOT) -> list[Diagnostic]:
    """Return unique first-party warnings and errors from a compiler log.

    Notes, remarks, non-Swift diagnostics and diagnostics in dependency
    checkouts are ignored. Repeated diagnostics (SwiftPM and xcodebuild print
    some twice) are collapsed.

    Args:
        lines: Compiler output lines.
        root: Repository root that first-party paths are relative to.

    Returns:
        Sorted, de-duplicated first-party diagnostics.
    """
    roots = _root_spellings(root)
    found: set[Diagnostic] = set()
    for raw in lines:
        match = _DIAGNOSTIC_RE.match(raw.rstrip("\r\n"))
        if match is None:
            continue
        relative = _relative_to_root(match["path"], roots)
        if relative is None or not any(
            relative.is_relative_to(prefix) for prefix in FIRST_PARTY_PREFIXES
        ):
            continue
        message = match["message"]
        for spelling in roots:
            message = message.replace(spelling + os.sep, "")
        found.add(
            Diagnostic(
                path=relative.as_posix(),
                line=int(match["line"]),
                column=int(match["column"]),
                severity=match["severity"],
                message=message,
            )
        )
    return sorted(found)


def build_command(platform: str, work_dir: Path) -> tuple[list[str], Path]:
    """Return the build command and working directory for one platform.

    Args:
        platform: ``macos`` or ``ios``.
        work_dir: Fresh scratch (macOS) or DerivedData (iOS) directory.

    Returns:
        The argument vector and the directory to run it from.

    Raises:
        ValueError: If the platform is not supported.
    """
    if platform == "macos":
        return (
            [
                "swift",
                "build",
                "--package-path",
                str(ROOT),
                "--scratch-path",
                str(work_dir),
                "--build-tests",
                "-Xswiftc",
                "-strict-concurrency=complete",
            ],
            ROOT,
        )
    if platform == "ios":
        return (
            [
                "xcodebuild",
                "build",
                "-scheme",
                IOS_SCHEME,
                "-destination",
                IOS_DESTINATION,
                "-derivedDataPath",
                str(work_dir),
                "-skipPackagePluginValidation",
                "SWIFT_STRICT_CONCURRENCY=complete",
            ],
            ROOT / NESTED_PACKAGE,
        )
    raise ValueError(f"unsupported_platform platform={platform}")


def _run_build(
    platform: str,
    work_dir: Path,
    log_path: Path,
    runner: Runner,
) -> int | None:
    command, cwd = build_command(platform, work_dir)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        try:
            completed = runner(
                command,
                cwd=cwd,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        except FileNotFoundError:
            return None
    return int(completed.returncode)


def _log_tail(log_path: Path, *, hidden: Sequence[Path]) -> list[str]:
    with log_path.open(encoding="utf-8", errors="replace") as log:
        lines = [line.rstrip("\r\n") for line in log][-FAILURE_TAIL_LINES:]
    spellings = sorted(
        {spelling for path in hidden for spelling in _root_spellings(path)},
        key=len,
        reverse=True,
    )
    tail = []
    for line in lines:
        for spelling in spellings:
            line = line.replace(spelling + os.sep, "").replace(spelling, ".")
        tail.append(line)
    return tail


def _report(platform: str, diagnostics: Sequence[Diagnostic], out: IO[str]) -> None:
    if not diagnostics:
        print(f"{platform}: 0 first-party concurrency diagnostics", file=out)
        return
    print(
        f"{platform}: {len(diagnostics)} first-party diagnostic(s) under "
        "complete concurrency checking:",
        file=out,
    )
    for diagnostic in diagnostics:
        print(f"  {diagnostic.render()}", file=out)


def check_platform(
    platform: str,
    *,
    work_dir: Path,
    log_dir: Path,
    runner: Runner = subprocess.run,
    out: IO[str] = sys.stdout,
) -> int:
    """Build one platform and report first-party diagnostics.

    Args:
        platform: ``macos`` or ``ios``.
        work_dir: Scratch or DerivedData directory for the build.
        log_dir: Directory that receives ``<platform>.log``.
        runner: ``subprocess.run`` compatible callable (injected in tests).
        out: Stream for the report.

    Returns:
        The exit code for this platform (0, 1 or 2).
    """
    log_path = log_dir / f"{platform}.log"
    print(f"{platform}: building with complete concurrency checking", file=out)
    returncode = _run_build(platform, work_dir, log_path, runner)
    if returncode is None:
        tool = "swift" if platform == "macos" else "xcodebuild"
        print(f"{platform}: toolchain_unavailable tool={tool}", file=out)
        return 2
    with log_path.open(encoding="utf-8", errors="replace") as log:
        diagnostics = parse_diagnostics(log)
    _report(platform, diagnostics, out)
    if diagnostics:
        return 1
    if returncode != 0:
        print(
            f"{platform}: build_failed exit={returncode} (no first-party diagnostic); "
            f"last {FAILURE_TAIL_LINES} log lines:",
            file=out,
        )
        for line in _log_tail(log_path, hidden=(ROOT, work_dir)):
            print(f"  {line}", file=out)
        return 2
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--platform",
        action="append",
        choices=PLATFORMS,
        help="Platform to build; repeat for several.",
    )
    source.add_argument(
        "--from-log",
        type=Path,
        help="Check an existing compiler log instead of building.",
    )
    parser.add_argument(
        "--scratch-path",
        type=Path,
        help="Reuse this build directory instead of a fresh temporary one.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Keep compiler logs here (default: the temporary work directory).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None, *, runner: Runner = subprocess.run) -> int:
    """Run the strict-concurrency check.

    Args:
        argv: Command-line arguments (``sys.argv[1:]`` when omitted).
        runner: ``subprocess.run`` compatible callable (injected in tests).

    Returns:
        The process exit code.
    """
    args = _parse_args(argv)
    if args.from_log is not None:
        with args.from_log.open(encoding="utf-8", errors="replace") as log:
            diagnostics = parse_diagnostics(log)
        _report("log", diagnostics, sys.stdout)
        return 1 if diagnostics else 0

    base = Path(os.environ.get("RUNNER_TEMP") or tempfile.gettempdir())
    work_root = Path(tempfile.mkdtemp(prefix="openmed-strict-concurrency.", dir=base))
    log_dir = args.log_dir or work_root
    status = 0
    try:
        for platform in dict.fromkeys(args.platform):
            work_dir = args.scratch_path or (work_root / platform)
            status = max(
                status,
                check_platform(
                    platform, work_dir=work_dir, log_dir=log_dir, runner=runner
                ),
            )
    finally:
        # Only the temporary root is removed; --scratch-path and --log-dir stay.
        shutil.rmtree(work_root, ignore_errors=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
