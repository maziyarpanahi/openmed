#!/usr/bin/env python3
"""Verify 16 KB ELF LOAD-segment alignment of Android native libraries.

Android 15+ devices that run with 16 KB memory pages can only map a shared
library whose ``PT_LOAD`` program headers are aligned to at least 16 KB. This
checker reads the ELF program headers of every native library in AARs
(``jni/<abi>/*.so``), APKs and JARs (``lib/<abi>/*.so``), loose ``.so`` files
and directories, and fails when a 64-bit library has a ``PT_LOAD`` segment whose
``p_align`` is below the required page size. 32-bit libraries are reported and
skipped, because 16 KB pages only exist on 64-bit devices.

A library that cannot be fixed yet may be listed in a reviewed allowlist with an
owner and an expiry date. Allowlisted failures are reported as ``ALLOW`` and do
not fail the run until the entry expires.

Diagnostics name only library file names, ABIs, alignment values and the file
name of the containing archive. They never include filesystem paths or file
contents.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import struct
import sys
import tomllib
import zipfile
import zlib
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, TextIO

DEFAULT_MIN_ALIGNMENT_KB = 16
SUPPORTED_MIN_ALIGNMENTS_KB = (4, 16, 64)
NATIVE_LIBRARY_SUFFIX = ".so"
ARCHIVE_SUFFIXES = frozenset({".aar", ".apk", ".jar", ".zip"})
# Android packages native libraries from these archive locations only; other
# ``.so`` entries (for example desktop natives inside a JVM jar) are never
# mapped by the Android dynamic linker and are ignored.
PACKAGED_NATIVE_ENTRY = re.compile(r"(?:lib|jni)/[^/]+/[^/]+\.so")

ELF_MAGIC = b"\x7fELF"
ELFCLASS32 = 1
ELFCLASS64 = 2
ELFDATA2LSB = 1
ELFDATA2MSB = 2
PT_LOAD = 1
PN_XNUM = 0xFFFF
ELF_IDENT_SIZE = 16
# e_type, e_machine, e_version, e_entry, e_phoff, e_shoff, e_flags, e_ehsize,
# e_phentsize, e_phnum (the section-header fields that follow are not needed).
ELF32_HEADER = "HHIIIIIHHH"
ELF64_HEADER = "HHIQQQIHHH"
# p_type plus p_align; the remaining program-header fields are skipped.
ELF32_PROGRAM_HEADER = "I24xI"
ELF64_PROGRAM_HEADER = "I44xQ"
ELF32_PROGRAM_HEADER_SIZE = 32
ELF64_PROGRAM_HEADER_SIZE = 56
MAX_PROGRAM_HEADER_END = 1024 * 1024

ABI_BY_MACHINE = {
    3: "x86",
    40: "armeabi-v7a",
    62: "x86_64",
    183: "arm64-v8a",
    243: "riscv64",
}
KNOWN_ABIS = frozenset(ABI_BY_MACHINE.values())

ALLOWLIST_SCHEMA_VERSION = 1
ALLOWLIST_REQUIRED_FIELDS = frozenset(
    {"library", "abis", "owner", "expires", "tracking", "reason"}
)
ALLOWLIST_OPTIONAL_FIELDS = frozenset({"artifact", "upstream"})

EXIT_PASS = 0
EXIT_MISALIGNED = 1
EXIT_INPUT_ERROR = 2


class NativeInputError(ValueError):
    """Raised when an input path, archive or allowlist cannot be used."""


class ElfFormatError(ValueError):
    """Raised when a ``.so`` file is not a well-formed ELF shared object.

    The message is a fixed reason code such as ``not-elf`` or
    ``truncated-program-headers``.
    """


@dataclass(frozen=True)
class ElfAlignment:
    """LOAD-segment alignment read from one ELF file.

    Attributes:
        elf_class: ``32`` or ``64``.
        machine: Raw ``e_machine`` value.
        load_alignments: ``p_align`` of every ``PT_LOAD`` program header.
    """

    elf_class: int
    machine: int
    load_alignments: tuple[int, ...]

    @property
    def abi(self) -> str:
        """Return the Android ABI name for ``machine``."""
        return ABI_BY_MACHINE.get(self.machine, f"machine-{self.machine}")

    @property
    def min_load_alignment(self) -> int:
        """Return the smallest ``PT_LOAD`` alignment."""
        return min(self.load_alignments)


@dataclass(frozen=True)
class LibraryResult:
    """Alignment result for one native library.

    Attributes:
        library: Library file name, for example ``libonnxruntime.so``.
        source: File name of the containing archive, or ``None`` for a loose
            ``.so`` file.
        abi: Android ABI derived from the ELF machine, or the archive directory
            name when the file is malformed.
        elf_class: ``32`` or ``64``, or ``None`` when the file is malformed.
        min_load_alignment: Smallest ``PT_LOAD`` ``p_align``, or ``None`` when
            the file is malformed.
        malformed: Fixed reason code when the ELF could not be read.
    """

    library: str
    source: str | None
    abi: str
    elf_class: int | None
    min_load_alignment: int | None
    malformed: str | None = None


@dataclass(frozen=True)
class AllowlistEntry:
    """One reviewed exception for a misaligned 64-bit library.

    Attributes:
        library: Library file name the exception covers.
        abis: ABIs the exception covers.
        owner: Person or team responsible for removing the exception.
        expires: Last day on which the exception applies.
        tracking: Issue that tracks the fix.
        reason: Why the library cannot be fixed yet.
        artifact: Maven coordinate of the dependency that ships the library.
        upstream: Upstream issue or release that would fix the library.
    """

    library: str
    abis: tuple[str, ...]
    owner: str
    expires: dt.date
    tracking: str
    reason: str
    artifact: str | None = None
    upstream: str | None = None

    def covers(self, result: LibraryResult) -> bool:
        """Return whether this entry applies to *result*."""
        return result.library == self.library and result.abi in self.abis


@dataclass(frozen=True)
class CheckSummary:
    """Totals for one checker run.

    Attributes:
        checked: 64-bit libraries whose alignment was evaluated.
        passed: 64-bit libraries at or above the required alignment.
        failed: Libraries that failed (misaligned, malformed or expired).
        allowed: Misaligned libraries covered by an unexpired allowlist entry.
        skipped: 32-bit libraries that were not evaluated.
        allowlist_errors: Expired or (in strict mode) unused allowlist entries.
    """

    checked: int
    passed: int
    failed: int
    allowed: int
    skipped: int
    allowlist_errors: int


def read_elf_alignment(handle: BinaryIO) -> ElfAlignment:
    """Read the ``PT_LOAD`` alignments of the ELF file open in *handle*.

    Only the ELF header and the program-header table are read; *handle* is
    consumed forward, so it may be a non-seekable zip entry stream.

    Args:
        handle: Binary stream positioned at the start of the ELF file.

    Returns:
        The ELF class, machine and ``PT_LOAD`` alignments.

    Raises:
        ElfFormatError: If the stream is not a readable ELF shared object.
    """
    ident = handle.read(ELF_IDENT_SIZE)
    if len(ident) < ELF_IDENT_SIZE or not ident.startswith(ELF_MAGIC):
        raise ElfFormatError("not-elf")
    elf_class, encoding = ident[4], ident[5]
    if elf_class not in (ELFCLASS32, ELFCLASS64):
        raise ElfFormatError("unsupported-elf-class")
    if encoding not in (ELFDATA2LSB, ELFDATA2MSB):
        raise ElfFormatError("unsupported-elf-encoding")
    order = "<" if encoding == ELFDATA2LSB else ">"
    is_64 = elf_class == ELFCLASS64
    header = struct.Struct(order + (ELF64_HEADER if is_64 else ELF32_HEADER))
    raw_header = handle.read(header.size)
    if len(raw_header) != header.size:
        raise ElfFormatError("truncated-elf-header")
    fields = header.unpack(raw_header)
    machine, phoff, phentsize, phnum = fields[1], fields[4], fields[8], fields[9]
    expected_entry_size = (
        ELF64_PROGRAM_HEADER_SIZE if is_64 else ELF32_PROGRAM_HEADER_SIZE
    )
    if phnum == 0:
        raise ElfFormatError("no-program-headers")
    if phnum == PN_XNUM:
        raise ElfFormatError("extended-program-header-count")
    if phentsize != expected_entry_size:
        raise ElfFormatError("bad-program-header-size")
    consumed = ELF_IDENT_SIZE + header.size
    table_end = phoff + phnum * phentsize
    if phoff < consumed or table_end > MAX_PROGRAM_HEADER_END:
        raise ElfFormatError("program-headers-out-of-bounds")
    _skip(handle, phoff - consumed)
    table = handle.read(phnum * phentsize)
    if len(table) != phnum * phentsize:
        raise ElfFormatError("truncated-program-headers")
    entry = struct.Struct(
        order + (ELF64_PROGRAM_HEADER if is_64 else ELF32_PROGRAM_HEADER)
    )
    alignments = tuple(
        p_align for p_type, p_align in entry.iter_unpack(table) if p_type == PT_LOAD
    )
    if not alignments:
        raise ElfFormatError("no-load-segments")
    return ElfAlignment(
        elf_class=64 if is_64 else 32,
        machine=machine,
        load_alignments=alignments,
    )


def _skip(handle: BinaryIO, count: int) -> None:
    """Advance *handle* by *count* bytes without seeking."""
    while count > 0:
        chunk = handle.read(min(count, 64 * 1024))
        if not chunk:
            raise ElfFormatError("truncated-program-headers")
        count -= len(chunk)


def _result(
    handle: BinaryIO, *, library: str, source: str | None, fallback_abi: str
) -> LibraryResult:
    try:
        elf = read_elf_alignment(handle)
    except ElfFormatError as error:
        return LibraryResult(
            library=library,
            source=source,
            abi=fallback_abi,
            elf_class=None,
            min_load_alignment=None,
            malformed=str(error),
        )
    return LibraryResult(
        library=library,
        source=source,
        abi=elf.abi,
        elf_class=elf.elf_class,
        min_load_alignment=elf.min_load_alignment,
    )


def _fallback_abi(directory: str) -> str:
    """Return *directory* when it is a known ABI name, else ``unknown-abi``.

    Directory names outside the fixed ABI set are never echoed, so a malformed
    library cannot leak a local folder name into the report.
    """
    return directory if directory in KNOWN_ABIS else "unknown-abi"


def is_packaged_native_entry(name: str) -> bool:
    """Return whether archive entry *name* is a native library Android loads.

    Android packages ``lib/<abi>/*.so`` from APKs and JARs and
    ``jni/<abi>/*.so`` from AARs.
    """
    return PACKAGED_NATIVE_ENTRY.fullmatch(name) is not None


def scan_archive(path: Path) -> list[LibraryResult]:
    """Return the alignment of every packaged native library in one archive.

    Args:
        path: AAR, APK, JAR or zip archive.

    Returns:
        One result per packaged ``.so`` entry, in archive order.

    Raises:
        NativeInputError: If the file cannot be read as a zip archive.
    """
    results: list[LibraryResult] = []
    try:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if info.is_dir() or not is_packaged_native_entry(info.filename):
                    continue
                parts = info.filename.split("/")
                with archive.open(info) as handle:
                    results.append(
                        _result(
                            handle,
                            library=parts[-1],
                            source=path.name,
                            fallback_abi=_fallback_abi(parts[-2]),
                        )
                    )
    except (OSError, EOFError, RuntimeError, zipfile.BadZipFile, zlib.error) as error:
        raise NativeInputError("not a readable zip archive") from error
    return results


def scan_library(path: Path) -> LibraryResult:
    """Return the alignment of one loose ``.so`` file.

    Raises:
        NativeInputError: If the file cannot be read.
    """
    try:
        with path.open("rb") as handle:
            return _result(
                handle,
                library=path.name,
                source=None,
                fallback_abi=_fallback_abi(path.parent.name),
            )
    except OSError as error:
        raise NativeInputError("native library is not readable") from error


def iter_results(paths: Iterable[Path]) -> Iterator[LibraryResult]:
    """Yield alignment results for every native library under *paths*.

    Args:
        paths: Archives (``.aar``, ``.apk``, ``.jar``, ``.zip``), loose ``.so``
            files, or directories, which are searched recursively for both
            (for example an AGP APK output directory or a CMake build tree).

    Yields:
        One result per native library.

    Raises:
        NativeInputError: If a path does not exist or has an unsupported type.
    """
    for path in paths:
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if not child.is_file():
                    continue
                if child.suffix == NATIVE_LIBRARY_SUFFIX:
                    yield scan_library(child)
                elif child.suffix in ARCHIVE_SUFFIXES:
                    yield from scan_archive(child)
        elif path.is_file() and path.suffix in ARCHIVE_SUFFIXES:
            yield from scan_archive(path)
        elif path.is_file() and path.suffix == NATIVE_LIBRARY_SUFFIX:
            yield scan_library(path)
        elif path.is_file():
            raise NativeInputError("unsupported input file type")
        else:
            raise NativeInputError("input path does not exist")


def _required_text(raw: dict[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise NativeInputError(f"allowlist field '{key}' must be a non-empty string")
    return value


def _optional_text(raw: dict[str, object], key: str) -> str | None:
    if key not in raw:
        return None
    return _required_text(raw, key)


def parse_allowlist(text: str) -> tuple[AllowlistEntry, ...]:
    """Parse and validate allowlist TOML.

    The document must set ``schema_version = 1`` and may contain any number of
    ``[[exceptions]]`` tables with ``library``, ``abis``, ``owner``,
    ``expires`` (a TOML date), ``tracking`` and ``reason``, plus the optional
    ``artifact`` and ``upstream`` fields.

    Args:
        text: Allowlist TOML document.

    Returns:
        The validated entries.

    Raises:
        NativeInputError: If the document or an entry is invalid.
    """
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise NativeInputError("allowlist is not valid TOML") from error
    if document.get("schema_version") != ALLOWLIST_SCHEMA_VERSION:
        raise NativeInputError("allowlist schema_version must be 1")
    unknown_top = set(document) - {"schema_version", "exceptions"}
    if unknown_top:
        raise NativeInputError("allowlist has unknown top-level keys")
    raw_entries = document.get("exceptions", [])
    if not isinstance(raw_entries, list):
        raise NativeInputError("allowlist 'exceptions' must be an array of tables")
    entries: list[AllowlistEntry] = []
    seen: set[tuple[str, str]] = set()
    for raw in raw_entries:
        if not isinstance(raw, dict):
            raise NativeInputError("allowlist exceptions must be tables")
        missing = ALLOWLIST_REQUIRED_FIELDS - set(raw)
        if missing:
            raise NativeInputError(
                "allowlist entry is missing " + ", ".join(sorted(missing))
            )
        if set(raw) - ALLOWLIST_REQUIRED_FIELDS - ALLOWLIST_OPTIONAL_FIELDS:
            raise NativeInputError("allowlist entry has unknown fields")
        library = _required_text(raw, "library")
        if "/" in library or not library.endswith(NATIVE_LIBRARY_SUFFIX):
            raise NativeInputError("allowlist 'library' must be a .so file name")
        abis = raw["abis"]
        if (
            not isinstance(abis, list)
            or not abis
            or not all(isinstance(abi, str) and abi in KNOWN_ABIS for abi in abis)
        ):
            raise NativeInputError("allowlist 'abis' must list known Android ABIs")
        expires = raw["expires"]
        if not isinstance(expires, dt.date) or isinstance(expires, dt.datetime):
            raise NativeInputError("allowlist 'expires' must be a TOML date")
        for abi in abis:
            if (library, abi) in seen:
                raise NativeInputError("allowlist has duplicate library/ABI entries")
            seen.add((library, abi))
        entries.append(
            AllowlistEntry(
                library=library,
                abis=tuple(abis),
                owner=_required_text(raw, "owner"),
                expires=expires,
                tracking=_required_text(raw, "tracking"),
                reason=_required_text(raw, "reason"),
                artifact=_optional_text(raw, "artifact"),
                upstream=_optional_text(raw, "upstream"),
            )
        )
    return tuple(entries)


def load_allowlist(path: Path) -> tuple[AllowlistEntry, ...]:
    """Read and validate the allowlist file at *path*.

    Raises:
        NativeInputError: If the file is missing or invalid.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise NativeInputError("allowlist is not readable") from error
    return parse_allowlist(text)


def _alignment_text(alignment: int) -> str:
    if alignment >= 1024 and alignment % 1024 == 0:
        return f"{alignment} ({alignment // 1024} KB)"
    return str(alignment)


def _label(result: LibraryResult) -> str:
    label = f"{result.abi} {result.library}"
    if result.source is not None:
        label += f" [{result.source}]"
    return label


def evaluate(
    results: Iterable[LibraryResult],
    *,
    min_alignment: int,
    allowlist: Sequence[AllowlistEntry] = (),
    today: dt.date,
    strict_allowlist: bool = False,
) -> tuple[list[str], CheckSummary]:
    """Classify *results* and return value-free report lines plus totals.

    Args:
        results: Library results to classify.
        min_alignment: Required minimum ``PT_LOAD`` alignment in bytes.
        allowlist: Reviewed exceptions for misaligned 64-bit libraries.
        today: Date used to decide whether an allowlist entry has expired.
        strict_allowlist: Fail when an allowlist entry matched no misaligned
            library, so fixed libraries cannot keep a stale exception.

    Returns:
        The report lines and the run totals.
    """
    lines: list[str] = []
    checked = passed = failed = allowed = skipped = allowlist_errors = 0
    used: set[int] = set()
    expired_reported: set[int] = set()
    for result in results:
        label = _label(result)
        if result.malformed is not None:
            failed += 1
            lines.append(f"FAIL {label}: malformed ELF ({result.malformed})")
            continue
        if result.elf_class == 32 or result.min_load_alignment is None:
            skipped += 1
            lines.append(f"SKIP {label}: 32-bit library")
            continue
        checked += 1
        alignment = _alignment_text(result.min_load_alignment)
        if result.min_load_alignment >= min_alignment:
            passed += 1
            lines.append(f"PASS {label}: LOAD alignment {alignment}")
            continue
        matches = [
            index for index, entry in enumerate(allowlist) if entry.covers(result)
        ]
        shortfall = f"LOAD alignment {alignment} < {min_alignment}"
        if not matches:
            failed += 1
            lines.append(f"FAIL {label}: {shortfall}")
            continue
        index = matches[0]
        entry = allowlist[index]
        used.add(index)
        if today > entry.expires:
            failed += 1
            if index not in expired_reported:
                expired_reported.add(index)
                allowlist_errors += 1
            lines.append(
                f"FAIL {label}: {shortfall}; allowlist entry expired on "
                f"{entry.expires.isoformat()} (owner {entry.owner})"
            )
            continue
        allowed += 1
        lines.append(
            f"ALLOW {label}: {shortfall}; allowlisted until "
            f"{entry.expires.isoformat()} (owner {entry.owner}, {entry.tracking})"
        )
    for index, entry in enumerate(allowlist):
        if index in used:
            continue
        abis = ", ".join(entry.abis)
        if strict_allowlist:
            allowlist_errors += 1
            lines.append(
                f"FAIL unused allowlist entry {entry.library} ({abis}): no "
                "misaligned library matched; remove the entry"
            )
        else:
            lines.append(
                f"NOTE unused allowlist entry {entry.library} ({abis}) in this run"
            )
    summary = CheckSummary(
        checked=checked,
        passed=passed,
        failed=failed,
        allowed=allowed,
        skipped=skipped,
        allowlist_errors=allowlist_errors,
    )
    return lines, summary


def format_summary(summary: CheckSummary, *, min_alignment: int) -> str:
    """Return the one-line, value-free summary for a run."""
    status = "FAIL" if summary.failed or summary.allowlist_errors else "PASS"
    return (
        f"{status} summary: {summary.checked} 64-bit libraries checked, "
        f"{summary.passed} passed, {summary.failed} failed, "
        f"{summary.allowed} allowlisted, {summary.skipped} 32-bit skipped "
        f"({min_alignment // 1024} KB minimum LOAD alignment)"
    )


def run(
    paths: Sequence[Path],
    *,
    min_alignment_kb: int = DEFAULT_MIN_ALIGNMENT_KB,
    allowlist_path: Path | None = None,
    strict_allowlist: bool = False,
    require_native_libraries: bool = False,
    today: dt.date | None = None,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    """Check every native library under *paths* and print diagnostics.

    Args:
        paths: Archives, loose ``.so`` files or directories.
        min_alignment_kb: Required ``PT_LOAD`` alignment in KiB (4, 16 or 64).
        allowlist_path: Optional reviewed allowlist TOML file.
        strict_allowlist: Fail on allowlist entries that matched nothing.
        require_native_libraries: Fail when no 64-bit library was checked, so
            the gate cannot pass vacuously.
        today: Date for allowlist expiry; defaults to the current UTC date.
        stdout: Stream for per-library results and the summary.
        stderr: Stream for input errors.

    Returns:
        ``0`` when every 64-bit library passes or is allowlisted, ``1`` on a
        misaligned, malformed or expired library, ``2`` on input errors.
    """
    if min_alignment_kb not in SUPPORTED_MIN_ALIGNMENTS_KB:
        print("error: minimum alignment must be 4, 16 or 64 KB", file=stderr)
        return EXIT_INPUT_ERROR
    if not paths:
        print("error: no inputs were given", file=stderr)
        return EXIT_INPUT_ERROR
    min_alignment = min_alignment_kb * 1024
    try:
        allowlist = load_allowlist(allowlist_path) if allowlist_path is not None else ()
        results = list(iter_results(paths))
    except NativeInputError as error:
        print(f"error: {error}", file=stderr)
        return EXIT_INPUT_ERROR
    lines, summary = evaluate(
        results,
        min_alignment=min_alignment,
        allowlist=allowlist,
        today=today or dt.datetime.now(dt.UTC).date(),
        strict_allowlist=strict_allowlist,
    )
    for line in lines:
        print(line, file=stdout)
    exit_code = EXIT_PASS
    if summary.failed or summary.allowlist_errors:
        exit_code = EXIT_MISALIGNED
    if require_native_libraries and summary.checked == 0:
        print("FAIL no 64-bit native libraries to verify", file=stdout)
        exit_code = EXIT_MISALIGNED
    print(format_summary(summary, min_alignment=min_alignment), file=stdout)
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="AAR, APK, JAR or zip archives, .so files, or directories",
    )
    parser.add_argument(
        "--min-alignment-kb",
        type=int,
        default=DEFAULT_MIN_ALIGNMENT_KB,
        help="required PT_LOAD alignment of 64-bit libraries in KiB (default: 16)",
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        help="reviewed TOML allowlist of temporary exceptions",
    )
    parser.add_argument(
        "--strict-allowlist",
        action="store_true",
        help="fail when an allowlist entry matches no misaligned library",
    )
    parser.add_argument(
        "--require-native-libraries",
        action="store_true",
        help="fail when no 64-bit native library was found",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    args = build_parser().parse_args(argv)
    return run(
        args.paths,
        min_alignment_kb=args.min_alignment_kb,
        allowlist_path=args.allowlist,
        strict_allowlist=args.strict_allowlist,
        require_native_libraries=args.require_native_libraries,
    )


if __name__ == "__main__":
    raise SystemExit(main())
