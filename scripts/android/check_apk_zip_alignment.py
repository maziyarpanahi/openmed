#!/usr/bin/env python3
"""Verify 16 KB zip alignment of uncompressed native libraries in Android APKs.

The check mirrors ``zipalign -c -P <page_kb> <alignment>``: the data of every
stored (uncompressed) ``.so`` entry must start on a page-size boundary, and the
data of every other stored entry on an ``alignment``-byte boundary. Compressed
entries are extracted at install time and are not checked.

Diagnostics name only the APK file name, archive entry names, offsets and
counts. They never include absolute filesystem paths or entry contents.
"""

from __future__ import annotations

import argparse
import struct
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Sequence, TextIO

DEFAULT_PAGE_SIZE_KB = 16
DEFAULT_ALIGNMENT = 4
SUPPORTED_PAGE_SIZES_KB = (4, 16, 64)
NATIVE_LIBRARY_SUFFIX = ".so"

LOCAL_FILE_HEADER = struct.Struct("<IHHHHHIIIHH")
LOCAL_FILE_HEADER_SIGNATURE = 0x04034B50

EXIT_PASS = 0
EXIT_MISALIGNED = 1
EXIT_INPUT_ERROR = 2


class ApkInputError(ValueError):
    """Raised when an input is missing, unreadable, or not a zip archive."""


@dataclass(frozen=True)
class EntryAlignment:
    """Alignment of one stored archive entry.

    Attributes:
        name: Archive entry name, for example ``lib/arm64-v8a/libfoo.so``.
        data_offset: Byte offset of the entry data from the archive start.
        required: Alignment in bytes the entry data must satisfy.
    """

    name: str
    data_offset: int
    required: int

    @property
    def remainder(self) -> int:
        """Return ``data_offset`` modulo the required alignment."""
        return self.data_offset % self.required

    @property
    def aligned(self) -> bool:
        """Return whether the entry data starts on the required boundary."""
        return self.remainder == 0


@dataclass(frozen=True)
class ApkAlignmentReport:
    """Alignment result for one APK.

    Attributes:
        apk_name: File name of the APK, without its directory.
        page_size: Required alignment in bytes for stored native libraries.
        stored_entries: Number of stored entries that were checked.
        stored_native_libraries: Number of stored ``.so`` entries.
        compressed_native_libraries: Number of compressed ``.so`` entries.
        misaligned: Stored entries whose data is not on the required boundary.
    """

    apk_name: str
    page_size: int
    stored_entries: int
    stored_native_libraries: int
    compressed_native_libraries: int
    misaligned: tuple[EntryAlignment, ...]

    @property
    def passed(self) -> bool:
        """Return whether every stored entry is aligned."""
        return not self.misaligned


def is_native_library(name: str) -> bool:
    """Return whether *name* is page-aligned by ``zipalign -P``.

    ``zipalign`` page-aligns entries whose final extension is exactly ``.so``.
    """
    return name.endswith(NATIVE_LIBRARY_SUFFIX)


def _data_offset(handle: BinaryIO, info: zipfile.ZipInfo) -> int:
    """Return the data offset of *info* by reading its local file header."""
    handle.seek(info.header_offset)
    header = handle.read(LOCAL_FILE_HEADER.size)
    if len(header) != LOCAL_FILE_HEADER.size:
        raise ApkInputError("truncated local file header")
    fields = LOCAL_FILE_HEADER.unpack(header)
    if fields[0] != LOCAL_FILE_HEADER_SIGNATURE:
        raise ApkInputError("invalid local file header signature")
    name_length, extra_length = int(fields[9]), int(fields[10])
    return info.header_offset + LOCAL_FILE_HEADER.size + name_length + extra_length


def check_apk(
    path: Path,
    *,
    page_size: int = DEFAULT_PAGE_SIZE_KB * 1024,
    alignment: int = DEFAULT_ALIGNMENT,
) -> ApkAlignmentReport:
    """Check the zip alignment of every stored entry in one APK.

    Args:
        path: APK (or any zip archive) to check.
        page_size: Alignment in bytes required for stored ``.so`` entries.
        alignment: Alignment in bytes required for other stored entries.

    Returns:
        The alignment report for the APK.

    Raises:
        ApkInputError: If the file cannot be read as a zip archive.
    """
    stored = 0
    stored_native = 0
    compressed_native = 0
    misaligned: list[EntryAlignment] = []
    try:
        with zipfile.ZipFile(path) as archive, path.open("rb") as handle:
            for info in archive.infolist():
                native = is_native_library(info.filename)
                if info.compress_type != zipfile.ZIP_STORED:
                    compressed_native += int(native)
                    continue
                stored += 1
                stored_native += int(native)
                entry = EntryAlignment(
                    name=info.filename,
                    data_offset=_data_offset(handle, info),
                    required=page_size if native else alignment,
                )
                if not entry.aligned:
                    misaligned.append(entry)
    except (OSError, zipfile.BadZipFile) as error:
        raise ApkInputError("not a readable zip archive") from error
    return ApkAlignmentReport(
        apk_name=path.name,
        page_size=page_size,
        stored_entries=stored,
        stored_native_libraries=stored_native,
        compressed_native_libraries=compressed_native,
        misaligned=tuple(misaligned),
    )


def collect_apks(paths: Iterable[Path]) -> list[Path]:
    """Expand *paths* into APK files.

    Args:
        paths: APK files, or directories whose direct ``*.apk`` children are
            checked (the layout of an AGP APK output directory).

    Returns:
        The APK files in a stable order.

    Raises:
        ApkInputError: If a path does not exist or no APK is found.
    """
    apks: list[Path] = []
    for path in paths:
        if path.is_dir():
            found = sorted(child for child in path.glob("*.apk") if child.is_file())
            if not found:
                raise ApkInputError("directory contains no APK files")
            apks.extend(found)
        elif path.is_file():
            apks.append(path)
        else:
            raise ApkInputError("input path does not exist")
    if not apks:
        raise ApkInputError("no APK files were given")
    return apks


def format_report(report: ApkAlignmentReport) -> list[str]:
    """Return the value-free diagnostic lines for *report*."""
    page_kb = report.page_size // 1024
    status = "PASS" if report.passed else "FAIL"
    lines = [
        f"{status} {report.apk_name}: "
        f"{report.stored_native_libraries} uncompressed native libraries, "
        f"{report.compressed_native_libraries} compressed, "
        f"{report.stored_entries} stored entries checked "
        f"({page_kb} KB page alignment)"
    ]
    for entry in report.misaligned:
        lines.append(
            f"  misaligned {entry.name}: data offset % {entry.required} = "
            f"{entry.remainder}"
        )
    return lines


def run(
    paths: Sequence[Path],
    *,
    page_size_kb: int = DEFAULT_PAGE_SIZE_KB,
    alignment: int = DEFAULT_ALIGNMENT,
    require_native_libraries: bool = False,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    """Check every APK in *paths* and print value-free diagnostics.

    Args:
        paths: APK files or AGP APK output directories.
        page_size_kb: Required native-library alignment in KiB (4, 16 or 64).
        alignment: Required alignment in bytes for other stored entries.
        require_native_libraries: Fail an APK that has no uncompressed native
            libraries, so the gate cannot pass vacuously.
        stdout: Stream for per-APK results.
        stderr: Stream for input errors.

    Returns:
        ``0`` when every APK passes, ``1`` when an APK is misaligned or has no
        native libraries while they are required, ``2`` on input errors.
    """
    if page_size_kb not in SUPPORTED_PAGE_SIZES_KB:
        print("error: page size must be 4, 16 or 64 KB", file=stderr)
        return EXIT_INPUT_ERROR
    if alignment <= 0:
        print("error: alignment must be a positive integer", file=stderr)
        return EXIT_INPUT_ERROR
    try:
        apks = collect_apks(paths)
    except ApkInputError as error:
        print(f"error: {error}", file=stderr)
        return EXIT_INPUT_ERROR

    exit_code = EXIT_PASS
    for apk in apks:
        try:
            report = check_apk(apk, page_size=page_size_kb * 1024, alignment=alignment)
        except ApkInputError as error:
            print(f"error: {apk.name}: {error}", file=stderr)
            return EXIT_INPUT_ERROR
        for line in format_report(report):
            print(line, file=stdout)
        if not report.passed:
            exit_code = EXIT_MISALIGNED
        elif require_native_libraries and report.stored_native_libraries == 0:
            print(
                f"FAIL {report.apk_name}: no uncompressed native libraries to verify",
                file=stdout,
            )
            exit_code = EXIT_MISALIGNED
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="APK files or directories containing APK files",
    )
    parser.add_argument(
        "--page-size-kb",
        type=int,
        default=DEFAULT_PAGE_SIZE_KB,
        help="required alignment of uncompressed .so entries in KiB (default: 16)",
    )
    parser.add_argument(
        "--alignment",
        type=int,
        default=DEFAULT_ALIGNMENT,
        help="required alignment of other stored entries in bytes (default: 4)",
    )
    parser.add_argument(
        "--require-native-libraries",
        action="store_true",
        help="fail an APK that contains no uncompressed native libraries",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    args = build_parser().parse_args(argv)
    return run(
        args.paths,
        page_size_kb=args.page_size_kb,
        alignment=args.alignment,
        require_native_libraries=args.require_native_libraries,
    )


if __name__ == "__main__":
    raise SystemExit(main())
