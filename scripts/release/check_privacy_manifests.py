#!/usr/bin/env python3
"""Check Apple privacy manifests against required-reason API use in Swift sources.

Every OpenMedKit target and the scan demo ship a ``PrivacyInfo.xcprivacy`` that
must declare no tracking, no tracking domains and no collected data types. Each
required-reason API category referenced by the Swift sources a manifest covers
must be declared in it with an approved reason.
"""

from __future__ import annotations

import plistlib
import re
import sys
from pathlib import Path
from xml.parsers.expat import ExpatError

ROOT = Path(__file__).resolve().parents[2]
KIT_SOURCES = Path("swift/OpenMedKit/Sources")
SCAN_DEMO = Path("swift/OpenMedScanDemo/OpenMedScanDemo")
MANIFEST_NAME = "PrivacyInfo.xcprivacy"

# Swift symbols and approved reasons from Apple's "Describing use of required
# reason API" documentation.
REQUIRED_REASON_SYMBOLS = {
    "NSPrivacyAccessedAPICategoryFileTimestamp": (
        "creationDate",
        "modificationDate",
        "fileModificationDate",
        "contentModificationDateKey",
        "creationDateKey",
        "getattrlist",
        "getattrlistbulk",
        "fgetattrlist",
        "getattrlistat",
        "stat",
        "fstat",
        "fstatat",
        "lstat",
    ),
    "NSPrivacyAccessedAPICategorySystemBootTime": (
        "systemUptime",
        "mach_absolute_time",
    ),
    "NSPrivacyAccessedAPICategoryDiskSpace": (
        "volumeAvailableCapacityKey",
        "volumeAvailableCapacityForImportantUsageKey",
        "volumeAvailableCapacityForOpportunisticUsageKey",
        "volumeTotalCapacityKey",
        "systemFreeSize",
        "systemSize",
        "statfs",
        "statvfs",
        "fstatfs",
        "fstatvfs",
        "getattrlist",
        "fgetattrlist",
        "getattrlistat",
    ),
    "NSPrivacyAccessedAPICategoryActiveKeyboards": ("activeInputModes",),
    # SwiftUI's AppStorage reads and writes UserDefaults without naming it.
    "NSPrivacyAccessedAPICategoryUserDefaults": ("UserDefaults", "AppStorage"),
}
APPROVED_REASONS = {
    "NSPrivacyAccessedAPICategoryFileTimestamp": {
        "DDA9.1",
        "C617.1",
        "3B52.1",
        "0A2A.1",
    },
    "NSPrivacyAccessedAPICategorySystemBootTime": {"35F9.1", "8FFB.1", "3D61.1"},
    "NSPrivacyAccessedAPICategoryDiskSpace": {"85F4.1", "E174.1", "7D9E.1", "B728.1"},
    "NSPrivacyAccessedAPICategoryActiveKeyboards": {"3EC4.1", "54BD.1"},
    "NSPrivacyAccessedAPICategoryUserDefaults": {
        "CA92.1",
        "1C8F.1",
        "C56D.1",
        "AC6B.1",
    },
}
SYMBOL_PATTERNS = {
    category: re.compile(r"\b(?:" + "|".join(symbols) + r")\b")
    for category, symbols in REQUIRED_REASON_SYMBOLS.items()
}
# Opening delimiter of a Swift string literal: optional raw-string hashes, then
# a multiline or single-line quote.
STRING_START = re.compile(r'(#*)("""|")')


def _blank(text: str) -> str:
    return re.sub(r"[^\n]", " ", text)


def _skip_block_comment(source: str, index: int) -> int:
    depth = 0
    while index < len(source):
        if source.startswith("/*", index):
            depth += 1
            index += 2
        elif source.startswith("*/", index):
            depth -= 1
            index += 2
            if depth == 0:
                break
        else:
            index += 1
    return index


def _scan_string(source: str, index: int, out: list[str]) -> int:
    match = STRING_START.match(source, index)
    assert match is not None
    hashes, quote = match.groups()
    close = quote + hashes
    escape = "\\" + hashes
    out.append(_blank(match.group()))
    index = match.end()
    while index < len(source):
        if source.startswith(close, index):
            out.append(_blank(close))
            return index + len(close)
        if quote == '"' and source[index] == "\n":
            break  # unterminated single-line literal
        if source.startswith(escape, index):
            after = index + len(escape)
            if source.startswith("(", after):
                out.append(_blank(source[index:after]) + "(")
                index = _scan_code(source, after + 1, out, nested=True)
                continue
            # Ordinary escape: skip the escaped character with the delimiter.
            out.append(_blank(source[index : after + 1]))
            index = after + 1
            continue
        out.append(_blank(source[index]))
        index += 1
    return index


def _scan_code(source: str, index: int, out: list[str], nested: bool) -> int:
    depth = 0
    while index < len(source):
        char = source[index]
        if source.startswith("//", index):
            end = source.find("\n", index)
            end = len(source) if end == -1 else end
            out.append(_blank(source[index:end]))
            index = end
        elif source.startswith("/*", index):
            end = _skip_block_comment(source, index)
            out.append(_blank(source[index:end]))
            index = end
        elif STRING_START.match(source, index):
            index = _scan_string(source, index, out)
        else:
            out.append(char)
            index += 1
            if nested and char == "(":
                depth += 1
            elif nested and char == ")":
                if depth == 0:
                    return index
                depth -= 1
    return index


def strip_comments_and_strings(source: str) -> str:
    """Blank Swift comments and string-literal text, keeping line numbers.

    String interpolations such as ``"\\(value)"`` are executable code and are
    kept, including in raw (``#"..."#``) and multiline literals.

    Args:
        source: Swift source text.

    Returns:
        The source with comments and literal text replaced by spaces, newlines
        preserved.
    """
    out: list[str] = []
    _scan_code(source, 0, out, nested=False)
    return "".join(out)


def find_required_reason_apis(source_dir: Path, root: Path) -> dict[str, list[str]]:
    """Find required-reason API references in Swift sources.

    Args:
        source_dir: Directory scanned recursively for ``*.swift`` files.
        root: Repository root used to report relative paths.

    Returns:
        A mapping from API category to ``path:line symbol`` references.
    """
    found: dict[str, list[str]] = {}
    for path in sorted(source_dir.rglob("*.swift")):
        code = strip_comments_and_strings(path.read_text(encoding="utf-8"))
        relative = path.relative_to(root).as_posix()
        for category, pattern in SYMBOL_PATTERNS.items():
            for match in pattern.finditer(code):
                line = code.count("\n", 0, match.start()) + 1
                found.setdefault(category, []).append(
                    f"{relative}:{line} {match.group()}"
                )
    return found


def declared_categories(manifest: Path, root: Path) -> tuple[set[str], list[str]]:
    """Validate one privacy manifest and return its declared API categories.

    Args:
        manifest: Path to a ``PrivacyInfo.xcprivacy`` property list.
        root: Repository root used to report relative paths.

    Returns:
        The declared API categories and any policy errors.
    """
    name = manifest.relative_to(root).as_posix()
    try:
        with manifest.open("rb") as handle:
            data = plistlib.load(handle)
    except (OSError, ValueError, ExpatError):
        return set(), [f"{name}: missing or unreadable privacy manifest"]

    errors: list[str] = []
    if data.get("NSPrivacyTracking") is not False:
        errors.append(f"{name}: NSPrivacyTracking must be false")
    if data.get("NSPrivacyTrackingDomains") != []:
        errors.append(f"{name}: NSPrivacyTrackingDomains must be empty")
    if data.get("NSPrivacyCollectedDataTypes") != []:
        errors.append(f"{name}: NSPrivacyCollectedDataTypes must be empty")

    declared: set[str] = set()
    for entry in data.get("NSPrivacyAccessedAPITypes", []):
        category = entry.get("NSPrivacyAccessedAPIType")
        reasons = set(entry.get("NSPrivacyAccessedAPITypeReasons", []))
        if reasons and reasons <= APPROVED_REASONS.get(category, set()):
            declared.add(category)
        else:
            errors.append(f"{name}: {category} needs approved reasons")
    return declared, errors


def check_target(source_dir: Path, manifest: Path, root: Path) -> list[str]:
    """Check that a manifest declares the API categories its sources use.

    Args:
        source_dir: Swift source directory covered by the manifest.
        manifest: Path to the target's ``PrivacyInfo.xcprivacy``.
        root: Repository root used to report relative paths.

    Returns:
        Policy errors; empty when the target is compliant.
    """
    declared, errors = declared_categories(manifest, root)
    name = manifest.relative_to(root).as_posix()
    for category, references in find_required_reason_apis(source_dir, root).items():
        if category not in declared:
            errors.extend(
                f"{reference}: {category} is not declared in {name}"
                for reference in references
            )
    return errors


def check_repository(root: Path = ROOT) -> list[str]:
    """Check every OpenMedKit target and the scan demo.

    Args:
        root: Repository root.

    Returns:
        Policy errors; empty when every manifest is compliant.
    """
    targets = [
        (source_dir, source_dir / "Resources" / MANIFEST_NAME)
        for source_dir in sorted((root / KIT_SOURCES).iterdir())
        if source_dir.is_dir()
    ]
    targets.append((root / SCAN_DEMO, root / SCAN_DEMO / MANIFEST_NAME))
    errors: list[str] = []
    for source_dir, manifest in targets:
        errors.extend(check_target(source_dir, manifest, root))
    return errors


def main() -> int:
    errors = check_repository()
    if errors:
        print("Privacy manifest check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("Privacy manifests declare every required-reason API in use.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
