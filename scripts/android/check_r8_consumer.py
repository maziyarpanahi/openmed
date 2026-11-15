#!/usr/bin/env python3
"""Verify OpenMedKit inside an R8-minified Android consumer app.

Subcommands:

``verify``
    Check one minified release variant: its ABI-split APKs, R8's
    ``mapping.txt``, ``usage.txt`` and ``configuration.txt``, and the merged
    manifest. Every Java class that a bundled JNI library looks up by name must
    survive minification whole and under its original name, native method
    bindings must keep their names, ``META-INF/services`` providers must still
    exist, no rule may silence R8 missing-class errors unless it is
    allowlisted, the OpenMedKit consumer rules must reach R8, desktop JNI
    binaries must not be packaged as Java resources, and the app must not
    declare the ``INTERNET`` permission. Member checks compare names, not
    descriptors. ``--expect-failure`` turns the check into a negative control.

``size``
    Enforce per-ABI install-size budgets: the APK with OpenMedKit minus the
    same app built without it.

``drop-keep-rule``
    Copy an AAR with one rule removed from its ``proguard.txt``, for the
    negative control.

Diagnostics hold only reason codes, class and member names from the minified
program, ABI names, archive entry names, rule text and byte counts. They never
include absolute paths, file contents or app input.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence, TextIO

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_INPUT_ERROR = 2

USAGE_FILE = "usage.txt"
CONFIGURATION_FILE = "configuration.txt"
OUTPUT_METADATA_FILE = "output-metadata.json"
SERVICES_PREFIX = "META-INF/services/"
INTERNET_PERMISSION = "android.permission.INTERNET"
DESKTOP_NATIVE_SUFFIXES = (".so", ".dylib", ".dll", ".jnilib", ".a")
CATCH_ALL_PATTERNS = frozenset({"", "*", "**", "***"})
ACC_NATIVE = 0x100
OWN_NAMESPACE_PREFIX = "com.openmed."
SIZE_BUDGET_KEY = "consumerApk.{abi}.maxDeltaBytes"

# Reason codes. They name what failed, never the data involved.
APK_NATIVE_LIBRARY_MISSING = "apk_native_library_missing"
CONSUMER_RULE_NOT_APPLIED = "consumer_rule_not_applied"
DESKTOP_NATIVE_RESOURCE_PACKAGED = "desktop_native_resource_packaged"
JNI_CLASS_NOT_KEPT = "jni_class_not_kept"
JNI_MEMBER_NOT_KEPT = "jni_member_not_kept"
JNI_NATIVE_BINDING_RENAMED = "jni_native_binding_renamed"
NETWORK_PERMISSION_DECLARED = "network_permission_declared"
R8_DONTWARN_NOT_ALLOWLISTED = "r8_dontwarn_not_allowlisted"
R8_WARNINGS_IGNORED = "r8_warnings_ignored"
SERVICE_PROVIDER_MISSING = "service_provider_missing"
SIZE_BUDGET_EXCEEDED = "size_budget_exceeded"

IDENTIFIER_BYTES = frozenset(
    b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_$/"
)
JNI_SYMBOL_PATTERN = re.compile(rb"Java_[A-Za-z0-9_]+")
DEX_ENTRY_PATTERN = re.compile(r"classes\d*\.dex")
NATIVE_LIBRARY_PATTERN = re.compile(r"lib/([^/]+)/lib([^/]+)\.so")
MAPPING_CLASS_PATTERN = re.compile(r"(\S+) -> (\S+):$")
MAPPING_MEMBER_PATTERN = re.compile(
    r"\s+(?:\d+(?::\d+)*:)?\S+ ([^\s(]+)(?:\([^)]*\))?(?::\d+(?::\d+)?)? -> (\S+)$"
)
USAGE_ARGUMENTS_PATTERN = re.compile(r"\(.*\)$")
PERMISSION_PATTERN = re.compile(
    r"<uses-permission(?:-sdk-23)?\b[^>]*android:name\s*=\s*\"([^\"]+)\"[^>]*>",
    re.DOTALL,
)
REMOVE_NODE_PATTERN = re.compile(r"tools:node\s*=\s*\"remove\"")


class GateInputError(ValueError):
    """Raised when a gate input is missing or malformed."""


@dataclass(frozen=True, order=True)
class Finding:
    """One value-free gate failure.

    Attributes:
        code: Stable reason code, for example ``jni_class_not_kept``.
        subject: Class, member, rule, ABI or archive entry the code is about.
        detail: Optional short qualifier such as ``renamed`` or ``removed``.
    """

    code: str
    subject: str
    detail: str = ""

    def format(self, status: str = "FAIL") -> str:
        """Return the diagnostic line for this finding.

        Args:
            status: Line prefix; negative controls report ``EXPECTED``.
        """
        suffix = f" ({self.detail})" if self.detail else ""
        return f"{status} {self.code} {self.subject}{suffix}"


@dataclass(frozen=True)
class Expectation:
    """A negative-control expectation: a code, optionally for a subject prefix."""

    code: str
    subject_prefix: str = ""

    @classmethod
    def parse(cls, value: str) -> Expectation:
        """Parse ``CODE`` or ``CODE:SUBJECT_PREFIX``."""
        code, _, prefix = value.partition(":")
        if not code:
            raise GateInputError("expectation needs a reason code")
        return cls(code=code, subject_prefix=prefix)

    def matches(self, finding: Finding) -> bool:
        """Return whether *finding* satisfies this expectation."""
        return finding.code == self.code and finding.subject.startswith(
            self.subject_prefix
        )

    def describe(self) -> str:
        """Return the expectation in its command-line form."""
        return (
            f"{self.code}:{self.subject_prefix}" if self.subject_prefix else self.code
        )


@dataclass
class DexClass:
    """Member names of one class defined in a dex file.

    Attributes:
        fields: Names of the fields the class defines.
        methods: Names of the methods the class defines.
        native_methods: Names of its ``native`` methods.
    """

    fields: set[str] = field(default_factory=set)
    methods: set[str] = field(default_factory=set)
    native_methods: set[str] = field(default_factory=set)


@dataclass
class R8Mapping:
    """Class and member renames recorded in R8's ``mapping.txt``.

    Attributes:
        classes: Original class name mapped to its name in the output.
        renamed_members: Original class name mapped to member names R8 renamed.
    """

    classes: dict[str, str] = field(default_factory=dict)
    renamed_members: dict[str, set[str]] = field(default_factory=dict)


@dataclass
class R8Usage:
    """Classes and members R8 removed, from ``usage.txt``.

    Attributes:
        removed_classes: Classes removed entirely.
        removed_members: Class name mapped to member names R8 removed.
    """

    removed_classes: set[str] = field(default_factory=set)
    removed_members: dict[str, set[str]] = field(default_factory=dict)


@dataclass
class VerifyReport:
    """Result of the ``verify`` subcommand.

    Attributes:
        findings: Gate failures, sorted and de-duplicated.
        counts: Value-free counters describing what was checked.
    """

    findings: list[Finding]
    counts: dict[str, int]

    @property
    def passed(self) -> bool:
        """Return whether no gate failed."""
        return not self.findings


@dataclass(frozen=True)
class AbiSize:
    """Per-ABI install-size measurement.

    Attributes:
        abi: ABI name, for example ``arm64-v8a``.
        apk_bytes: Size of the APK with OpenMedKit.
        baseline_bytes: Size of the same app built without OpenMedKit.
        max_delta_bytes: Committed budget for ``apk_bytes - baseline_bytes``.
        native_library_bytes: Uncompressed ``lib/<abi>/*.so`` bytes per library.
        dex_bytes: Uncompressed ``classes*.dex`` bytes.
    """

    abi: str
    apk_bytes: int
    baseline_bytes: int
    max_delta_bytes: int
    native_library_bytes: dict[str, int]
    dex_bytes: int

    @property
    def delta_bytes(self) -> int:
        """Return the bytes attributable to OpenMedKit."""
        return self.apk_bytes - self.baseline_bytes

    @property
    def passed(self) -> bool:
        """Return whether the delta is within budget."""
        return self.delta_bytes <= self.max_delta_bytes


# --------------------------------------------------------------------------
# Readers


def _uleb128(data: bytes, offset: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        if offset >= len(data):
            raise GateInputError("truncated dex string length")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7


def _dex_string(dex: bytes, offset: int) -> str:
    _, start = _uleb128(dex, offset)
    end = dex.index(b"\0", start)
    return dex[start:end].decode("utf-8", errors="replace")


def read_dex(dex: bytes) -> dict[str, DexClass]:
    """Return every class defined in one dex file with its member names.

    Args:
        dex: Raw bytes of a ``classes*.dex`` file.

    Returns:
        Dotted class names, such as ``ai.onnxruntime.OnnxTensor``, mapped to
        the fields and methods the class defines.

    Raises:
        GateInputError: If the bytes are not a well-formed dex file.
    """
    if len(dex) < 0x70 or not dex.startswith(b"dex\n"):
        raise GateInputError("not a dex file")
    try:
        string_ids_size, string_ids_off, type_ids_size, type_ids_off = (
            struct.unpack_from("<IIII", dex, 0x38)
        )
        field_ids_size, field_ids_off, method_ids_size, method_ids_off = (
            struct.unpack_from("<IIII", dex, 0x50)
        )
        class_defs_size, class_defs_off = struct.unpack_from("<II", dex, 0x60)
        strings = struct.unpack_from(f"<{string_ids_size}I", dex, string_ids_off)
        types = struct.unpack_from(f"<{type_ids_size}I", dex, type_ids_off)
        field_names = [
            struct.unpack_from("<I", dex, field_ids_off + index * 8 + 4)[0]
            for index in range(field_ids_size)
        ]
        method_names = [
            struct.unpack_from("<I", dex, method_ids_off + index * 8 + 4)[0]
            for index in range(method_ids_size)
        ]
        classes: dict[str, DexClass] = {}
        for index in range(class_defs_size):
            base = class_defs_off + index * 32
            (type_index,) = struct.unpack_from("<I", dex, base)
            (class_data_off,) = struct.unpack_from("<I", dex, base + 24)
            descriptor = _dex_string(dex, strings[types[type_index]])
            if not (descriptor.startswith("L") and descriptor.endswith(";")):
                raise GateInputError("malformed dex class descriptor")
            members = DexClass()
            if class_data_off:
                offset = class_data_off
                sizes = []
                for _ in range(4):
                    size, offset = _uleb128(dex, offset)
                    sizes.append(size)
                for count in sizes[:2]:
                    field_index = 0
                    for _ in range(count):
                        delta, offset = _uleb128(dex, offset)
                        _, offset = _uleb128(dex, offset)
                        field_index += delta
                        name = _dex_string(dex, strings[field_names[field_index]])
                        members.fields.add(name)
                for count in sizes[2:]:
                    method_index = 0
                    for _ in range(count):
                        delta, offset = _uleb128(dex, offset)
                        access, offset = _uleb128(dex, offset)
                        _, offset = _uleb128(dex, offset)
                        method_index += delta
                        name = _dex_string(dex, strings[method_names[method_index]])
                        members.methods.add(name)
                        if access & ACC_NATIVE:
                            members.native_methods.add(name)
            classes[descriptor[1:-1].replace("/", ".")] = members
    except GateInputError:
        raise
    except (IndexError, ValueError, struct.error) as error:
        raise GateInputError("malformed dex tables") from error
    return classes


def parse_mapping(lines: Iterable[str]) -> R8Mapping:
    """Parse R8's ``mapping.txt``.

    Members are recorded when any line maps them to a different name. That
    includes inline frames, whose right-hand side is the enclosing method, so
    callers confirm a rename against the member names in the dex.
    """
    mapping = R8Mapping()
    current: str | None = None
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line[0].isspace():
            match = MAPPING_CLASS_PATTERN.fullmatch(line.rstrip())
            current = None
            if match:
                current = match.group(1)
                mapping.classes[current] = match.group(2)
            continue
        if current is None:
            continue
        member = MAPPING_MEMBER_PATTERN.fullmatch(line.rstrip())
        if member is None:
            continue
        original, renamed = member.group(1), member.group(2)
        # Qualified names are inlined frames of other classes, and `$r8$`
        # names are members R8 synthesized; neither exists in the input.
        if "." not in original and "$r8$" not in original and original != renamed:
            mapping.renamed_members.setdefault(current, set()).add(original)
    return mapping


def parse_usage(lines: Iterable[str]) -> R8Usage:
    """Parse R8's ``usage.txt`` (classes and members removed by shrinking)."""
    usage = R8Usage()
    current: str | None = None
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line[0].isspace():
            name = line.strip()
            if name.endswith(":"):
                current = name[:-1]
            else:
                current = None
                usage.removed_classes.add(name)
            continue
        if current is None:
            continue
        # Members read "[modifiers] type name" or "[modifiers] type name(args)".
        declaration = USAGE_ARGUMENTS_PATTERN.sub("", line.strip()).split()
        if declaration:
            usage.removed_members.setdefault(current, set()).add(declaration[-1])
    return usage


def _strip_comment(line: str) -> str:
    return line.split("#", maxsplit=1)[0].strip()


def _normalize_rule(line: str) -> str:
    return " ".join(_strip_comment(line).split())


def read_rules(text: str) -> list[str]:
    """Return the normalized, non-comment rule lines of a ProGuard file."""
    return [rule for rule in map(_normalize_rule, text.splitlines()) if rule]


def dontwarn_patterns(rules: Iterable[str]) -> list[str]:
    """Return every class filter silenced by ``-dontwarn`` in *rules*."""
    patterns: list[str] = []
    for rule in rules:
        if rule != "-dontwarn" and not rule.startswith("-dontwarn "):
            continue
        # A bare -dontwarn silences every class; report it as the empty filter.
        filters = rule[len("-dontwarn") :].strip()
        patterns.extend(part.strip() for part in filters.split(","))
    return patterns


def read_allowlist(text: str) -> set[str]:
    """Return allowlisted ``-dontwarn`` filters.

    Raises:
        GateInputError: If an entry is a catch-all or silences OpenMedKit's
            own namespace, which no allowlist may do.
    """
    entries = {_strip_comment(line) for line in text.splitlines()} - {""}
    for entry in entries:
        if entry in CATCH_ALL_PATTERNS or entry.startswith(OWN_NAMESPACE_PREFIX):
            raise GateInputError("allowlist may not silence everything or com.openmed")
    return entries


def demangle_jni_symbol(symbol: str) -> tuple[str, str] | None:
    """Return ``(class, method)`` for a ``Java_`` JNI export, or ``None``.

    Implements the JNI name mangling rules: ``_`` separates name parts,
    ``_1`` is ``_``, ``_2`` is ``;``, ``_3`` is ``[`` and ``_0xxxx`` is a UTF-16
    code unit. Overload signatures after ``__`` are ignored.
    """
    if not symbol.startswith("Java_"):
        return None
    body = symbol[len("Java_") :]
    decoded: list[str] = []
    index = 0
    while index < len(body):
        char = body[index]
        if char != "_":
            decoded.append(char)
            index += 1
            continue
        following = body[index + 1 : index + 2]
        if following == "1":
            decoded.append("_")
            index += 2
        elif following == "2":
            decoded.append(";")
            index += 2
        elif following == "3":
            decoded.append("[")
            index += 2
        elif following == "0":
            code = body[index + 2 : index + 6]
            if len(code) != 4:
                return None
            decoded.append(chr(int(code, 16)))
            index += 6
        elif following == "_":
            break
        else:
            decoded.append("/")
            index += 1
    name = "".join(decoded)
    if "/" not in name:
        return None
    class_path, method = name.rsplit("/", maxsplit=1)
    if not class_path or not method:
        return None
    return class_path.replace("/", "."), method


def jni_class_references(
    blob: bytes, universe: set[str], packages: Sequence[str]
) -> set[str]:
    """Return program classes a native library refers to by binary name.

    Native code looks classes up with names such as ``ai/onnxruntime/OnnxTensor``
    or ``Lai/onnxruntime/TensorInfo;``. Rust libraries concatenate string
    literals without separators, so each occurrence of a package prefix is
    resolved to the longest name in *universe* that starts there.

    Args:
        blob: Raw bytes of a native library.
        universe: Internal names (``a/b/C``) of every class in the program.
        packages: Internal package prefixes to look for, for example
            ``ai/onnxruntime``.

    Returns:
        Dotted names of the referenced classes.
    """
    references: set[str] = set()
    for package in packages:
        prefix = package.strip("/").encode("ascii") + b"/"
        start = blob.find(prefix)
        while start != -1:
            end = start
            while end < len(blob) and blob[end] in IDENTIFIER_BYTES:
                end += 1
            candidate = blob[start:end].decode("ascii")
            for length in range(len(candidate), len(prefix), -1):
                if candidate[:length] in universe:
                    references.add(candidate[:length].replace("/", "."))
                    break
            start = blob.find(prefix, start + 1)
    return references


def jni_native_bindings(blob: bytes) -> set[tuple[str, str]]:
    """Return ``(class, method)`` pairs exported as ``Java_`` JNI symbols."""
    bindings = set()
    for match in JNI_SYMBOL_PATTERN.finditer(blob):
        decoded = demangle_jni_symbol(match.group().decode("ascii"))
        if decoded is not None:
            bindings.add(decoded)
    return bindings


def merged_manifest_permissions(manifest: str) -> set[str]:
    """Return permissions a merged manifest still requests."""
    return {
        match.group(1)
        for match in PERMISSION_PATTERN.finditer(manifest)
        if not REMOVE_NODE_PATTERN.search(match.group(0))
    }


def read_output_metadata(apk_dir: Path) -> dict[str, Path]:
    """Map each ABI to its split APK using AGP's ``output-metadata.json``.

    Raises:
        GateInputError: If the metadata is missing, malformed, or names an APK
            without exactly one ABI filter.
    """
    metadata_file = apk_dir / OUTPUT_METADATA_FILE
    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        elements = metadata["elements"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise GateInputError("APK output metadata is missing or malformed") from error
    apks: dict[str, Path] = {}
    for element in elements:
        abis = [
            item.get("value")
            for item in element.get("filters", [])
            if item.get("filterType") == "ABI"
        ]
        if len(abis) != 1 or not isinstance(abis[0], str):
            raise GateInputError("every APK must be split by exactly one ABI")
        apk = apk_dir / PurePosixPath(str(element.get("outputFile", ""))).name
        if not apk.is_file():
            raise GateInputError("APK listed in output metadata does not exist")
        apks[abis[0]] = apk
    if not apks:
        raise GateInputError("APK output metadata lists no APKs")
    return apks


def read_properties(text: str) -> dict[str, str]:
    """Parse a simple ``key=value`` properties file."""
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if line and not line.startswith(("#", "!")) and "=" in line:
            key, value = line.split("=", maxsplit=1)
            values[key.strip()] = value.strip()
    return values


def _read_text(path: Path, description: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise GateInputError(f"{description} is missing or unreadable") from error


# --------------------------------------------------------------------------
# verify


@dataclass
class _ApkContents:
    classes: dict[str, DexClass] = field(default_factory=dict)
    native_libraries: dict[str, set[str]] = field(default_factory=dict)
    native_blobs: list[bytes] = field(default_factory=list)
    services: dict[str, list[str]] = field(default_factory=dict)
    desktop_native_entries: set[str] = field(default_factory=set)


def _read_apks(apks: Mapping[str, Path]) -> _ApkContents:
    contents = _ApkContents()
    for abi, apk in sorted(apks.items()):
        contents.native_libraries.setdefault(abi, set())
        try:
            with zipfile.ZipFile(apk) as archive:
                for info in archive.infolist():
                    name = info.filename
                    if DEX_ENTRY_PATTERN.fullmatch(name):
                        contents.classes.update(read_dex(archive.read(info)))
                    elif native := NATIVE_LIBRARY_PATTERN.fullmatch(name):
                        contents.native_libraries.setdefault(native.group(1), set())
                        contents.native_libraries[native.group(1)].add(native.group(2))
                        contents.native_blobs.append(archive.read(info))
                    elif name.startswith(SERVICES_PREFIX) and not name.endswith("/"):
                        service = name[len(SERVICES_PREFIX) :]
                        providers = read_rules(archive.read(info).decode("utf-8"))
                        contents.services.setdefault(service, [])
                        for provider in providers:
                            if provider not in contents.services[service]:
                                contents.services[service].append(provider)
                    elif name.lower().endswith(DESKTOP_NATIVE_SUFFIXES):
                        contents.desktop_native_entries.add(name)
        except (OSError, zipfile.BadZipFile, UnicodeDecodeError) as error:
            raise GateInputError("APK is not a readable zip archive") from error
    return contents


def verify(
    *,
    apks: Mapping[str, Path],
    mapping_text: str,
    usage_text: str,
    configuration_text: str,
    merged_manifest: str,
    consumer_rules_text: str,
    dontwarn_allowlist_text: str,
    jni_packages: Sequence[str],
    required_native_libraries: Sequence[str] = (),
) -> VerifyReport:
    """Run every ``verify`` check over one minified release variant.

    Args:
        apks: ABI name mapped to the split release APK.
        mapping_text: Contents of R8's ``mapping.txt``.
        usage_text: Contents of R8's ``usage.txt``.
        configuration_text: Contents of R8's merged ``configuration.txt``.
        merged_manifest: Contents of the merged ``AndroidManifest.xml``.
        consumer_rules_text: OpenMedKit's ``consumer-rules.pro``.
        dontwarn_allowlist_text: Reviewed ``-dontwarn`` filters, one per line.
        jni_packages: Internal package prefixes whose JNI lookups are checked.
        required_native_libraries: Library names (``libX.so`` -> ``X``) every
            ABI APK must contain.

    Returns:
        The findings and counters.

    Raises:
        GateInputError: If an input cannot be read.
    """
    contents = _read_apks(apks)
    mapping = parse_mapping(mapping_text.splitlines())
    usage = parse_usage(usage_text.splitlines())
    configuration = read_rules(configuration_text)
    allowlist = read_allowlist(dontwarn_allowlist_text)
    findings: set[Finding] = set()

    for abi in sorted(contents.native_libraries):
        for library in required_native_libraries:
            if library not in contents.native_libraries[abi]:
                findings.add(
                    Finding(APK_NATIVE_LIBRARY_MISSING, f"{abi}:lib{library}.so")
                )
    for entry in sorted(contents.desktop_native_entries):
        findings.add(Finding(DESKTOP_NATIVE_RESOURCE_PACKAGED, entry))
    if INTERNET_PERMISSION in merged_manifest_permissions(merged_manifest):
        findings.add(Finding(NETWORK_PERMISSION_DECLARED, INTERNET_PERMISSION))

    if any(rule.startswith("-ignorewarnings") for rule in configuration):
        findings.add(Finding(R8_WARNINGS_IGNORED, "-ignorewarnings"))
    patterns = dontwarn_patterns(configuration)
    for pattern in patterns:
        if pattern not in allowlist:
            findings.add(
                Finding(R8_DONTWARN_NOT_ALLOWLISTED, f"-dontwarn {pattern}".strip())
            )
    applied = set(configuration)
    for rule in read_rules(consumer_rules_text):
        if rule not in applied:
            findings.add(Finding(CONSUMER_RULE_NOT_APPLIED, rule))

    universe = (
        set(mapping.classes)
        | usage.removed_classes
        | set(usage.removed_members)
        | set(contents.classes)
    )
    internal_universe = {name.replace(".", "/") for name in universe}
    references: set[str] = set()
    bindings: set[tuple[str, str]] = set()
    for blob in contents.native_blobs:
        references |= jni_class_references(blob, internal_universe, jni_packages)
        bindings |= jni_native_bindings(blob)

    # Native code resolves these classes, and their fields and methods, by
    # name, so each must survive whole: present under its original name and
    # with no member renamed or removed.
    for name in sorted(references):
        dex_class = contents.classes.get(name)
        if dex_class is None:
            renamed = mapping.classes.get(name, name) != name
            detail = "renamed" if renamed else "removed"
            findings.add(Finding(JNI_CLASS_NOT_KEPT, name, detail))
            continue
        present = dex_class.fields | dex_class.methods
        for member in sorted(mapping.renamed_members.get(name, set()) - present):
            findings.add(Finding(JNI_MEMBER_NOT_KEPT, f"{name}.{member}", "renamed"))
        for member in sorted(usage.removed_members.get(name, set()) - present):
            findings.add(Finding(JNI_MEMBER_NOT_KEPT, f"{name}.{member}", "removed"))

    # Java_<class>_<method> exports bind to native methods by name. A class or
    # method R8 removed is never called; one it renamed no longer links.
    for class_name, method in sorted(bindings):
        if class_name not in universe:
            continue
        dex_class = contents.classes.get(class_name)
        if dex_class is None:
            if mapping.classes.get(class_name, class_name) != class_name:
                findings.add(Finding(JNI_NATIVE_BINDING_RENAMED, class_name, "class"))
        elif method not in dex_class.native_methods and method in (
            mapping.renamed_members.get(class_name, set())
        ):
            findings.add(
                Finding(JNI_NATIVE_BINDING_RENAMED, f"{class_name}.{method}", "method")
            )

    provider_count = 0
    for service, providers in sorted(contents.services.items()):
        if service not in contents.classes:
            continue
        for provider in providers:
            provider_count += 1
            if provider not in contents.classes:
                findings.add(Finding(SERVICE_PROVIDER_MISSING, provider, service))

    counts = {
        "apks": len(apks),
        "dexClasses": len(contents.classes),
        "nativeLibraries": sum(
            len(libs) for libs in contents.native_libraries.values()
        ),
        "jniClassReferences": len(references),
        "jniNativeBindings": len(bindings),
        "serviceProviders": provider_count,
        "dontwarnFilters": len(patterns),
        "consumerRules": len(read_rules(consumer_rules_text)),
    }
    return VerifyReport(findings=sorted(findings), counts=counts)


def _write_properties(path: Path, values: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{key}={value}\n" for key, value in values.items()), encoding="utf-8"
    )


def _verify_command(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    try:
        expectations = [Expectation.parse(value) for value in args.expect_failure]
        mapping_dir = args.mapping.parent
        report = verify(
            apks=read_output_metadata(args.apk_dir),
            mapping_text=_read_text(args.mapping, "R8 mapping.txt"),
            usage_text=_read_text(mapping_dir / USAGE_FILE, "R8 usage.txt"),
            configuration_text=_read_text(
                mapping_dir / CONFIGURATION_FILE, "R8 configuration.txt"
            ),
            merged_manifest=_read_text(args.merged_manifest, "merged manifest"),
            consumer_rules_text=_read_text(args.consumer_rules, "consumer rules"),
            dontwarn_allowlist_text=_read_text(
                args.dontwarn_allowlist, "-dontwarn allowlist"
            ),
            jni_packages=args.jni_package,
            required_native_libraries=args.require_native_library,
        )
    except GateInputError as error:
        print(f"error: {error}", file=stderr)
        return EXIT_INPUT_ERROR

    line_status = "EXPECTED" if expectations else "FAIL"
    for finding in report.findings:
        print(finding.format(line_status), file=stdout)
    counters = ", ".join(f"{key} {value}" for key, value in report.counts.items())
    unmet = [
        expectation
        for expectation in expectations
        if not any(expectation.matches(finding) for finding in report.findings)
    ]
    if expectations:
        passed = not report.passed and not unmet
        label = "negative control"
        if passed:
            summary = (
                f"PASS {label}: the gate failed as expected "
                f"({len(report.findings)} findings; {counters})"
            )
        else:
            missing = ", ".join(expectation.describe() for expectation in unmet)
            summary = f"FAIL {label}: expected findings not reported: {missing}"
    else:
        passed = report.passed
        status = "PASS" if passed else "FAIL"
        summary = (
            f"{status} R8 consumer gate: {len(report.findings)} findings; {counters}"
        )
    print(summary, file=stdout)

    if args.report is not None:
        by_code = Counter(finding.code for finding in report.findings)
        values: dict[str, object] = {"status": "PASS" if passed else "FAIL"}
        values["findings"] = len(report.findings)
        values.update(report.counts)
        values.update(
            {f"finding.{code}": count for code, count in sorted(by_code.items())}
        )
        _write_properties(args.report, values)
    return EXIT_PASS if passed else EXIT_FAIL


# --------------------------------------------------------------------------
# size


def _apk_breakdown(apk: Path) -> tuple[dict[str, int], int]:
    natives: dict[str, int] = {}
    dex_bytes = 0
    try:
        with zipfile.ZipFile(apk) as archive:
            for info in archive.infolist():
                if DEX_ENTRY_PATTERN.fullmatch(info.filename):
                    dex_bytes += info.file_size
                elif native := NATIVE_LIBRARY_PATTERN.fullmatch(info.filename):
                    natives[native.group(2)] = (
                        natives.get(native.group(2), 0) + info.file_size
                    )
    except (OSError, zipfile.BadZipFile) as error:
        raise GateInputError("APK is not a readable zip archive") from error
    return natives, dex_bytes


def measure_sizes(
    apks: Mapping[str, Path],
    baseline_apks: Mapping[str, Path],
    budgets: Mapping[str, str],
) -> list[AbiSize]:
    """Measure each ABI against the baseline app and its committed budget.

    Raises:
        GateInputError: If the ABI sets differ or a budget is missing, not a
            positive integer, or committed for an ABI that is not built.
    """
    if set(apks) != set(baseline_apks):
        raise GateInputError("OpenMedKit and baseline apps must split the same ABIs")
    budget_keys = {
        key
        for key in budgets
        if re.fullmatch(r"consumerApk\.[^.]+\.maxDeltaBytes", key)
    }
    expected_keys = {SIZE_BUDGET_KEY.format(abi=abi) for abi in apks}
    if budget_keys != expected_keys:
        raise GateInputError(
            "consumerApk budgets must cover exactly the built ABIs: "
            + ", ".join(sorted(budget_keys ^ expected_keys))
        )
    sizes = []
    for abi in sorted(apks):
        raw_budget = budgets[SIZE_BUDGET_KEY.format(abi=abi)]
        if not raw_budget.isdigit() or int(raw_budget) <= 0:
            raise GateInputError(f"budget for {abi} must be a positive integer")
        natives, dex_bytes = _apk_breakdown(apks[abi])
        sizes.append(
            AbiSize(
                abi=abi,
                apk_bytes=apks[abi].stat().st_size,
                baseline_bytes=baseline_apks[abi].stat().st_size,
                max_delta_bytes=int(raw_budget),
                native_library_bytes=natives,
                dex_bytes=dex_bytes,
            )
        )
    return sizes


def format_size(size: AbiSize) -> str:
    """Return the value-free diagnostic line for one ABI."""
    status = "PASS" if size.passed else "FAIL"
    natives = ", ".join(
        f"lib{name}.so {value}"
        for name, value in sorted(size.native_library_bytes.items())
    )
    return (
        f"{status} {size.abi}: OpenMedKit adds {size.delta_bytes} bytes "
        f"(budget {size.max_delta_bytes}); APK {size.apk_bytes}, baseline "
        f"{size.baseline_bytes}, dex {size.dex_bytes}; native libraries: {natives}"
    )


def _size_command(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    try:
        sizes = measure_sizes(
            read_output_metadata(args.apk_dir),
            read_output_metadata(args.baseline_apk_dir),
            read_properties(_read_text(args.budgets, "budget file")),
        )
    except GateInputError as error:
        print(f"error: {error}", file=stderr)
        return EXIT_INPUT_ERROR
    for size in sizes:
        print(format_size(size), file=stdout)
    passed = all(size.passed for size in sizes)
    values: dict[str, object] = {
        "status": "PASS" if passed else "FAIL",
        "abis": ",".join(size.abi for size in sizes),
    }
    for size in sizes:
        values[f"{size.abi}.apkBytes"] = size.apk_bytes
        values[f"{size.abi}.baselineApkBytes"] = size.baseline_bytes
        values[f"{size.abi}.deltaBytes"] = size.delta_bytes
        values[f"{size.abi}.maxDeltaBytes"] = size.max_delta_bytes
        values[f"{size.abi}.dexBytes"] = size.dex_bytes
        values[f"{size.abi}.nativeLibraryBytes"] = sum(
            size.native_library_bytes.values()
        )
        values[f"{size.abi}.status"] = "PASS" if size.passed else "FAIL"
    if args.report is not None:
        _write_properties(args.report, values)
    if not passed:
        for size in sizes:
            if not size.passed:
                print(Finding(SIZE_BUDGET_EXCEEDED, size.abi).format(), file=stdout)
    return EXIT_PASS if passed else EXIT_FAIL


# --------------------------------------------------------------------------
# drop-keep-rule


def drop_keep_rule(aar: Path, output: Path, rule: str) -> int:
    """Copy *aar* to *output* without *rule* in its packaged ``proguard.txt``.

    Args:
        aar: Source AAR.
        output: Destination AAR; parent directories are created.
        rule: Rule text to remove, compared after whitespace normalization.

    Returns:
        The number of removed lines.

    Raises:
        GateInputError: If the AAR has no ``proguard.txt`` or the rule is absent,
            so a negative control can never silently become a no-op.
    """
    target = _normalize_rule(rule)
    if not target:
        raise GateInputError("rule must not be empty")
    try:
        with zipfile.ZipFile(aar) as source:
            entries: list[tuple[zipfile.ZipInfo, bytes]] = [
                (info, source.read(info)) for info in source.infolist()
            ]
    except (OSError, zipfile.BadZipFile) as error:
        raise GateInputError("AAR is not a readable zip archive") from error
    removed = 0
    rewritten: list[tuple[zipfile.ZipInfo, bytes]] = []
    for info, payload in entries:
        if info.filename == "proguard.txt":
            kept = []
            for line in payload.decode("utf-8").splitlines(keepends=True):
                if _normalize_rule(line) == target:
                    removed += 1
                else:
                    kept.append(line)
            payload = "".join(kept).encode("utf-8")
        rewritten.append((info, payload))
    if not any(info.filename == "proguard.txt" for info, _ in entries):
        raise GateInputError("AAR does not contain proguard.txt")
    if removed == 0:
        raise GateInputError("rule is not present in the AAR proguard.txt")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as destination:
        for info, payload in rewritten:
            destination.writestr(info, payload)
    return removed


def _drop_command(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    try:
        removed = drop_keep_rule(args.aar, args.output, args.rule)
    except GateInputError as error:
        print(f"error: {error}", file=stderr)
        return EXIT_INPUT_ERROR
    print(
        f"wrote {args.output.name} without {removed} line(s) of {args.rule}",
        file=stdout,
    )
    return EXIT_PASS


# --------------------------------------------------------------------------
# CLI


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    verify_parser = commands.add_parser("verify", help="check one minified variant")
    verify_parser.add_argument("--apk-dir", type=Path, required=True)
    verify_parser.add_argument(
        "--mapping",
        type=Path,
        required=True,
        help="R8 mapping.txt; usage.txt and configuration.txt are read beside it",
    )
    verify_parser.add_argument("--merged-manifest", type=Path, required=True)
    verify_parser.add_argument("--consumer-rules", type=Path, required=True)
    verify_parser.add_argument("--dontwarn-allowlist", type=Path, required=True)
    verify_parser.add_argument(
        "--jni-package", action="append", default=[], required=True
    )
    verify_parser.add_argument("--require-native-library", action="append", default=[])
    verify_parser.add_argument(
        "--expect-failure",
        action="append",
        default=[],
        metavar="CODE[:SUBJECT_PREFIX]",
        help="negative control: pass only when the gate fails with this finding",
    )
    verify_parser.add_argument("--report", type=Path)

    size_parser = commands.add_parser("size", help="enforce per-ABI size budgets")
    size_parser.add_argument("--apk-dir", type=Path, required=True)
    size_parser.add_argument("--baseline-apk-dir", type=Path, required=True)
    size_parser.add_argument("--budgets", type=Path, required=True)
    size_parser.add_argument("--report", type=Path)

    drop_parser = commands.add_parser(
        "drop-keep-rule", help="copy an AAR without one consumer rule"
    )
    drop_parser.add_argument("--aar", type=Path, required=True)
    drop_parser.add_argument("--output", type=Path, required=True)
    drop_parser.add_argument("--rule", required=True)
    return parser


COMMANDS = {
    "verify": _verify_command,
    "size": _size_command,
    "drop-keep-rule": _drop_command,
}


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    """Run the command-line interface."""
    args = build_parser().parse_args(argv)
    return COMMANDS[args.command](args, stdout, stderr)


if __name__ == "__main__":
    raise SystemExit(main())
