"""16 KB ELF LOAD-alignment checker tests with synthetic ELF fixtures."""

from __future__ import annotations

import datetime as dt
import io
import os
import re
import shutil
import struct
import subprocess
import tomllib
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.android.check_elf_page_alignment import (
    EXIT_INPUT_ERROR,
    EXIT_MISALIGNED,
    EXIT_PASS,
    ElfFormatError,
    LibraryResult,
    NativeInputError,
    evaluate,
    is_packaged_native_entry,
    iter_results,
    load_allowlist,
    main,
    parse_allowlist,
    read_elf_alignment,
    run,
)

ROOT = Path(__file__).parents[3]
ALLOWLIST = ROOT / "android" / "gradle" / "native-page-alignment-allowlist.toml"
CATALOG = ROOT / "android" / "gradle" / "libs.versions.toml"
ROOT_BUILD = ROOT / "android" / "build.gradle.kts"
LIBRARY_BUILD = ROOT / "android" / "openmedkit" / "build.gradle.kts"
ANDROID_CI = ROOT / ".github" / "workflows" / "android-ci.yml"
FLUTTER_CI = ROOT / ".github" / "workflows" / "flutter-ci.yml"
FLUTTER_CMAKE = ROOT / "js" / "openmedkit-flutter" / "native" / "CMakeLists.txt"
SCRIPT = "scripts/android/check_elf_page_alignment.py"

PAGE_4_KB = 4 * 1024
PAGE_16_KB = 16 * 1024
PAGE_64_KB = 64 * 1024
EM_386 = 3
EM_ARM = 40
EM_X86_64 = 62
EM_AARCH64 = 183
PT_LOAD = 1
PT_DYNAMIC = 2
PT_GNU_STACK = 0x6474E551
TODAY = dt.date(2026, 11, 7)
SENTINEL = "MRN-SENTINEL-4242"


def synthetic_elf(
    load_alignments: Sequence[int] = (PAGE_16_KB, PAGE_16_KB),
    *,
    elf_class: int = 64,
    machine: int = EM_AARCH64,
    big_endian: bool = False,
    program_header_size: int | None = None,
    program_header_count: int | None = None,
    program_header_offset: int | None = None,
) -> bytes:
    """Return a minimal ELF shared object with the given LOAD alignments.

    Each ``PT_LOAD`` segment gets ``p_align`` from *load_alignments*. A
    ``PT_DYNAMIC`` and a ``PT_GNU_STACK`` header with small alignments are
    added so the checker must filter by segment type.
    """
    order = ">" if big_endian else "<"
    is_64 = elf_class == 64
    header_size = 64 if is_64 else 52
    entry_size = 56 if is_64 else 32
    segments = [(PT_LOAD, align) for align in load_alignments]
    segments += [(PT_DYNAMIC, 8), (PT_GNU_STACK, 16)]
    count = len(segments) if program_header_count is None else program_header_count
    phoff = header_size if program_header_offset is None else program_header_offset
    file_size = header_size + len(segments) * entry_size
    ident = b"\x7fELF" + bytes([2 if is_64 else 1, 2 if big_endian else 1, 1, 0])
    ident += bytes(8)
    header_format = "HHIQQQIHHHHHH" if is_64 else "HHIIIIIHHHHHH"
    header = ident + struct.pack(
        order + header_format,
        3,  # ET_DYN
        machine,
        1,
        0,
        phoff,
        0,
        0,
        header_size,
        entry_size if program_header_size is None else program_header_size,
        count,
        64 if is_64 else 40,
        0,
        0,
    )
    table = b""
    for index, (p_type, align) in enumerate(segments):
        vaddr = index * 0x100000
        if is_64:
            table += struct.pack(
                order + "IIQQQQQQ",
                p_type,
                5,
                0,
                vaddr,
                vaddr,
                file_size,
                file_size,
                align,
            )
        else:
            table += struct.pack(
                order + "IIIIIIII",
                p_type,
                0,
                vaddr,
                vaddr,
                file_size,
                file_size,
                5,
                align,
            )
    return header + table


def _archive(
    path: Path, entries: dict[str, bytes], *, compressed: bool = False
) -> Path:
    method = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w", compression=method) as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
    return path


def _run(paths: Sequence[Path], **kwargs: object) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    kwargs.setdefault("today", TODAY)
    code = run(paths, stdout=stdout, stderr=stderr, **kwargs)  # type: ignore[arg-type]
    return code, stdout.getvalue(), stderr.getvalue()


def _allowlist(tmp_path: Path, body: str, name: str = "allowlist.toml") -> Path:
    path = tmp_path / name
    path.write_text("schema_version = 1\n" + body, encoding="utf-8")
    return path


DJL_EXCEPTION = """
[[exceptions]]
library = "libdjl_tokenizer.so"
abis = ["arm64-v8a", "x86_64"]
owner = "maintainer"
expires = 2027-02-01
tracking = "https://example.invalid/issues/1"
reason = "Synthetic exception for tests."
"""


def _djl_aar(path: Path, alignment: int = PAGE_4_KB) -> Path:
    return _archive(
        path,
        {
            "classes.jar": b"PK",
            "jni/arm64-v8a/libdjl_tokenizer.so": synthetic_elf((alignment,) * 4),
            "jni/x86_64/libdjl_tokenizer.so": synthetic_elf(
                (alignment,) * 4, machine=EM_X86_64
            ),
            "jni/armeabi-v7a/libdjl_tokenizer.so": synthetic_elf(
                (PAGE_4_KB,), elf_class=32, machine=EM_ARM
            ),
        },
    )


# ELF parsing ---------------------------------------------------------------


def test_reads_every_load_alignment_and_ignores_other_segments() -> None:
    elf = read_elf_alignment(io.BytesIO(synthetic_elf((PAGE_16_KB, PAGE_64_KB))))

    assert elf.elf_class == 64
    assert elf.abi == "arm64-v8a"
    assert elf.load_alignments == (PAGE_16_KB, PAGE_64_KB)
    assert elf.min_load_alignment == PAGE_16_KB


@pytest.mark.parametrize(
    ("machine", "elf_class", "abi"),
    [
        (EM_AARCH64, 64, "arm64-v8a"),
        (EM_X86_64, 64, "x86_64"),
        (EM_ARM, 32, "armeabi-v7a"),
        (EM_386, 32, "x86"),
        (999, 64, "machine-999"),
    ],
)
def test_abi_comes_from_the_elf_machine(machine: int, elf_class: int, abi: str) -> None:
    elf = read_elf_alignment(
        io.BytesIO(synthetic_elf((PAGE_16_KB,), elf_class=elf_class, machine=machine))
    )

    assert elf.abi == abi
    assert elf.elf_class == elf_class


def test_big_endian_headers_are_parsed() -> None:
    elf = read_elf_alignment(
        io.BytesIO(synthetic_elf((PAGE_4_KB, PAGE_16_KB), big_endian=True))
    )

    assert elf.load_alignments == (PAGE_4_KB, PAGE_16_KB)


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        (b"MRN-SENTINEL-4242 not an ELF", "not-elf"),
        (b"\x7fELF\x03\x01\x01" + bytes(9), "unsupported-elf-class"),
        (b"\x7fELF\x02\x03\x01" + bytes(9), "unsupported-elf-encoding"),
        (synthetic_elf()[:40], "truncated-elf-header"),
        (synthetic_elf()[:100], "truncated-program-headers"),
        (synthetic_elf(()), "no-load-segments"),
        (synthetic_elf(program_header_count=0), "no-program-headers"),
        (synthetic_elf(program_header_count=0xFFFF), "extended-program-header-count"),
        (synthetic_elf(program_header_size=40), "bad-program-header-size"),
        (synthetic_elf(program_header_offset=8), "program-headers-out-of-bounds"),
        (
            synthetic_elf(program_header_offset=8 * 1024 * 1024),
            "program-headers-out-of-bounds",
        ),
    ],
)
def test_malformed_elf_files_raise_fixed_reason_codes(
    payload: bytes, reason: str
) -> None:
    with pytest.raises(ElfFormatError) as error:
        read_elf_alignment(io.BytesIO(payload))

    assert str(error.value) == reason


# Gate behaviour --------------------------------------------------------------


def test_16_kb_aligned_library_passes(tmp_path: Path) -> None:
    library = tmp_path / "libsynthetic.so"
    library.write_bytes(synthetic_elf())

    code, stdout, _ = _run([library], require_native_libraries=True)

    assert code == EXIT_PASS
    assert "PASS arm64-v8a libsynthetic.so: LOAD alignment 16384 (16 KB)" in stdout
    assert stdout.splitlines()[-1].startswith("PASS summary: 1 64-bit libraries")


def test_synthetic_4_kb_aligned_library_is_always_rejected(tmp_path: Path) -> None:
    library = tmp_path / "libsynthetic.so"
    library.write_bytes(synthetic_elf((PAGE_4_KB, PAGE_4_KB)))

    for allowlist in (None, ALLOWLIST):
        code, stdout, _ = _run([library], allowlist_path=allowlist)

        assert code == EXIT_MISALIGNED
        assert (
            "FAIL arm64-v8a libsynthetic.so: LOAD alignment 4096 (4 KB) < 16384"
            in stdout
        )


def test_one_misaligned_load_segment_fails_the_library(tmp_path: Path) -> None:
    library = tmp_path / "libmixed.so"
    library.write_bytes(synthetic_elf((PAGE_16_KB, PAGE_4_KB, PAGE_16_KB)))

    code, stdout, _ = _run([library])

    assert code == EXIT_MISALIGNED
    assert "LOAD alignment 4096 (4 KB) < 16384" in stdout


def test_minimum_alignment_is_configurable(tmp_path: Path) -> None:
    library = tmp_path / "libsixteen.so"
    library.write_bytes(synthetic_elf())

    assert _run([library], min_alignment_kb=4)[0] == EXIT_PASS
    code, stdout, _ = _run([library], min_alignment_kb=64)
    assert code == EXIT_MISALIGNED
    assert "< 65536" in stdout
    assert _run([library], min_alignment_kb=8)[0] == EXIT_INPUT_ERROR


def test_32_bit_libraries_are_skipped_but_cannot_satisfy_the_gate(
    tmp_path: Path,
) -> None:
    library = tmp_path / "libarm32.so"
    library.write_bytes(synthetic_elf((PAGE_4_KB,), elf_class=32, machine=EM_ARM))

    code, stdout, _ = _run([library])
    assert code == EXIT_PASS
    assert "SKIP armeabi-v7a libarm32.so: 32-bit library" in stdout

    code, stdout, _ = _run([library], require_native_libraries=True)
    assert code == EXIT_MISALIGNED
    assert "FAIL no 64-bit native libraries to verify" in stdout


def test_malformed_library_fails_without_echoing_its_contents(tmp_path: Path) -> None:
    library = tmp_path / "libbroken.so"
    library.write_bytes(SENTINEL.encode())

    code, stdout, stderr = _run([library])

    assert code == EXIT_MISALIGNED
    assert "FAIL unknown-abi libbroken.so: malformed ELF (not-elf)" in stdout
    assert SENTINEL not in stdout + stderr


# Archives and directories ------------------------------------------------------


def test_aar_checks_only_the_jni_libraries_android_packages(tmp_path: Path) -> None:
    aar = _archive(
        tmp_path / "vendor-native-1.0.0.aar",
        {
            "AndroidManifest.xml": b"<manifest/>",
            "jni/arm64-v8a/libgood.so": synthetic_elf(),
            "jni/x86_64/libgood.so": synthetic_elf(machine=EM_X86_64),
            # Not loaded by Android: desktop natives and nested paths.
            "native/lib/linux-x86_64/libdesktop.so": synthetic_elf((PAGE_4_KB,)),
            "assets/models/libnot-a-library.so": b"not an ELF",
            "jni/arm64-v8a/nested/libdeep.so": synthetic_elf((PAGE_4_KB,)),
        },
    )

    results = list(iter_results([aar]))

    assert [(item.abi, item.library) for item in results] == [
        ("arm64-v8a", "libgood.so"),
        ("x86_64", "libgood.so"),
    ]
    assert {item.source for item in results} == {"vendor-native-1.0.0.aar"}
    assert _run([aar], require_native_libraries=True)[0] == EXIT_PASS


def test_compressed_apk_and_jar_entries_are_read(tmp_path: Path) -> None:
    apk = _archive(
        tmp_path / "app-debug.apk",
        {"lib/arm64-v8a/libapp.so": synthetic_elf((PAGE_4_KB,))},
        compressed=True,
    )
    jar = _archive(
        tmp_path / "bundled-1.0.jar",
        {"lib/x86_64/libjar.so": synthetic_elf(machine=EM_X86_64)},
    )

    code, stdout, _ = _run([apk, jar])

    assert code == EXIT_MISALIGNED
    assert "FAIL arm64-v8a libapp.so [app-debug.apk]" in stdout
    assert "PASS x86_64 libjar.so [bundled-1.0.jar]" in stdout


def test_packaged_native_entry_rule() -> None:
    assert is_packaged_native_entry("lib/arm64-v8a/libonnxruntime.so")
    assert is_packaged_native_entry("jni/x86_64/libdjl_tokenizer.so")
    assert not is_packaged_native_entry("native/lib/linux-x86_64/cpu/libtok.so")
    assert not is_packaged_native_entry("lib/arm64-v8a/libfoo.so.1")
    assert not is_packaged_native_entry("base/lib/arm64-v8a/libfoo.so")


def test_directory_input_finds_libraries_and_archives_recursively(
    tmp_path: Path,
) -> None:
    build = tmp_path / "private-patient-folder" / "build"
    (build / "arm64-v8a").mkdir(parents=True)
    (build / "arm64-v8a" / "libopenmed_ffi.so").write_bytes(synthetic_elf())
    (build / "apk").mkdir()
    _archive(
        build / "apk" / "app-debug.apk",
        {"lib/x86_64/libapp.so": synthetic_elf(machine=EM_X86_64)},
    )
    (build / "apk" / "output-metadata.json").write_text("{}", encoding="utf-8")

    code, stdout, stderr = _run([tmp_path], require_native_libraries=True)

    assert code == EXIT_PASS
    assert "PASS x86_64 libapp.so [app-debug.apk]" in stdout
    assert "PASS arm64-v8a libopenmed_ffi.so: LOAD alignment 16384" in stdout
    assert str(tmp_path) not in stdout + stderr
    assert "private-patient-folder" not in stdout + stderr


def test_input_errors_are_value_free(tmp_path: Path) -> None:
    private_dir = tmp_path / "private-patient-folder"
    private_dir.mkdir()
    broken_aar = private_dir / "broken.aar"
    broken_aar.write_bytes(SENTINEL.encode())
    other_file = private_dir / "notes.txt"
    other_file.write_text(SENTINEL, encoding="utf-8")

    for paths in ([broken_aar], [other_file], [private_dir / "missing.so"], []):
        code, stdout, stderr = _run(paths)

        assert code == EXIT_INPUT_ERROR
        assert str(tmp_path) not in stdout + stderr
        assert "private-patient-folder" not in stdout + stderr
        assert SENTINEL not in stdout + stderr
    with pytest.raises(NativeInputError):
        list(iter_results([broken_aar]))


def test_corrupt_compressed_entry_is_a_value_free_input_error(tmp_path: Path) -> None:
    name = "lib/arm64-v8a/libcorrupt.so"
    apk = _archive(
        tmp_path / "app-debug.apk",
        {name: synthetic_elf() + SENTINEL.encode() * 64},
        compressed=True,
    )
    payload = bytearray(apk.read_bytes())
    data_start = 30 + len(name)
    payload[data_start : data_start + 32] = b"\xff" * 32
    apk.write_bytes(bytes(payload))

    code, stdout, stderr = _run([apk])

    assert code == EXIT_INPUT_ERROR
    assert stderr == "error: not a readable zip archive\n"
    assert SENTINEL not in stdout + stderr


def test_report_lines_carry_only_names_abis_and_alignment(tmp_path: Path) -> None:
    private_dir = tmp_path / "private-patient-folder"
    private_dir.mkdir()
    aar = _djl_aar(private_dir / "tokenizer-native.aar")
    allowlist = _allowlist(tmp_path, DJL_EXCEPTION)

    _, stdout, stderr = _run([aar], allowlist_path=allowlist)

    assert stderr == ""
    line = re.compile(
        r"(PASS|FAIL|ALLOW|SKIP|NOTE) "
        r"(?:(?:arm64-v8a|x86_64|armeabi-v7a|x86|unknown-abi) [\w.-]+\.so"
        r" \[tokenizer-native\.aar\]: .+|summary: .+)"
    )
    for output_line in stdout.splitlines():
        assert line.fullmatch(output_line), output_line
    assert str(tmp_path) not in stdout
    assert "private-patient-folder" not in stdout


# Allowlist ---------------------------------------------------------------------


def test_unexpired_allowlist_entry_turns_the_failure_into_allow(tmp_path: Path) -> None:
    aar = _djl_aar(tmp_path / "tokenizer-native-0.33.0.aar")
    allowlist = _allowlist(tmp_path, DJL_EXCEPTION)

    assert _run([aar])[0] == EXIT_MISALIGNED
    code, stdout, _ = _run([aar], allowlist_path=allowlist, strict_allowlist=True)

    assert code == EXIT_PASS
    assert (
        "ALLOW arm64-v8a libdjl_tokenizer.so [tokenizer-native-0.33.0.aar]: LOAD "
        "alignment 4096 (4 KB) < 16384; allowlisted until 2027-02-01 (owner "
        "maintainer, https://example.invalid/issues/1)"
    ) in stdout
    assert "SKIP armeabi-v7a libdjl_tokenizer.so" in stdout
    assert stdout.splitlines()[-1] == (
        "PASS summary: 2 64-bit libraries checked, 0 passed, 0 failed, "
        "2 allowlisted, 1 32-bit skipped (16 KB minimum LOAD alignment)"
    )


def test_allowlist_applies_through_its_expiry_date_only(tmp_path: Path) -> None:
    aar = _djl_aar(tmp_path / "tokenizer-native-0.33.0.aar")
    allowlist = _allowlist(tmp_path, DJL_EXCEPTION)

    assert _run([aar], allowlist_path=allowlist, today=dt.date(2027, 2, 1))[0] == 0
    code, stdout, _ = _run([aar], allowlist_path=allowlist, today=dt.date(2027, 2, 2))

    assert code == EXIT_MISALIGNED
    assert "allowlist entry expired on 2027-02-01 (owner maintainer)" in stdout
    assert "ALLOW" not in stdout


def test_allowlist_covers_only_the_named_library_and_abis(tmp_path: Path) -> None:
    allowlist = _allowlist(
        tmp_path, DJL_EXCEPTION.replace('["arm64-v8a", "x86_64"]', '["arm64-v8a"]')
    )
    aar = _djl_aar(tmp_path / "tokenizer-native-0.33.0.aar")
    renamed = _archive(
        tmp_path / "other.aar",
        {"jni/arm64-v8a/libother.so": synthetic_elf((PAGE_4_KB,))},
    )

    code, stdout, _ = _run([aar, renamed], allowlist_path=allowlist)

    assert code == EXIT_MISALIGNED
    assert "ALLOW arm64-v8a libdjl_tokenizer.so" in stdout
    assert "FAIL x86_64 libdjl_tokenizer.so" in stdout
    assert "FAIL arm64-v8a libother.so" in stdout


def test_strict_allowlist_rejects_entries_that_no_longer_match(
    tmp_path: Path,
) -> None:
    fixed = _djl_aar(tmp_path / "tokenizer-native-9.9.9.aar", alignment=PAGE_16_KB)
    allowlist = _allowlist(tmp_path, DJL_EXCEPTION)

    code, stdout, _ = _run([fixed], allowlist_path=allowlist)
    assert code == EXIT_PASS
    assert "NOTE unused allowlist entry libdjl_tokenizer.so" in stdout

    code, stdout, _ = _run([fixed], allowlist_path=allowlist, strict_allowlist=True)
    assert code == EXIT_MISALIGNED
    assert "FAIL unused allowlist entry libdjl_tokenizer.so (arm64-v8a, x86_64)" in (
        stdout
    )


def test_expired_entries_are_counted_once_in_evaluate() -> None:
    entries = parse_allowlist("schema_version = 1\n" + DJL_EXCEPTION)
    results = [
        LibraryResult("libdjl_tokenizer.so", "a.apk", abi, 64, PAGE_4_KB)
        for abi in ("arm64-v8a", "x86_64")
    ]

    _, summary = evaluate(
        results,
        min_alignment=PAGE_16_KB,
        allowlist=entries,
        today=dt.date(2030, 1, 1),
    )

    assert (summary.failed, summary.allowed, summary.allowlist_errors) == (2, 0, 1)


@pytest.mark.parametrize(
    "body",
    [
        DJL_EXCEPTION.replace('owner = "maintainer"\n', ""),
        DJL_EXCEPTION.replace('owner = "maintainer"', 'owner = "  "'),
        DJL_EXCEPTION.replace("expires = 2027-02-01", 'expires = "2027-02-01"'),
        DJL_EXCEPTION.replace("expires = 2027-02-01", "expires = 2027-02-01T00:00:00"),
        DJL_EXCEPTION.replace('"x86_64"]', '"mips"]'),
        DJL_EXCEPTION.replace('["arm64-v8a", "x86_64"]', "[]"),
        DJL_EXCEPTION.replace('"libdjl_tokenizer.so"', '"jni/x86/libdjl.so"'),
        DJL_EXCEPTION.replace('reason = "', 'surprise = 1\nreason = "'),
        DJL_EXCEPTION + DJL_EXCEPTION,
        "exceptions = 3\n",
        "unexpected = true\n",
        "[[exceptions\n",
    ],
)
def test_invalid_allowlists_are_input_errors(tmp_path: Path, body: str) -> None:
    library = tmp_path / "libsynthetic.so"
    library.write_bytes(synthetic_elf())
    allowlist = _allowlist(tmp_path, body)

    code, stdout, stderr = _run([library], allowlist_path=allowlist)

    assert code == EXIT_INPUT_ERROR
    assert stdout == ""
    assert stderr.startswith("error: allowlist")
    assert str(tmp_path) not in stderr


def test_allowlist_schema_version_is_required(tmp_path: Path) -> None:
    with pytest.raises(NativeInputError, match="schema_version"):
        parse_allowlist(DJL_EXCEPTION)
    with pytest.raises(NativeInputError, match="not readable"):
        load_allowlist(tmp_path / "missing.toml")


# Committed allowlist and wiring -------------------------------------------------


def _catalog() -> dict[str, Any]:
    return tomllib.loads(CATALOG.read_text(encoding="utf-8"))


def test_committed_allowlist_entries_are_owned_dated_and_track_the_catalog() -> None:
    catalog = _catalog()
    versions = catalog["versions"]
    modules = {
        library["module"]: versions[library["version"]["ref"]]
        for library in catalog["libraries"].values()
        if "version" in library
    }

    for entry in load_allowlist(ALLOWLIST):
        assert set(entry.abis) <= {"arm64-v8a", "x86_64"}
        assert entry.owner and entry.reason and entry.upstream
        assert entry.tracking.startswith("https://github.com/maziyarpanahi/openmed/")
        assert entry.artifact is not None
        module, version = entry.artifact.rsplit(":", maxsplit=1)
        # A dependency bump must revisit the exception it relied on.
        assert modules.get(module) == version, entry.artifact


def test_djl_tokenizer_exception_matches_the_pinned_android_native_artifact() -> None:
    (entry,) = [
        item
        for item in load_allowlist(ALLOWLIST)
        if item.library == "libdjl_tokenizer.so"
    ]
    versions = _catalog()["versions"]

    assert (
        entry.artifact
        == f"ai.djl.android:tokenizer-native:{versions['djl-tokenizers']}"
    )
    assert entry.abis == ("arm64-v8a", "x86_64")
    assert entry.upstream == "https://github.com/deepjavalibrary/djl/issues/3815"


def test_library_build_gates_release_aar_and_runtime_dependencies() -> None:
    build = LIBRARY_BUILD.read_text(encoding="utf-8")
    task = build[build.index("val verifyReleaseNativeLibraryAlignment") :]
    task = task[: task.index('tasks.register("verifyAndroidBudgets")')]
    budgets = build[build.index('tasks.register("verifyAndroidBudgets")') :]
    budgets = budgets[: budgets.index("\n}\n")]

    assert SCRIPT in build
    assert "gradle/native-page-alignment-allowlist.toml" in build
    assert 'onVariants(selector().withName("release"))' in build
    assert "variant.runtimeConfiguration.incoming.artifactView" in build
    assert 'attribute(ArtifactTypeDefinition.ARTIFACT_TYPE_ATTRIBUTE, "aar")' in build
    assert 'dependsOn("bundleReleaseAar")' in task
    assert "inputs.files(releaseRuntimeNativeArchives)" in task
    for flag in ("--strict-allowlist", "--require-native-libraries"):
        assert f'"{flag}"' in task
    assert '"--min-alignment-kb",\n        "16",' in task
    assert "releaseAar.get().asFile.absolutePath" in task
    assert "dependsOn(verifyReleaseNativeLibraryAlignment)" in budgets


def test_root_build_gates_every_apk_elf_alignment() -> None:
    build = ROOT_BUILD.read_text(encoding="utf-8")
    helper = build[build.index("fun Project.registerApkElfAlignmentCheck") :]
    helper = helper[: helper.index("\nsubprojects {")]

    assert SCRIPT in build
    assert "val verifyApkElfAlignment by tasks.registering" in build
    assert build.count("module.registerApkElfAlignmentCheck(") == 2
    assert '"--require-native-libraries"' in helper
    assert '"--allowlist"' in helper
    assert "--strict-allowlist" not in helper
    assert "verifyApkElfAlignment.configure { dependsOn(check) }" in helper


def test_android_ci_runs_both_elf_gates() -> None:
    workflow = yaml.safe_load(ANDROID_CI.read_text(encoding="utf-8"))
    steps = {
        step.get("name"): step for step in workflow["jobs"]["build-and-test"]["steps"]
    }
    triggers = workflow[True] if True in workflow else workflow["on"]

    for event in ("push", "pull_request"):
        assert SCRIPT in triggers[event]["paths"]
    assert (
        ":openmedkit:verifyAndroidBudgets"
        in (steps["Verify Android library budgets"]["run"])
    )
    assert (
        "./gradlew verifyApkElfAlignment"
        in (steps["Verify 16 KB ELF alignment of APK native libraries"]["run"])
    )
    names = list(steps)
    assert names.index("Assemble demo APKs") < names.index(
        "Verify 16 KB ELF alignment of APK native libraries"
    )


def test_flutter_plugin_links_16_kb_pages_and_ci_checks_the_apk() -> None:
    cmake = FLUTTER_CMAKE.read_text(encoding="utf-8")
    android_block = cmake[cmake.index("if(ANDROID)\n  # Android 15+") :]
    android_block = android_block[: android_block.index("endif()")]
    workflow = yaml.safe_load(FLUTTER_CI.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["mobile-build"]["steps"]
    names = [step.get("name") for step in steps]
    check = steps[names.index("Verify 16 KB ELF alignment of Android native libraries")]
    triggers = workflow[True] if True in workflow else workflow["on"]

    assert (
        'target_link_options(openmed_ffi PRIVATE "-Wl,-z,max-page-size=16384")'
        in android_block
    )
    assert names.index("Build Android consumer") < names.index(check["name"])
    assert SCRIPT in check["run"]
    assert "--require-native-libraries" in check["run"]
    assert "flutter-apk/app-debug.apk" in check["run"]
    for event in ("push", "pull_request"):
        assert SCRIPT in triggers[event]["paths"]


def test_cli_entry_point(tmp_path: Path) -> None:
    aligned = tmp_path / "libaligned.so"
    aligned.write_bytes(synthetic_elf())
    misaligned = tmp_path / "libmisaligned.so"
    misaligned.write_bytes(synthetic_elf((PAGE_4_KB,)))

    assert main([str(aligned), "--require-native-libraries"]) == EXIT_PASS
    assert main([str(misaligned), "--min-alignment-kb", "16"]) == EXIT_MISALIGNED
    assert main([str(aligned), "--allowlist", str(ALLOWLIST)]) == EXIT_PASS


# Parity with the NDK's readelf -----------------------------------------------


def _llvm_readelf() -> str | None:
    candidates: list[Path] = []
    for variable in ("ANDROID_NDK_HOME", "ANDROID_NDK_ROOT"):
        ndk = os.environ.get(variable)
        if ndk:
            candidates.extend(
                Path(ndk).glob("toolchains/llvm/prebuilt/*/bin/llvm-readelf")
            )
    for variable in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
        sdk = os.environ.get(variable)
        if sdk:
            candidates.extend(
                sorted(
                    Path(sdk).glob("ndk/*/toolchains/llvm/prebuilt/*/bin/llvm-readelf"),
                    reverse=True,
                )
            )
    for name in ("llvm-readelf", "readelf"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))
    return str(candidates[0]) if candidates else None


@pytest.mark.integration
@pytest.mark.parametrize(
    "alignments", [(PAGE_4_KB, PAGE_4_KB), (PAGE_16_KB, PAGE_64_KB)]
)
def test_checker_agrees_with_readelf_on_synthetic_fixtures(
    tmp_path: Path, alignments: tuple[int, int]
) -> None:
    readelf = _llvm_readelf()
    if readelf is None:
        pytest.skip("llvm-readelf or readelf is not installed")
    library = tmp_path / "libsynthetic.so"
    library.write_bytes(synthetic_elf(alignments))

    output = subprocess.run(
        [readelf, "--program-headers", "--wide", str(library)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    reported = tuple(
        int(line.split()[-1], 16)
        for line in output.splitlines()
        if line.strip().startswith("LOAD")
    )

    assert reported == alignments
    with library.open("rb") as handle:
        assert read_elf_alignment(handle).load_alignments == reported
