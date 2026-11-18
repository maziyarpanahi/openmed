"""16 KB APK zip-alignment checker tests with synthetic APK fixtures."""

from __future__ import annotations

import io
import os
import shutil
import struct
import subprocess
import zipfile
from pathlib import Path

import pytest

from scripts.android.check_apk_zip_alignment import (
    EXIT_INPUT_ERROR,
    EXIT_MISALIGNED,
    EXIT_PASS,
    ApkInputError,
    check_apk,
    collect_apks,
    is_native_library,
    main,
    run,
)

PAGE_16_KB = 16 * 1024
PAGE_4_KB = 4 * 1024
# Android's zip-alignment extra field (also written by zipalign and apksigner).
ALIGNMENT_EXTRA_ID = 0xD935
LOCAL_HEADER_SIZE = 30
NATIVE_LIBRARY = "lib/arm64-v8a/libsynthetic.so"
SYNTHETIC_ELF = b"\x7fELF" + bytes(60)


def _padding_extra(data_start_without_extra: int, alignment: int, skew: int) -> bytes:
    """Return an extra field that places entry data at ``k * alignment + skew``."""
    padding = (skew - data_start_without_extra) % alignment
    while padding < 6:
        padding += alignment
    return struct.pack("<HHH", ALIGNMENT_EXTRA_ID, padding - 4, alignment) + bytes(
        padding - 6
    )


def _write_entry(
    archive: zipfile.ZipFile,
    name: str,
    payload: bytes,
    *,
    compressed: bool = False,
    alignment: int | None = None,
    skew: int = 0,
) -> None:
    info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
    if alignment is not None:
        assert archive.fp is not None
        start = archive.fp.tell() + LOCAL_HEADER_SIZE + len(name.encode("utf-8"))
        info.extra = _padding_extra(start, alignment, skew)
    archive.writestr(info, payload)


def _synthetic_apk(
    path: Path,
    *,
    native_alignment: int = PAGE_16_KB,
    native_skew: int = 0,
    native_compressed: bool = False,
    include_native: bool = True,
    resources_skew: int = 0,
) -> Path:
    """Write a tiny synthetic APK whose stored entries sit at chosen offsets."""
    with zipfile.ZipFile(path, "w") as archive:
        _write_entry(archive, "AndroidManifest.xml", b"<manifest/>", compressed=True)
        _write_entry(
            archive,
            "resources.arsc",
            bytes(12),
            alignment=4 if resources_skew == 0 else 8,
            skew=resources_skew,
        )
        if include_native:
            _write_entry(
                archive,
                NATIVE_LIBRARY,
                SYNTHETIC_ELF,
                compressed=native_compressed,
                alignment=None if native_compressed else native_alignment,
                skew=native_skew,
            )
        _write_entry(archive, "classes.dex", b"dex\n035\x00", compressed=True)
    return path


def _misaligned_apk(path: Path) -> Path:
    """Return a deliberately misaligned APK: 4 KB-aligned but not 16 KB-aligned.

    This is the layout AGP releases before 8.5.1 produce for uncompressed
    native libraries.
    """
    return _synthetic_apk(path, native_alignment=PAGE_16_KB, native_skew=PAGE_4_KB)


def _run(paths: list[Path], **kwargs: object) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    code = run(paths, stdout=stdout, stderr=stderr, **kwargs)  # type: ignore[arg-type]
    return code, stdout.getvalue(), stderr.getvalue()


def test_aligned_native_library_passes(tmp_path: Path) -> None:
    apk = _synthetic_apk(tmp_path / "app-debug.apk")

    report = check_apk(apk)

    assert report.passed
    assert report.stored_native_libraries == 1
    assert report.compressed_native_libraries == 0
    assert report.stored_entries == 2


def test_deliberately_misaligned_fixture_fails_16_kb_check(tmp_path: Path) -> None:
    apk = _misaligned_apk(tmp_path / "app-debug.apk")

    report = check_apk(apk)

    assert not report.passed
    assert [entry.name for entry in report.misaligned] == [NATIVE_LIBRARY]
    assert report.misaligned[0].required == PAGE_16_KB
    assert report.misaligned[0].remainder == PAGE_4_KB
    code, stdout, _ = _run([apk])
    assert code == EXIT_MISALIGNED
    assert f"misaligned {NATIVE_LIBRARY}: data offset % 16384 = 4096" in stdout


def test_misaligned_fixture_still_passes_the_legacy_4_kb_page_check(
    tmp_path: Path,
) -> None:
    apk = _misaligned_apk(tmp_path / "app-debug.apk")

    assert check_apk(apk, page_size=PAGE_4_KB).passed
    assert _run([apk], page_size_kb=4)[0] == EXIT_PASS


def test_compressed_native_libraries_are_not_page_checked(tmp_path: Path) -> None:
    apk = _synthetic_apk(tmp_path / "legacy.apk", native_compressed=True)

    report = check_apk(apk)

    assert report.passed
    assert report.stored_native_libraries == 0
    assert report.compressed_native_libraries == 1


def test_other_stored_entries_must_keep_four_byte_alignment(tmp_path: Path) -> None:
    apk = _synthetic_apk(tmp_path / "app-debug.apk", resources_skew=2)

    report = check_apk(apk)

    assert [
        (entry.name, entry.required, entry.remainder) for entry in report.misaligned
    ] == [("resources.arsc", 4, 2)]


def test_require_native_libraries_rejects_a_vacuous_pass(tmp_path: Path) -> None:
    apk = _synthetic_apk(tmp_path / "no-native.apk", include_native=False)

    assert _run([apk])[0] == EXIT_PASS
    code, stdout, _ = _run([apk], require_native_libraries=True)
    assert code == EXIT_MISALIGNED
    assert "no uncompressed native libraries to verify" in stdout


def test_directory_input_checks_every_apk_and_fails_on_any_misalignment(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "apk" / "debug"
    output_dir.mkdir(parents=True)
    _synthetic_apk(output_dir / "a-aligned.apk")
    _misaligned_apk(output_dir / "b-misaligned.apk")
    (output_dir / "output-metadata.json").write_text("{}", encoding="utf-8")

    assert [path.name for path in collect_apks([output_dir])] == [
        "a-aligned.apk",
        "b-misaligned.apk",
    ]
    code, stdout, _ = _run([output_dir], require_native_libraries=True)
    assert code == EXIT_MISALIGNED
    assert "PASS a-aligned.apk" in stdout
    assert "FAIL b-misaligned.apk" in stdout


def test_input_errors_are_value_free(tmp_path: Path) -> None:
    empty_dir = tmp_path / "private-patient-folder"
    empty_dir.mkdir()
    not_a_zip = tmp_path / "broken.apk"
    not_a_zip.write_bytes(b"MRN-SENTINEL-123 not a zip")

    for paths in ([empty_dir], [tmp_path / "missing.apk"], [not_a_zip]):
        code, stdout, stderr = _run(paths)
        assert code == EXIT_INPUT_ERROR
        assert str(tmp_path) not in stdout + stderr
        assert "private-patient-folder" not in stdout + stderr
        assert "MRN-SENTINEL" not in stdout + stderr
    with pytest.raises(ApkInputError):
        check_apk(not_a_zip)


def test_reports_never_include_absolute_paths(tmp_path: Path) -> None:
    apk = _misaligned_apk(tmp_path / "app-debug.apk")

    code, stdout, stderr = _run([apk])

    assert code == EXIT_MISALIGNED
    assert str(tmp_path) not in stdout + stderr


def test_rejects_unsupported_page_sizes(tmp_path: Path) -> None:
    apk = _synthetic_apk(tmp_path / "app-debug.apk")

    assert _run([apk], page_size_kb=8)[0] == EXIT_INPUT_ERROR
    assert _run([apk], alignment=0)[0] == EXIT_INPUT_ERROR


def test_native_library_detection_matches_zipalign_extension_rule() -> None:
    assert is_native_library("lib/x86_64/libonnxruntime.so")
    assert not is_native_library("lib/x86_64/libonnxruntime.so.1")
    assert not is_native_library("assets/model.onnx")


def test_cli_entry_point(tmp_path: Path) -> None:
    aligned = _synthetic_apk(tmp_path / "aligned.apk")
    misaligned = _misaligned_apk(tmp_path / "misaligned.apk")

    assert main([str(aligned), "--require-native-libraries"]) == EXIT_PASS
    assert main([str(misaligned), "--page-size-kb", "16"]) == EXIT_MISALIGNED


def _zipalign_with_page_size_support() -> str | None:
    candidates = []
    for variable in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
        sdk = os.environ.get(variable)
        if sdk:
            build_tools = Path(sdk) / "build-tools"
            if build_tools.is_dir():
                candidates.extend(sorted(build_tools.glob("*/zipalign"), reverse=True))
    on_path = shutil.which("zipalign")
    if on_path:
        candidates.append(Path(on_path))
    for candidate in candidates:
        usage = subprocess.run(
            [str(candidate)], capture_output=True, text=True, check=False
        )
        if "-P <pagesize_kb>" in usage.stdout + usage.stderr:
            return str(candidate)
    return None


@pytest.mark.integration
def test_checker_agrees_with_sdk_zipalign_on_synthetic_fixtures(
    tmp_path: Path,
) -> None:
    zipalign = _zipalign_with_page_size_support()
    if zipalign is None:
        pytest.skip("Android build-tools 35+ zipalign is not installed")
    aligned = _synthetic_apk(tmp_path / "aligned.apk")
    misaligned = _misaligned_apk(tmp_path / "misaligned.apk")

    for apk, expected_pass in ((aligned, True), (misaligned, False)):
        result = subprocess.run(
            [zipalign, "-c", "-P", "16", "4", str(apk)],
            capture_output=True,
            check=False,
        )
        assert (result.returncode == 0) is expected_pass
        assert check_apk(apk).passed is expected_pass
