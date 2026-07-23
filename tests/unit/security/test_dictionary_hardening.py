"""Security and property tests for untrusted dictionary ingestion."""

from __future__ import annotations

import logging
import os
import re
import struct
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from openmed.processing.tokenization import (
    DictionaryArchiveError,
    DictionaryEncodingError,
    DictionaryEntryLimitError,
    DictionaryEntryValidationError,
    DictionaryExpansionLimitError,
    DictionaryIngestionError,
    DictionaryLimits,
    DictionaryRecordLimitError,
    UserDictionaryEntry,
    load_user_dictionary,
    validate_user_dictionary_entry,
)


def _fuzz_examples() -> int:
    value = os.environ.get("OPENMED_INGESTION_FUZZ_EXAMPLES", "500")
    try:
        return max(500, int(value))
    except ValueError as exc:  # pragma: no cover - explicit local misuse
        raise RuntimeError(
            "OPENMED_INGESTION_FUZZ_EXAMPLES must be an integer"
        ) from exc


FUZZ_EXAMPLES = _fuzz_examples()
FUZZ_LIMITS = DictionaryLimits(
    max_compressed_bytes=2_048,
    max_decompressed_bytes=1_024,
    max_entries=32,
    max_entry_bytes=128,
    max_term_characters=64,
    max_expansion_ratio=100,
)


def test_loads_valid_literal_dictionary(tmp_path: Path) -> None:
    path = tmp_path / "terms.txt"
    path.write_text("心房颤动 200 n\nCOVID-19\n# comment\n", encoding="utf-8")

    assert load_user_dictionary(path) == (
        UserDictionaryEntry("心房颤动", 200, "n"),
        UserDictionaryEntry("COVID-19"),
    )


def test_zip_bomb_is_rejected_before_member_decompression(tmp_path: Path) -> None:
    archive_path = tmp_path / "terms.zip"
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr("terms.txt", b"safe-term\n" * 200_000)

    with patch.object(
        zipfile.ZipFile,
        "open",
        side_effect=AssertionError("archive member must not be opened"),
    ):
        with pytest.raises(DictionaryExpansionLimitError):
            load_user_dictionary(archive_path)


def test_single_member_stored_zip_loads_and_multiple_members_fail_closed(
    tmp_path: Path,
) -> None:
    valid_path = tmp_path / "valid.zip"
    with zipfile.ZipFile(valid_path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("terms.txt", "患者 10 n\n")
    assert load_user_dictionary(valid_path) == (UserDictionaryEntry("患者", 10, "n"),)

    invalid_path = tmp_path / "multiple.zip"
    with zipfile.ZipFile(invalid_path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("first.txt", "patient\n")
        archive.writestr("second.txt", "alias\n")
    with pytest.raises(DictionaryArchiveError):
        load_user_dictionary(invalid_path)


def test_zip_entry_count_is_rejected_before_central_directory_parsing(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "directory-metadata.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("metadata/", b"")
        archive.writestr("terms.txt", b"patient\n")

    with patch.object(
        zipfile,
        "ZipFile",
        side_effect=AssertionError("central directory must not be materialized"),
    ):
        with pytest.raises(DictionaryArchiveError):
            load_user_dictionary(archive_path)


def test_forged_single_entry_count_cannot_hide_extra_metadata(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "forged-count.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("metadata/", b"")
        archive.writestr("terms.txt", b"patient\n")

    payload = bytearray(archive_path.read_bytes())
    end_record = payload.rfind(b"PK\x05\x06")
    assert end_record >= 0
    struct.pack_into("<HH", payload, end_record + 8, 1, 1)
    archive_path.write_bytes(payload)

    with patch.object(
        zipfile,
        "ZipFile",
        side_effect=AssertionError("central directory must not be materialized"),
    ):
        with pytest.raises(DictionaryArchiveError):
            load_user_dictionary(archive_path)


@pytest.mark.timeout(10)
def test_ten_million_entries_stop_at_the_default_cap(tmp_path: Path) -> None:
    path = tmp_path / "ten-million.txt"
    chunk = b"x\n" * 100_000
    with path.open("wb") as handle:
        for _ in range(100):
            handle.write(chunk)

    with pytest.raises(DictionaryEntryLimitError) as exc_info:
        load_user_dictionary(path)

    assert exc_info.value.observed_count == 100_001


def test_blank_and_comment_records_consume_the_record_budget(tmp_path: Path) -> None:
    path = tmp_path / "empty-records.txt"
    path.write_text("\n# comment\n\n", encoding="utf-8")
    limits = DictionaryLimits(max_records=2)

    with pytest.raises(DictionaryRecordLimitError) as exc_info:
        load_user_dictionary(path, limits=limits)

    assert exc_info.value.observed_count == 3


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("max_compressed_bytes", 16 * 1024 * 1024 + 1),
        ("max_decompressed_bytes", 64 * 1024 * 1024 + 1),
        ("max_entries", 100_001),
        ("max_records", 200_001),
        ("max_entry_bytes", 4 * 1024 + 1),
        ("max_term_characters", 257),
        ("max_expansion_ratio", 100.01),
        ("max_expansion_ratio", float("nan")),
    ],
)
def test_dictionary_limits_are_lower_only(name: str, value: int | float) -> None:
    with pytest.raises(ValueError):
        DictionaryLimits(**{name: value})


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("max_entries", True),
        ("max_entry_bytes", 1.5),
        ("max_expansion_ratio", "10"),
    ],
)
def test_dictionary_limit_types_fail_closed(name: str, value: object) -> None:
    with pytest.raises(TypeError):
        DictionaryLimits(**{name: value})


@pytest.mark.parametrize(
    ("entry", "rule"),
    [
        ("(a+)+$ 10 n", "executable_regex_construct"),
        ("patient\x00alias 10 n", "control_character"),
    ],
)
def test_dangerous_user_entries_name_the_rejected_rule(
    tmp_path: Path,
    entry: str,
    rule: str,
) -> None:
    path = tmp_path / "terms.txt"
    path.write_text(f"{entry}\n", encoding="utf-8")

    with pytest.raises(DictionaryEntryValidationError) as exc_info:
        load_user_dictionary(path)

    assert exc_info.value.rule == rule
    assert rule in str(exc_info.value)
    assert entry not in str(exc_info.value)


def test_rejection_logs_only_phi_safe_file_metadata(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "SyntheticPatientAlice"
    path = tmp_path / f"{secret}-dictionary.txt"
    path.write_text(f"{secret}\x00alias\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="openmed.processing.tokenization"):
        with pytest.raises(DictionaryEntryValidationError):
            load_user_dictionary(path)

    assert secret not in caplog.text
    assert str(path) not in caplog.text
    assert "size_bytes=" in caplog.text
    assert "entry_count=0" in caplog.text
    assert "reason=entry_validation" in caplog.text
    assert re.search(r"path_hash=[0-9a-f]{64}", caplog.text)


def test_invalid_utf8_raises_typed_content_free_error(tmp_path: Path) -> None:
    path = tmp_path / "private-terms.txt"
    path.write_bytes(b"synthetic-name-\xff\n")

    with pytest.raises(DictionaryEncodingError) as exc_info:
        load_user_dictionary(path)

    assert "synthetic-name" not in str(exc_info.value)
    assert str(path) not in str(exc_info.value)


@pytest.mark.fuzz
@pytest.mark.timeout(30)
@settings(
    max_examples=FUZZ_EXAMPLES,
    deadline=100,
    derandomize=True,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@example(data=b"\xff\xfe\x00truncated")
@example(data=b"x" * 1_025)
@example(data=b"(a+)+$ 10 n\n")
@example(data="Latin\u0430Mixed 10 n\n".encode())
@given(data=st.binary(max_size=1_200))
def test_dictionary_parser_fuzz_target_is_fail_closed(
    data: bytes,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.CRITICAL, logger="openmed.processing.tokenization")
    path = tmp_path / "fuzz-dictionary.txt"
    path.write_bytes(data)

    try:
        entries = load_user_dictionary(path, limits=FUZZ_LIMITS)
    except DictionaryIngestionError:
        return

    assert len(entries) <= FUZZ_LIMITS.max_entries
    for entry in entries:
        assert 0 < len(entry.term) <= FUZZ_LIMITS.max_term_characters
        assert not any(char in r".^$*+?{}[]\|()" for char in entry.term)


@pytest.mark.fuzz
@pytest.mark.timeout(30)
@settings(max_examples=FUZZ_EXAMPLES, deadline=50, derandomize=True)
@example(line="(a+)+$ 10 n")
@example(line="patient\x00alias 10 n")
@example(line="x" * 65)
@example(line="混合Latin\u0430 2 n")
@given(line=st.text(max_size=160))
def test_user_dictionary_entry_validation_fuzz_target(line: str) -> None:
    try:
        entry = validate_user_dictionary_entry(line, limits=FUZZ_LIMITS)
    except DictionaryEntryValidationError as exc:
        assert exc.rule
        assert line not in str(exc)
        return

    if entry is None:
        return
    assert 0 < len(entry.term) <= FUZZ_LIMITS.max_term_characters
    assert not any(char in r".^$*+?{}[]\|()" for char in entry.term)
