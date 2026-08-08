"""Focused tests for the local dataset upload privacy guard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.guard.dataset import (
    BLOCK_ONLY_MODE,
    REDACT_TO_STAGING_MODE,
    DatasetFinding,
    DatasetGuardError,
    DatasetUploadBlockedError,
    DatasetUploadError,
    DatasetUploadGuard,
    scan_dataset_files,
)

SYNTHETIC_EMAIL = "synthetic.contact@example.invalid"
SYNTHETIC_TOKEN = "SYNTHETIC_SUBJECT_001"


def test_default_scan_is_deterministic_and_report_is_privacy_safe(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.csv"
    source.write_text(f"subject,contact\n{SYNTHETIC_TOKEN},{SYNTHETIC_EMAIL}\n")

    first = scan_dataset_files(source)
    second = scan_dataset_files(source)

    assert first == second
    assert first.mode == BLOCK_ONLY_MODE
    assert first.allowed is False
    assert first.file_count == 1
    assert first.finding_count == 1
    assert first.finding_counts == {"EMAIL": 1}
    payload = json.dumps(first.to_dict(), sort_keys=True)
    assert SYNTHETIC_EMAIL not in payload
    assert str(source) not in payload
    assert first.file_ids[0].startswith("sha256:")


def test_block_mode_does_not_call_upload_and_exception_has_no_finding_text(
    tmp_path: Path,
) -> None:
    source = tmp_path / "blocked.csv"
    source.write_text(f"contact\n{SYNTHETIC_EMAIL}\n")
    calls: list[tuple[Path, ...]] = []

    def upload(paths: tuple[Path, ...]) -> str:
        calls.append(paths)
        return "uploaded"

    guard = DatasetUploadGuard(upload, mode="block")
    with pytest.raises(DatasetUploadBlockedError) as caught:
        guard(source)

    assert calls == []
    assert caught.value.report.allowed is False
    assert SYNTHETIC_EMAIL not in str(caught.value)
    assert SYNTHETIC_EMAIL not in json.dumps(caught.value.report.to_dict())


def test_clean_block_mode_returns_counts_and_calls_configured_upload(
    tmp_path: Path,
) -> None:
    source = tmp_path / "clean.csv"
    source.write_text("subject\nsynthetic-row-001\n")
    received: list[tuple[Path, ...]] = []

    def upload(paths: tuple[Path, ...]) -> dict[str, str]:
        received.append(paths)
        return {"status": "ok"}

    result = DatasetUploadGuard(upload, mode=BLOCK_ONLY_MODE)(source)

    assert result.upload_result == {"status": "ok"}
    assert received == [(source,)]
    assert result.report.allowed is True
    assert result.report.finding_count == 0
    assert result.to_dict()["upload_completed"] is True
    assert result.file_ids == result.report.file_ids


def test_redaction_mode_stages_safe_content_without_mutating_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "notes.csv"
    source.write_text(f"subject\n{SYNTHETIC_TOKEN}\n")
    staging = tmp_path / "staging"
    received: list[tuple[Path, ...]] = []

    def scanner(text: str):
        start = text.index(SYNTHETIC_TOKEN)
        return [DatasetFinding("synthetic_id", start, start + len(SYNTHETIC_TOKEN))]

    def upload(paths: tuple[Path, ...]) -> str:
        received.append(paths)
        return "staged-upload"

    result = DatasetUploadGuard(
        upload,
        mode=REDACT_TO_STAGING_MODE,
        scanner=scanner,
        staging_dir=staging,
    )(source)

    staged = received[0][0]
    assert result.upload_result == "staged-upload"
    assert result.report.allowed is True
    assert result.report.finding_counts == {"SYNTHETIC_ID": 1}
    assert result.report.staged_file_ids == result.report.file_ids
    assert staged.parent == staging
    assert staged.name.startswith("openmed-0000-")
    assert staged.name != source.name
    assert SYNTHETIC_TOKEN not in staged.read_text()
    assert SYNTHETIC_TOKEN in source.read_text()
    assert SYNTHETIC_TOKEN not in json.dumps(result.report.to_dict())


def test_custom_scanner_failure_is_safe_and_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "scanner.csv"
    source.write_text("synthetic-row-001\n")

    def broken_scanner(_: str):
        raise RuntimeError("scanner detail must not escape")

    with pytest.raises(DatasetGuardError) as caught:
        scan_dataset_files(source, scanner=broken_scanner)

    assert str(caught.value) == "dataset scanner failed safely"


def test_upload_failure_is_wrapped_without_echoing_exception(tmp_path: Path) -> None:
    source = tmp_path / "upload.csv"
    source.write_text("synthetic-row-001\n")

    def upload(_: tuple[Path, ...]) -> None:
        raise RuntimeError("synthetic sensitive upload detail")

    with pytest.raises(DatasetUploadError) as caught:
        DatasetUploadGuard(upload)(source)

    assert "synthetic sensitive upload detail" not in str(caught.value)
    assert caught.value.report.allowed is True


def test_overlapping_custom_findings_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "overlap.csv"
    source.write_text("synthetic-row-001\n")

    def scanner(_: str):
        return [(0, 8, "one"), (4, 12, "two")]

    with pytest.raises(DatasetGuardError, match="overlapping findings"):
        scan_dataset_files(source, scanner=scanner)
