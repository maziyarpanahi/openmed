"""Tests for PHI-safe batch progress callbacks."""

import logging
from dataclasses import FrozenInstanceError
from unittest.mock import Mock, patch

import pytest

from openmed.processing.batch import BatchItem, BatchProcessor, BatchProgress
from openmed.processing.outputs import PredictionResult

LOGGER_NAME = "openmed.processing.batch"


def _mock_prediction() -> Mock:
    return Mock(spec=PredictionResult)


def _configure_analyzer(mock_get_analyze: Mock) -> None:
    mock_get_analyze.return_value = Mock(return_value=_mock_prediction())


@patch("openmed.processing.batch.BatchProcessor._get_analyze_text")
def test_process_texts_on_progress_emits_phi_safe_records(
    mock_get_analyze: Mock,
) -> None:
    _configure_analyzer(mock_get_analyze)
    records: list[BatchProgress] = []
    texts = [
        "Patient Jane Doe called from 555-0100.",
        "Patient John Roe emailed john.roe@example.org.",
        "Patient Alex Kim lives at 12 Oak Street.",
    ]

    processor = BatchProcessor(model_name="test-model", batch_size=2)
    result = processor.process_texts(texts, on_progress=records.append)

    assert result.total_items == len(texts)
    assert [record.completed for record in records] == [1, 2, 3]
    assert [record.total for record in records] == [3, 3, 3]
    assert [record.current_index for record in records] == [0, 1, 2]
    assert all(record.elapsed >= 0 for record in records)

    with pytest.raises(FrozenInstanceError):
        records[0].completed = 99  # type: ignore[misc]

    rendered_records = "\n".join(repr(record) for record in records)
    assert "Jane Doe" not in rendered_records
    assert "555-0100" not in rendered_records
    assert "john.roe@example.org" not in rendered_records
    assert not hasattr(records[0], "text")
    assert not hasattr(records[0], "result")


@patch("openmed.processing.batch.BatchProcessor._get_analyze_text")
def test_process_items_on_progress_emits_completed_counts(
    mock_get_analyze: Mock,
) -> None:
    _configure_analyzer(mock_get_analyze)
    records: list[BatchProgress] = []
    items = [
        BatchItem(id="note-1", text="Patient Alice Example."),
        BatchItem(id="note-2", text="Patient Bob Example."),
    ]

    processor = BatchProcessor(model_name="test-model")
    result = processor.process_items(items, on_progress=records.append)

    assert [item.id for item in result.items] == ["note-1", "note-2"]
    assert [record.completed for record in records] == [1, 2]
    assert [record.total for record in records] == [2, 2]


@patch("openmed.processing.batch.BatchProcessor._get_analyze_text")
def test_on_progress_and_legacy_progress_callback_can_run_together(
    mock_get_analyze: Mock,
) -> None:
    _configure_analyzer(mock_get_analyze)
    records: list[BatchProgress] = []
    legacy_calls: list[tuple[int, int, str, bool]] = []

    def progress_callback(completed, total, item_result) -> None:
        legacy_calls.append((completed, total, item_result.id, item_result.success))

    processor = BatchProcessor(model_name="test-model")
    processor.process_texts(
        ["Patient One.", "Patient Two."],
        ids=["one", "two"],
        progress_callback=progress_callback,
        on_progress=records.append,
    )

    assert [record.completed for record in records] == [1, 2]
    assert legacy_calls == [(1, 2, "one", True), (2, 2, "two", True)]


@patch("openmed.processing.batch.BatchProcessor._get_analyze_text")
def test_progress_callback_errors_are_logged_without_exception_text(
    mock_get_analyze: Mock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _configure_analyzer(mock_get_analyze)
    records: list[BatchProgress] = []
    legacy_calls = []

    def on_progress(progress: BatchProgress) -> None:
        records.append(progress)
        raise RuntimeError("Patient Jane Doe called from 555-0100")

    def progress_callback(completed, total, item_result) -> None:
        legacy_calls.append((completed, total, item_result.id))
        raise RuntimeError("john.roe@example.org")

    processor = BatchProcessor(model_name="test-model")
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        result = processor.process_texts(
            ["Patient Jane Doe called from 555-0100."],
            ids=["note-1"],
            progress_callback=progress_callback,
            on_progress=on_progress,
        )

    assert result.total_items == 1
    assert [record.completed for record in records] == [1]
    assert legacy_calls == [(1, 1, "note-1")]
    assert "on_progress callback raised; continuing batch" in caplog.text
    assert "progress_callback raised; continuing batch" in caplog.text
    assert "Jane Doe" not in caplog.text
    assert "555-0100" not in caplog.text
    assert "john.roe@example.org" not in caplog.text


@patch("openmed.processing.batch.BatchProcessor._get_analyze_text")
def test_process_files_on_progress_includes_read_errors(
    mock_get_analyze: Mock,
    tmp_path,
) -> None:
    _configure_analyzer(mock_get_analyze)
    valid = tmp_path / "valid.txt"
    missing = tmp_path / "missing.txt"
    valid.write_text("Patient File One.", encoding="utf-8")
    records: list[BatchProgress] = []

    processor = BatchProcessor(model_name="test-model")
    result = processor.process_files([valid, missing], on_progress=records.append)

    assert [item.id for item in result.items] == ["valid.txt", "missing.txt"]
    assert result.failed_items == 1
    assert [record.completed for record in records] == [1, 2]
    assert [record.total for record in records] == [2, 2]
    assert [record.current_index for record in records] == [0, 1]


@patch("openmed.processing.batch.BatchProcessor._get_analyze_text")
def test_process_directory_threads_on_progress(
    mock_get_analyze: Mock,
    tmp_path,
) -> None:
    _configure_analyzer(mock_get_analyze)
    (tmp_path / "b.txt").write_text("Patient B.", encoding="utf-8")
    (tmp_path / "a.txt").write_text("Patient A.", encoding="utf-8")
    records: list[BatchProgress] = []

    processor = BatchProcessor(model_name="test-model")
    result = processor.process_directory(tmp_path, on_progress=records.append)

    assert [item.id for item in result.items] == ["a.txt", "b.txt"]
    assert [record.completed for record in records] == [1, 2]
    assert [record.total for record in records] == [2, 2]


@patch("openmed.processing.batch.BatchProcessor._get_analyze_text")
def test_iter_process_threads_on_progress(
    mock_get_analyze: Mock,
) -> None:
    _configure_analyzer(mock_get_analyze)
    records: list[BatchProgress] = []

    processor = BatchProcessor(model_name="test-model", batch_size=2)
    results = list(
        processor.iter_process(
            ["Patient One.", "Patient Two.", "Patient Three."],
            on_progress=records.append,
        )
    )

    assert len(results) == 3
    assert [record.completed for record in records] == [1, 2, 3]
    assert [record.total for record in records] == [3, 3, 3]
