"""Tests for PHI-safe batch progress callbacks."""

from dataclasses import FrozenInstanceError
from unittest.mock import Mock, patch

import pytest

from openmed.processing.batch import BatchItem, BatchProcessor, BatchProgress
from openmed.processing.outputs import PredictionResult


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
