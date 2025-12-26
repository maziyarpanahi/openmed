"""Unit tests for batch processing functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from tempfile import TemporaryDirectory

from openmed.processing.batch import (
    BatchProcessor,
    BatchItem,
    BatchItemResult,
    BatchResult,
    process_batch,
)
from openmed.processing.outputs import PredictionResult, EntityPrediction


class TestBatchItem:
    """Tests for BatchItem dataclass."""

    def test_basic_creation(self):
        """Test basic BatchItem creation."""
        item = BatchItem(id="test-1", text="Sample text")
        assert item.id == "test-1"
        assert item.text == "Sample text"
        assert item.source is None
        assert item.metadata is None

    def test_with_metadata(self):
        """Test BatchItem with metadata."""
        item = BatchItem(
            id="test-2",
            text="Sample text",
            source="/path/to/file.txt",
            metadata={"key": "value"},
        )
        assert item.source == "/path/to/file.txt"
        assert item.metadata == {"key": "value"}


class TestBatchItemResult:
    """Tests for BatchItemResult dataclass."""

    def test_successful_result(self):
        """Test successful result creation."""
        mock_result = Mock(spec=PredictionResult)
        mock_result.to_dict.return_value = {"entities": []}

        item_result = BatchItemResult(
            id="test-1",
            result=mock_result,
            processing_time=0.5,
        )

        assert item_result.success is True
        assert item_result.error is None
        assert item_result.processing_time == 0.5

    def test_failed_result(self):
        """Test failed result creation."""
        item_result = BatchItemResult(
            id="test-1",
            error="Processing failed",
            processing_time=0.1,
        )

        assert item_result.success is False
        assert item_result.error == "Processing failed"

    def test_to_dict(self):
        """Test to_dict conversion."""
        mock_result = Mock(spec=PredictionResult)
        mock_result.to_dict.return_value = {"entities": []}

        item_result = BatchItemResult(
            id="test-1",
            result=mock_result,
            processing_time=0.5,
            source="/path/file.txt",
        )

        data = item_result.to_dict()
        assert data["id"] == "test-1"
        assert data["success"] is True
        assert data["processing_time"] == 0.5
        assert data["source"] == "/path/file.txt"


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_empty_result(self):
        """Test empty batch result."""
        result = BatchResult()
        assert result.total_items == 0
        assert result.successful_items == 0
        assert result.failed_items == 0
        assert result.success_rate == 0.0
        assert result.average_processing_time == 0.0

    def test_with_items(self):
        """Test batch result with items."""
        mock_pred = Mock(spec=PredictionResult)
        mock_pred.to_dict.return_value = {}

        items = [
            BatchItemResult(id="1", result=mock_pred, processing_time=0.1),
            BatchItemResult(id="2", result=mock_pred, processing_time=0.2),
            BatchItemResult(id="3", error="Failed", processing_time=0.05),
        ]

        result = BatchResult(items=items, total_processing_time=0.35)

        assert result.total_items == 3
        assert result.successful_items == 2
        assert result.failed_items == 1
        assert result.success_rate == pytest.approx(66.67, rel=0.1)
        assert result.average_processing_time == pytest.approx(0.1167, rel=0.01)

    def test_get_successful_results(self):
        """Test filtering successful results."""
        mock_pred = Mock(spec=PredictionResult)

        items = [
            BatchItemResult(id="1", result=mock_pred, processing_time=0.1),
            BatchItemResult(id="2", error="Failed"),
            BatchItemResult(id="3", result=mock_pred, processing_time=0.1),
        ]

        result = BatchResult(items=items)
        successful = result.get_successful_results()

        assert len(successful) == 2
        assert all(r.success for r in successful)

    def test_get_failed_results(self):
        """Test filtering failed results."""
        mock_pred = Mock(spec=PredictionResult)

        items = [
            BatchItemResult(id="1", result=mock_pred),
            BatchItemResult(id="2", error="Error 1"),
            BatchItemResult(id="3", error="Error 2"),
        ]

        result = BatchResult(items=items)
        failed = result.get_failed_results()

        assert len(failed) == 2
        assert all(not r.success for r in failed)

    def test_summary(self):
        """Test summary generation."""
        result = BatchResult(
            model_name="test-model",
            started_at="2024-01-01T00:00:00",
            completed_at="2024-01-01T00:00:01",
            total_processing_time=1.0,
        )

        summary = result.summary()
        assert "Batch Processing Summary" in summary
        assert "test-model" in summary

    def test_to_dict(self):
        """Test to_dict conversion."""
        result = BatchResult(
            model_name="test-model",
            total_processing_time=1.0,
        )

        data = result.to_dict()
        assert data["model_name"] == "test-model"
        assert data["total_processing_time"] == 1.0
        assert "items" in data


class TestBatchProcessor:
    """Tests for BatchProcessor class."""

    @patch("openmed.processing.batch.BatchProcessor._get_analyze_text")
    def test_process_texts(self, mock_get_analyze):
        """Test processing multiple texts."""
        mock_result = Mock(spec=PredictionResult)
        mock_analyze = Mock(return_value=mock_result)
        mock_get_analyze.return_value = mock_analyze

        processor = BatchProcessor(model_name="test-model")
        texts = ["Text one", "Text two", "Text three"]

        result = processor.process_texts(texts)

        assert result.total_items == 3
        assert mock_analyze.call_count == 3

    @patch("openmed.processing.batch.BatchProcessor._get_analyze_text")
    def test_process_texts_with_ids(self, mock_get_analyze):
        """Test processing texts with custom IDs."""
        mock_result = Mock(spec=PredictionResult)
        mock_analyze = Mock(return_value=mock_result)
        mock_get_analyze.return_value = mock_analyze

        processor = BatchProcessor(model_name="test-model")
        texts = ["Text one", "Text two"]
        ids = ["doc-1", "doc-2"]

        result = processor.process_texts(texts, ids=ids)

        assert result.items[0].id == "doc-1"
        assert result.items[1].id == "doc-2"

    @patch("openmed.processing.batch.BatchProcessor._get_analyze_text")
    def test_process_texts_with_error(self, mock_get_analyze):
        """Test error handling during processing."""
        mock_analyze = Mock(side_effect=[Mock(spec=PredictionResult), Exception("Error")])
        mock_get_analyze.return_value = mock_analyze

        processor = BatchProcessor(model_name="test-model", continue_on_error=True)
        texts = ["Text one", "Text two"]

        result = processor.process_texts(texts)

        assert result.total_items == 2
        assert result.successful_items == 1
        assert result.failed_items == 1

    @patch("openmed.processing.batch.BatchProcessor._get_analyze_text")
    def test_process_texts_stop_on_error(self, mock_get_analyze):
        """Test stopping on first error."""
        mock_analyze = Mock(side_effect=Exception("Error"))
        mock_get_analyze.return_value = mock_analyze

        processor = BatchProcessor(model_name="test-model", continue_on_error=False)
        texts = ["Text one", "Text two"]

        with pytest.raises(Exception, match="Error"):
            processor.process_texts(texts)

    @patch("openmed.processing.batch.BatchProcessor._get_analyze_text")
    def test_process_files(self, mock_get_analyze):
        """Test processing files."""
        mock_result = Mock(spec=PredictionResult)
        mock_analyze = Mock(return_value=mock_result)
        mock_get_analyze.return_value = mock_analyze

        with TemporaryDirectory() as tmp_dir:
            # Create test files
            file1 = Path(tmp_dir) / "test1.txt"
            file2 = Path(tmp_dir) / "test2.txt"
            file1.write_text("Content one", encoding="utf-8")
            file2.write_text("Content two", encoding="utf-8")

            processor = BatchProcessor(model_name="test-model")
            result = processor.process_files([file1, file2])

            assert result.total_items == 2
            assert result.items[0].source == str(file1)
            assert result.items[1].source == str(file2)

    @patch("openmed.processing.batch.BatchProcessor._get_analyze_text")
    def test_process_directory(self, mock_get_analyze):
        """Test processing directory."""
        mock_result = Mock(spec=PredictionResult)
        mock_analyze = Mock(return_value=mock_result)
        mock_get_analyze.return_value = mock_analyze

        with TemporaryDirectory() as tmp_dir:
            # Create test files
            (Path(tmp_dir) / "test1.txt").write_text("Content one", encoding="utf-8")
            (Path(tmp_dir) / "test2.txt").write_text("Content two", encoding="utf-8")
            (Path(tmp_dir) / "other.md").write_text("Markdown", encoding="utf-8")

            processor = BatchProcessor(model_name="test-model")
            result = processor.process_directory(tmp_dir, pattern="*.txt")

            assert result.total_items == 2

    @patch("openmed.processing.batch.BatchProcessor._get_analyze_text")
    def test_process_directory_recursive(self, mock_get_analyze):
        """Test recursive directory processing."""
        mock_result = Mock(spec=PredictionResult)
        mock_analyze = Mock(return_value=mock_result)
        mock_get_analyze.return_value = mock_analyze

        with TemporaryDirectory() as tmp_dir:
            # Create nested structure
            subdir = Path(tmp_dir) / "subdir"
            subdir.mkdir()
            (Path(tmp_dir) / "test1.txt").write_text("Content one", encoding="utf-8")
            (subdir / "test2.txt").write_text("Content two", encoding="utf-8")

            processor = BatchProcessor(model_name="test-model")
            result = processor.process_directory(tmp_dir, pattern="*.txt", recursive=True)

            assert result.total_items == 2

    @patch("openmed.processing.batch.BatchProcessor._get_analyze_text")
    def test_progress_callback(self, mock_get_analyze):
        """Test progress callback is called."""
        mock_result = Mock(spec=PredictionResult)
        mock_analyze = Mock(return_value=mock_result)
        mock_get_analyze.return_value = mock_analyze

        callback_calls = []

        def progress_callback(current, total, result):
            callback_calls.append((current, total))

        processor = BatchProcessor(model_name="test-model")
        processor.process_texts(["A", "B", "C"], progress_callback=progress_callback)

        assert len(callback_calls) == 3
        assert callback_calls[0] == (1, 3)
        assert callback_calls[1] == (2, 3)
        assert callback_calls[2] == (3, 3)

    @patch("openmed.processing.batch.BatchProcessor._get_analyze_text")
    def test_iter_process(self, mock_get_analyze):
        """Test iterator-based processing."""
        mock_result = Mock(spec=PredictionResult)
        mock_analyze = Mock(return_value=mock_result)
        mock_get_analyze.return_value = mock_analyze

        processor = BatchProcessor(model_name="test-model")
        texts = ["Text one", "Text two"]

        results = list(processor.iter_process(texts))

        assert len(results) == 2
        assert all(isinstance(r, BatchItemResult) for r in results)


class TestProcessBatchFunction:
    """Tests for process_batch convenience function."""

    @patch("openmed.processing.batch.BatchProcessor._get_analyze_text")
    def test_process_batch(self, mock_get_analyze):
        """Test process_batch convenience function."""
        mock_result = Mock(spec=PredictionResult)
        mock_analyze = Mock(return_value=mock_result)
        mock_get_analyze.return_value = mock_analyze

        result = process_batch(
            texts=["Text one", "Text two"],
            model_name="test-model",
        )

        assert result.total_items == 2
        assert result.model_name == "test-model"
