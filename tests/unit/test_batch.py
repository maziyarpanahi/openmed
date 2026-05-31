"""Unit tests for batch processing functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import datetime

from openmed.processing.batch import (
    BatchProcessor,
    BatchItem,
    BatchItemResult,
    BatchResult,
    process_batch,
)
from openmed.processing.outputs import PredictionResult, EntityPrediction
from openmed.core.pii import DeidentificationResult, PIIEntity


def _prediction_result(text="Sample text", label="NAME", entity_text="John Doe"):
    return PredictionResult(
        text=text,
        entities=[
            EntityPrediction(
                text=entity_text,
                label=label,
                confidence=0.95,
                start=0,
                end=len(entity_text),
            )
        ],
        model_name="test-model",
        timestamp=datetime.now().isoformat(),
    )


def _deidentification_result(text="Patient John Doe", method="mask"):
    entity = PIIEntity(
        text="John Doe",
        label="NAME",
        confidence=0.95,
        start=8,
        end=16,
        entity_type="NAME",
        original_text="John Doe",
        redacted_text="[NAME]",
    )
    return DeidentificationResult(
        original_text=text,
        deidentified_text=text.replace("John Doe", "[NAME]"),
        pii_entities=[entity],
        method=method,
        timestamp=datetime.now(),
        mapping={"[NAME]": "John Doe"} if method == "mask" else None,
    )


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

    def test_to_dict_with_deidentification_result(self):
        """Test to_dict supports de-identification results."""
        item_result = BatchItemResult(
            id="test-1",
            result=_deidentification_result(),
            processing_time=0.5,
        )

        data = item_result.to_dict()

        assert data["success"] is True
        assert data["result"]["deidentified_text"] == "Patient [NAME]"
        assert data["result"]["num_entities_redacted"] == 1


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

    def test_invalid_operation_raises(self):
        """Test operation validation."""
        with pytest.raises(ValueError, match="Unsupported batch operation"):
            BatchProcessor(operation="redact")

    def test_invalid_batch_size_raises(self):
        """Test batch size validation."""
        with pytest.raises(ValueError, match="Batch size must be positive"):
            BatchProcessor(batch_size=0)

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

    def test_extract_pii_operation_batches_by_batch_size(self, monkeypatch):
        """Test extract_pii operation uses the PII batch helper in chunks."""
        calls = []

        def fake_extract_batch(texts, **kwargs):
            calls.append((list(texts), kwargs))
            return [_prediction_result(text=text) for text in texts]

        monkeypatch.setattr(
            "openmed.core.pii._extract_pii_batch",
            fake_extract_batch,
        )
        monkeypatch.setattr(BatchProcessor, "_get_shared_loader", lambda self: "loader")

        processor = BatchProcessor(
            model_name="pii-model",
            operation="extract_pii",
            batch_size=2,
            confidence_threshold=0.6,
            lang="fr",
            use_smart_merging=False,
            normalize_accents=True,
        )

        result = processor.process_texts(["A", "B", "C"], ids=["a", "b", "c"])

        assert result.total_items == 3
        assert result.successful_items == 3
        assert [call[0] for call in calls] == [["A", "B"], ["C"]]
        assert calls[0][1]["model_name"] == "pii-model"
        assert calls[0][1]["confidence_threshold"] == 0.6
        assert calls[0][1]["loader"] == "loader"
        assert calls[0][1]["batch_size"] == 2
        assert calls[0][1]["lang"] == "fr"
        assert calls[0][1]["use_smart_merging"] is False
        assert calls[0][1]["normalize_accents"] is True
        assert result.items[0].id == "a"

    def test_deidentify_operation_forwards_privacy_kwargs(self, monkeypatch):
        """Test deidentify operation forwards method and anonymizer kwargs."""
        calls = []

        def fake_deidentify_batch(texts, **kwargs):
            calls.append((list(texts), kwargs))
            return [_deidentification_result(text=text, method=kwargs["method"]) for text in texts]

        monkeypatch.setattr(
            "openmed.core.pii._deidentify_batch",
            fake_deidentify_batch,
        )
        monkeypatch.setattr(BatchProcessor, "_get_shared_loader", lambda self: "loader")

        processor = BatchProcessor(
            model_name="pii-model",
            operation="deidentify",
            batch_size=3,
            confidence_threshold=0.8,
            method="replace",
            keep_mapping=True,
            seed=42,
            locale="pt_BR",
            date_shift_days=30,
            shift_dates=True,
            lang="pt",
            normalize_accents=False,
        )

        result = processor.process_texts(["Patient John Doe", "Patient Jane Roe"])

        assert result.successful_items == 2
        kwargs = calls[0][1]
        assert kwargs["method"] == "replace"
        assert kwargs["keep_mapping"] is True
        assert kwargs["seed"] == 42
        assert kwargs["locale"] == "pt_BR"
        assert kwargs["date_shift_days"] == 30
        assert kwargs["shift_dates"] is True
        assert kwargs["lang"] == "pt"
        assert kwargs["normalize_accents"] is False

    def test_pii_progress_callback(self, monkeypatch):
        """Test progress callback runs for PII operations."""
        monkeypatch.setattr(
            "openmed.core.pii._extract_pii_batch",
            lambda texts, **kwargs: [_prediction_result(text=text) for text in texts],
        )

        callback_calls = []
        processor = BatchProcessor(operation="extract_pii", batch_size=2)
        processor.process_texts(
            ["A", "B", "C"],
            progress_callback=lambda current, total, result: callback_calls.append(
                (current, total, result.success)
            ),
        )

        assert callback_calls == [(1, 3, True), (2, 3, True), (3, 3, True)]

    def test_extract_pii_process_files(self, monkeypatch):
        """Test file processing works for PII extraction."""
        monkeypatch.setattr(
            "openmed.core.pii._extract_pii_batch",
            lambda texts, **kwargs: [_prediction_result(text=text) for text in texts],
        )

        with TemporaryDirectory() as tmp_dir:
            file1 = Path(tmp_dir) / "a.txt"
            file2 = Path(tmp_dir) / "b.txt"
            file1.write_text("Patient John Doe", encoding="utf-8")
            file2.write_text("Patient Jane Roe", encoding="utf-8")

            processor = BatchProcessor(operation="extract_pii", batch_size=2)
            result = processor.process_files([file1, file2])

        assert result.total_items == 2
        assert result.successful_items == 2
        assert result.items[0].source == str(file1)

    def test_deidentify_process_directory(self, monkeypatch):
        """Test directory processing works for de-identification."""
        monkeypatch.setattr(
            "openmed.core.pii._deidentify_batch",
            lambda texts, **kwargs: [_deidentification_result(text=text) for text in texts],
        )

        with TemporaryDirectory() as tmp_dir:
            (Path(tmp_dir) / "a.txt").write_text("Patient John Doe", encoding="utf-8")
            (Path(tmp_dir) / "b.md").write_text("ignored", encoding="utf-8")

            processor = BatchProcessor(operation="deidentify", batch_size=2)
            result = processor.process_directory(tmp_dir, pattern="*.txt")

        assert result.total_items == 1
        assert result.successful_items == 1
        assert result.items[0].result.deidentified_text == "Patient [NAME]"

    def test_batch_failure_falls_back_to_item_errors(self, monkeypatch):
        """Test batch failures fall back to per-item handling when allowed."""
        def fail_batch(*args, **kwargs):
            raise RuntimeError("batch boom")

        monkeypatch.setattr("openmed.core.pii._extract_pii_batch", fail_batch)
        with patch("openmed.extract_pii") as mock_extract:
            mock_extract.side_effect = [
                _prediction_result(text="A"),
                RuntimeError("single boom"),
            ]

            processor = BatchProcessor(
                operation="extract_pii",
                batch_size=2,
                continue_on_error=True,
            )
            result = processor.process_texts(["A", "B"])

        assert result.successful_items == 1
        assert result.failed_items == 1
        assert "single boom" in result.items[1].error

    def test_batch_failure_raises_when_continue_on_error_false(self, monkeypatch):
        """Test batch failures raise when continue_on_error is disabled."""
        def fail_batch(*args, **kwargs):
            raise RuntimeError("batch boom")

        monkeypatch.setattr("openmed.core.pii._extract_pii_batch", fail_batch)

        processor = BatchProcessor(
            operation="extract_pii",
            batch_size=2,
            continue_on_error=False,
        )

        with pytest.raises(RuntimeError, match="batch boom"):
            processor.process_texts(["A", "B"])


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


class TestPIIBatchHelpers:
    """Focused tests for private PII batch helpers used by BatchProcessor."""

    def test_extract_pii_batch_matches_single_extract(self, monkeypatch):
        """Test batched PII extraction matches single-call extraction."""
        import openmed
        from openmed.core.pii import _extract_pii_batch, extract_pii

        def fake_analyze(text, **kwargs):
            return _prediction_result(text=text, entity_text=text[:4] or "X")

        monkeypatch.setattr(openmed, "analyze_text", fake_analyze)
        loader = object()
        texts = ["John Doe", "Jane Roe"]

        batched = _extract_pii_batch(
            texts,
            model_name="custom-pii-model",
            use_smart_merging=False,
            loader=loader,
        )
        singles = [
            extract_pii(
                text,
                model_name="custom-pii-model",
                use_smart_merging=False,
                loader=loader,
            )
            for text in texts
        ]

        assert [(result.text, [e.to_dict() for e in result.entities]) for result in batched] == [
            (result.text, [e.to_dict() for e in result.entities])
            for result in singles
        ]

    def test_deidentify_batch_matches_single_deidentify(self, monkeypatch):
        """Test batched de-identification matches single-call de-identification."""
        import openmed
        from openmed.core.pii import _deidentify_batch, deidentify

        def fake_analyze(text, **kwargs):
            return PredictionResult(
                text=text,
                entities=[
                    EntityPrediction(
                        text=text[8:16],
                        label="NAME",
                        confidence=0.95,
                        start=8,
                        end=16,
                    )
                ],
                model_name="custom-pii-model",
                timestamp="now",
            )

        monkeypatch.setattr(openmed, "analyze_text", fake_analyze)
        loader = object()
        texts = ["Patient John Doe", "Patient Jane Roe"]

        batched = _deidentify_batch(
            texts,
            model_name="custom-pii-model",
            method="mask",
            keep_mapping=True,
            use_smart_merging=False,
            loader=loader,
        )
        singles = [
            deidentify(
                text,
                model_name="custom-pii-model",
                method="mask",
                keep_mapping=True,
                use_smart_merging=False,
                loader=loader,
            )
            for text in texts
        ]

        assert [result.deidentified_text for result in batched] == [
            result.deidentified_text for result in singles
        ]
        assert [result.mapping for result in batched] == [
            result.mapping for result in singles
        ]

    def test_privacy_filter_batch_reuses_one_pipeline(self):
        """Test privacy-filter batch extraction creates one pipeline."""
        from openmed.core.pii import _extract_pii_batch

        pipeline_calls = []

        def fake_pipeline(texts):
            pipeline_calls.append(list(texts))
            return [
                [
                    {
                        "entity_group": "NAME",
                        "score": 0.95,
                        "word": text,
                        "start": 0,
                        "end": len(text),
                    }
                ]
                for text in texts
            ]

        with patch("openmed.core.backends.create_privacy_filter_pipeline") as factory:
            factory.return_value = fake_pipeline
            results = _extract_pii_batch(
                ["John Doe", "Jane Roe"],
                model_name="openai/privacy-filter",
                confidence_threshold=0.5,
            )

        factory.assert_called_once_with("openai/privacy-filter")
        assert pipeline_calls == [["John Doe", "Jane Roe"]]
        assert [result.entities[0].text for result in results] == [
            "John Doe",
            "Jane Roe",
        ]

    def test_batch_processor_reuses_privacy_filter_pipeline_across_chunks(self):
        """Test BatchProcessor caches one privacy-filter pipeline per job."""
        pipeline_calls = []

        def fake_pipeline(texts, **kwargs):
            pipeline_calls.append((list(texts), kwargs.get("batch_size")))
            return [
                [
                    {
                        "entity_group": "NAME",
                        "score": 0.95,
                        "word": text,
                        "start": 0,
                        "end": len(text),
                    }
                ]
                for text in texts
            ]

        with patch("openmed.core.backends.create_privacy_filter_pipeline") as factory:
            factory.return_value = fake_pipeline
            processor = BatchProcessor(
                model_name="openai/privacy-filter",
                operation="extract_pii",
                batch_size=2,
                confidence_threshold=0.5,
            )
            result = processor.process_texts(["John Doe", "Jane Roe", "Alex Kim"])

        factory.assert_called_once_with("openai/privacy-filter")
        assert pipeline_calls == [
            (["John Doe", "Jane Roe"], 2),
            (["Alex Kim"], 2),
        ]
        assert result.successful_items == 3

    def test_deidentify_batch_forwards_kwargs_to_extraction(self, monkeypatch):
        """Test deidentify batch forwards language and pipeline kwargs."""
        from openmed.core import pii

        captured = {}

        def fake_extract_batch(texts, **kwargs):
            captured["texts"] = texts
            captured["kwargs"] = kwargs
            return [
                PredictionResult(
                    text=text,
                    entities=[],
                    model_name="custom",
                    timestamp="now",
                )
                for text in texts
            ]

        monkeypatch.setattr(pii, "_extract_pii_batch", fake_extract_batch)

        result = pii._deidentify_batch(
            ["Paciente Maria"],
            model_name="custom",
            method="shift_dates",
            keep_mapping=True,
            date_shift_days=30,
            use_smart_merging=False,
            lang="pt",
            normalize_accents=False,
            seed=42,
            locale="pt_BR",
            loader="loader",
            privacy_filter_pipeline="pipeline",
            batch_size=4,
        )

        assert result[0].method == "shift_dates"
        assert captured["kwargs"]["use_smart_merging"] is False
        assert captured["kwargs"]["lang"] == "pt"
        assert captured["kwargs"]["normalize_accents"] is False
        assert captured["kwargs"]["loader"] == "loader"
        assert captured["kwargs"]["privacy_filter_pipeline"] == "pipeline"
        assert captured["kwargs"]["batch_size"] == 4
