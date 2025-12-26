"""Unit tests for performance profiling utilities."""

import time
import pytest
from unittest.mock import patch

from openmed.utils.profiling import (
    Profiler,
    ProfileReport,
    TimingResult,
    InferenceMetrics,
    BatchMetrics,
    Timer,
    get_profiler,
    enable_profiling,
    disable_profiling,
    get_profile_report,
    profile,
    timed,
)


class TestTimingResult:
    """Tests for TimingResult dataclass."""

    def test_basic_creation(self):
        """Test basic TimingResult creation."""
        result = TimingResult(name="test_op", duration=0.5)

        assert result.name == "test_op"
        assert result.duration == 0.5
        assert result.metadata is None

    def test_with_metadata(self):
        """Test TimingResult with metadata."""
        result = TimingResult(
            name="test_op",
            duration=0.5,
            metadata={"key": "value"},
        )

        assert result.metadata == {"key": "value"}

    def test_to_dict(self):
        """Test to_dict conversion."""
        result = TimingResult(name="test_op", duration=0.5)
        data = result.to_dict()

        assert data["name"] == "test_op"
        assert data["duration_ms"] == 500.0
        assert data["duration_s"] == 0.5


class TestProfileReport:
    """Tests for ProfileReport dataclass."""

    def test_empty_report(self):
        """Test empty report."""
        report = ProfileReport()

        assert report.total_duration == 0
        assert report.timing_count == 0

    def test_with_timings(self):
        """Test report with timings."""
        timings = [
            TimingResult(name="op1", duration=0.1),
            TimingResult(name="op2", duration=0.2),
        ]
        report = ProfileReport(timings=timings)

        assert report.total_duration == pytest.approx(0.3)
        assert report.timing_count == 2

    def test_get_timing(self):
        """Test getting specific timing."""
        timings = [
            TimingResult(name="op1", duration=0.1),
            TimingResult(name="op2", duration=0.2),
        ]
        report = ProfileReport(timings=timings)

        assert report.get_timing("op1").duration == 0.1
        assert report.get_timing("nonexistent") is None

    def test_get_timings_by_prefix(self):
        """Test filtering timings by prefix."""
        timings = [
            TimingResult(name="model.load", duration=0.1),
            TimingResult(name="model.infer", duration=0.2),
            TimingResult(name="preprocess", duration=0.05),
        ]
        report = ProfileReport(timings=timings)

        model_timings = report.get_timings_by_prefix("model.")
        assert len(model_timings) == 2

    def test_summary(self):
        """Test summary generation."""
        timings = [
            TimingResult(name="op1", duration=0.1),
            TimingResult(name="op2", duration=0.2),
        ]
        report = ProfileReport(timings=timings)
        summary = report.summary()

        assert summary["count"] == 2
        assert summary["total_ms"] == pytest.approx(300.0)
        assert summary["min_ms"] == pytest.approx(100.0)
        assert summary["max_ms"] == pytest.approx(200.0)
        assert summary["avg_ms"] == pytest.approx(150.0)

    def test_format_report(self):
        """Test formatted report generation."""
        timings = [
            TimingResult(name="op1", duration=0.1),
            TimingResult(name="op2", duration=0.2),
        ]
        report = ProfileReport(timings=timings, metadata={"test": "value"})
        formatted = report.format_report(include_metadata=True)

        assert "Performance Profile Report" in formatted
        assert "op1" in formatted
        assert "op2" in formatted
        assert "test:" in formatted


class TestProfiler:
    """Tests for Profiler class."""

    def test_enabled_profiler(self):
        """Test enabled profiler measures time."""
        profiler = Profiler(enabled=True)

        with profiler.measure("test_op"):
            time.sleep(0.01)

        report = profiler.report()
        assert report.timing_count == 1
        assert report.timings[0].name == "test_op"
        assert report.timings[0].duration >= 0.01

    def test_disabled_profiler(self):
        """Test disabled profiler is no-op."""
        profiler = Profiler(enabled=False)

        with profiler.measure("test_op"):
            time.sleep(0.01)

        report = profiler.report()
        assert report.timing_count == 0

    def test_multiple_measurements(self):
        """Test multiple measurements."""
        profiler = Profiler(enabled=True)

        with profiler.measure("op1"):
            time.sleep(0.01)

        with profiler.measure("op2"):
            time.sleep(0.02)

        report = profiler.report()
        assert report.timing_count == 2

    def test_add_timing_manual(self):
        """Test manually adding timing."""
        profiler = Profiler(enabled=True)
        profiler.add_timing("manual_op", 0.5, metadata={"source": "external"})

        report = profiler.report()
        assert report.timing_count == 1
        assert report.timings[0].duration == 0.5

    def test_add_metadata(self):
        """Test adding metadata."""
        profiler = Profiler(enabled=True)
        profiler.add_metadata("model", "test-model")

        report = profiler.report()
        assert report.metadata["model"] == "test-model"

    def test_start_stop_session(self):
        """Test session start/stop."""
        profiler = Profiler(enabled=True)
        profiler.start()

        with profiler.measure("op"):
            time.sleep(0.01)

        profiler.stop()
        report = profiler.report()

        assert "session_duration_ms" in report.metadata

    def test_reset(self):
        """Test reset clears state."""
        profiler = Profiler(enabled=True)

        with profiler.measure("op"):
            pass

        profiler.add_metadata("key", "value")
        profiler.reset()

        report = profiler.report()
        assert report.timing_count == 0
        assert len(report.metadata) == 0


class TestGlobalProfiler:
    """Tests for global profiler functions."""

    def test_get_profiler_creates_instance(self):
        """Test get_profiler creates instance."""
        profiler = get_profiler()
        assert isinstance(profiler, Profiler)

    def test_enable_profiling(self):
        """Test enable_profiling returns enabled profiler."""
        profiler = enable_profiling()
        assert profiler.enabled is True

    def test_disable_profiling(self):
        """Test disable_profiling disables profiler."""
        enable_profiling()
        disable_profiling()

        profiler = get_profiler()
        assert profiler.enabled is False

    def test_get_profile_report(self):
        """Test get_profile_report returns report."""
        report = get_profile_report()
        assert isinstance(report, ProfileReport)


class TestProfileDecorator:
    """Tests for @profile decorator."""

    def test_profile_decorator(self):
        """Test profile decorator records timing."""
        profiler = enable_profiling()
        profiler.reset()

        @profile("decorated_func")
        def test_func():
            time.sleep(0.01)
            return "result"

        result = test_func()

        assert result == "result"
        report = get_profile_report()
        assert any(t.name == "decorated_func" for t in report.timings)

    def test_profile_decorator_default_name(self):
        """Test profile decorator uses function name by default."""
        profiler = enable_profiling()
        profiler.reset()

        @profile()
        def my_function():
            pass

        my_function()

        report = get_profile_report()
        assert any(t.name == "my_function" for t in report.timings)


class TestTimer:
    """Tests for Timer class."""

    def test_basic_timing(self):
        """Test basic timing."""
        timer = Timer()
        timer.start()
        time.sleep(0.01)
        elapsed = timer.stop()

        assert elapsed >= 10  # At least 10ms

    def test_elapsed_ms(self):
        """Test elapsed_ms property."""
        timer = Timer()
        timer.start()
        time.sleep(0.01)
        timer.stop()

        assert timer.elapsed_ms >= 10

    def test_elapsed_s(self):
        """Test elapsed_s property."""
        timer = Timer()
        timer.start()
        time.sleep(0.01)
        timer.stop()

        assert timer.elapsed_s >= 0.01

    def test_reset(self):
        """Test timer reset."""
        timer = Timer()
        timer.start()
        time.sleep(0.01)
        timer.stop()
        timer.reset()

        assert timer.elapsed_ms == 0

    def test_chaining(self):
        """Test method chaining."""
        timer = Timer().start()
        time.sleep(0.01)
        elapsed = timer.stop()

        assert elapsed >= 10

    def test_stop_without_start_raises(self):
        """Test stopping without starting raises error."""
        timer = Timer()

        with pytest.raises(RuntimeError, match="not started"):
            timer.stop()


class TestTimedContextManager:
    """Tests for timed context manager."""

    def test_timed_context(self):
        """Test timed context manager."""
        with timed("test_operation") as timer:
            time.sleep(0.01)

        assert timer.elapsed_ms >= 10

    def test_timed_yields_timer(self):
        """Test timed yields Timer instance."""
        with timed("op") as timer:
            pass

        assert isinstance(timer, Timer)


class TestInferenceMetrics:
    """Tests for InferenceMetrics dataclass."""

    def test_basic_creation(self):
        """Test basic creation."""
        metrics = InferenceMetrics(
            text_length=100,
            token_count=20,
            entity_count=5,
            inference_time_ms=50.0,
            total_time_ms=60.0,
        )

        assert metrics.text_length == 100
        assert metrics.entity_count == 5

    def test_tokens_per_second(self):
        """Test tokens per second calculation."""
        metrics = InferenceMetrics(
            text_length=100,
            token_count=20,
            inference_time_ms=100.0,
            total_time_ms=100.0,
        )

        assert metrics.tokens_per_second == 200  # 20 tokens / 0.1 seconds

    def test_chars_per_second(self):
        """Test chars per second calculation."""
        metrics = InferenceMetrics(
            text_length=1000,
            total_time_ms=100.0,
        )

        assert metrics.chars_per_second == 10000  # 1000 chars / 0.1 seconds

    def test_to_dict(self):
        """Test to_dict conversion."""
        metrics = InferenceMetrics(
            text_length=100,
            entity_count=5,
            total_time_ms=50.0,
        )
        data = metrics.to_dict()

        assert data["text_length"] == 100
        assert data["entity_count"] == 5


class TestBatchMetrics:
    """Tests for BatchMetrics dataclass."""

    def test_empty_batch(self):
        """Test empty batch metrics."""
        metrics = BatchMetrics()

        assert metrics.item_count == 0
        assert metrics.total_chars == 0
        assert metrics.throughput_items_per_second == 0

    def test_with_items(self):
        """Test batch metrics with items."""
        items = [
            InferenceMetrics(text_length=100, entity_count=2, total_time_ms=50),
            InferenceMetrics(text_length=200, entity_count=3, total_time_ms=100),
        ]
        metrics = BatchMetrics(items=items, total_time_ms=150)

        assert metrics.item_count == 2
        assert metrics.total_chars == 300
        assert metrics.total_entities == 5
        assert metrics.avg_time_per_item_ms == 75

    def test_throughput_items_per_second(self):
        """Test items throughput calculation."""
        items = [InferenceMetrics(text_length=100, total_time_ms=100)]
        metrics = BatchMetrics(items=items, total_time_ms=1000)

        assert metrics.throughput_items_per_second == 1.0

    def test_throughput_chars_per_second(self):
        """Test chars throughput calculation."""
        items = [InferenceMetrics(text_length=1000, total_time_ms=100)]
        metrics = BatchMetrics(items=items, total_time_ms=1000)

        assert metrics.throughput_chars_per_second == 1000

    def test_format_report(self):
        """Test formatted report."""
        items = [
            InferenceMetrics(text_length=100, entity_count=2, total_time_ms=50),
        ]
        metrics = BatchMetrics(items=items, total_time_ms=50)
        report = metrics.format_report()

        assert "Batch Processing Metrics" in report
        assert "Items processed: 1" in report
