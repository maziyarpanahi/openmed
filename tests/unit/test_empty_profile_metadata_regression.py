"""Empty profile reports must retain session and caller metadata."""

import pytest

from openmed.utils import profiling
from openmed.utils.profiling import Profiler, ProfileReport, TimingResult


@pytest.mark.parametrize("method", ["summary", "to_dict"])
def test_empty_profile_retains_metadata(method):
    report = ProfileReport(metadata={"profile": "synthetic", "session_duration_ms": 4})
    data = getattr(report, method)()
    assert data["metadata"] == report.metadata
    assert data["count"] == 0
    assert data["total_ms"] == 0
    assert data["timings"] == []


def test_empty_report_has_empty_metadata():
    assert ProfileReport().to_dict()["metadata"] == {}


def test_session_without_measurements_retains_duration(monkeypatch):
    ticks = iter([1.0, 1.25])
    monkeypatch.setattr(profiling.time, "perf_counter", lambda: next(ticks))
    profiler = Profiler()
    profiler.start()
    profiler.add_metadata("profile", "synthetic")
    profiler.stop()
    data = profiler.report().to_dict()
    assert data["metadata"] == {"profile": "synthetic", "session_duration_ms": 250.0}
    assert data["timings"] == []


def test_nonempty_statistics_are_unchanged():
    report = ProfileReport(
        timings=[TimingResult("one", 0.1), TimingResult("two", 0.3)],
        metadata={"profile": "synthetic"},
    )
    data = report.to_dict()
    assert data["count"] == 2
    assert data["min_ms"] == 100.0
    assert data["max_ms"] == 300.0
    assert data["avg_ms"] == 200.0
    assert data["metadata"] == {"profile": "synthetic"}
