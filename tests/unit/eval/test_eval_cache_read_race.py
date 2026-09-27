"""Deterministic cache read/invalidate interleavings, with no thread sleeps."""

import json
from unittest.mock import Mock

import pytest

from openmed.eval import cache


def prepare(tmp_path, monkeypatch, reader):
    path = cache.cache_path("synthetic-key", cache_dir=tmp_path)
    path.write_text('{"synthetic": true}', encoding="utf-8")
    monkeypatch.setattr(cache.BenchmarkReport, "read_json", staticmethod(reader))
    return path


def test_invalidation_between_lookup_and_read_is_a_miss(tmp_path, monkeypatch):
    def reader(path):
        assert cache.invalidate("synthetic-key", cache_dir=tmp_path)
        return json.loads(path.read_text(encoding="utf-8"))

    path = prepare(tmp_path, monkeypatch, reader)
    assert cache.load("synthetic-key", cache_dir=tmp_path) is None
    assert not path.exists()


def test_clear_between_lookup_and_read_is_a_miss(tmp_path, monkeypatch):
    def reader(path):
        assert cache.clear(cache_dir=tmp_path) == 1
        return json.loads(path.read_text(encoding="utf-8"))

    prepare(tmp_path, monkeypatch, reader)
    assert cache.load("synthetic-key", cache_dir=tmp_path) is None


def test_preexisting_miss_is_none(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cache.BenchmarkReport,
        "read_json",
        staticmethod(lambda path: json.loads(path.read_text(encoding="utf-8"))),
    )
    assert cache.load("absent", cache_dir=tmp_path) is None


def test_successful_read_preserves_identity(tmp_path, monkeypatch):
    report = object()
    prepare(tmp_path, monkeypatch, lambda _path: report)
    assert cache.load("synthetic-key", cache_dir=tmp_path) is report


@pytest.mark.parametrize(
    "error", [PermissionError("synthetic"), ValueError("synthetic")]
)
def test_non_missing_errors_are_not_silenced(tmp_path, monkeypatch, error):
    def reader(path):
        raise error

    prepare(tmp_path, monkeypatch, reader)
    with pytest.raises(type(error)) as raised:
        cache.load("synthetic-key", cache_dir=tmp_path)
    assert raised.value is error


def test_malformed_json_error_is_not_silenced(tmp_path, monkeypatch):
    path = prepare(
        tmp_path, monkeypatch, lambda p: json.loads(p.read_text(encoding="utf-8"))
    )
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        cache.load("synthetic-key", cache_dir=tmp_path)


def test_load_or_compute_recomputes_after_invalidation(tmp_path, monkeypatch):
    # Use a report double only at serialization boundaries, not inside cache.load.
    class ReportDouble:
        @staticmethod
        def read_json(path):
            path.unlink()
            return path.read_text(encoding="utf-8")

    monkeypatch.setattr(cache, "BenchmarkReport", ReportDouble)
    cache.cache_path("synthetic-key", cache_dir=tmp_path).write_text("{}")
    report = ReportDouble()
    compute = Mock(return_value=report)
    store = Mock()
    monkeypatch.setattr(cache, "store", store)
    assert cache.load_or_compute("synthetic-key", compute, cache_dir=tmp_path) is report
    compute.assert_called_once_with()
    store.assert_called_once_with("synthetic-key", report, cache_dir=tmp_path)
