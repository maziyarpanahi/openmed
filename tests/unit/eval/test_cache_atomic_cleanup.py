"""Atomic report publication and temporary-file failure cleanup."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from openmed.eval import cache


@pytest.fixture
def report():
    result = Mock()
    result.to_json.return_value = json.dumps({"suite": "synthetic", "results": []})
    return result


def install_failure(monkeypatch, phase, failure):
    original_factory = cache.tempfile.NamedTemporaryFile

    if phase == "replace":

        def replace(self, target):
            raise failure

        monkeypatch.setattr(Path, "replace", replace)
        return

    @contextmanager
    def temporary_file(*args, **kwargs):
        with original_factory(*args, **kwargs) as handle:
            if phase == "write":

                def write(payload):
                    handle.write(payload[:3])
                    handle.flush()
                    raise failure

                yield SimpleNamespace(name=handle.name, write=write)
            else:
                yield handle
                raise failure  # Simulate a flush/close error from __exit__.

    monkeypatch.setattr(cache.tempfile, "NamedTemporaryFile", temporary_file)


@pytest.mark.parametrize("phase", ["write", "close", "replace"])
@pytest.mark.parametrize("existing", [False, True])
def test_failed_store_preserves_old_report_and_removes_its_temp(
    monkeypatch, tmp_path, report, phase, existing
):
    path = cache.cache_path("synthetic", cache_dir=tmp_path)
    old_payload = b'{"previous":true}\n'
    if existing:
        path.write_bytes(old_payload)
    unrelated = tmp_path / "unrelated.tmp"
    unrelated.write_text("leave unchanged", encoding="utf-8")
    failure = OSError("synthetic publication failure")
    install_failure(monkeypatch, phase, failure)

    with pytest.raises(OSError) as caught:
        cache.store("synthetic", report, cache_dir=tmp_path)
    assert caught.value is failure
    assert path.exists() is existing
    if existing:
        assert path.read_bytes() == old_payload
    assert unrelated.read_text(encoding="utf-8") == "leave unchanged"
    assert set(tmp_path.iterdir()) == ({unrelated, path} if existing else {unrelated})


@pytest.mark.parametrize("existing", [False, True])
def test_successful_store_publishes_complete_json(tmp_path, report, existing):
    path = cache.cache_path("synthetic", cache_dir=tmp_path)
    if existing:
        path.write_text("old report", encoding="utf-8")
    assert cache.store("synthetic", report, cache_dir=tmp_path) == path
    assert path.read_text(encoding="utf-8") == report.to_json.return_value + "\n"
    assert json.loads(path.read_text(encoding="utf-8"))["suite"] == "synthetic"
    assert list(tmp_path.iterdir()) == [path]
    report.to_json.assert_called_once_with(indent=2)


def test_serialization_failure_does_not_create_a_temporary_file(
    monkeypatch, tmp_path, report
):
    failure = ValueError("synthetic serialization failure")
    report.to_json.side_effect = failure
    factory = Mock(side_effect=AssertionError("must not create a temporary file"))
    monkeypatch.setattr(cache.tempfile, "NamedTemporaryFile", factory)
    with pytest.raises(ValueError) as caught:
        cache.store("synthetic", report, cache_dir=tmp_path)
    assert caught.value is failure
    factory.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_temp_creation_failure_does_not_mask_the_original_error(
    monkeypatch, tmp_path, report
):
    failure = OSError("synthetic descriptor limit")
    monkeypatch.setattr(cache.tempfile, "NamedTemporaryFile", Mock(side_effect=failure))
    with pytest.raises(OSError) as caught:
        cache.store("synthetic", report, cache_dir=tmp_path)
    assert caught.value is failure
    assert list(tmp_path.iterdir()) == []


def test_cleanup_failure_does_not_mask_the_write_error(monkeypatch, tmp_path, report):
    failure = OSError("synthetic write failure")
    install_failure(monkeypatch, "write", failure)
    monkeypatch.setattr(Path, "unlink", Mock(side_effect=PermissionError("cleanup")))
    with pytest.raises(OSError) as caught:
        cache.store("synthetic", report, cache_dir=tmp_path)
    assert caught.value is failure
    assert not cache.cache_path("synthetic", cache_dir=tmp_path).exists()


def test_interrupted_write_also_cleans_up(monkeypatch, tmp_path, report):
    failure = KeyboardInterrupt()
    install_failure(monkeypatch, "write", failure)
    with pytest.raises(KeyboardInterrupt) as caught:
        cache.store("synthetic", report, cache_dir=tmp_path)
    assert caught.value is failure
    assert list(tmp_path.iterdir()) == []
