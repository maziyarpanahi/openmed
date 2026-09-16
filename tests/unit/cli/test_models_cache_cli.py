"""Tests for the ``openmed models cache`` commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from openmed.cli import main_module
from openmed.core.hf_hub import CachedModel
from openmed.core.offline import OfflineModeError


def test_models_cache_list_prints_repo_sizes_and_last_used(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        main_module,
        "list_cached_models",
        lambda: [
            CachedModel(
                repo_id="OpenMed/unit-model",
                size_on_disk=12_345_678,
                last_accessed=1_700_000_000.0,
                path=tmp_path / "unit-model",
            )
        ],
    )

    result = main_module.main(["models", "cache", "list"])
    captured = capsys.readouterr()

    assert result == 0
    assert "repo_id" in captured.out
    assert "size_mb" in captured.out
    assert "last_used" in captured.out
    assert "OpenMed/unit-model" in captured.out
    assert "12.346 MB" in captured.out
    assert "2023-11-14T22:13:20+00:00" in captured.out
    assert captured.err == ""


def test_models_cache_list_json_is_scriptable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        main_module,
        "list_cached_models",
        lambda: [
            CachedModel(
                repo_id="OpenMed/unit-model",
                size_on_disk=2_000_000,
                last_accessed=None,
                path=tmp_path / "unit-model",
            )
        ],
    )

    result = main_module.main(["models", "cache", "list", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    assert payload["ok"] is True
    assert payload["command"] == "models cache list"
    assert payload["data"]["models"] == [
        {
            "last_accessed": None,
            "last_used": "never",
            "repo_id": "OpenMed/unit-model",
            "size_bytes": 2_000_000,
            "size_mb": 2.0,
        }
    ]


def test_models_cache_download_resolves_repo_and_returns_local_path(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: dict[str, Any] = {}

    def fake_prefetch(repo_id: str, **kwargs: Any) -> str:
        calls["repo_id"] = repo_id
        calls.update(kwargs)
        return "/synthetic/hf-cache/snapshot"

    monkeypatch.setattr(main_module, "prefetch_model", fake_prefetch)

    result = main_module.main(
        [
            "models",
            "cache",
            "download",
            "disease_detection_tiny",
            "--revision",
            "synthetic-revision",
            "--allow-patterns",
            "*.json",
        ]
    )
    captured = capsys.readouterr()

    assert result == 0
    assert calls["repo_id"] == "OpenMed/OpenMed-NER-DiseaseDetect-TinyMed-135M"
    assert calls["revision"] == "synthetic-revision"
    assert calls["allow_patterns"] == ["*.json"]
    assert calls["config"] is not None
    assert "Model ready: /synthetic/hf-cache/snapshot" in captured.out
    assert captured.err == ""


def test_models_cache_download_reports_offline_cache_miss(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OPENMED_OFFLINE", "1")

    def fail_prefetch(*_args: Any, **_kwargs: Any) -> str:
        raise OfflineModeError(
            "OpenMed/unit-model is not available in the local cache and offline "
            "mode blocks the download."
        )

    monkeypatch.setattr(main_module, "prefetch_model", fail_prefetch)

    result = main_module.main(["models", "cache", "download", "OpenMed/unit-model"])
    captured = capsys.readouterr()

    assert result == 1
    assert "not available in the local cache" in captured.err
    assert "offline mode blocks the download" in captured.err
    assert captured.out == ""


def test_models_cache_clear_requires_confirmation_without_deleting(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_clear(*_args: Any, **_kwargs: Any) -> bool:
        pytest.fail("cache clear was called without --yes")

    monkeypatch.setattr(main_module, "clear_cached_model", fail_clear)

    result = main_module.main(["models", "cache", "clear", "OpenMed/unit-model"])
    captured = capsys.readouterr()

    assert result == 2
    assert "without --yes confirmation" in captured.err
    assert captured.out == ""


def test_models_cache_clear_with_confirmation_targets_resolved_repo(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: dict[str, Any] = {}

    def fake_clear(repo_id: str, **kwargs: Any) -> bool:
        calls["repo_id"] = repo_id
        calls.update(kwargs)
        return True

    monkeypatch.setattr(main_module, "clear_cached_model", fake_clear)

    result = main_module.main(
        ["models", "cache", "clear", "OpenMed/unit-model", "--yes"]
    )
    captured = capsys.readouterr()

    assert result == 0
    assert calls == {"repo_id": "OpenMed/unit-model"}
    assert "Cleared model cache: OpenMed/unit-model" in captured.out
    assert captured.err == ""
