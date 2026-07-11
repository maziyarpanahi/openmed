"""Tests for the Hugging Face Hub model-pull convenience helpers.

The network is always mocked: no real ``huggingface_hub`` download or cache scan
runs here, so the suite stays offline-friendly (see ``AGENTS.md``).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, List

import pytest

from openmed.core import hf_hub
from openmed.core.hf_hub import (
    CachedModel,
    clear_cached_model,
    list_cached_models,
    prefetch_model,
    resolve_repo_id,
)
from openmed.core.offline import OfflineModeError

# A stable legacy registry alias -> Hub repo id pair.
_ALIAS = "disease_detection_superclinical"
_REPO_ID = "OpenMed/OpenMed-NER-DiseaseDetect-SuperClinical-434M"


class _FakeRepo:
    """Minimal stand-in for ``huggingface_hub`` ``CachedRepoInfo``."""

    def __init__(
        self,
        *,
        repo_id: str,
        repo_path: Path,
        size_on_disk: int = 1024,
        last_accessed: float = 123.0,
        repo_type: str = "model",
    ) -> None:
        self.repo_id = repo_id
        self.repo_path = repo_path
        self.size_on_disk = size_on_disk
        self.last_accessed = last_accessed
        self.repo_type = repo_type


class _FakeCacheInfo:
    def __init__(self, repos: List[_FakeRepo]) -> None:
        self.repos = repos


class _FakeCacheNotFound(Exception):
    """Stand-in for ``huggingface_hub.CacheNotFound``."""


def _install_fake_hub(
    monkeypatch: pytest.MonkeyPatch, **attrs: Any
) -> types.ModuleType:
    """Install a fake ``huggingface_hub`` module exposing *attrs*."""

    module = types.ModuleType("huggingface_hub")
    module.CacheNotFound = _FakeCacheNotFound
    for name, value in attrs.items():
        setattr(module, name, value)
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return module


# --- resolve_repo_id -------------------------------------------------------


def test_resolve_repo_id_maps_registry_alias() -> None:
    assert resolve_repo_id(_ALIAS) == _REPO_ID


def test_resolve_repo_id_passes_through_full_id() -> None:
    assert resolve_repo_id("some-org/custom-model") == "some-org/custom-model"


def test_resolve_repo_id_prefixes_bare_name() -> None:
    assert resolve_repo_id("Custom-Model") == "OpenMed/Custom-Model"
    assert resolve_repo_id("Custom-Model", org="Acme") == "Acme/Custom-Model"


def test_resolve_repo_id_rejects_empty() -> None:
    with pytest.raises(ValueError):
        resolve_repo_id("   ")


# --- prefetch_model --------------------------------------------------------


def test_prefetch_model_resolves_alias_before_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[dict[str, Any]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        return "/cache/models--OpenMed--DiseaseDetect"

    _install_fake_hub(monkeypatch, snapshot_download=fake_snapshot_download)
    monkeypatch.delenv("OPENMED_OFFLINE", raising=False)

    path = prefetch_model(_ALIAS, allow_patterns=["*.safetensors"])

    assert path == "/cache/models--OpenMed--DiseaseDetect"
    assert len(calls) == 1
    kwargs = calls[0]
    # The friendly alias is resolved to the Hub repo id before download.
    assert kwargs["repo_id"] == _REPO_ID
    assert kwargs["repo_type"] == "model"
    assert kwargs["local_files_only"] is False
    assert kwargs["allow_patterns"] == ["*.safetensors"]


def test_prefetch_model_forwards_revision_and_cache_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[dict[str, Any]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        return "/cache/snapshot"

    _install_fake_hub(monkeypatch, snapshot_download=fake_snapshot_download)
    monkeypatch.delenv("OPENMED_OFFLINE", raising=False)

    prefetch_model(_ALIAS, revision="v2", cache_dir="/tmp/hf")

    assert calls[0]["revision"] == "v2"
    assert calls[0]["cache_dir"] == "/tmp/hf"


def test_prefetch_model_offline_passes_local_files_only_and_raises_when_uncached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[dict[str, Any]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        # huggingface_hub raises when local_files_only=True and nothing is cached.
        raise FileNotFoundError("no local snapshot")

    _install_fake_hub(monkeypatch, snapshot_download=fake_snapshot_download)
    monkeypatch.setenv("OPENMED_OFFLINE", "1")

    with pytest.raises(OfflineModeError) as excinfo:
        prefetch_model(_ALIAS)

    assert calls[0]["local_files_only"] is True
    message = str(excinfo.value)
    assert _REPO_ID in message
    assert "offline" in message.lower()


def test_prefetch_model_offline_returns_path_when_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_snapshot_download(**kwargs: Any) -> str:
        assert kwargs["local_files_only"] is True
        return "/cache/already-here"

    _install_fake_hub(monkeypatch, snapshot_download=fake_snapshot_download)
    monkeypatch.setenv("OPENMED_OFFLINE", "1")

    assert prefetch_model(_ALIAS) == "/cache/already-here"


def test_prefetch_model_online_error_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_snapshot_download(**kwargs: Any) -> str:
        raise RuntimeError("network exploded")

    _install_fake_hub(monkeypatch, snapshot_download=fake_snapshot_download)
    monkeypatch.delenv("OPENMED_OFFLINE", raising=False)

    with pytest.raises(RuntimeError, match="network exploded"):
        prefetch_model(_ALIAS)


# --- list_cached_models ----------------------------------------------------


def test_list_cached_models_filters_openmed_repos(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repos = [
        _FakeRepo(repo_id="other/model", repo_path=tmp_path / "other"),
        _FakeRepo(
            repo_id="OpenMed/B-model",
            repo_path=tmp_path / "b",
            size_on_disk=200,
            last_accessed=2.0,
        ),
        _FakeRepo(
            repo_id="OpenMed/A-model",
            repo_path=tmp_path / "a",
            size_on_disk=100,
            last_accessed=1.0,
        ),
        # A non-model repo type must be ignored even under the org.
        _FakeRepo(
            repo_id="OpenMed/some-dataset",
            repo_path=tmp_path / "d",
            repo_type="dataset",
        ),
    ]

    def fake_scan_cache_dir(cache_dir: Any = None) -> _FakeCacheInfo:
        return _FakeCacheInfo(repos)

    _install_fake_hub(monkeypatch, scan_cache_dir=fake_scan_cache_dir)

    cached = list_cached_models()

    assert [item.repo_id for item in cached] == ["OpenMed/A-model", "OpenMed/B-model"]
    assert all(isinstance(item, CachedModel) for item in cached)
    assert cached[0].size_on_disk == 100
    assert cached[0].last_accessed == 1.0


def test_list_cached_models_absent_cache_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_scan_cache_dir(cache_dir: Any = None) -> _FakeCacheInfo:
        raise FileNotFoundError("cache dir missing")

    _install_fake_hub(monkeypatch, scan_cache_dir=fake_scan_cache_dir)

    assert list_cached_models() == []


def test_list_cached_models_hf_cache_not_found_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_scan_cache_dir(cache_dir: Any = None) -> _FakeCacheInfo:
        raise _FakeCacheNotFound("cache dir missing")

    _install_fake_hub(monkeypatch, scan_cache_dir=fake_scan_cache_dir)

    assert list_cached_models(cache_dir="/missing/hf-cache") == []


# --- clear_cached_model ----------------------------------------------------


def test_clear_cached_model_removes_matching_repo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_path = tmp_path / "models--OpenMed--DiseaseDetect"
    repo_path.mkdir()
    (repo_path / "config.json").write_text("{}", encoding="utf-8")

    def fake_scan_cache_dir(cache_dir: Any = None) -> _FakeCacheInfo:
        return _FakeCacheInfo([_FakeRepo(repo_id=_REPO_ID, repo_path=repo_path)])

    _install_fake_hub(monkeypatch, scan_cache_dir=fake_scan_cache_dir)

    # Resolve via the friendly alias; the resolved repo id must match.
    removed = clear_cached_model(_ALIAS)

    assert removed is True
    assert not repo_path.exists()


def test_clear_cached_model_missing_repo_returns_false(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_scan_cache_dir(cache_dir: Any = None) -> _FakeCacheInfo:
        return _FakeCacheInfo(
            [_FakeRepo(repo_id="OpenMed/Other", repo_path=tmp_path / "other")]
        )

    _install_fake_hub(monkeypatch, scan_cache_dir=fake_scan_cache_dir)

    assert clear_cached_model(_ALIAS) is False


# --- optional [hf] extra guarding ------------------------------------------


def test_helpers_raise_actionable_error_without_hf_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate huggingface_hub not being installed.
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    with pytest.raises(ImportError, match=r"openmed\[hf\]"):
        prefetch_model(_ALIAS)

    with pytest.raises(ImportError, match=r"openmed\[hf\]"):
        list_cached_models()


def test_importing_openmed_succeeds_without_hf_hub() -> None:
    # The module is imported at collection time; reaching here proves that
    # ``import openmed`` (and openmed.core.hf_hub) works without the extra.
    assert hasattr(hf_hub, "prefetch_model")
    assert "prefetch_model" in hf_hub.__all__
