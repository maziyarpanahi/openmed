"""Tests for the Hugging Face Hub model-pull convenience helpers.

The network is always mocked: no real ``huggingface_hub`` download or cache scan
runs here, so the suite stays offline-friendly (see ``AGENTS.md``).
"""

from __future__ import annotations

import hashlib
import socket
import sys
import types
from pathlib import Path
from typing import Any, List

import pytest

from openmed.core import hf_hub
from openmed.core.hf_hub import (
    CachedModel,
    DownloadIntegrityError,
    DownloadProgress,
    clear_cached_model,
    list_cached_models,
    prefetch_model,
    resolve_repo_id,
)
from openmed.core.offline import (
    HF_OFFLINE_ENV_VARS,
    OfflineModeError,
    network_blocked_if_offline,
)

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


class _FakeLocalEntryNotFoundError(FileNotFoundError):
    """Stand-in for a genuine Hugging Face local-cache miss."""


def _install_fake_hub(
    monkeypatch: pytest.MonkeyPatch, **attrs: Any
) -> types.ModuleType:
    """Install a fake ``huggingface_hub`` module exposing *attrs*."""

    module = types.ModuleType("huggingface_hub")
    module.CacheNotFound = _FakeCacheNotFound
    model_info = attrs.pop("model_info", None)
    if model_info is not None:

        class _FakeHfApi:
            def model_info(self, *args: Any, **kwargs: Any) -> Any:
                return model_info(*args, **kwargs)

        module.HfApi = _FakeHfApi
    for name, value in attrs.items():
        setattr(module, name, value)
    constants_module = types.ModuleType("huggingface_hub.constants")
    constants_module.HF_HUB_CACHE = "/fake/huggingface/hub"
    errors_module = types.ModuleType("huggingface_hub.errors")
    errors_module.LocalEntryNotFoundError = _FakeLocalEntryNotFoundError
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants_module)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors_module)
    return module


def _model_info(filename: str, content: bytes, *, sha: str = "a" * 40) -> Any:
    return types.SimpleNamespace(
        sha=sha,
        siblings=[
            types.SimpleNamespace(
                rfilename=filename,
                size=len(content),
                blob_id=None,
                lfs=types.SimpleNamespace(sha256=hashlib.sha256(content).hexdigest()),
            )
        ],
    )


def _write_snapshot_file(root: Path, filename: str, content: bytes) -> Path:
    path = root / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: List[dict[str, Any]] = []
    content = b"synthetic weights"
    snapshot = tmp_path / "snapshot"

    def fake_hf_hub_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(_write_snapshot_file(snapshot, kwargs["filename"], content))

    _install_fake_hub(
        monkeypatch,
        snapshot_download=lambda **kwargs: "/offline/cache",
        model_info=lambda *args, **kwargs: _model_info("model.safetensors", content),
        hf_hub_download=fake_hf_hub_download,
    )
    monkeypatch.delenv("OPENMED_OFFLINE", raising=False)

    path = prefetch_model(_ALIAS, allow_patterns=["*.safetensors"])

    assert path == str(snapshot)
    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["repo_id"] == _REPO_ID
    assert kwargs["repo_type"] == "model"
    assert kwargs["local_files_only"] is False
    assert kwargs["filename"] == "model.safetensors"
    assert "endpoint" not in kwargs


def test_prefetch_model_forwards_revision_and_cache_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    api_calls: List[dict[str, Any]] = []
    download_calls: List[dict[str, Any]] = []
    content = b"{}"
    snapshot = tmp_path / "snapshot"

    def fake_model_info(*args: Any, **kwargs: Any) -> Any:
        api_calls.append(kwargs)
        return _model_info("config.json", content, sha="b" * 40)

    def fake_hf_hub_download(**kwargs: Any) -> str:
        download_calls.append(kwargs)
        return str(_write_snapshot_file(snapshot, kwargs["filename"], content))

    _install_fake_hub(
        monkeypatch,
        snapshot_download=lambda **kwargs: "/offline/cache",
        model_info=fake_model_info,
        hf_hub_download=fake_hf_hub_download,
    )
    monkeypatch.delenv("OPENMED_OFFLINE", raising=False)

    prefetch_model(_ALIAS, revision="v2", cache_dir="/tmp/hf")

    assert api_calls[0]["revision"] == "v2"
    assert download_calls[0]["revision"] == "b" * 40
    assert download_calls[0]["cache_dir"] == "/tmp/hf"


def test_prefetch_model_preserves_single_allow_pattern(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: List[dict[str, Any]] = []
    weights = b"weights"
    config = b"{}"
    snapshot = tmp_path / "snapshot"

    def fake_hf_hub_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(_write_snapshot_file(snapshot, kwargs["filename"], weights))

    info = _model_info("model.safetensors", weights)
    info.siblings.extend(_model_info("config.json", config).siblings)
    _install_fake_hub(
        monkeypatch,
        snapshot_download=lambda **kwargs: "/offline/cache",
        model_info=lambda *args, **kwargs: info,
        hf_hub_download=fake_hf_hub_download,
    )
    monkeypatch.delenv("OPENMED_OFFLINE", raising=False)

    prefetch_model(_ALIAS, allow_patterns="*.safetensors")

    assert [call["filename"] for call in calls] == ["model.safetensors"]


def test_prefetch_model_offline_passes_local_files_only_and_raises_when_uncached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[dict[str, Any]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        # huggingface_hub raises when local_files_only=True and nothing is cached.
        raise _FakeLocalEntryNotFoundError("no local snapshot")

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
    def fake_hf_hub_download(**kwargs: Any) -> str:
        raise RuntimeError("network exploded")

    _install_fake_hub(
        monkeypatch,
        snapshot_download=lambda **kwargs: "/offline/cache",
        model_info=lambda *args, **kwargs: _model_info("config.json", b"{}"),
        hf_hub_download=fake_hf_hub_download,
    )
    monkeypatch.delenv("OPENMED_OFFLINE", raising=False)

    with pytest.raises(RuntimeError, match="network exploded"):
        prefetch_model(_ALIAS)


def test_prefetch_model_resumes_partial_file_after_connection_drop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    content = b"x" * 100
    snapshot = tmp_path / "snapshot"
    path = snapshot / "model.bin"
    fetched_per_attempt: List[int] = []
    progress: List[DownloadProgress] = []

    def fake_hf_hub_download(**kwargs: Any) -> str:
        existing = path.read_bytes() if path.exists() else b""
        progress_bar = kwargs["tqdm_class"](
            total=len(content),
            initial=len(existing),
        )
        if not existing:
            chunk = content[: len(content) // 2]
            _write_snapshot_file(snapshot, "model.bin", chunk)
            progress_bar.update(len(chunk))
            fetched_per_attempt.append(len(chunk))
            raise TimeoutError("connection dropped at 50%")

        remainder = content[len(existing) :]
        path.write_bytes(existing + remainder)
        progress_bar.update(len(remainder))
        fetched_per_attempt.append(len(remainder))
        return str(path)

    _install_fake_hub(
        monkeypatch,
        snapshot_download=lambda **kwargs: "/offline/cache",
        model_info=lambda *args, **kwargs: _model_info("model.bin", content),
        hf_hub_download=fake_hf_hub_download,
    )
    monkeypatch.delenv("OPENMED_OFFLINE", raising=False)
    monkeypatch.setattr(hf_hub.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(hf_hub.random, "uniform", lambda start, end: 0.0)

    result = prefetch_model(_ALIAS, retries=1, progress_callback=progress.append)

    assert result == str(snapshot)
    assert fetched_per_attempt == [50, 50]
    assert fetched_per_attempt[1] <= len(content) // 2 + 1
    assert path.read_bytes() == content
    assert progress[-1] == DownloadProgress(
        filename="model.bin",
        bytes_done=100,
        bytes_total=100,
        files_done=1,
        files_total=1,
    )


def test_prefetch_model_refetches_corrupt_cached_file_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    content = b"good"
    snapshot = tmp_path / "snapshot"
    path = _write_snapshot_file(snapshot, "model.bin", b"bad!")
    calls: List[bool] = []

    def fake_hf_hub_download(**kwargs: Any) -> str:
        calls.append(kwargs["force_download"])
        if kwargs["force_download"]:
            _write_snapshot_file(snapshot, "model.bin", content)
        return str(path)

    _install_fake_hub(
        monkeypatch,
        snapshot_download=lambda **kwargs: "/offline/cache",
        model_info=lambda *args, **kwargs: _model_info("model.bin", content),
        hf_hub_download=fake_hf_hub_download,
    )
    monkeypatch.delenv("OPENMED_OFFLINE", raising=False)

    assert prefetch_model(_ALIAS) == str(snapshot)
    assert calls == [False, True]
    assert path.read_bytes() == content


def test_models_pull_exits_nonzero_when_refetched_file_remains_corrupt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from openmed.cli import main_module

    content = b"good"
    snapshot = tmp_path / "snapshot"
    path = _write_snapshot_file(snapshot, "model.bin", b"bad!")
    calls: List[bool] = []

    def fake_hf_hub_download(**kwargs: Any) -> str:
        calls.append(kwargs["force_download"])
        _write_snapshot_file(snapshot, "model.bin", b"bad!")
        return str(path)

    _install_fake_hub(
        monkeypatch,
        snapshot_download=lambda **kwargs: "/offline/cache",
        model_info=lambda *args, **kwargs: _model_info("model.bin", content),
        hf_hub_download=fake_hf_hub_download,
    )
    monkeypatch.delenv("OPENMED_OFFLINE", raising=False)

    result = main_module.main(["models", "pull", _ALIAS])

    captured = capsys.readouterr()
    assert result == 1
    assert calls == [False, True]
    assert "model.bin" in captured.err
    assert "forced re-fetch" in captured.err


def test_prefetch_model_retries_transient_failures_with_backoff(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    content = b"{}"
    snapshot = tmp_path / "snapshot"
    attempts = 0
    delays: List[float] = []

    def fake_hf_hub_download(**kwargs: Any) -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TimeoutError("temporary outage")
        return str(_write_snapshot_file(snapshot, "config.json", content))

    _install_fake_hub(
        monkeypatch,
        snapshot_download=lambda **kwargs: "/offline/cache",
        model_info=lambda *args, **kwargs: _model_info("config.json", content),
        hf_hub_download=fake_hf_hub_download,
    )
    monkeypatch.delenv("OPENMED_OFFLINE", raising=False)
    monkeypatch.setattr(hf_hub.time, "sleep", delays.append)
    monkeypatch.setattr(hf_hub.random, "uniform", lambda start, end: 0.0)

    prefetch_model(_ALIAS, retries=2)

    assert attempts == 3
    assert delays == [1.0, 2.0]


def test_prefetch_model_retries_wrapped_transient_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    content = b"{}"
    snapshot = tmp_path / "snapshot"
    attempts = 0

    def fake_hf_hub_download(**kwargs: Any) -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            try:
                raise TimeoutError("temporary outage")
            except TimeoutError as cause:
                raise _FakeLocalEntryNotFoundError("cache miss") from cause
        return str(_write_snapshot_file(snapshot, "config.json", content))

    _install_fake_hub(
        monkeypatch,
        snapshot_download=lambda **kwargs: "/offline/cache",
        model_info=lambda *args, **kwargs: _model_info("config.json", content),
        hf_hub_download=fake_hf_hub_download,
    )
    monkeypatch.delenv("OPENMED_OFFLINE", raising=False)
    monkeypatch.setattr(hf_hub.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(hf_hub.random, "uniform", lambda start, end: 0.0)

    assert prefetch_model(_ALIAS, retries=1) == str(snapshot)
    assert attempts == 2


def test_prefetch_model_default_retries_five_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    def fake_hf_hub_download(**kwargs: Any) -> str:
        nonlocal attempts
        attempts += 1
        raise TimeoutError("still offline")

    _install_fake_hub(
        monkeypatch,
        snapshot_download=lambda **kwargs: "/offline/cache",
        model_info=lambda *args, **kwargs: _model_info("config.json", b"{}"),
        hf_hub_download=fake_hf_hub_download,
    )
    monkeypatch.delenv("OPENMED_OFFLINE", raising=False)
    monkeypatch.setattr(hf_hub.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(hf_hub.random, "uniform", lambda start, end: 0.0)

    with pytest.raises(TimeoutError, match="still offline"):
        prefetch_model(_ALIAS)

    assert attempts == 6


@pytest.mark.parametrize("status_code", [401, 404])
def test_prefetch_model_does_not_retry_permanent_http_errors(
    monkeypatch: pytest.MonkeyPatch, status_code: int
) -> None:
    attempts = 0

    def fake_hf_hub_download(**kwargs: Any) -> str:
        nonlocal attempts
        attempts += 1
        error = RuntimeError(f"HTTP {status_code}")
        error.response = types.SimpleNamespace(status_code=status_code)  # type: ignore[attr-defined]
        raise error

    _install_fake_hub(
        monkeypatch,
        snapshot_download=lambda **kwargs: "/offline/cache",
        model_info=lambda *args, **kwargs: _model_info("config.json", b"{}"),
        hf_hub_download=fake_hf_hub_download,
    )
    monkeypatch.delenv("OPENMED_OFFLINE", raising=False)

    with pytest.raises(RuntimeError, match=f"HTTP {status_code}"):
        prefetch_model(_ALIAS)

    assert attempts == 1


def test_prefetch_model_does_not_retry_permanent_error_with_transient_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    def fake_hf_hub_download(**kwargs: Any) -> str:
        nonlocal attempts
        attempts += 1
        try:
            raise TimeoutError("earlier timeout")
        except TimeoutError as cause:
            error = RuntimeError("HTTP 404")
            error.response = types.SimpleNamespace(status_code=404)  # type: ignore[attr-defined]
            raise error from cause

    _install_fake_hub(
        monkeypatch,
        snapshot_download=lambda **kwargs: "/offline/cache",
        model_info=lambda *args, **kwargs: _model_info("config.json", b"{}"),
        hf_hub_download=fake_hf_hub_download,
    )
    monkeypatch.delenv("OPENMED_OFFLINE", raising=False)

    with pytest.raises(RuntimeError, match="HTTP 404"):
        prefetch_model(_ALIAS, retries=2)

    assert attempts == 1


def test_models_pull_caps_bandwidth_and_prints_file_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from openmed.cli import main_module

    content = b"x" * 1_048_576
    snapshot = tmp_path / "snapshot"
    clock = [0.0]

    def fake_monotonic() -> float:
        return clock[0]

    def fake_sleep(seconds: float) -> None:
        clock[0] += seconds

    def fake_hf_hub_download(**kwargs: Any) -> str:
        progress_bar = kwargs["tqdm_class"](total=len(content), initial=0)
        for _ in range(4):
            progress_bar.update(len(content) // 4)
        return str(_write_snapshot_file(snapshot, "model.bin", content))

    _install_fake_hub(
        monkeypatch,
        snapshot_download=lambda **kwargs: "/offline/cache",
        model_info=lambda *args, **kwargs: _model_info("model.bin", content),
        hf_hub_download=fake_hf_hub_download,
    )
    monkeypatch.delenv("OPENMED_OFFLINE", raising=False)
    monkeypatch.setattr(hf_hub.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(hf_hub.time, "sleep", fake_sleep)

    result = main_module.main(["models", "pull", _ALIAS, "--max-bandwidth", "524288"])

    output = capsys.readouterr().out
    assert result == 0
    assert len(content) / clock[0] <= 524_288
    assert "model.bin: 1048576/1048576 bytes; 1/1 files" in output
    assert f"Model ready: {snapshot}" in output


def test_prefetch_model_fails_when_integrity_metadata_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    info = types.SimpleNamespace(
        sha="a" * 40,
        siblings=[
            types.SimpleNamespace(
                rfilename="model.bin",
                size=4,
                lfs=None,
                blob_id=None,
            )
        ],
    )
    _install_fake_hub(
        monkeypatch,
        snapshot_download=lambda **kwargs: "/offline/cache",
        model_info=lambda *args, **kwargs: info,
        hf_hub_download=lambda **kwargs: "/unused",
    )
    monkeypatch.delenv("OPENMED_OFFLINE", raising=False)

    with pytest.raises(DownloadIntegrityError, match="model.bin"):
        prefetch_model(_ALIAS)


@pytest.mark.parametrize(
    "invalid_filename",
    ["../secret", "/absolute", "..\\secret", "bad\nname", "\x1b[31mred"],
)
def test_prefetch_model_rejects_unsafe_remote_filenames(
    monkeypatch: pytest.MonkeyPatch, invalid_filename: str
) -> None:
    info = _model_info(invalid_filename, b"content")
    _install_fake_hub(
        monkeypatch,
        snapshot_download=lambda **kwargs: "/offline/cache",
        model_info=lambda *args, **kwargs: info,
        hf_hub_download=lambda **kwargs: "/unused",
    )
    monkeypatch.delenv("OPENMED_OFFLINE", raising=False)

    with pytest.raises(DownloadIntegrityError, match="invalid filename") as exc_info:
        prefetch_model(_ALIAS)

    assert invalid_filename not in str(exc_info.value)


def test_prefetch_model_offline_attempts_no_socket_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[dict[str, Any]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        if not kwargs["local_files_only"]:
            socket.create_connection(("127.0.0.1", 9))
        return "/cache/already-here"

    _install_fake_hub(monkeypatch, snapshot_download=fake_snapshot_download)
    monkeypatch.setenv("OPENMED_OFFLINE", "1")

    try:
        with network_blocked_if_offline(local_only=True):
            result = prefetch_model(_ALIAS)
    finally:
        for env_var in HF_OFFLINE_ENV_VARS:
            monkeypatch.delenv(env_var, raising=False)

    assert result == "/cache/already-here"
    assert len(calls) == 1
    assert calls[0]["local_files_only"] is True


@pytest.mark.parametrize(
    "error",
    [
        PermissionError("cache unreadable"),
        OSError("cache corrupted"),
        ValueError("invalid download arguments"),
    ],
)
def test_prefetch_model_offline_preserves_non_cache_miss_errors(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    def fake_snapshot_download(**kwargs: Any) -> str:
        raise error

    _install_fake_hub(monkeypatch, snapshot_download=fake_snapshot_download)
    monkeypatch.setenv("OPENMED_OFFLINE", "1")

    with pytest.raises(type(error), match=str(error)):
        prefetch_model(_ALIAS)


# --- list_cached_models ----------------------------------------------------


def test_list_cached_models_filters_openmed_repos(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    openmed_a = tmp_path / "models--OpenMed--A-model"
    openmed_b = tmp_path / "models--OpenMed--B-model"
    openmed_a.mkdir()
    openmed_b.mkdir()
    repos = [
        _FakeRepo(repo_id="other/model", repo_path=tmp_path / "other"),
        _FakeRepo(
            repo_id="OpenMed/B-model",
            repo_path=openmed_b,
            size_on_disk=200,
            last_accessed=2.0,
        ),
        _FakeRepo(
            repo_id="OpenMed/A-model",
            repo_path=openmed_a,
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

    cached = list_cached_models(cache_dir=str(tmp_path))

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
    repo_path = tmp_path / f"models--{_REPO_ID.replace('/', '--')}"
    repo_path.mkdir()
    (repo_path / "config.json").write_text("{}", encoding="utf-8")

    def fake_scan_cache_dir(cache_dir: Any = None) -> _FakeCacheInfo:
        return _FakeCacheInfo([_FakeRepo(repo_id=_REPO_ID, repo_path=repo_path)])

    _install_fake_hub(monkeypatch, scan_cache_dir=fake_scan_cache_dir)

    # Resolve via the friendly alias; the resolved repo id must match.
    removed = clear_cached_model(_ALIAS, cache_dir=str(tmp_path))

    assert removed is True
    assert not repo_path.exists()


def test_clear_cached_model_missing_repo_returns_false(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    other_path = tmp_path / "models--OpenMed--Other"
    other_path.mkdir()

    def fake_scan_cache_dir(cache_dir: Any = None) -> _FakeCacheInfo:
        return _FakeCacheInfo(
            [_FakeRepo(repo_id="OpenMed/Other", repo_path=other_path)]
        )

    _install_fake_hub(monkeypatch, scan_cache_dir=fake_scan_cache_dir)

    assert clear_cached_model(_ALIAS, cache_dir=str(tmp_path)) is False


def test_clear_cached_model_rejects_missing_repo_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = types.SimpleNamespace(repo_id=_REPO_ID, repo_type="model")

    def fake_scan_cache_dir(cache_dir: Any = None) -> _FakeCacheInfo:
        return _FakeCacheInfo([repo])

    removed_paths: List[Path] = []
    _install_fake_hub(monkeypatch, scan_cache_dir=fake_scan_cache_dir)
    monkeypatch.setattr(hf_hub.shutil, "rmtree", removed_paths.append)

    assert clear_cached_model(_ALIAS, cache_dir=str(tmp_path)) is False
    assert removed_paths == []


def test_clear_cached_model_rejects_paths_outside_cache_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_root = tmp_path / "hub"
    outside_root = tmp_path / "outside"
    cache_root.mkdir()
    outside_root.mkdir()
    expected_name = f"models--{_REPO_ID.replace('/', '--')}"
    cwd_entry = cache_root / expected_name
    outside_entry = outside_root / expected_name
    cache_root_entry = tmp_path / expected_name
    cwd_entry.mkdir()
    outside_entry.mkdir()
    cache_root_entry.mkdir()
    current_path = [Path(Path.cwd().anchor)]

    def fake_scan_cache_dir(cache_dir: Any = None) -> _FakeCacheInfo:
        return _FakeCacheInfo([_FakeRepo(repo_id=_REPO_ID, repo_path=current_path[0])])

    removed_paths: List[Path] = []
    _install_fake_hub(monkeypatch, scan_cache_dir=fake_scan_cache_dir)
    monkeypatch.setattr(hf_hub.shutil, "rmtree", removed_paths.append)

    for unsafe_path in [Path(Path.cwd().anchor), outside_entry]:
        current_path[0] = unsafe_path
        assert clear_cached_model(_ALIAS, cache_dir=str(cache_root)) is False

    current_path[0] = cache_root_entry
    assert clear_cached_model(_ALIAS, cache_dir=str(cache_root_entry)) is False

    current_path[0] = cwd_entry
    monkeypatch.chdir(cwd_entry)
    assert clear_cached_model(_ALIAS, cache_dir=str(cache_root)) is False
    assert removed_paths == []


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
