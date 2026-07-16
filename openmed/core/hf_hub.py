"""Convenience helpers to pull and cache OpenMed models from the Hugging Face Hub.

This module is the pull-side counterpart to :mod:`openmed.core.hf_publish`. It
lets consumers pre-fetch an OpenMed model for offline use, inspect what is
already cached, and drop a single repo from the cache, without hand-rolling
``huggingface_hub`` calls or re-implementing offline handling.

Design constraints (see ``AGENTS.md``):

* **Local-first.** No telemetry and no mandatory network access. Downloads are
  explicit; when ``OPENMED_OFFLINE`` is active every helper stays cache-only.
* **Optional ``[hf]`` extra.** ``huggingface_hub`` is imported lazily inside the
  functions so ``import openmed`` keeps working without the extra installed.
  Calling a helper without it raises an actionable :class:`ImportError`.
* **Mirror aware.** ``HF_HOME`` / ``HF_HUB_CACHE`` and ``HF_ENDPOINT`` are
  honoured by ``huggingface_hub`` itself; these helpers never override them.
"""

from __future__ import annotations

import fnmatch
import hashlib
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Callable, List, Mapping, Optional, Sequence

from .model_registry import get_model_info
from .offline import OfflineModeError, is_local_only

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .config import OpenMedConfig

__all__ = [
    "CachedModel",
    "DownloadIntegrityError",
    "DownloadProgress",
    "prefetch_model",
    "list_cached_models",
    "clear_cached_model",
    "resolve_repo_id",
]

DEFAULT_ORG = "OpenMed"
DEFAULT_RETRIES = 5
_BACKOFF_BASE_SECONDS = 1.0
_BACKOFF_MAX_SECONDS = 30.0

_HF_INSTALL_HINT = (
    "huggingface-hub is required to pull OpenMed models from the Hub. "
    'Install the optional extra with: pip install "openmed[hf]"'
)


@dataclass(frozen=True)
class CachedModel:
    """A single OpenMed repo present in the local Hugging Face cache.

    Attributes:
        repo_id: Fully-qualified Hub repo id, such as ``OpenMed/OpenMed-NER-...``.
        size_on_disk: Total size of the cached snapshot in bytes.
        last_accessed: Epoch seconds of the most recent access, when known.
        path: Filesystem path to the cached repo directory.
    """

    repo_id: str
    size_on_disk: int
    last_accessed: Optional[float]
    path: Path


@dataclass(frozen=True)
class DownloadProgress:
    """PHI-free progress for one model-repository file.

    Attributes:
        filename: Repository-relative model filename.
        bytes_done: Bytes already present for the current file.
        bytes_total: Expected size of the current file.
        files_done: Fully downloaded and integrity-checked files.
        files_total: Total number of files selected for the snapshot.
    """

    filename: str
    bytes_done: int
    bytes_total: int
    files_done: int
    files_total: int


class DownloadIntegrityError(RuntimeError):
    """Raised when a downloaded model file cannot be verified safely."""


@dataclass(frozen=True)
class _RemoteFile:
    filename: str
    size: int
    algorithm: str
    digest: str


class _BandwidthLimiter:
    """Simple process-local limiter shared by every file in one pull."""

    def __init__(self, bytes_per_second: int) -> None:
        self.bytes_per_second = bytes_per_second
        self.started_at = time.monotonic()
        self.transferred = 0

    def consume(self, byte_count: int) -> None:
        if byte_count <= 0:
            return
        self.transferred += byte_count
        target_elapsed = self.transferred / self.bytes_per_second
        delay = target_elapsed - (time.monotonic() - self.started_at)
        if delay > 0:
            time.sleep(delay)


def resolve_repo_id(name_or_alias: str, *, org: str = DEFAULT_ORG) -> str:
    """Resolve a registry alias or bare name to a fully-qualified Hub repo id.

    A value that already looks like a Hub id (contains a ``/``) is returned
    unchanged. A known registry key or repo id is resolved through the committed
    model registry. A bare name is prefixed with *org*.

    Args:
        name_or_alias: Registry key, repo id, or bare model name.
        org: Organization used to qualify a bare name.

    Returns:
        A ``org/name`` Hugging Face Hub repo id.

    Raises:
        ValueError: If *name_or_alias* is empty.
    """

    if not name_or_alias or not name_or_alias.strip():
        raise ValueError("name_or_alias must be a non-empty model name or alias")

    name_or_alias = name_or_alias.strip()

    registry_info = get_model_info(name_or_alias)
    if registry_info is not None:
        return registry_info.model_id

    if "/" in name_or_alias:
        return name_or_alias

    return f"{org}/{name_or_alias}"


def prefetch_model(
    name_or_alias: str,
    *,
    revision: Optional[str] = None,
    allow_patterns: str | Sequence[str] | None = None,
    cache_dir: Optional[str] = None,
    org: str = DEFAULT_ORG,
    config: "OpenMedConfig | None" = None,
    retries: int = DEFAULT_RETRIES,
    max_bandwidth: int | None = None,
    progress_callback: Callable[[DownloadProgress], None] | None = None,
) -> str:
    """Pre-download an OpenMed model snapshot and return its local path.

    The friendly *name_or_alias* is resolved to a Hub repo id via the model
    registry before download. This mirrors :mod:`openmed.core.hf_publish`'s push
    side so consumers can warm the cache for later offline inference.

    ``HF_HOME`` / ``HF_HUB_CACHE`` and ``HF_ENDPOINT`` (mirror support) are
    honoured by ``huggingface_hub`` and are never overridden here.

    When offline mode is active (``OPENMED_OFFLINE`` or ``config.local_only``),
    the download runs with ``local_files_only=True``; if the snapshot is not
    already cached an :class:`~openmed.core.offline.OfflineModeError` is raised
    instead of attempting a network fetch.

    Args:
        name_or_alias: Registry key, repo id, or bare model name to fetch.
        revision: Optional git revision (branch, tag, or commit) to pin.
        allow_patterns: Optional glob patterns limiting which files download.
        cache_dir: Optional explicit cache directory. Defaults to the standard
            Hugging Face cache (``HF_HOME`` / ``HF_HUB_CACHE``).
        org: Organization used to qualify a bare name.
        config: Optional :class:`~openmed.core.config.OpenMedConfig` used to
            detect ``local_only`` mode.
        retries: Number of retries after the initial attempt for transient
            network failures. Defaults to five. Permanent HTTP errors such as
            401 and 404 are never retried.
        max_bandwidth: Optional aggregate download cap in bytes per second.
        progress_callback: Optional callback receiving repository-relative
            filenames and byte/file totals. No file contents are exposed.

    Returns:
        Filesystem path to the downloaded snapshot directory.

    Raises:
        ImportError: If the optional ``[hf]`` extra is not installed.
        OfflineModeError: If offline mode is active and the model is not cached.
        DownloadIntegrityError: If Hub metadata is insufficient for verification
            or a downloaded file remains corrupt after one forced re-fetch.
        ValueError: If *retries* or *max_bandwidth* is invalid.
    """

    snapshot_download, local_entry_not_found = _import_snapshot_download()
    repo_id = resolve_repo_id(name_or_alias, org=org)
    local_only = is_local_only(config)

    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ValueError("retries must be a non-negative integer")
    if max_bandwidth is not None and (
        isinstance(max_bandwidth, bool)
        or not isinstance(max_bandwidth, int)
        or max_bandwidth <= 0
    ):
        raise ValueError("max_bandwidth must be a positive integer")

    download_kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "repo_type": "model",
        "revision": revision,
        "local_files_only": local_only,
    }
    if allow_patterns is not None:
        download_kwargs["allow_patterns"] = (
            allow_patterns if isinstance(allow_patterns, str) else list(allow_patterns)
        )
    if cache_dir is not None:
        download_kwargs["cache_dir"] = cache_dir

    if not local_only:
        return _prefetch_online(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=allow_patterns,
            cache_dir=cache_dir,
            retries=retries,
            max_bandwidth=max_bandwidth,
            progress_callback=progress_callback,
        )

    try:
        return snapshot_download(**download_kwargs)
    except local_entry_not_found as exc:
        if local_only:
            raise OfflineModeError(
                f"{repo_id} is not available in the local cache and offline mode "
                "blocks the download. Pre-fetch it once with offline mode "
                "disabled, or point HF_HOME at a cache that already contains it."
            ) from exc
        raise


def _prefetch_online(
    *,
    repo_id: str,
    revision: str | None,
    allow_patterns: str | Sequence[str] | None,
    cache_dir: str | None,
    retries: int,
    max_bandwidth: int | None,
    progress_callback: Callable[[DownloadProgress], None] | None,
) -> str:
    hf_api_type, hf_hub_download = _import_online_downloads()
    api = hf_api_type()
    repo_info = _with_retry(
        lambda: api.model_info(
            repo_id,
            revision=revision,
            files_metadata=True,
        ),
        retries=retries,
    )
    commit_hash = getattr(repo_info, "sha", None)
    if not isinstance(commit_hash, str) or not commit_hash:
        raise DownloadIntegrityError(
            f"Cannot verify {repo_id}: the Hub did not report a commit hash."
        )

    remote_files = _remote_files_from_info(
        repo_id=repo_id,
        siblings=getattr(repo_info, "siblings", None),
        allow_patterns=allow_patterns,
    )
    limiter = _BandwidthLimiter(max_bandwidth) if max_bandwidth else None
    downloaded: list[tuple[_RemoteFile, Path]] = []
    files_total = len(remote_files)

    for files_done, remote_file in enumerate(remote_files):
        local_path = _download_remote_file(
            hf_hub_download=hf_hub_download,
            repo_id=repo_id,
            revision=commit_hash,
            cache_dir=cache_dir,
            remote_file=remote_file,
            retries=retries,
            limiter=limiter,
            progress_callback=progress_callback,
            files_done=files_done,
            files_total=files_total,
            force_download=False,
        )
        if not _file_matches_metadata(local_path, remote_file):
            _unlink_corrupt_file(local_path, remote_file.filename)
            local_path = _download_remote_file(
                hf_hub_download=hf_hub_download,
                repo_id=repo_id,
                revision=commit_hash,
                cache_dir=cache_dir,
                remote_file=remote_file,
                retries=retries,
                limiter=limiter,
                progress_callback=progress_callback,
                files_done=files_done,
                files_total=files_total,
                force_download=True,
            )
            if not _file_matches_metadata(local_path, remote_file):
                raise DownloadIntegrityError(
                    f"Integrity verification failed for {remote_file.filename} "
                    "after one forced re-fetch. Clear the model cache and retry."
                )

        downloaded.append((remote_file, local_path))
        _emit_progress(
            progress_callback,
            remote_file=remote_file,
            bytes_done=remote_file.size,
            files_done=files_done + 1,
            files_total=files_total,
        )

    first_file, first_path = downloaded[0]
    return str(_snapshot_root(first_path, first_file.filename))


def _download_remote_file(
    *,
    hf_hub_download: Any,
    repo_id: str,
    revision: str,
    cache_dir: str | None,
    remote_file: _RemoteFile,
    retries: int,
    limiter: _BandwidthLimiter | None,
    progress_callback: Callable[[DownloadProgress], None] | None,
    files_done: int,
    files_total: int,
    force_download: bool,
) -> Path:
    progress_class = _make_progress_class(
        remote_file=remote_file,
        limiter=limiter,
        progress_callback=progress_callback,
        files_done=files_done,
        files_total=files_total,
    )
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "filename": remote_file.filename,
        "repo_type": "model",
        "revision": revision,
        "force_download": force_download,
        "local_files_only": False,
        "tqdm_class": progress_class,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir

    path = _with_retry(lambda: hf_hub_download(**kwargs), retries=retries)
    return Path(path)


def _remote_files_from_info(
    *,
    repo_id: str,
    siblings: Any,
    allow_patterns: str | Sequence[str] | None,
) -> list[_RemoteFile]:
    patterns = _normalise_patterns(allow_patterns)
    remote_files: list[_RemoteFile] = []
    for sibling in siblings or ():
        filename = getattr(sibling, "rfilename", None)
        if not isinstance(filename, str) or not _safe_repo_filename(filename):
            raise DownloadIntegrityError(
                f"Cannot verify {repo_id}: the Hub reported an invalid filename."
            )
        if patterns and not any(
            fnmatch.fnmatchcase(filename, pattern) for pattern in patterns
        ):
            continue

        size = getattr(sibling, "size", None)
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise DownloadIntegrityError(
                f"Cannot verify {filename}: the Hub did not report its size."
            )
        algorithm, digest = _metadata_digest(sibling)
        if algorithm is None or digest is None:
            raise DownloadIntegrityError(
                f"Cannot verify {filename}: the Hub did not report an ETag or "
                "SHA-256 digest."
            )
        remote_files.append(
            _RemoteFile(
                filename=filename,
                size=size,
                algorithm=algorithm,
                digest=digest,
            )
        )

    if not remote_files:
        pattern_hint = f" matching {patterns!r}" if patterns else ""
        raise DownloadIntegrityError(
            f"Cannot verify {repo_id}: the Hub reported no model files{pattern_hint}."
        )
    return remote_files


def _metadata_digest(sibling: Any) -> tuple[str | None, str | None]:
    lfs = getattr(sibling, "lfs", None)
    if isinstance(lfs, Mapping):
        lfs_sha256 = lfs.get("sha256")
    else:
        lfs_sha256 = getattr(lfs, "sha256", None)
    if _is_hex_digest(lfs_sha256, 64):
        return "sha256", str(lfs_sha256).lower()

    blob_id = getattr(sibling, "blob_id", None)
    if _is_hex_digest(blob_id, 40):
        return "git-sha1", str(blob_id).lower()
    if _is_hex_digest(blob_id, 64):
        return "sha256", str(blob_id).lower()
    return None, None


def _file_matches_metadata(path: Path, remote_file: _RemoteFile) -> bool:
    try:
        if not path.is_file() or path.stat().st_size != remote_file.size:
            return False
        if remote_file.algorithm == "git-sha1":
            digest = hashlib.sha1(usedforsecurity=False)
            digest.update(f"blob {remote_file.size}\0".encode())
        else:
            digest = hashlib.sha256()
        with path.open("rb") as downloaded_file:
            for chunk in iter(lambda: downloaded_file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return False
    return digest.hexdigest() == remote_file.digest


def _unlink_corrupt_file(path: Path, filename: str) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        raise DownloadIntegrityError(
            f"Corrupt file {filename} could not be removed before re-fetch: {exc}"
        ) from exc


def _snapshot_root(path: Path, filename: str) -> Path:
    root = path
    for _ in PurePosixPath(filename).parts:
        root = root.parent
    return root


def _normalise_patterns(
    allow_patterns: str | Sequence[str] | None,
) -> list[str]:
    if allow_patterns is None:
        return []
    if isinstance(allow_patterns, str):
        return [allow_patterns]
    return list(allow_patterns)


def _safe_repo_filename(filename: str) -> bool:
    path = PurePosixPath(filename)
    return bool(filename) and not path.is_absolute() and ".." not in path.parts


def _is_hex_digest(value: Any, length: int) -> bool:
    if not isinstance(value, str) or len(value) != length:
        return False
    return all(character in "0123456789abcdefABCDEF" for character in value)


def _with_retry(operation: Callable[[], Any], *, retries: int) -> Any:
    for attempt in range(retries + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt >= retries or not _is_transient_error(exc):
                raise
            exponential = min(
                _BACKOFF_BASE_SECONDS * (2**attempt),
                _BACKOFF_MAX_SECONDS,
            )
            time.sleep(exponential + random.uniform(0.0, _BACKOFF_BASE_SECONDS))
    raise AssertionError("retry loop must return or raise")


def _is_transient_error(error: Exception) -> bool:
    status_code = _error_status_code(error)
    if status_code is not None:
        return status_code in {408, 429} or 500 <= status_code <= 599
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    error_type = type(error)
    qualified_name = f"{error_type.__module__}.{error_type.__name__}".lower()
    return any(
        marker in qualified_name
        for marker in (
            "connecterror",
            "connectionerror",
            "networkerror",
            "proxyerror",
            "timeoutexception",
            "timeouterror",
        )
    )


def _error_status_code(error: Exception) -> int | None:
    response = getattr(error, "response", None)
    value = getattr(response, "status_code", None)
    if value is None:
        value = getattr(error, "status_code", None)
    return value if isinstance(value, int) else None


def _make_progress_class(
    *,
    remote_file: _RemoteFile,
    limiter: _BandwidthLimiter | None,
    progress_callback: Callable[[DownloadProgress], None] | None,
    files_done: int,
    files_total: int,
) -> type[Any]:
    furthest_bytes = 0

    class _DownloadProgressBar:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args
            initial = kwargs.get("initial", 0)
            self.n = int(initial or 0)
            self.total = int(kwargs.get("total") or remote_file.size)
            self._last_limited_update = 0
            self._record_progress(self.n)

        def _record_progress(self, bytes_done: int) -> None:
            nonlocal furthest_bytes
            furthest_bytes = max(furthest_bytes, min(bytes_done, remote_file.size))
            _emit_progress(
                progress_callback,
                remote_file=remote_file,
                bytes_done=furthest_bytes,
                files_done=files_done,
                files_total=files_total,
            )

        def update(self, byte_count: int | float | None = 1) -> bool:
            count = int(byte_count or 0)
            if limiter is not None and count > 0:
                limiter.consume(count)
                self._last_limited_update = count
            self.n += count
            self._record_progress(self.n)
            return True

        def update_transfer(self, byte_count: int | float | None = 1) -> None:
            count = int(byte_count or 0)
            if self._last_limited_update == count:
                self._last_limited_update = 0
            elif limiter is not None and count > 0:
                limiter.consume(count)

        def __enter__(self) -> "_DownloadProgressBar":
            return self

        def __exit__(self, *args: Any) -> None:
            self.close()

        def close(self) -> None:
            return None

        def refresh(self, *args: Any, **kwargs: Any) -> None:
            return None

        def set_description(self, *args: Any, **kwargs: Any) -> None:
            return None

        def set_postfix_str(self, *args: Any, **kwargs: Any) -> None:
            return None

        def set_transfer_postfix_str(self, *args: Any, **kwargs: Any) -> None:
            return None

    return _DownloadProgressBar


def _emit_progress(
    callback: Callable[[DownloadProgress], None] | None,
    *,
    remote_file: _RemoteFile,
    bytes_done: int,
    files_done: int,
    files_total: int,
) -> None:
    if callback is None:
        return
    callback(
        DownloadProgress(
            filename=remote_file.filename,
            bytes_done=bytes_done,
            bytes_total=remote_file.size,
            files_done=files_done,
            files_total=files_total,
        )
    )


def list_cached_models(
    *,
    org: str = DEFAULT_ORG,
    cache_dir: Optional[str] = None,
) -> List[CachedModel]:
    """Return the OpenMed model repos present in the local Hugging Face cache.

    The scan is read-only and never touches the network. An absent cache
    directory yields an empty list rather than an error.

    Args:
        org: Only repos owned by this organization are reported.
        cache_dir: Optional explicit cache directory to scan. Defaults to the
            standard Hugging Face cache (``HF_HOME`` / ``HF_HUB_CACHE``).

    Returns:
        Cached OpenMed models sorted by repo id.

    Raises:
        ImportError: If the optional ``[hf]`` extra is not installed.
    """

    scan_cache_dir, cache_not_found, default_cache_dir = _import_scan_cache_dir()
    prefix = f"{org}/"

    try:
        cache_info = scan_cache_dir(cache_dir) if cache_dir else scan_cache_dir()
    except (FileNotFoundError, cache_not_found):
        return []

    cache_root = Path(cache_dir or default_cache_dir).expanduser()
    try:
        cache_root = cache_root.resolve(strict=True)
    except (OSError, RuntimeError):
        return []

    cached: List[CachedModel] = []
    for repo in getattr(cache_info, "repos", ()):
        repo_id = getattr(repo, "repo_id", "")
        repo_type = getattr(repo, "repo_type", "model")
        if (
            repo_type != "model"
            or not isinstance(repo_id, str)
            or not repo_id.startswith(prefix)
        ):
            continue
        repo_path = _safe_cached_repo_path(
            repo_id=repo_id,
            repo_path=getattr(repo, "repo_path", None),
            cache_root=cache_root,
        )
        if repo_path is None:
            continue
        cached.append(
            CachedModel(
                repo_id=repo_id,
                size_on_disk=int(getattr(repo, "size_on_disk", 0) or 0),
                last_accessed=_as_optional_float(getattr(repo, "last_accessed", None)),
                path=repo_path,
            )
        )

    cached.sort(key=lambda item: item.repo_id)
    return cached


def clear_cached_model(
    name: str,
    *,
    org: str = DEFAULT_ORG,
    cache_dir: Optional[str] = None,
) -> bool:
    """Delete a single OpenMed repo from the local Hugging Face cache.

    Args:
        name: Registry key, repo id, or bare model name to remove.
        org: Organization used to qualify a bare name.
        cache_dir: Optional explicit cache directory to scan. Defaults to the
            standard Hugging Face cache (``HF_HOME`` / ``HF_HUB_CACHE``).

    Returns:
        ``True`` if a cached repo was found and removed, ``False`` otherwise.

    Raises:
        ImportError: If the optional ``[hf]`` extra is not installed.
    """

    repo_id = resolve_repo_id(name, org=org)
    for cached in list_cached_models(org=org, cache_dir=cache_dir):
        if cached.repo_id == repo_id:
            if cached.path.exists():
                shutil.rmtree(cached.path)
            return True
    return False


def _import_snapshot_download() -> tuple[Any, type[Exception]]:
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError as exc:
        raise ImportError(_HF_INSTALL_HINT) from exc
    return snapshot_download, LocalEntryNotFoundError


def _import_online_downloads() -> tuple[Any, Any]:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise ImportError(_HF_INSTALL_HINT) from exc
    return HfApi, hf_hub_download


def _import_scan_cache_dir() -> tuple[Any, type[Exception], Path]:
    try:
        from huggingface_hub import CacheNotFound, scan_cache_dir
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError as exc:
        raise ImportError(_HF_INSTALL_HINT) from exc
    return scan_cache_dir, CacheNotFound, Path(HF_HUB_CACHE)


def _safe_cached_repo_path(
    *,
    repo_id: str,
    repo_path: Any,
    cache_root: Path,
) -> Path | None:
    """Return a canonical cache entry path, failing closed on unsafe values."""

    try:
        path = Path(repo_path)
    except (TypeError, ValueError):
        return None
    if not path.is_absolute() or path.is_symlink():
        return None

    try:
        resolved_path = path.resolve(strict=True)
        cwd = Path.cwd().resolve(strict=True)
    except (OSError, RuntimeError):
        return None

    expected_name = f"models--{repo_id.replace('/', '--')}"
    filesystem_root = Path(resolved_path.anchor)
    if (
        not resolved_path.is_dir()
        or resolved_path.name != expected_name
        or resolved_path.parent != cache_root
        or resolved_path in {cache_root, filesystem_root, cwd}
    ):
        return None
    return resolved_path


def _as_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
