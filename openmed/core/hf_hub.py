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

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Sequence

from .model_registry import get_model_info
from .offline import OfflineModeError, is_local_only

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .config import OpenMedConfig

__all__ = [
    "CachedModel",
    "prefetch_model",
    "list_cached_models",
    "clear_cached_model",
    "resolve_repo_id",
]

DEFAULT_ORG = "OpenMed"

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
    allow_patterns: Optional[Sequence[str]] = None,
    cache_dir: Optional[str] = None,
    org: str = DEFAULT_ORG,
    config: "OpenMedConfig | None" = None,
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

    Returns:
        Filesystem path to the downloaded snapshot directory.

    Raises:
        ImportError: If the optional ``[hf]`` extra is not installed.
        OfflineModeError: If offline mode is active and the model is not cached.
    """

    snapshot_download = _import_snapshot_download()
    repo_id = resolve_repo_id(name_or_alias, org=org)
    local_only = is_local_only(config)

    download_kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "repo_type": "model",
        "revision": revision,
        "local_files_only": local_only,
    }
    if allow_patterns is not None:
        download_kwargs["allow_patterns"] = list(allow_patterns)
    if cache_dir is not None:
        download_kwargs["cache_dir"] = cache_dir

    try:
        return snapshot_download(**download_kwargs)
    except Exception as exc:  # noqa: BLE001 - re-raised with actionable guidance
        if local_only:
            raise OfflineModeError(
                f"{repo_id} is not available in the local cache and offline mode "
                "blocks the download. Pre-fetch it once with offline mode "
                "disabled, or point HF_HOME at a cache that already contains it."
            ) from exc
        raise


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

    scan_cache_dir, cache_not_found = _import_scan_cache_dir()
    prefix = f"{org}/"

    try:
        cache_info = scan_cache_dir(cache_dir) if cache_dir else scan_cache_dir()
    except (FileNotFoundError, cache_not_found):
        return []

    cached: List[CachedModel] = []
    for repo in getattr(cache_info, "repos", ()):
        repo_id = getattr(repo, "repo_id", "")
        repo_type = getattr(repo, "repo_type", "model")
        if repo_type != "model" or not repo_id.startswith(prefix):
            continue
        cached.append(
            CachedModel(
                repo_id=repo_id,
                size_on_disk=int(getattr(repo, "size_on_disk", 0) or 0),
                last_accessed=_as_optional_float(getattr(repo, "last_accessed", None)),
                path=Path(str(getattr(repo, "repo_path", ""))),
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


def _import_snapshot_download() -> Any:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(_HF_INSTALL_HINT) from exc
    return snapshot_download


def _import_scan_cache_dir() -> tuple[Any, type[Exception]]:
    try:
        from huggingface_hub import CacheNotFound, scan_cache_dir
    except ImportError as exc:
        raise ImportError(_HF_INSTALL_HINT) from exc
    return scan_cache_dir, CacheNotFound


def _as_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
