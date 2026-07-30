"""Offline-mode helpers for local-only OpenMed inference."""

from __future__ import annotations

import os
import socket
from contextlib import contextmanager
from typing import Any, Iterator

OFFLINE_ENV_VAR = "OPENMED_OFFLINE"
HF_OFFLINE_ENV_VARS = (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
)
_FALSE_ENV_VALUES = {"", "0", "false", "no", "off"}

OFFLINE_NETWORK_ERROR = (
    "OPENMED_OFFLINE/local_only=True blocks outbound network access after "
    "model loading. Pre-download required model files into the configured "
    "cache, pass a local model path, or disable offline mode before remote "
    "fetches."
)


class OfflineModeError(RuntimeError):
    """Raised when offline mode blocks a remote operation."""


def env_flag_enabled(value: str | None) -> bool:
    """Return True when an environment flag value enables a boolean option."""
    if value is None:
        return False
    return value.strip().lower() not in _FALSE_ENV_VALUES


def is_local_only(config: Any = None) -> bool:
    """Return whether offline/local-only mode is active."""
    return bool(
        getattr(config, "local_only", False)
        or env_flag_enabled(os.getenv(OFFLINE_ENV_VAR))
    )


def enable_hf_offline_flags() -> None:
    """Set Hub/Transformers offline flags for cache-only loading."""
    for name in HF_OFFLINE_ENV_VARS:
        os.environ[name] = "1"


def configure_offline_mode(config: Any = None) -> bool:
    """Enable process-level offline flags when local-only mode is active."""
    if is_local_only(config):
        enable_hf_offline_flags()
        return True
    return False


def offline_mode_assertion(config: Any = None) -> dict[str, Any]:
    """Return PHI-free evidence describing the configured offline posture.

    The assertion distinguishes OpenMed's socket guard from dependency-specific
    cache-only flags.  Hugging Face flags alone do not prove that OpenMed's
    outbound socket guard was requested, while ``OPENMED_OFFLINE`` or
    ``config.local_only`` does.
    """

    config_local_only = bool(getattr(config, "local_only", False))
    environment_local_only = env_flag_enabled(os.getenv(OFFLINE_ENV_VAR))
    local_only = config_local_only or environment_local_only
    environment_flags = {
        OFFLINE_ENV_VAR: environment_local_only,
        **{name: env_flag_enabled(os.getenv(name)) for name in HF_OFFLINE_ENV_VARS},
    }
    dependency_flags_enabled = all(
        environment_flags[name] for name in HF_OFFLINE_ENV_VARS
    )
    if config_local_only and environment_local_only:
        source = "config+environment"
    elif config_local_only:
        source = "config"
    elif environment_local_only:
        source = "environment"
    else:
        source = "none"

    return {
        "asserted": bool(local_only and dependency_flags_enabled),
        "local_only": local_only,
        "source": source,
        "network_guard_requested": local_only,
        "dependency_flags_enabled": dependency_flags_enabled,
        "environment_flags": environment_flags,
    }


def raise_offline_error(action: str) -> None:
    """Raise a clear error for a blocked remote action."""
    raise OfflineModeError(f"{OFFLINE_NETWORK_ERROR} Blocked action: {action}.")


@contextmanager
def network_blocked_if_offline(
    config: Any = None,
    *,
    local_only: bool = False,
) -> Iterator[None]:
    """Block outbound sockets for configured or explicitly local-only work."""
    if local_only:
        enable_hf_offline_flags()
    elif not configure_offline_mode(config):
        yield
        return

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection

    def _blocked_connect(*args: Any, **kwargs: Any) -> Any:
        raise_offline_error("socket connection")

    socket.socket.connect = _blocked_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = _blocked_connect  # type: ignore[method-assign]
    socket.create_connection = _blocked_connect  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket.connect = original_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = original_connect_ex  # type: ignore[method-assign]
        socket.create_connection = original_create_connection  # type: ignore[assignment]
