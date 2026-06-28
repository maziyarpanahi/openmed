"""Attention backend selection for PyTorch/Transformers models."""

from __future__ import annotations

import importlib
import importlib.util
import logging
from typing import Any

ATTENTION_BACKEND_ENV_VAR = "OPENMED_TORCH_ATTENTION_BACKEND"

_ATTN_FLASH = "flash_attention_2"
_ATTN_SDPA = "sdpa"
_ATTN_EAGER = "eager"
_AUTO_VALUES = {"", "auto", "default"}
_LOGGED_BACKENDS: set[str] = set()

logger = logging.getLogger(__name__)


def select_attn_implementation(
    prefer: str | None = "auto",
    *,
    log: logging.Logger | None = None,
) -> str:
    """Return a safe Transformers ``attn_implementation`` value.

    Args:
        prefer: ``"auto"``, ``"flash_attention_2"``, ``"sdpa"``, or
            ``"eager"``. Unavailable accelerated preferences downgrade with a
            warning instead of raising.
        log: Optional logger used for downgrade warnings and one-time selection
            messages.

    Returns:
        The attention implementation string to pass to
        ``from_pretrained(..., attn_implementation=...)``.
    """
    selected_logger = log or logger
    requested = _normalize_attn_preference(prefer)

    if requested in _AUTO_VALUES:
        return _log_selected(_best_available_backend(), selected_logger)

    if requested == _ATTN_EAGER:
        return _log_selected(_ATTN_EAGER, selected_logger)

    if requested == _ATTN_FLASH:
        if _flash_attention_2_is_available():
            return _log_selected(_ATTN_FLASH, selected_logger)
        fallback = _best_available_backend(allow_flash=False)
        _warn_downgrade(_ATTN_FLASH, fallback, selected_logger)
        return _log_selected(fallback, selected_logger)

    if requested == _ATTN_SDPA:
        if _sdpa_is_available():
            return _log_selected(_ATTN_SDPA, selected_logger)
        _warn_downgrade(_ATTN_SDPA, _ATTN_EAGER, selected_logger)
        return _log_selected(_ATTN_EAGER, selected_logger)

    fallback = _best_available_backend()
    selected_logger.warning(
        "Unknown PyTorch attention backend %r; using %s instead.",
        prefer,
        fallback,
    )
    return _log_selected(fallback, selected_logger)


def _best_available_backend(*, allow_flash: bool = True) -> str:
    if allow_flash and _flash_attention_2_is_available():
        return _ATTN_FLASH
    if _sdpa_is_available():
        return _ATTN_SDPA
    return _ATTN_EAGER


def _normalize_attn_preference(prefer: str | None) -> str:
    if prefer is None:
        return "auto"
    normalized = str(prefer).strip().lower().replace("-", "_")
    aliases = {
        "flash": _ATTN_FLASH,
        "flash_attn": _ATTN_FLASH,
        "flash_attn_2": _ATTN_FLASH,
        "flashattention2": _ATTN_FLASH,
        "flash_attention2": _ATTN_FLASH,
        "torch_sdpa": _ATTN_SDPA,
    }
    return aliases.get(normalized, normalized)


def _flash_attention_2_is_available() -> bool:
    try:
        if importlib.util.find_spec("flash_attn") is None:
            return False
    except (ImportError, ValueError):
        return False
    torch = _import_torch()
    cuda = getattr(torch, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    return callable(is_available) and bool(is_available())


def _sdpa_is_available() -> bool:
    torch = _import_torch()
    if torch is None:
        return False
    nn = getattr(torch, "nn", None)
    functional = getattr(nn, "functional", None)
    return callable(getattr(functional, "scaled_dot_product_attention", None))


def _import_torch() -> Any | None:
    try:
        return importlib.import_module("torch")
    except Exception:
        return None


def _warn_downgrade(
    requested: str, fallback: str, selected_logger: logging.Logger
) -> None:
    selected_logger.warning(
        "Requested PyTorch attention backend %s is unavailable; using %s instead.",
        requested,
        fallback,
    )


def _log_selected(backend: str, selected_logger: logging.Logger) -> str:
    if backend not in _LOGGED_BACKENDS:
        selected_logger.info("Using PyTorch attention backend: %s", backend)
        _LOGGED_BACKENDS.add(backend)
    return backend


__all__ = [
    "ATTENTION_BACKEND_ENV_VAR",
    "select_attn_implementation",
]
