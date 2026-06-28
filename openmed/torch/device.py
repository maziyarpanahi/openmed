"""Device selection helpers for the PyTorch backend."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any, Mapping

TORCH_DEVICE_ENV_VAR = "OPENMED_TORCH_DEVICE"
LEGACY_DEVICE_ENV_VAR = "OPENMED_DEVICE"

MPS_ENV_DEFAULTS: Mapping[str, str] = {
    "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "1.0",
    "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.9",
}


@dataclass(frozen=True)
class MpsTuning:
    """MPS tuning choices applied by :func:`apply_mps_tuning`."""

    env: Mapping[str, str]
    recommended_dtype: str = "float32"


def resolve_torch_device(prefer: str | None = "auto") -> str:
    """Resolve the best PyTorch device for inference.

    Args:
        prefer: Explicit device preference such as ``"cpu"``, ``"cuda"``,
            ``"cuda:1"``, ``"mps"``, or ``"auto"``. ``None`` is treated as
            ``"auto"`` and consults environment overrides before probing torch.

    Returns:
        ``"mps"`` when MPS is available, else ``"cuda"`` when CUDA is
        available, else ``"cpu"``. Explicit device preferences are returned
        after light normalization.
    """
    requested = _normalize_device_preference(prefer)
    if _is_auto_device(requested):
        requested = _env_device_preference()

    if not _is_auto_device(requested):
        return requested

    torch = _import_torch()
    if torch is None:
        return "cpu"
    if _mps_is_available(torch):
        return "mps"
    if _cuda_is_available(torch):
        return "cuda"
    return "cpu"


def apply_mps_tuning() -> MpsTuning:
    """Apply conservative MPS runtime defaults and return dtype guidance.

    Existing environment values are preserved so operators can override
    OpenMed's defaults from their process manager, shell, or notebook.
    """
    applied: dict[str, str] = {}
    for key, value in MPS_ENV_DEFAULTS.items():
        applied[key] = os.environ.setdefault(key, value)
    return MpsTuning(env=applied)


def _env_device_preference() -> str:
    """Return the first configured torch device override, or ``"auto"``."""
    for env_var in (TORCH_DEVICE_ENV_VAR, LEGACY_DEVICE_ENV_VAR):
        value = _normalize_device_preference(os.getenv(env_var))
        if not _is_auto_device(value):
            return value
    return "auto"


def _normalize_device_preference(prefer: str | None) -> str:
    if prefer is None:
        return "auto"
    normalized = str(prefer).strip().lower()
    if normalized == "gpu":
        return "cuda"
    return normalized or "auto"


def _is_auto_device(device: str) -> bool:
    return device in {"auto", "default"}


def _import_torch() -> Any | None:
    try:
        return importlib.import_module("torch")
    except Exception:
        return None


def _mps_is_available(torch: Any) -> bool:
    backends = getattr(torch, "backends", None)
    mps_backend = getattr(backends, "mps", None)
    is_available = getattr(mps_backend, "is_available", None)
    if not callable(is_available):
        return False
    try:
        return bool(is_available())
    except Exception:
        return False


def _cuda_is_available(torch: Any) -> bool:
    cuda = getattr(torch, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    if not callable(is_available):
        return False
    try:
        return bool(is_available())
    except Exception:
        return False


__all__ = [
    "LEGACY_DEVICE_ENV_VAR",
    "MPS_ENV_DEFAULTS",
    "MpsTuning",
    "TORCH_DEVICE_ENV_VAR",
    "apply_mps_tuning",
    "resolve_torch_device",
]
