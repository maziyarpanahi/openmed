"""Quantization helpers for PyTorch model loading."""

from __future__ import annotations

import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_bnb_4bit_quantization_config(
    *,
    load_in_4bit: bool,
    bnb_4bit_use_double_quant: bool,
    device: str,
) -> Any | None:
    """Return a Transformers bitsandbytes NF4 config when usable.

    The optional bitsandbytes path is CUDA-only in the supported OpenMed torch
    loader. Unsupported environments deliberately fall back to regular loading
    so CPU-only installs and default development environments keep working.
    """
    if not load_in_4bit:
        return None

    if not _is_cuda_device(device):
        logger.warning(
            "load_in_4bit was requested, but bitsandbytes 4-bit loading "
            "requires a CUDA device; loading without 4-bit quantization."
        )
        return None

    if not _module_importable("bitsandbytes"):
        logger.warning(
            "load_in_4bit was requested, but the optional bitsandbytes package "
            "is not installed; loading without 4-bit quantization."
        )
        return None

    try:
        transformers = importlib.import_module("transformers")
        bitsandbytes_config = getattr(transformers, "BitsAndBytesConfig")
    except Exception as exc:
        logger.warning(
            "load_in_4bit was requested, but Transformers BitsAndBytesConfig "
            "is unavailable (%s); loading without 4-bit quantization.",
            exc,
        )
        return None

    return bitsandbytes_config(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=bool(bnb_4bit_use_double_quant),
    )


def _is_cuda_device(device: str) -> bool:
    return str(device).lower().startswith("cuda")


def _module_importable(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except Exception:
        return False
    return True


__all__ = ["build_bnb_4bit_quantization_config"]
