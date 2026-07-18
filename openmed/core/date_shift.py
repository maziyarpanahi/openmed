"""Stable patient-keyed date-shift offsets."""

from __future__ import annotations

import hashlib
import hmac
from typing import Final

DEFAULT_DATE_SHIFT_MAX_DAYS: Final = 365


def stable_offset_for(
    patient_key: str | bytes,
    *,
    max_days: int,
    secret: str | bytes,
) -> int:
    """Return a deterministic non-zero signed day offset for a patient key.

    The raw patient key is used only as the HMAC message. Callers should retain
    their own stable patient key and secret; this helper stores neither value.
    """
    if isinstance(max_days, bool) or not isinstance(max_days, int):
        raise TypeError("max_days must be an integer")
    if max_days <= 0:
        raise ValueError("max_days must be positive")

    patient_key_bytes = _nonempty_bytes(patient_key, name="patient_key")
    secret_bytes = _nonempty_bytes(secret, name="secret")
    digest = hmac.new(secret_bytes, patient_key_bytes, hashlib.sha256).digest()
    bucket = int.from_bytes(digest, "big") % (max_days * 2)

    if bucket < max_days:
        return bucket - max_days
    return bucket - max_days + 1


def _nonempty_bytes(value: str | bytes, *, name: str) -> bytes:
    if isinstance(value, str):
        encoded = value.encode("utf-8")
    elif isinstance(value, bytes):
        encoded = value
    else:
        raise TypeError(f"{name} must be str or bytes")

    if not encoded:
        raise ValueError(f"{name} must be non-empty")
    return encoded


__all__ = [
    "DEFAULT_DATE_SHIFT_MAX_DAYS",
    "stable_offset_for",
]
