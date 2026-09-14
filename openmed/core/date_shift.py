"""Stable patient-keyed date-shift offsets."""

from __future__ import annotations

import hashlib
import hmac
from typing import Final

from .errors import InputError

DEFAULT_DATE_SHIFT_MAX_DAYS: Final = 365
_SEED_DOMAIN: Final = b"openmed-date-shift-seed-v1\x00"


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
    _validate_max_days(max_days)

    patient_key_bytes = _nonempty_bytes(patient_key, name="patient_key")
    secret_bytes = _nonempty_bytes(secret, name="secret")
    digest = hmac.new(secret_bytes, patient_key_bytes, hashlib.sha256).digest()
    return _offset_from_digest(digest, max_days=max_days)


def stable_offset_from_seed(seed: int, *, max_days: int) -> int:
    """Return a deterministic non-zero day offset for a request seed.

    The domain-separated digest keeps date shifting independent from every
    other seeded operation in a de-identification request while allowing one
    caller-supplied seed to reproduce the complete output.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise InputError(
            "seed must be an integer. Pass a stable integer seed before retrying.",
            details={"argument": "seed"},
        )
    _validate_max_days(max_days)

    digest = hashlib.sha256(_SEED_DOMAIN + str(seed).encode("ascii")).digest()
    return _offset_from_digest(digest, max_days=max_days)


def _offset_from_digest(digest: bytes, *, max_days: int) -> int:
    bucket = int.from_bytes(digest, "big") % (max_days * 2)

    if bucket < max_days:
        return bucket - max_days
    return bucket - max_days + 1


def _validate_max_days(max_days: int) -> None:
    if isinstance(max_days, bool) or not isinstance(max_days, int):
        raise InputError(
            "max_days must be an integer. Pass a positive integer day limit.",
            details={"argument": "max_days"},
        )
    if max_days <= 0:
        raise InputError(
            "max_days must be positive. Pass an integer greater than zero.",
            details={"argument": "max_days", "constraint": "positive"},
        )


def _nonempty_bytes(value: str | bytes, *, name: str) -> bytes:
    if isinstance(value, str):
        encoded = value.encode("utf-8")
    elif isinstance(value, bytes):
        encoded = value
    else:
        raise InputError(
            f"{name} must be text or bytes. Pass a non-empty stable value.",
            details={"argument": name, "expected": "str or bytes"},
        )

    if not encoded:
        raise InputError(
            f"{name} must be non-empty. Pass a stable value before retrying.",
            details={"argument": name, "constraint": "non_empty"},
        )
    return encoded


__all__ = [
    "DEFAULT_DATE_SHIFT_MAX_DAYS",
    "stable_offset_for",
    "stable_offset_from_seed",
]
