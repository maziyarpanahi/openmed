"""Shared grammar for opaque ``X-Request-ID`` correlation identifiers.

The REST service, the privacy proxy and gateway, and the typed Python client
all apply this one grammar. A conforming request id is 1 to 128 ASCII letters,
digits, ``_``, ``.``, ``:`` or ``-``. Anything else (oversized, non-ASCII, or
carrying whitespace, CR/LF, commas or other separators) is non-conforming.

This module depends only on the standard library so clients can import it
without the optional ``service`` extra.
"""

from __future__ import annotations

import re
import uuid
from typing import TypeGuard

REQUEST_ID_HEADER = "X-Request-ID"
REQUEST_ID_MAX_LENGTH = 128
REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9_.:-]{1,128}")


def is_valid_request_id(value: object) -> TypeGuard[str]:
    """Return whether ``value`` conforms to the shared request-id grammar.

    Args:
        value: Candidate request id. Non-string values never conform.

    Returns:
        True only for a string of 1 to 128 characters drawn from ASCII
        letters, digits, ``_``, ``.``, ``:`` and ``-``.
    """
    return type(value) is str and REQUEST_ID_PATTERN.fullmatch(value) is not None


def generate_request_id() -> str:
    """Return a fresh random request id that conforms to the grammar.

    Returns:
        A random UUID4 string, which carries no caller-supplied content.
    """
    return str(uuid.uuid4())


def request_id_or_generated(value: object) -> str:
    """Return ``value`` when it conforms, otherwise a freshly generated id.

    Non-conforming values are replaced, never truncated or repaired, so no
    part of them reaches headers, envelopes, logs or spans.

    Args:
        value: Candidate request id, for example an inbound header value.

    Returns:
        The unchanged conforming id, or a new id from
        :func:`generate_request_id`.
    """
    if is_valid_request_id(value):
        return value
    return generate_request_id()


__all__ = [
    "REQUEST_ID_HEADER",
    "REQUEST_ID_MAX_LENGTH",
    "REQUEST_ID_PATTERN",
    "generate_request_id",
    "is_valid_request_id",
    "request_id_or_generated",
]
