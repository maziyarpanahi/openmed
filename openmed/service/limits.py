"""Request limits for the OpenMed service schemas."""

from __future__ import annotations

import os
from typing import Optional


SERVICE_MAX_TEXT_LENGTH_ENV_VAR = "OPENMED_SERVICE_MAX_TEXT_LENGTH"
DEFAULT_MAX_TEXT_LENGTH = 1_000_000


def parse_max_text_length(raw_value: Optional[str]) -> int:
    """Parse the configured request text limit.

    Invalid, empty, or non-positive values fall back to the default cap. The
    cap is defensive; falling back keeps the service bounded even when the
    environment value is misconfigured.
    """
    if raw_value is None:
        return DEFAULT_MAX_TEXT_LENGTH

    raw_value = raw_value.strip()
    if not raw_value:
        return DEFAULT_MAX_TEXT_LENGTH

    try:
        parsed = int(raw_value)
    except ValueError:
        return DEFAULT_MAX_TEXT_LENGTH

    if parsed <= 0:
        return DEFAULT_MAX_TEXT_LENGTH
    return parsed


def get_max_text_length() -> int:
    """Return the current service request text cap in characters."""
    return parse_max_text_length(os.getenv(SERVICE_MAX_TEXT_LENGTH_ENV_VAR))
