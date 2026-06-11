"""Benchmark suite registry for the OpenMed eval harness.

The first concrete fixture loaders land in follow-up tasks. OM-018 defines the
suite names and keeps this package importable for local and CI runners.
"""

from __future__ import annotations


GOLDEN = "golden"
I2B2 = "i2b2"
N2C2 = "n2c2"

DEFAULT_SUITES: tuple[str, ...] = (GOLDEN, I2B2, N2C2)


def validate_suite_name(name: str) -> str:
    """Return *name* if it is one of the scaffolded benchmark suites."""
    if name not in DEFAULT_SUITES:
        allowed = ", ".join(DEFAULT_SUITES)
        raise ValueError(f"unknown benchmark suite {name!r}; expected one of: {allowed}")
    return name


__all__ = [
    "GOLDEN",
    "I2B2",
    "N2C2",
    "DEFAULT_SUITES",
    "validate_suite_name",
]
