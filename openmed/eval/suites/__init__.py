"""Benchmark suite registry for the OpenMed eval harness."""

from __future__ import annotations

from typing import Any

from openmed.eval.harness import BenchmarkFixture
from openmed.eval.suites.shield import (
    SHIELD,
    load_shield_fixtures,
    shield_suite_metadata,
)


GOLDEN = "golden"
I2B2 = "i2b2"
N2C2 = "n2c2"

DEFAULT_SUITES: tuple[str, ...] = (GOLDEN, I2B2, N2C2, SHIELD)


def validate_suite_name(name: str) -> str:
    """Return *name* if it is one of the scaffolded benchmark suites."""
    if name not in DEFAULT_SUITES:
        allowed = ", ".join(DEFAULT_SUITES)
        raise ValueError(f"unknown benchmark suite {name!r}; expected one of: {allowed}")
    return name


def load_suite_fixtures(name: str, **kwargs: Any) -> list[BenchmarkFixture]:
    """Load benchmark fixtures for a named suite."""
    suite = validate_suite_name(name)
    if suite == SHIELD:
        return load_shield_fixtures(**kwargs)
    raise ValueError(f"benchmark suite {suite!r} does not have a concrete loader yet")


def suite_metadata(name: str, **kwargs: Any) -> dict[str, Any]:
    """Return suite-specific report metadata."""
    suite = validate_suite_name(name)
    if suite == SHIELD:
        return shield_suite_metadata(**kwargs)
    return {"suite": suite}


__all__ = [
    "GOLDEN",
    "I2B2",
    "N2C2",
    "SHIELD",
    "DEFAULT_SUITES",
    "validate_suite_name",
    "load_suite_fixtures",
    "suite_metadata",
    "load_shield_fixtures",
    "shield_suite_metadata",
]
