"""Benchmark suite registry for the OpenMed eval harness."""

from __future__ import annotations

from typing import Any

from openmed.eval.datasets.biomedical_ner import (
    BIOMEDICAL_NER,
    biomedical_ner_suite_metadata,
    load_biomedical_ner_fixtures,
    run_biomedical_ner_benchmark,
)
from openmed.eval.datasets.drugprot import (
    DRUGPROT,
    drugprot_suite_metadata,
    load_drugprot_fixtures,
)
from openmed.eval.datasets.i2b2 import (
    I2B2,
    I2B2_PATH_ENV,
    I2B2_YEAR_ENV,
    i2b2_suite_metadata,
    load_i2b2_deid,
)
from openmed.eval.harness import BenchmarkFixture
from openmed.eval.suites.shield import (
    SHIELD,
    load_shield_fixtures,
    shield_suite_metadata,
)

GOLDEN = "golden"
N2C2 = "n2c2"

DEFAULT_SUITES: tuple[str, ...] = (
    GOLDEN,
    I2B2,
    N2C2,
    SHIELD,
    DRUGPROT,
    BIOMEDICAL_NER,
)


def validate_suite_name(name: str) -> str:
    """Return *name* if it is one of the scaffolded benchmark suites."""
    if name not in DEFAULT_SUITES:
        allowed = ", ".join(DEFAULT_SUITES)
        raise ValueError(
            f"unknown benchmark suite {name!r}; expected one of: {allowed}"
        )
    return name


def load_suite_fixtures(name: str, **kwargs: Any) -> list[Any]:
    """Load benchmark fixtures for a named suite."""
    suite = validate_suite_name(name)
    if suite == I2B2:
        return load_i2b2_deid(
            path=kwargs.get("path"),
            year=kwargs.get("year", kwargs.get("corpus_year")),
        )
    if suite == SHIELD:
        return load_shield_fixtures(**kwargs)
    if suite == DRUGPROT:
        return load_drugprot_fixtures(**kwargs)
    if suite == BIOMEDICAL_NER:
        return load_biomedical_ner_fixtures(**kwargs)
    raise ValueError(f"benchmark suite {suite!r} does not have a concrete loader yet")


def suite_metadata(name: str, **kwargs: Any) -> dict[str, Any]:
    """Return suite-specific report metadata."""
    suite = validate_suite_name(name)
    if suite == I2B2:
        metadata = i2b2_suite_metadata()
        metadata["path_config"] = kwargs.get("path_config", I2B2_PATH_ENV)
        metadata["year_config"] = kwargs.get("year_config", I2B2_YEAR_ENV)
        return metadata
    if suite == SHIELD:
        return shield_suite_metadata(**kwargs)
    if suite == DRUGPROT:
        return drugprot_suite_metadata(**kwargs)
    if suite == BIOMEDICAL_NER:
        return biomedical_ner_suite_metadata(**kwargs)
    return {"suite": suite}


__all__ = [
    "GOLDEN",
    "I2B2",
    "N2C2",
    "SHIELD",
    "DRUGPROT",
    "BIOMEDICAL_NER",
    "DEFAULT_SUITES",
    "validate_suite_name",
    "load_suite_fixtures",
    "suite_metadata",
    "load_i2b2_deid",
    "i2b2_suite_metadata",
    "biomedical_ner_suite_metadata",
    "load_drugprot_fixtures",
    "load_biomedical_ner_fixtures",
    "drugprot_suite_metadata",
    "load_shield_fixtures",
    "shield_suite_metadata",
    "run_biomedical_ner_benchmark",
]
