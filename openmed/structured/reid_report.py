"""Privacy-safe attacker-model risk reports for structured releases.

The report implements three common exact-match disclosure scenarios over
quasi-identifier equivalence classes:

* a prosecutor targets a person known to be in the released sample, so a row
  in a sample class of size ``f`` has probability ``1 / f``;
* a journalist does not know sample membership, so a row in a corresponding
  population class of size ``F`` has probability ``1 / F``; and
* a marketer targets the whole release, so the expected successful fraction is
  the sample-weighted mean of ``1 / F``.

Without a supplied population, ``F >= f`` is the only available assumption and
the sample frequencies are used as conservative upper bounds. Public class
identifiers are stable hashes. Raw quasi-identifier values and row identifiers
are never copied into the report.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from openmed.core.audit import stable_hash
from openmed.risk.kanon import kanon_report


def reid_report(
    table: Any,
    *,
    quasi_identifiers: Sequence[str],
    population: Any | None = None,
) -> dict[str, Any]:
    """Return exact-match re-identification risk for a structured table.

    Args:
        table: Released sample rows in a shape accepted by
            :func:`openmed.risk.kanon.kanon_report`.
        quasi_identifiers: Non-empty quasi-identifier column names used to form
            equivalence classes.
        population: Optional auxiliary/reference population with the same
            quasi-identifier columns. It is treated as the attack population
            and must contain every sample class at least as often as the
            sample to be model-consistent.

    Returns:
        A deterministic, JSON-serializable report with headline, expected, and
        maximum risk for the prosecutor, journalist, and marketer scenarios;
        k-minimum and singleton counts; and privacy-safe identifiers for the
        highest-risk equivalence classes.

    Raises:
        TypeError: If ``quasi_identifiers`` is not a sequence of strings.
        ValueError: If the quasi-identifier list or sample is empty, or if a
            population is supplied without any rows.

    Note:
        This statistical report does not authorize publication or constitute
        an Expert Determination. A release policy must independently configure
        acceptable thresholds for the relevant attacker scenarios.
    """

    qis = _validated_quasi_identifiers(quasi_identifiers)
    sample_report = kanon_report(table, quasi_identifiers=qis)
    record_count = int(sample_report["record_count"])
    if record_count == 0:
        raise ValueError("table must contain at least one record")

    sample_classes = _class_counts(sample_report)
    population_supplied = population is not None
    if population_supplied:
        population_report = kanon_report(population, quasi_identifiers=qis)
        if int(population_report["record_count"]) == 0:
            raise ValueError("population must contain at least one record")
        population_counts = _class_counts(population_report)
    else:
        population_counts = dict(sample_classes)

    class_risks: list[dict[str, Any]] = []
    inconsistent_class_count = 0
    inconsistent_record_count = 0
    for class_key, sample_count in sample_classes.items():
        supplied_population_count = population_counts.get(class_key, 0)
        model_consistent = supplied_population_count >= sample_count
        if population_supplied and not model_consistent:
            inconsistent_class_count += 1
            inconsistent_record_count += sample_count

        prosecutor_probability = 1.0 / sample_count
        if population_supplied and not model_consistent:
            journalist_probability = 1.0
        else:
            journalist_probability = 1.0 / supplied_population_count

        class_risks.append(
            {
                "equivalence_class_key": _equivalence_class_digest(class_key),
                "sample_count": sample_count,
                "population_count": (
                    supplied_population_count if population_supplied else None
                ),
                "prosecutor_probability": prosecutor_probability,
                "journalist_probability": journalist_probability,
                "marketer_probability": journalist_probability,
                "population_model_consistent": model_consistent,
            }
        )

    prosecutor_expected = _weighted_probability(
        class_risks,
        probability_key="prosecutor_probability",
        record_count=record_count,
    )
    prosecutor_maximum = max(
        float(item["prosecutor_probability"]) for item in class_risks
    )
    population_expected = _weighted_probability(
        class_risks,
        probability_key="journalist_probability",
        record_count=record_count,
    )
    population_maximum = max(
        float(item["journalist_probability"]) for item in class_risks
    )

    highest_risk_records = [
        item
        for item in class_risks
        if math.isclose(
            float(item["prosecutor_probability"]),
            prosecutor_maximum,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or math.isclose(
            float(item["journalist_probability"]),
            population_maximum,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ]
    highest_risk_records.sort(
        key=lambda item: (
            -float(item["journalist_probability"]),
            -float(item["prosecutor_probability"]),
            str(item["equivalence_class_key"]),
        )
    )

    k_min = int(sample_report["k"])
    singleton_records = sum(count for count in sample_classes.values() if count == 1)
    basis = "population_estimate" if population_supplied else "sample_upper_bound"
    return {
        "schema_version": 1,
        "record_count": record_count,
        "equivalence_class_count": len(sample_classes),
        "quasi_identifiers": list(qis),
        "population_supplied": population_supplied,
        "population_model_consistent": inconsistent_class_count == 0,
        "population_inconsistent_class_count": inconsistent_class_count,
        "population_inconsistent_record_count": inconsistent_record_count,
        "k_min": k_min,
        "singleton_records": singleton_records,
        "prosecutor": _scenario(
            risk=prosecutor_maximum,
            expected=prosecutor_expected,
            maximum=prosecutor_maximum,
            basis="sample_exact",
        ),
        "journalist": _scenario(
            risk=population_maximum,
            expected=population_expected,
            maximum=population_maximum,
            basis=basis,
        ),
        "marketer": _scenario(
            risk=population_expected,
            expected=population_expected,
            maximum=population_maximum,
            basis=basis,
        ),
        "highest_risk_records": highest_risk_records,
    }


def _validated_quasi_identifiers(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError("quasi_identifiers must be a sequence of column names")
    fields: list[str] = []
    for field in value:
        if not isinstance(field, str):
            raise TypeError("quasi-identifier column names must be strings")
        if not field:
            raise ValueError("quasi-identifier column names must not be empty")
        if field not in fields:
            fields.append(field)
    if not fields:
        raise ValueError("quasi_identifiers must not be empty")
    return tuple(fields)


def _class_counts(report: Mapping[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for equivalence_class in report["equivalence_classes"]:
        key = json.dumps(
            equivalence_class["key"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        result[key] = int(equivalence_class["size"])
    return result


def _equivalence_class_digest(class_key: str) -> str:
    return stable_hash(
        {
            "artifact": "openmed.structured.reid_equivalence_class",
            "schema_version": 1,
            "key": json.loads(class_key),
        }
    )


def _weighted_probability(
    classes: Sequence[Mapping[str, Any]],
    *,
    probability_key: str,
    record_count: int,
) -> float:
    return (
        sum(
            int(item["sample_count"]) * float(item[probability_key]) for item in classes
        )
        / record_count
    )


def _scenario(
    *,
    risk: float,
    expected: float,
    maximum: float,
    basis: str,
) -> dict[str, float | str]:
    if not 0.0 <= expected <= maximum <= 1.0:
        raise ValueError(
            "scenario probabilities must satisfy 0 <= expected <= max <= 1"
        )
    return {
        "risk": float(risk),
        "expected_probability": float(expected),
        "maximum_probability": float(maximum),
        "basis": basis,
    }


__all__ = ["reid_report"]
