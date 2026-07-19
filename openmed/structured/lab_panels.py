"""Lab-panel structurer (roadmap section 4.2).

Lab results are reported as panels -- a CBC, a BMP -- each a set of analyte /
value / unit / reference-range rows. OpenMed already parses a single lab value
(:mod:`openmed.clinical.lab_values`); this module groups parsed results into
recognized panels and emits normalized analyte rows (canonical analyte name,
value, unit, parsed reference range, and abnormal flag) suitable for downstream
analytics and measurement export.

Reference-range parsing and abnormal-flag derivation are reused from
``clinical.lab_values`` rather than re-implemented. Structuring is deterministic
and offline; it performs no clinical interpretation beyond the range comparison
the flag already encodes.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any, TypedDict

from openmed.clinical.lab_values import (
    AbnormalFlag,
    ReferenceRange,
    derive_abnormal_flag,
    parse_reference_range,
)

LAB_PANEL_ADVISORY = (
    "Lab-panel structuring groups parsed results into panels and normalizes "
    "analyte rows deterministically. Abnormal flags reflect only the stated "
    "reference range and are not a clinical interpretation; review before use."
)

# Alias (whitespace/case/punctuation-insensitive) -> canonical analyte name.
_ANALYTE_ALIASES: Mapping[str, str] = {
    # Complete blood count
    "wbc": "WBC",
    "whitebloodcell": "WBC",
    "whitebloodcellcount": "WBC",
    "rbc": "RBC",
    "redbloodcell": "RBC",
    "hgb": "Hemoglobin",
    "hb": "Hemoglobin",
    "hemoglobin": "Hemoglobin",
    "hct": "Hematocrit",
    "hematocrit": "Hematocrit",
    "plt": "Platelets",
    "platelet": "Platelets",
    "platelets": "Platelets",
    "mcv": "MCV",
    "mch": "MCH",
    "mchc": "MCHC",
    "rdw": "RDW",
    # Basic metabolic panel
    "na": "Sodium",
    "sodium": "Sodium",
    "k": "Potassium",
    "potassium": "Potassium",
    "cl": "Chloride",
    "chloride": "Chloride",
    "co2": "CO2",
    "hco3": "CO2",
    "bicarbonate": "CO2",
    "bun": "BUN",
    "ureanitrogen": "BUN",
    "cr": "Creatinine",
    "creat": "Creatinine",
    "creatinine": "Creatinine",
    "glu": "Glucose",
    "glucose": "Glucose",
    "ca": "Calcium",
    "calcium": "Calcium",
}

_PANEL_MEMBERS: Mapping[str, frozenset[str]] = {
    "CBC": frozenset(
        {
            "WBC",
            "RBC",
            "Hemoglobin",
            "Hematocrit",
            "Platelets",
            "MCV",
            "MCH",
            "MCHC",
            "RDW",
        }
    ),
    "BMP": frozenset(
        {
            "Sodium",
            "Potassium",
            "Chloride",
            "CO2",
            "BUN",
            "Creatinine",
            "Glucose",
            "Calcium",
        }
    ),
}

#: Deterministic panel emission order; unrecognized analytes fall into "other".
PANEL_ORDER = ("CBC", "BMP", "other")

# A single "analyte value [unit] [(range)]" result. The unit must contain a
# ``/``, ``%`` or ``^`` so it is never confused with a following analyte token
# on a multi-result line ("Na 140  K 4.0").
_RESULT_RE = re.compile(
    r"(?P<analyte>[A-Za-z][A-Za-z0-9]*)\s+"
    r"(?P<value>-?\d+(?:\.\d+)?)"
    r"(?:\s+(?P<unit>[^\s()]*[/%^][^\s()]*))?"
    r"(?:\s*\((?P<range>[^)]*)\))?"
)


class AnalyteRow(TypedDict):
    """One normalized analyte result within a panel."""

    analyte: str
    value: float | None
    unit: str | None
    reference_range: ReferenceRange
    flag: AbnormalFlag


class LabPanel(TypedDict):
    """A recognized panel with its normalized analyte rows."""

    panel: str
    analytes: list[AnalyteRow]


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def canonical_analyte(name: str) -> str:
    """Return the canonical analyte name for an alias, or the input unchanged."""

    return _ANALYTE_ALIASES.get(_key(str(name)), str(name))


def _panel_for(canonical: str) -> str:
    for panel, members in _PANEL_MEMBERS.items():
        if canonical in members:
            return panel
    return "other"


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return (
        result
        if result == result and result not in (float("inf"), float("-inf"))
        else None
    )


def _analyte_row(raw: Mapping[str, Any]) -> AnalyteRow:
    analyte = canonical_analyte(str(raw.get("analyte", "")))
    value = _finite_float(raw.get("value"))
    unit = raw.get("unit")
    unit_str = str(unit) if unit else None

    range_source = raw.get("reference_range")
    if isinstance(range_source, str):
        reference_range: ReferenceRange = parse_reference_range(range_source)
    elif isinstance(range_source, Mapping):
        reference_range = parse_reference_range(range_source)
    else:
        reference_range = parse_reference_range("")

    # The value unit is recorded on the row, but the flag is derived from the
    # range comparison only: passing a value unit without a reference unit would
    # trigger a unit-mismatch and yield "unknown".
    flag = derive_abnormal_flag(value, range_source, explicit_flag=raw.get("flag"))
    return AnalyteRow(
        analyte=analyte,
        value=value,
        unit=unit_str,
        reference_range=reference_range,
        flag=flag,
    )


def structure_lab_panels(results: Iterable[Mapping[str, Any]]) -> list[LabPanel]:
    """Group parsed lab results into recognized panels of normalized rows.

    ``results`` are per-analyte mappings with ``analyte`` and ``value`` and
    optional ``unit`` / ``reference_range`` / ``flag``. Each result is
    canonicalized, assigned to its panel (``CBC``, ``BMP``, or ``other``), and
    normalized. Panels are returned in :data:`PANEL_ORDER`.
    """

    grouped: dict[str, list[AnalyteRow]] = {}
    for raw in results:
        row = _analyte_row(raw)
        panel = _panel_for(row["analyte"])
        grouped.setdefault(panel, []).append(row)

    return [
        LabPanel(panel=panel, analytes=grouped[panel])
        for panel in PANEL_ORDER
        if panel in grouped
    ]


def parse_lab_report(text: str) -> list[LabPanel]:
    """Parse free-text or table lab-report ``text`` into structured panels.

    Panel headers (``CBC:``) are ignored for grouping -- analytes are grouped by
    membership -- and multi-result lines are split. Each ``analyte value [unit]
    [(range)]`` result is extracted and passed to :func:`structure_lab_panels`.
    """

    results: list[dict[str, Any]] = []
    for match in _RESULT_RE.finditer(text):
        results.append(
            {
                "analyte": match.group("analyte"),
                "value": match.group("value"),
                "unit": match.group("unit"),
                "reference_range": match.group("range"),
            }
        )
    return structure_lab_panels(results)


__all__ = [
    "LAB_PANEL_ADVISORY",
    "PANEL_ORDER",
    "AnalyteRow",
    "LabPanel",
    "canonical_analyte",
    "structure_lab_panels",
    "parse_lab_report",
]
