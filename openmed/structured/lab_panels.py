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
_NUMERIC = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_RESULT_RE = re.compile(
    r"(?P<analyte>[A-Za-z][A-Za-z0-9]*)\s+"
    rf"(?P<value>{_NUMERIC})"
    r"(?:\s+(?P<unit>[^\s()]*[/%^][^\s()]*))?"
    r"(?:\s*\((?P<range>[^)]*)\))?"
)
_RESULT_SEGMENT_RE = re.compile(
    rf"^(?P<analyte>.+?)\s+(?P<value>{_NUMERIC})(?P<tail>.*)$"
)
_PARENTHESIZED_RANGE_RE = re.compile(r"\((?P<range>[^()]*)\)")
_RANGE_SUFFIX_RE = re.compile(
    rf"(?P<range>(?:[<>]=?|≤|≥)\s*{_NUMERIC}(?:\s+\S+)?|"
    rf"{_NUMERIC}\s*(?:-|to|–|—)\s*{_NUMERIC}(?:\s+\S+)?)\s*$",
    re.IGNORECASE,
)
_FLAG_SUFFIX_RE = re.compile(
    r"(?:^|\s)(?P<flag>HH|LL|H|L|N|HIGH|LOW|NORMAL|CRIT|CRITICAL)\s*$",
    re.IGNORECASE,
)
_PANEL_HEADER_RE = re.compile(
    r"^\s*(?P<panel>CBC|BMP)\s*:?[ \t]*$",
    re.IGNORECASE,
)
_INLINE_PANEL_HEADER_RE = re.compile(
    r"^\s*(?P<panel>CBC|BMP)\s*:\s*(?P<rest>.+)$",
    re.IGNORECASE,
)
_TABLE_HEADER_NAMES = frozenset({"analyte", "name", "test"})
_TABLE_VALUE_NAMES = frozenset({"result", "value"})
_MARKDOWN_SEPARATOR_RE = re.compile(r"^:?-{3,}:?$")


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
    """Return the canonical analyte name for an alias.

    Args:
        name: Raw analyte label from a parsed result or report.

    Returns:
        The canonical label for a known alias, or ``name`` unchanged.
    """

    return _ANALYTE_ALIASES.get(_key(str(name)), str(name))


def _panel_for(canonical: str) -> str:
    for panel, members in _PANEL_MEMBERS.items():
        if canonical in members:
            return panel
    return "other"


def _finite_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return (
        result
        if result == result and result not in (float("inf"), float("-inf"))
        else None
    )


def _normalize_reference_range(source: object) -> ReferenceRange:
    if isinstance(source, str):
        return parse_reference_range(source)
    if not isinstance(source, Mapping):
        return parse_reference_range("")

    raw_low = source.get("low")
    raw_high = source.get("high")
    low = _finite_float(raw_low)
    high = _finite_float(raw_high)
    if (raw_low is not None and low is None) or (raw_high is not None and high is None):
        return parse_reference_range("")
    if low is not None and high is not None and low > high:
        return parse_reference_range("")

    reference_range = ReferenceRange(
        low=low,
        high=high,
        low_inclusive=(
            source.get("low_inclusive")
            if isinstance(source.get("low_inclusive"), bool)
            else True
        ),
        high_inclusive=(
            source.get("high_inclusive")
            if isinstance(source.get("high_inclusive"), bool)
            else True
        ),
    )
    unit = source.get("unit") or source.get("units") or source.get("reference_unit")
    if isinstance(unit, str) and unit.strip():
        reference_range["unit"] = unit.strip()
    return reference_range


def _analyte_row(raw: Mapping[str, Any]) -> AnalyteRow:
    analyte = canonical_analyte(str(raw.get("analyte", "")).strip())
    value = _finite_float(raw.get("value"))
    unit = raw.get("unit")
    unit_str = str(unit).strip() if unit else None

    range_source = raw.get("reference_range")
    reference_range = _normalize_reference_range(range_source)
    reference_unit = reference_range.get("unit")
    explicit_flag = raw.get("flag")
    flag = derive_abnormal_flag(
        value,
        reference_range,
        explicit_flag=str(explicit_flag) if explicit_flag is not None else None,
        value_unit=unit_str if reference_unit else None,
        reference_unit=reference_unit,
    )
    return AnalyteRow(
        analyte=analyte,
        value=value,
        unit=unit_str,
        reference_range=reference_range,
        flag=flag,
    )


def structure_lab_panels(results: Iterable[Mapping[str, Any]]) -> list[LabPanel]:
    """Group parsed lab results into recognized panels of normalized rows.

    Args:
        results: Per-analyte mappings with ``analyte`` and ``value`` plus
            optional ``unit``, ``reference_range``, ``flag``, and recognized
            ``panel`` hint fields.

    Returns:
        Normalized panels in :data:`PANEL_ORDER`. Blank analyte labels are
        omitted, and analytes without a recognized panel fall into ``other``.
    """

    grouped: dict[str, list[AnalyteRow]] = {}
    for raw in results:
        row = _analyte_row(raw)
        if not row["analyte"]:
            continue
        panel_hint = str(raw.get("panel", "")).strip()
        panel = panel_hint.upper() if panel_hint.upper() in _PANEL_MEMBERS else None
        panel = panel or _panel_for(row["analyte"])
        grouped.setdefault(panel, []).append(row)

    return [
        LabPanel(panel=panel, analytes=grouped[panel])
        for panel in PANEL_ORDER
        if panel in grouped
    ]


def _table_cells(line: str) -> list[str] | None:
    delimiter = "|" if "|" in line else "\t" if "\t" in line else None
    if delimiter is None:
        return None
    cells = [cell.strip() for cell in line.split(delimiter)]
    while cells and not cells[0]:
        cells.pop(0)
    while cells and not cells[-1]:
        cells.pop()
    return cells


def _parse_table_row(line: str, panel: str | None) -> dict[str, Any] | None:
    cells = _table_cells(line)
    if cells is None or len(cells) < 2:
        return None
    if all(_MARKDOWN_SEPARATOR_RE.fullmatch(cell) for cell in cells):
        return None
    if _key(cells[0]) in _TABLE_HEADER_NAMES and _key(cells[1]) in _TABLE_VALUE_NAMES:
        return None
    if _finite_float(cells[1]) is None:
        return None
    return {
        "analyte": cells[0],
        "value": cells[1],
        "unit": cells[2] if len(cells) > 2 else None,
        "reference_range": cells[3] if len(cells) > 3 else None,
        "flag": cells[4] if len(cells) > 4 else None,
        "panel": panel,
    }


def _parse_result_segment(
    segment: str,
    panel: str | None,
) -> dict[str, Any] | None:
    match = _RESULT_SEGMENT_RE.fullmatch(segment.strip())
    if match is None:
        return None

    analyte = match.group("analyte").strip(" :-")
    if not analyte:
        return None
    tail = match.group("tail").strip()
    flag: str | None = None
    if flag_match := _FLAG_SUFFIX_RE.search(tail):
        flag = flag_match.group("flag")
        tail = tail[: flag_match.start()].strip()

    reference_range: str | None = None
    if range_match := _PARENTHESIZED_RANGE_RE.search(tail):
        reference_range = range_match.group("range").strip()
        tail = f"{tail[: range_match.start()]} {tail[range_match.end() :]}".strip()
    elif range_match := _RANGE_SUFFIX_RE.search(tail):
        reference_range = range_match.group("range").strip()
        tail = tail[: range_match.start()].strip()
    if flag is None and (flag_match := _FLAG_SUFFIX_RE.search(tail)):
        flag = flag_match.group("flag")
        tail = tail[: flag_match.start()].strip()

    return {
        "analyte": analyte,
        "value": match.group("value"),
        "unit": tail or None,
        "reference_range": reference_range,
        "flag": flag,
        "panel": panel,
    }


def _parse_free_text_line(line: str, panel: str | None) -> list[dict[str, Any]]:
    segments = [part for part in re.split(r"\s{2,}", line) if part.strip()]
    if len(segments) > 1:
        parsed = [_parse_result_segment(segment, panel) for segment in segments]
        if all(result is not None for result in parsed):
            return [result for result in parsed if result is not None]

    compact_matches = list(_RESULT_RE.finditer(line))
    if len(compact_matches) > 1:
        return [
            {
                "analyte": match.group("analyte"),
                "value": match.group("value"),
                "unit": match.group("unit"),
                "reference_range": match.group("range"),
                "panel": panel,
            }
            for match in compact_matches
        ]

    result = _parse_result_segment(line, panel)
    if result is not None:
        return [result]

    return [
        {
            "analyte": match.group("analyte"),
            "value": match.group("value"),
            "unit": match.group("unit"),
            "reference_range": match.group("range"),
            "panel": panel,
        }
        for match in compact_matches
    ]


def parse_lab_report(text: str) -> list[LabPanel]:
    """Parse free-text or table lab-report ``text`` into structured panels.

    Recognized panel headers (``CBC:`` / ``BMP:``) carry forward to following
    rows, pipe/tabular rows are parsed by column, and multi-result lines are
    split. Each normalized result is passed to :func:`structure_lab_panels`.

    Args:
        text: Synthetic or de-identified free-text, pipe-table, or tabular lab
            report content.

    Returns:
        Normalized panels in deterministic panel and source-row order.
    """

    results: list[dict[str, Any]] = []
    panel: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if header_match := _PANEL_HEADER_RE.fullmatch(line):
            panel = header_match.group("panel").upper()
            continue
        if inline_header := _INLINE_PANEL_HEADER_RE.fullmatch(line):
            panel = inline_header.group("panel").upper()
            line = inline_header.group("rest").strip()

        table_row = _parse_table_row(line, panel)
        if table_row is not None:
            results.append(table_row)
            continue
        if _table_cells(line) is not None:
            continue
        results.extend(_parse_free_text_line(line, panel))
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
