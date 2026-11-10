"""Tests for the lab-panel structurer."""

from __future__ import annotations

from openmed.structured.lab_panels import (
    LAB_PANEL_ADVISORY,
    LabPanel,
    canonical_analyte,
    parse_lab_report,
    structure_lab_panels,
)

# --------------------------------------------------------------------------
# Analyte canonicalization / panel recognition
# --------------------------------------------------------------------------


def test_aliases_resolve_to_canonical_analytes():
    assert canonical_analyte("Hgb") == "Hemoglobin"
    assert canonical_analyte("Na") == "Sodium"
    assert canonical_analyte("K") == "Potassium"
    assert canonical_analyte("WBC") == "WBC"


def test_groups_results_into_recognized_panels():
    results = [
        {
            "analyte": "WBC",
            "value": 7.5,
            "unit": "10^3/uL",
            "reference_range": "4.0-11.0",
        },
        {"analyte": "Na", "value": 140, "unit": "mmol/L", "reference_range": "135-145"},
        {"analyte": "Hgb", "value": 13.5, "unit": "g/dL", "reference_range": "12-16"},
    ]

    panels = {panel["panel"]: panel for panel in structure_lab_panels(results)}

    assert set(panels) == {"CBC", "BMP"}
    cbc = {row["analyte"] for row in panels["CBC"]["analytes"]}
    assert cbc == {"WBC", "Hemoglobin"}
    assert [r["analyte"] for r in panels["BMP"]["analytes"]] == ["Sodium"]


def test_unknown_analyte_falls_into_other_panel():
    panels = structure_lab_panels([{"analyte": "Widgetase", "value": 1.0}])
    assert panels[0]["panel"] == "other"
    assert panels[0]["analytes"][0]["analyte"] == "Widgetase"


# --------------------------------------------------------------------------
# Normalization: reference range + abnormal flag
# --------------------------------------------------------------------------


def test_abnormal_flag_derived_from_reference_range():
    results = [
        {
            "analyte": "WBC",
            "value": 15.0,
            "unit": "10^3/uL",
            "reference_range": "4.0-11.0",
        },
        {"analyte": "Platelets", "value": 250, "reference_range": "150-400"},
    ]
    rows = {r["analyte"]: r for r in structure_lab_panels(results)[0]["analytes"]}

    assert rows["WBC"]["flag"] == "high"
    assert rows["Platelets"]["flag"] == "normal"


def test_reference_range_is_parsed_into_bounds():
    row = structure_lab_panels(
        [{"analyte": "Sodium", "value": 140, "reference_range": "135-145"}]
    )[0]["analytes"][0]

    assert row["reference_range"]["low"] == 135.0
    assert row["reference_range"]["high"] == 145.0


def test_mapping_reference_range_is_preserved_and_compared_with_units():
    row = structure_lab_panels(
        [
            {
                "analyte": "Glucose",
                "value": 120,
                "unit": "mg/dL",
                "reference_range": {
                    "low": 70,
                    "high": 99,
                    "low_inclusive": True,
                    "high_inclusive": True,
                    "unit": "mg/dL",
                },
            }
        ]
    )[0]["analytes"][0]

    assert row["reference_range"] == {
        "low": 70.0,
        "high": 99.0,
        "low_inclusive": True,
        "high_inclusive": True,
        "unit": "mg/dL",
    }
    assert row["flag"] == "high"


# --------------------------------------------------------------------------
# Free-text / table parsing
# --------------------------------------------------------------------------


def test_parse_lab_report_handles_headers_and_result_lines():
    text = (
        "CBC:\n"
        "WBC 7.5 10^3/uL (4.0-11.0)\n"
        "Hgb 13.5 g/dL (12.0-16.0)\n"
        "BMP:\n"
        "Na 140 mmol/L (135-145)\n"
        "K 5.9 mmol/L (3.5-5.1)\n"
    )

    panels = {panel["panel"]: panel for panel in parse_lab_report(text)}

    assert {"CBC", "BMP"} <= set(panels)
    k_row = next(r for r in panels["BMP"]["analytes"] if r["analyte"] == "Potassium")
    assert k_row["value"] == 5.9
    assert k_row["flag"] == "high"  # 5.9 > 5.1


def test_multi_result_line_is_split():
    text = "Na 140  K 4.0  Cl 102"
    panels = parse_lab_report(text)
    analytes = {r["analyte"] for panel in panels for r in panel["analytes"]}
    assert {"Sodium", "Potassium", "Chloride"} <= analytes


def test_pipe_table_uses_panel_header_and_originating_flag():
    text = (
        "CBC:\n"
        "| Analyte | Value | Unit | Reference range | Flag |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| Bands | 12 | % | 0-10 | H |\n"
    )

    panels = parse_lab_report(text)

    assert [panel["panel"] for panel in panels] == ["CBC"]
    assert panels[0]["analytes"] == [
        {
            "analyte": "Bands",
            "value": 12.0,
            "unit": "%",
            "reference_range": {
                "low": 0.0,
                "high": 10.0,
                "low_inclusive": True,
                "high_inclusive": True,
            },
            "flag": "high",
        }
    ]


def test_multiword_analyte_and_unparenthesized_range_are_parsed():
    panels = parse_lab_report("White blood cell count 12.5 10^3/uL 4.0-11.0")

    row = panels[0]["analytes"][0]
    assert row["analyte"] == "WBC"
    assert row["value"] == 12.5
    assert row["unit"] == "10^3/uL"
    assert row["reference_range"]["high"] == 11.0
    assert row["flag"] == "high"


def test_inline_panel_header_and_originating_flag_are_honored():
    panels = parse_lab_report("BMP: K 4.0 mmol/L H (3.5-5.1)")

    assert [panel["panel"] for panel in panels] == ["BMP"]
    assert panels[0]["analytes"][0]["flag"] == "high"


# --------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------


def test_deterministic_and_typed():
    results = [{"analyte": "WBC", "value": 7.5, "reference_range": "4-11"}]
    assert structure_lab_panels(results) == structure_lab_panels(results)
    assert isinstance(structure_lab_panels(results)[0], dict)
    assert LabPanel is not None


def test_empty_input_returns_empty():
    assert structure_lab_panels([]) == []
    assert parse_lab_report("") == []


def test_blank_analytes_are_skipped_and_boolean_values_are_not_numeric():
    panels = structure_lab_panels(
        [
            {"analyte": "", "value": 1},
            {"analyte": "WBC", "value": True},
        ]
    )

    assert len(panels) == 1
    assert panels[0]["analytes"][0]["value"] is None


def test_advisory_exposed():
    assert isinstance(LAB_PANEL_ADVISORY, str) and LAB_PANEL_ADVISORY
