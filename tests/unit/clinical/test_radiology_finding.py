"""Tests for deterministic radiology finding extraction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.clinical import (
    RADIOLOGY_FINDING_ADVISORY,
    RADIOLOGY_LATERALITY_LEXICON,
    extract_radiology_findings,
)
from openmed.eval import radiology_finding_tuple_f1
from openmed.eval.golden import list_fixture_paths

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "clinical"
    / "radiology_finding.jsonl"
)
EVAL_FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "openmed"
    / "eval"
    / "golden"
    / "fixtures"
    / "radiology_finding.jsonl"
)


def _load_fixture(path: Path = FIXTURE) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_fixture_rows_are_synthetic():
    rows = _load_fixture()
    assert rows
    assert all(row["metadata"]["synthetic"] is True for row in rows)


@pytest.mark.parametrize("row", _load_fixture())
def test_extractor_matches_gold_row(row):
    assert extract_radiology_findings(row["text"]) == row["gold"]


def test_single_finding_binds_size_laterality_and_location():
    findings = extract_radiology_findings("A 9 mm mass is in the left upper lobe.")
    finding = findings[0]
    assert (
        finding["finding"],
        finding["laterality"],
        finding["size_value"],
        finding["size_unit"],
        finding["location"],
        finding["radlex_code"],
    ) == ("mass", "left", 9.0, "mm", "upper lobe", None)


def test_multiple_findings_bind_nearest_mixed_units_in_source_order():
    findings = extract_radiology_findings(
        "Right upper lobe 4 mm nodule; left lower lobe 1.2 cm mass."
    )
    assert [
        (
            item["finding"],
            item["laterality"],
            item["size_value"],
            item["size_unit"],
            item["location"],
        )
        for item in findings
    ] == [
        ("nodule", "right", 4.0, "mm", "upper lobe"),
        ("mass", "left", 1.2, "cm", "lower lobe"),
    ]


def test_graph_neighborhoods_prevent_cross_binding_in_one_clause():
    findings = extract_radiology_findings(
        "Right upper lobe 4 mm nodule and left lower lobe 1.2 cm mass."
    )
    assert [
        (
            item["finding"],
            item["laterality"],
            item["size_value"],
            item["size_unit"],
            item["location"],
        )
        for item in findings
    ] == [
        ("nodule", "right", 4.0, "mm", "upper lobe"),
        ("mass", "left", 1.2, "cm", "lower lobe"),
    ]


def test_bilateral_and_unknown_laterality_are_explicit():
    assert (
        extract_radiology_findings("Bilateral pulmonary nodules.")[0]["laterality"]
        == "bilateral"
    )
    assert (
        extract_radiology_findings("A lesion is present.")[0]["laterality"] == "unknown"
    )


@pytest.mark.parametrize(
    "text,laterality",
    [("Left breast mass.", "left"), ("Right breast mass.", "right")],
)
def test_left_and_right_laterality(text, laterality):
    assert extract_radiology_findings(text)[0]["laterality"] == laterality


def test_missing_size_and_location_are_none():
    finding = extract_radiology_findings("There is a nodule.")[0]
    assert finding["size_value"] is None
    assert finding["size_unit"] is None
    assert finding["location"] is None


def test_provenance_spans_index_into_original_text():
    text = "A 6 mm right lower lobe nodule."
    finding = extract_radiology_findings(text)[0]
    spans = finding["provenance_spans"]
    assert text[spans["finding"]["start"] : spans["finding"]["end"]] == "nodule"
    assert text[spans["size_value"]["start"] : spans["size_value"]["end"]] == "6"
    assert text[spans["size_unit"]["start"] : spans["size_unit"]["end"]] == "mm"
    assert text[spans["laterality"]["start"] : spans["laterality"]["end"]] == "right"
    assert text[spans["location"]["start"] : spans["location"]["end"]] == "lower lobe"


def test_optional_radlex_mapping_is_caller_supplied_and_case_insensitive():
    finding = extract_radiology_findings(
        "A nodule.", radlex_mapping={"NODULE": "RID4278"}
    )[0]
    assert finding["radlex_code"] == "RID4278"


def test_optional_radlex_mapping_file_is_local_and_caller_supplied(tmp_path):
    mapping_path = tmp_path / "radlex.json"
    mapping_path.write_text(json.dumps({"NODULE": "RID4278"}), encoding="utf-8")

    finding = extract_radiology_findings(
        "A nodule.",
        radlex_mapping=mapping_path,
    )[0]

    assert finding["radlex_code"] == "RID4278"


def test_documented_laterality_lexicon_is_exposed_and_immutable():
    assert RADIOLOGY_LATERALITY_LEXICON["b/l"] == "bilateral"
    with pytest.raises(TypeError):
        RADIOLOGY_LATERALITY_LEXICON["unilateral"] = "unknown"  # type: ignore[index]


def test_attribute_distance_prevents_remote_cross_binding():
    text = "Left " + "descriptive text " * 8 + "nodule"
    finding = extract_radiology_findings(text, max_attribute_distance=20)[0]
    assert finding["laterality"] == "unknown"


def test_negative_attribute_distance_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        extract_radiology_findings("A nodule.", max_attribute_distance=-1)


def test_offline_eval_fixture_meets_finding_tuple_f1_gate():
    rows = _load_fixture(EVAL_FIXTURE)
    predicted = [
        finding for row in rows for finding in extract_radiology_findings(row["text"])
    ]
    gold = [finding for row in rows for finding in row["gold"]]

    metrics = radiology_finding_tuple_f1(predicted, gold)

    assert metrics.f1 >= 0.85
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0


def test_domain_eval_fixture_is_not_loaded_as_pii_span_gold():
    assert EVAL_FIXTURE not in list_fixture_paths()


def test_order_is_deterministic_and_empty_input_is_empty():
    text = "A 2 mm nodule. A 1 cm mass."
    assert extract_radiology_findings(text) == extract_radiology_findings(text)
    assert [item["finding"] for item in extract_radiology_findings(text)] == [
        "nodule",
        "mass",
    ]
    assert extract_radiology_findings("") == []


def test_advisory_states_review_only_scope():
    assert "radiologist review" in RADIOLOGY_FINDING_ADVISORY
    assert "not diagnostic" in RADIOLOGY_FINDING_ADVISORY
