"""Focused tests for deterministic laboratory-result structuring."""

from openmed.clinical import LabResult, extract_lab_results
from openmed.processing.advanced_ner import EntitySpan


def _span(
    text: str,
    value: str,
    label: str,
    *,
    start: int | None = None,
    score: float = 0.95,
) -> EntitySpan:
    start = text.index(value) if start is None else start
    return EntitySpan(
        text=value,
        label=label,
        start=start,
        end=start + len(value),
        score=score,
    )


def test_extract_lab_result_parses_explicit_low_flag() -> None:
    text = "Na 130 mmol/L (135-145) L"
    spans = [
        _span(text, "Na", "LAB_TEST"),
        _span(text, "130", "LAB_VALUE"),
        _span(text, "mmol/L", "UNIT"),
        _span(text, "135-145", "REFERENCE_RANGE"),
        _span(text, "L", "ABNORMAL_FLAG", start=text.rindex("L")),
    ]

    results = extract_lab_results(text, spans)

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, LabResult)
    assert result.analyte.text == "Na"
    assert result.analyte.offset_key() == (0, 2)
    assert result.value == 130
    assert result.unit == "mmol/L"
    assert result.reference_range == {
        "low": 135.0,
        "high": 145.0,
        "low_inclusive": True,
        "high_inclusive": True,
    }
    assert result.abnormal_flag == "low"
    assert 0 < result.score <= 1


def test_extract_lab_result_derives_high_flag_without_marker() -> None:
    text = "Potassium 5.8 mmol/L (3.5-5.1)"

    (result,) = extract_lab_results(
        text,
        [_span(text, "Potassium", "analyte")],
    )

    assert result.value == 5.8
    assert result.reference_range is not None
    assert result.reference_range["high"] == 5.1
    assert result.abnormal_flag == "high"


def test_numeric_value_without_analyte_is_not_emitted() -> None:
    text = "130 mmol/L (135-145) L"

    assert (
        extract_lab_results(
            text,
            [
                _span(text, "130", "LAB_VALUE"),
                _span(text, "mmol/L", "UNIT"),
            ],
        )
        == ()
    )


def test_number_followed_by_prose_is_not_treated_as_a_measurement() -> None:
    text = "Sodium 130 today"

    assert extract_lab_results(text, [_span(text, "Sodium", "LAB_TEST")]) == ()


def test_nearest_analyte_receives_each_measurement() -> None:
    text = "Na 130 mmol/L (135-145) L; K 5.8 mmol/L (3.5-5.1)"
    results = extract_lab_results(
        text,
        [
            _span(text, "Na", "LAB_TEST"),
            _span(text, "K", "LAB_TEST"),
        ],
    )

    assert [
        (result.analyte.text, result.value, result.abnormal_flag) for result in results
    ] == [
        ("Na", 130.0, "low"),
        ("K", 5.8, "high"),
    ]


def test_explicit_critical_flag_is_preserved_without_range() -> None:
    text = "Lactate 8.2 mmol/L critical"

    (result,) = extract_lab_results(
        text,
        [_span(text, "Lactate", "LAB_TEST")],
    )

    assert result.reference_range is None
    assert result.abnormal_flag == "critical"


def test_lab_result_linking_does_not_cross_precomputed_sections() -> None:
    text = "Na 130 mmol/L"
    sections = (
        {"label": "first", "start": 0, "end": 2},
        {"label": "second", "start": 2, "end": len(text)},
    )

    assert (
        extract_lab_results(
            text,
            [_span(text, "Na", "LAB_TEST")],
            sections=sections,
        )
        == ()
    )


def test_lab_result_supports_non_numeric_lab_value_head_alias() -> None:
    text = "Na 130 mmol/L (135-145)"

    (result,) = extract_lab_results(
        text,
        [_span(text, "Na", "LAB_VALUE")],
    )

    assert result.analyte.text == "Na"
    assert result.abnormal_flag == "low"
