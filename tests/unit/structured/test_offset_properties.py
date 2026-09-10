"""Cross-format, offline properties for normalized text offsets."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from openmed.structured.offset_properties import (
    SUPPORTED_FORMATS,
    OffsetProjectionError,
    OffsetSpan,
    SyntheticOffsetCase,
    build_synthetic_offset_cases,
    run_offset_property_suite,
    safe_failure_category,
    validate_offset_projection,
)

_TOKEN_RE = re.compile(r"\S+", re.UNICODE)
_TOKENS = st.sampled_from(("alpha", "β", "東京", "café", "🧪", "line"))


@dataclass(frozen=True)
class _AdapterResult:
    """Minimal adapter result used by every synthetic format contract."""

    text: str
    spans: tuple[OffsetSpan, ...]


def _identity_adapter(case: SyntheticOffsetCase) -> _AdapterResult:
    return _AdapterResult(text=case.text, spans=case.source_spans)


def _identity_adapters() -> dict[str, Any]:
    return {format_name: _identity_adapter for format_name in SUPPORTED_FORMATS}


def test_synthetic_cases_cover_all_formats_and_are_deterministic() -> None:
    first = build_synthetic_offset_cases()
    second = build_synthetic_offset_cases()

    assert first == second
    assert tuple(case.format_name for case in first) == SUPPORTED_FORMATS
    for case in first:
        assert len(case.column_ranges) == 2
        assert case.text.count("\n") >= 3
        assert any(span.is_empty for span in case.redaction_spans)
        payload = json.dumps(case.to_dict(), ensure_ascii=False, sort_keys=True)
        assert case.text not in payload


def test_common_suite_preserves_round_trip_offsets_without_raw_report_text() -> None:
    cases = build_synthetic_offset_cases()
    reports = run_offset_property_suite(_identity_adapters(), cases=cases)

    assert len(reports) == len(SUPPORTED_FORMATS)
    for case, report in zip(cases, reports, strict=True):
        assert report.passed
        assert report.format_name == case.format_name
        assert report.source_span_count == len(case.source_spans)
        assert report.empty_redaction_count >= 3
        assert report.unmapped_redaction_count == 0
        assert report.source_text_sha256 == case.source_text_sha256
        payload = json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True)
        assert case.text not in payload

        projections = report.projections
        assert len(projections) == len(case.redaction_spans)
        for redaction, projection in zip(
            case.redaction_spans, projections, strict=True
        ):
            assert (projection.start, projection.end) == redaction.offsets
            if redaction.is_empty:
                assert projection.source_span_indexes == ()
            else:
                assert projection.source_span_indexes
                assert all(
                    case.source_spans[index].start < redaction.end
                    and case.source_spans[index].end > redaction.start
                    for index in projection.source_span_indexes
                )


def _case_from_columns(
    format_name: str,
    columns: tuple[tuple[str, ...], ...],
) -> SyntheticOffsetCase:
    parts: list[str] = []
    column_ranges: list[OffsetSpan] = []
    cursor = 0
    for column_index, lines in enumerate(columns):
        if column_index:
            parts.append("\n\n")
            cursor += 2
        column_text = "\n".join(lines)
        start = cursor
        parts.append(column_text)
        cursor += len(column_text)
        column_ranges.append(OffsetSpan(start, cursor, label=f"column-{column_index}"))

    text = "".join(parts)
    source_spans = tuple(
        OffsetSpan(match.start(), match.end(), label="source")
        for match in _TOKEN_RE.finditer(text)
    )
    empty_positions = {
        0,
        len(text),
        *(index for index, value in enumerate(text) if value == "\n"),
    }
    redaction_candidates = [
        *source_spans,
        *(
            OffsetSpan(position, position, label="empty")
            for position in sorted(empty_positions)
        ),
    ]
    if len(source_spans) >= 3:
        redaction_candidates.append(
            OffsetSpan(
                source_spans[0].start,
                source_spans[2].end,
                label="cross-line",
            )
        )
    redaction_spans = tuple(
        sorted(redaction_candidates, key=lambda span: (span.start, span.end))
    )
    return SyntheticOffsetCase(
        format_name=format_name,
        text=text,
        source_spans=source_spans,
        redaction_spans=redaction_spans,
        column_ranges=tuple(column_ranges),
    )


@st.composite
def _synthetic_case(draw: st.DrawFn) -> SyntheticOffsetCase:
    format_name = draw(st.sampled_from(SUPPORTED_FORMATS))
    columns = draw(
        st.lists(
            st.lists(_TOKENS, min_size=2, max_size=3),
            min_size=2,
            max_size=3,
        )
    )
    return _case_from_columns(
        format_name,
        tuple(tuple(lines) for lines in columns),
    )


@pytest.mark.contract
@settings(max_examples=40, deadline=None, derandomize=True)
@given(case=_synthetic_case())
def test_property_suite_handles_unicode_lines_columns_and_empty_spans(
    case: SyntheticOffsetCase,
) -> None:
    report = run_offset_property_suite(
        _identity_adapters(),
        cases=(case,),
    )[0]

    assert report.passed
    assert report.empty_redaction_count >= 3
    assert report.unmapped_redaction_count == 0
    assert any(
        len(projection.source_span_indexes) > 1
        for projection in report.projections
        if not projection.is_empty
    )
    assert case.text not in json.dumps(report.to_dict(), ensure_ascii=False)


@pytest.mark.parametrize(
    ("source_spans", "category"),
    [
        (({"start": -1, "end": 0},), "invalid_offsets"),
        (((0, 100),), "invalid_offsets"),
        (((2, 3), (0, 1)), "span_order"),
        (((0, 2), (1, 3)), "span_overlap"),
    ],
)
def test_invalid_inputs_raise_safe_categories(
    source_spans: tuple[Any, ...],
    category: str,
) -> None:
    with pytest.raises(OffsetProjectionError) as raised:
        validate_offset_projection("synthetic text", source_spans)

    assert raised.value.category == category
    assert "synthetic text" not in str(raised.value)


def test_unmapped_non_empty_spans_can_be_reported_without_raw_text() -> None:
    report = validate_offset_projection(
        "synthetic text",
        source_spans=((0, 8),),
        redaction_spans=((9, 13), (13, 13)),
        require_coverage=False,
    )

    assert not report.passed
    assert report.mapped_redaction_count == 0
    assert report.unmapped_redaction_count == 1
    assert report.empty_redaction_count == 1
    assert "synthetic text" not in json.dumps(report.to_dict())


def test_missing_adapter_and_adapter_failures_are_categorized() -> None:
    case = build_synthetic_offset_cases()[0]
    with pytest.raises(OffsetProjectionError) as missing:
        run_offset_property_suite({}, cases=(case,))
    assert missing.value.category == "unsupported_format"

    class UnsupportedDocumentError(ValueError):
        pass

    def failing_adapter(_: SyntheticOffsetCase) -> _AdapterResult:
        raise UnsupportedDocumentError("synthetic parser detail")

    with pytest.raises(OffsetProjectionError) as failed:
        run_offset_property_suite(
            {case.format_name: failing_adapter},
            cases=(case,),
        )
    assert failed.value.category == "unsupported_format"
    assert "synthetic parser detail" not in str(failed.value)


def test_safe_failure_classifier_does_not_use_exception_messages() -> None:
    class MissingDependencyError(RuntimeError):
        pass

    class UnsupportedDocumentError(ValueError):
        pass

    assert safe_failure_category(MissingDependencyError("sensitive detail")) == (
        "missing_dependency"
    )
    assert safe_failure_category(UnsupportedDocumentError("sensitive detail")) == (
        "unsupported_format"
    )
