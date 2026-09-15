"""Offline regression tests for de-identification-aware citation boundaries."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from openmed.clinical.citation_boundaries import (
    CITATION_CROSSES_REPLACEMENT,
    DOCUMENT_DIGEST_MISMATCH,
    SOURCE_VERSION_UNAVAILABLE,
    CitationBoundaryError,
    build_deidentification_offset_map,
    validate_citation_boundaries,
)

SYNTHETIC_PROTECTED_VALUE = "SYNTHETIC_PROTECTED_VALUE"


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _result() -> SimpleNamespace:
    source = f"prefix {SYNTHETIC_PROTECTED_VALUE} suffix"
    start = source.index(SYNTHETIC_PROTECTED_VALUE)
    end = start + len(SYNTHETIC_PROTECTED_VALUE)
    post = source[:start] + "[NAME]" + source[end:]
    return SimpleNamespace(
        original_text=source,
        deidentified_text=post,
        pii_entities=[
            SimpleNamespace(
                start=start,
                end=end,
                redacted_text="[NAME]",
            )
        ],
    )


def _map():
    return build_deidentification_offset_map(_result(), source_version="source-v1")


def _citation(start: int, end: int, *, source_version: str = "source-v1"):
    return {
        "source_offset": {"start": start, "end": end},
        "document_digest": _map().document_digest,
        "source_version": source_version,
    }


def test_result_builder_records_post_offsets_and_digest_without_source_values():
    offset_map = _map()

    assert offset_map.source_length == len(_result().original_text)
    assert offset_map.post_length == len(_result().deidentified_text)
    assert offset_map.document_digest == _digest(_result().deidentified_text)
    assert offset_map.replacements[0].source_start == 7
    assert offset_map.replacements[0].post_start == 7
    assert offset_map.replacements[0].post_end == 13
    assert SYNTHETIC_PROTECTED_VALUE not in offset_map.to_json()
    assert offset_map.to_json() == offset_map.to_json()


def test_validates_post_deidentified_offsets_and_projects_to_original_offsets():
    offset_map = _map()
    report = validate_citation_boundaries(
        [
            _citation(14, 20),  # the stable suffix after the shortened replacement
            _citation(7, 13),  # one complete replacement interval
            _citation(0, 6),  # the stable prefix
        ],
        offset_map,
    )

    assert report.valid
    assert report.accepted_count == 3
    assert [
        (item.citation.post_start, item.citation.post_end) for item in report.accepted
    ] == [(0, 6), (7, 13), (14, 20)]
    assert [(item.source_start, item.source_end) for item in report.accepted] == [
        (0, 6),
        (7, 7 + len(SYNTHETIC_PROTECTED_VALUE)),
        (
            7 + len(SYNTHETIC_PROTECTED_VALUE) + 1,
            7 + len(SYNTHETIC_PROTECTED_VALUE) + 7,
        ),
    ]


def test_rejects_citations_that_cross_or_cut_a_replacement_boundary():
    offset_map = _map()
    citation = _citation(6, 8)

    with pytest.raises(CitationBoundaryError) as error:
        validate_citation_boundaries([citation], offset_map)

    assert error.value.reason_code == CITATION_CROSSES_REPLACEMENT
    assert SYNTHETIC_PROTECTED_VALUE not in str(error.value)

    report = validate_citation_boundaries(
        [citation],
        offset_map,
        raise_on_error=False,
    )
    assert not report.valid
    assert report.reason_counts == {CITATION_CROSSES_REPLACEMENT: 1}
    assert SYNTHETIC_PROTECTED_VALUE not in report.to_json()


def test_rejects_original_offsets_after_text_length_changes():
    offset_map = _map()
    original_start = 7
    original_end = original_start + len(SYNTHETIC_PROTECTED_VALUE)

    report = validate_citation_boundaries(
        [
            {
                "start": original_start,
                "end": original_end,
                "document_digest": offset_map.document_digest,
                "source_version": "source-v1",
            }
        ],
        offset_map,
        raise_on_error=False,
    )

    assert not report.valid
    assert report.reason_counts == {"invalid_citation_offset": 1}


def test_rejects_digest_mismatch_without_echoing_supplied_value():
    offset_map = _map()
    report = validate_citation_boundaries(
        [
            {
                "start": 0,
                "end": 6,
                "document_digest": "SYNTHETIC_WRONG_DOCUMENT_DIGEST",
                "source_version": "source-v1",
            }
        ],
        offset_map,
        raise_on_error=False,
    )

    assert report.reason_counts == {DOCUMENT_DIGEST_MISMATCH: 1}
    serialized = report.to_json()
    assert "SYNTHETIC_WRONG_DOCUMENT_DIGEST" not in serialized
    assert "document_digest" not in str(CitationBoundaryError(DOCUMENT_DIGEST_MISMATCH))


def test_rejects_unavailable_source_versions_and_honours_explicit_allow_list():
    offset_map = _map()
    citation = _citation(0, 6, source_version="retired-v0")

    report = validate_citation_boundaries(
        [citation],
        offset_map,
        raise_on_error=False,
    )
    assert report.reason_counts == {SOURCE_VERSION_UNAVAILABLE: 1}

    current = _citation(0, 6)
    report = validate_citation_boundaries(
        [current],
        offset_map,
        available_source_versions={"retired-v0"},
        raise_on_error=False,
    )
    assert report.reason_counts == {SOURCE_VERSION_UNAVAILABLE: 1}


def test_validation_and_serialization_are_deterministic_for_input_order():
    offset_map = _map()
    citations = [_citation(14, 20), _citation(0, 6), _citation(7, 13)]

    first = validate_citation_boundaries(citations, offset_map)
    second = validate_citation_boundaries(list(reversed(citations)), offset_map)

    assert first.to_json() == second.to_json()
    assert json.loads(first.to_json())["requires_human_review"] is True
    assert SYNTHETIC_PROTECTED_VALUE not in first.to_json()


def test_explicit_text_replacements_and_existing_source_offset_records_are_supported():
    result = _result()
    offset_map = build_deidentification_offset_map(
        result.original_text,
        result.deidentified_text,
        [
            {
                "source_start": 7,
                "source_end": 7 + len(SYNTHETIC_PROTECTED_VALUE),
                "replacement": "[NAME]",
            }
        ],
        document_digest=_digest(result.deidentified_text),
        source_version="source-v1",
    )
    citation = SimpleNamespace(
        source_start=0,
        source_end=6,
        document_id=offset_map.document_digest,
    )

    report = validate_citation_boundaries([citation], offset_map)

    assert report.valid
    assert report.accepted[0].source_start == 0
    assert report.accepted[0].source_end == 6


def test_map_content_errors_are_fixed_and_value_free():
    result = _result()
    with pytest.raises(CitationBoundaryError) as error:
        build_deidentification_offset_map(
            result.original_text,
            result.deidentified_text.replace("[NAME]", "[OTHER]"),
            [
                {
                    "source_start": 7,
                    "source_end": 7 + len(SYNTHETIC_PROTECTED_VALUE),
                    "replacement": "[NAME]",
                }
            ],
        )

    assert error.value.reason_code == "offset_map_content_mismatch"
    assert SYNTHETIC_PROTECTED_VALUE not in str(error.value)
