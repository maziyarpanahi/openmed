"""Half-open byte spans must keep empty intervals empty."""

import pytest

from openmed.processing.legacy_encoding import ConversionOffsetMap


@pytest.fixture
def mapping():
    # Three synthetic characters occupying 1, 3, and 2 input bytes.
    return ConversionOffsetMap(6, ((0, 1), (1, 4), (4, 6)), (0, 1, 1, 1, 2, 2))


@pytest.mark.parametrize(
    "point,expected", [(0, 0), (1, 1), (2, 1), (3, 1), (4, 2), (5, 2), (6, 3)]
)
def test_empty_span_anchors_without_selecting_characters(mapping, point, expected):
    assert mapping.to_converted_span(point, point) == (expected, expected)


def test_all_nonempty_spans_keep_overlap_semantics(mapping):
    for start in range(6):
        for end in range(start + 1, 7):
            matches = [
                i
                for i, (a, b) in enumerate(mapping.converted_to_original_spans)
                if a < end and b > start
            ]
            assert mapping.to_converted_span(start, end) == (
                min(matches),
                max(matches) + 1,
            )


def test_empty_point_in_removed_bytes_uses_next_anchor():
    mapping = ConversionOffsetMap(5, ((0, 1), (3, 5)), (0, None, None, 1, 1))
    assert mapping.to_converted_span(1, 1) == (1, 1)
    assert mapping.to_converted_span(2, 2) == (1, 1)
    assert mapping.to_converted_span(4, 4) == (1, 1)


def test_empty_point_in_one_to_many_mapping_stays_empty():
    mapping = ConversionOffsetMap(2, ((0, 2), (0, 2)), (0, 0))
    assert mapping.to_converted_span(1, 1) == (0, 0)
    assert mapping.to_converted_span(0, 2) == (0, 2)


def test_empty_document_and_trailing_removed_bytes():
    assert ConversionOffsetMap(0, (), ()).to_converted_span(0, 0) == (0, 0)
    mapping = ConversionOffsetMap(3, ((0, 1),), (0, None, None))
    assert mapping.to_converted_span(2, 2) == (1, 1)


@pytest.mark.parametrize("start,end", [(-1, 0), (2, 1), (0, 7)])
def test_bounds_validation_is_unchanged(mapping, start, end):
    with pytest.raises(ValueError):
        mapping.to_converted_span(start, end)


def test_reverse_empty_mapping_is_unchanged(mapping):
    for point, expected in [(0, 0), (1, 1), (2, 4), (3, 6)]:
        assert mapping.to_original_span(point, point) == (expected, expected)
