"""Only ancestor/descendant list spans may overlap."""

import pytest

from openmed.processing.lists import ListItemSpan, parse_lists, validate_list_items

TEXT = "0123456789abcdefghijklmnop"


def _item(start, end, level=0, parent=None):
    return ListItemSpan(
        text=TEXT[start:end],
        start=start,
        end=end,
        nesting_level=level,
        style="line",
        parent_index=parent,
    )


@pytest.mark.parametrize("right_start", [2, 7, 11])
def test_overlapping_children_are_rejected(right_start):
    items = [_item(0, 20), _item(2, 12, 1, 0), _item(right_start, 16, 1, 0)]
    with pytest.raises(ValueError, match="overlaps its sibling"):
        validate_list_items(TEXT, items)


def test_overlap_at_deeper_level_is_rejected():
    items = [_item(0, 24), _item(1, 20, 1, 0), _item(2, 12, 2, 1), _item(8, 16, 2, 1)]
    with pytest.raises(ValueError, match="overlaps its sibling"):
        validate_list_items(TEXT, items)


def test_adjacent_children_and_contained_grandchild_are_valid():
    items = [
        _item(0, 24),
        _item(1, 12, 1, 0),
        _item(2, 8, 2, 1),
        _item(12, 20, 1, 0),
        _item(13, 19, 2, 3),
    ]
    validate_list_items(TEXT, items)


def test_different_parents_have_independent_sibling_tracking():
    items = [_item(0, 12), _item(1, 8, 1, 0), _item(12, 24), _item(13, 20, 1, 2)]
    validate_list_items(TEXT, items)


def test_existing_top_level_overlap_error_is_preserved():
    with pytest.raises(ValueError, match="top-level.*overlaps"):
        validate_list_items(TEXT, [_item(0, 12), _item(8, 20)])


def test_parser_generated_nested_lists_remain_valid():
    text = "- root\n  - first\n    - grandchild\n  - second\n- next\n"
    items = parse_lists(text)
    validate_list_items(text, items)
    assert len(items) == 5
