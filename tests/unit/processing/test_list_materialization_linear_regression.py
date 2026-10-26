"""Boundary planning should be linear while preserving exact item spans."""

import random
from collections.abc import Sequence

import pytest

from openmed.processing.lists import _ItemSeed, _materialize_items, parse_lists


class CountingSeeds(Sequence):
    def __init__(self, data):
        self.data = data
        self.visits = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if isinstance(index, slice):
            result = self.data[index]
            self.visits += len(result)
            return result
        self.visits += 1
        return self.data[index]


def _seeds(levels):
    return [
        _ItemSeed(
            start=index,
            indent=level,
            nesting_level=level,
            style="line",
            marker=None,
            marker_start=None,
            marker_end=None,
            content_start=index,
            parent_index=None,
        )
        for index, level in enumerate(levels)
    ]


@pytest.mark.parametrize("count", [64, 512])
def test_flat_list_boundary_work_is_linear(count):
    seeds = CountingSeeds(_seeds([0] * count))
    items = _materialize_items("x" * count, seeds)
    assert [(item.start, item.end) for item in items] == [
        (index, index + 1) for index in range(count)
    ]
    assert seeds.visits <= 8 * count, "quadratic seed copying/traversal"


@pytest.mark.parametrize(
    "levels", [[], [0], [0, 1, 2, 1, 0], [0, 1, 1, 2, 1, 0], [0, 0, 0]]
)
def test_boundary_semantics_match_simple_reference(levels):
    text = "x" * len(levels)
    items = _materialize_items(text, _seeds(levels))
    for index, item in enumerate(items):
        end = next(
            (j for j in range(index + 1, len(levels)) if levels[j] <= levels[index]),
            len(levels),
        )
        assert item.start == index and item.end == end
        assert item.text == text[index:end]


def test_randomized_boundaries_match_reference():
    rng = random.Random(1729)
    for _ in range(50):
        levels = [rng.randrange(6) for _ in range(rng.randrange(1, 60))]
        items = _materialize_items("x" * len(levels), _seeds(levels))
        for index, item in enumerate(items):
            expected = next(
                (
                    j
                    for j in range(index + 1, len(levels))
                    if levels[j] <= levels[index]
                ),
                len(levels),
            )
            assert item.end == expected


@pytest.mark.parametrize(
    "text",
    [
        "- alpha\n- beta\n- gamma\n",
        "1. alpha\n  - child\n    * grandchild\n  - second\n2. beta\n",
        "alpha\nbeta\ngamma\n",
        "- alpha\r\n  - child\r\n- beta\r\n",
    ],
)
def test_public_parser_preserves_source_and_hierarchy(text):
    items = parse_lists(text)
    assert items
    for item in items:
        assert item.text == text[item.start : item.end]
        if item.parent_index is not None:
            parent = items[item.parent_index]
            assert parent.start <= item.start < item.end <= parent.end
    top = [item for item in items if item.parent_index is None]
    assert "".join(item.text for item in top) == text
