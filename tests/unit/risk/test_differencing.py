"""Synthetic non-DP aggregate difference controls."""

import json

import pytest

from openmed.risk.differencing import check_differencing


def test_cross_release_small_difference_is_deduplicated():
    rows = [[{"region": "secret-region", "count": v}] for v in (20, 100, 21, 22)]
    report = check_differencing(rows, dimensions=["region"], threshold=5)
    assert report == {"flagged_cell_count": 1, "dimension_codes": ["region"]}
    assert "secret-region" not in json.dumps(report)


@pytest.mark.parametrize("counts", [(9, 9), (0, 5), (7, 50)])
def test_strict_boundary_and_large_differences(counts):
    assert (
        check_differencing(
            [[{"count": n}] for n in counts], dimensions=["region"], threshold=5
        )["flagged_cell_count"]
        == 0
    )


def test_marginal_combination_and_single_child():
    rows = [
        {"region": "secret", "count": 100},
        {"region": "secret", "age": "young", "count": 60},
        {"region": "secret", "age": "old", "count": 39},
        {"region": "other", "count": 15},
        {"region": "other", "age": "young", "count": 14},
    ]
    assert check_differencing([rows], dimensions=["region", "age"], threshold=5) == {
        "flagged_cell_count": 2,
        "dimension_codes": ["region"],
    }


def test_unobserved_cells_and_incomparable_grains():
    releases = [[{"age": "old", "count": 20}, {"region": "secret", "count": 19}], []]
    assert (
        check_differencing(releases, dimensions=["age", "region"], threshold=5)[
            "flagged_cell_count"
        ]
        == 0
    )


@pytest.mark.parametrize(
    "rows",
    [
        [{"count": True}],
        [{"secret": "foo", "count": 1}],
        [{"count": 1}, {"count": 2}],
        [{"count": 2}, {"region": "secret", "count": 5}],
    ],
)
def test_invalid_input_diagnostics(rows):
    with pytest.raises(ValueError) as error:
        check_differencing([rows], dimensions=["region"], threshold=5)
    assert "secret" not in str(error.value)
