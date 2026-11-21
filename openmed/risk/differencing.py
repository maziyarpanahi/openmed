"""Aggregate-only screening of count differences across comparable releases."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence


def check_differencing(
    releases: Iterable[Iterable[Mapping]], *, dimensions: Sequence[str], threshold: int
) -> dict:
    """Count cells exposing a positive difference below a caller's threshold.

    Args:
        releases: Snapshots from the same counting universe. A row has a
            nonnegative integer `count` and zero or more declared dimensions
            with string or integer values. Omitted dimensions denote a subtotal.
            Distinct values at a common grain must be mutually exclusive.
            Supply explicit zeros for empty cells that a reader can observe.
        dimensions: Unique metadata codes for the comparable dimensions.
        threshold: Exclusive upper bound for a flagged absolute difference.

    Returns:
        The number of distinct flagged cells and their dimension codes.
        Each cell is counted once across releases and marginal comparisons.
    """
    if type(threshold) is not int or threshold < 1:
        raise ValueError("invalid differencing threshold")
    if (
        isinstance(dimensions, str)
        or not dimensions
        or any(
            not isinstance(d, str)
            or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", d)
            or d == "count"
            for d in dimensions
        )
        or len(set(dimensions)) != len(dimensions)
    ):
        raise ValueError("invalid dimension codes")

    allowed = set(dimensions) | {"count"}
    history = defaultdict(set)
    flagged = set()
    for release in releases:
        grains = defaultdict(dict)
        for row in release:
            if not isinstance(row, Mapping) or not row.keys() <= allowed:
                raise ValueError("invalid count table columns")
            count = row.get("count")
            if type(count) is not int or count < 0:
                raise ValueError("invalid count")
            grain = tuple(sorted(set(row) - {"count"}))
            if any(type(row[d]) not in (int, str) for d in grain):
                raise ValueError("invalid dimension value")
            key = tuple((d, row[d]) for d in grain)
            if key in grains[grain]:
                raise ValueError("duplicate cell in release")
            grains[grain][key] = count
            if any(0 < abs(previous - count) < threshold for previous in history[key]):
                flagged.add(key)
            history[key].add(count)

        for parent_dims, parents in grains.items():
            for child_dims, children in grains.items():
                if not set(parent_dims) < set(child_dims):
                    continue
                subtotals = defaultdict(int)
                for child, count in children.items():
                    parent = tuple(pair for pair in child if pair[0] in parent_dims)
                    if parent not in parents:
                        continue
                    subtotals[parent] += count
                    if 0 < parents[parent] - count < threshold:
                        flagged.add(parent)
                for parent, subtotal in subtotals.items():
                    remainder = parents[parent] - subtotal
                    if remainder < 0:
                        raise ValueError("inconsistent marginal counts")
                    if 0 < remainder < threshold:
                        flagged.add(parent)
    return {
        "flagged_cell_count": len(flagged),
        "dimension_codes": sorted({d for key in flagged for d, _ in key}),
    }
