"""Distribution-preserving synthetic tabular data generation.

The sampler fits empirical numeric and categorical marginals and a Gaussian
rank-correlation model for numeric columns.  Sampling uses stratified empirical
quantiles, so the default marginal acceptance tolerance is a maximum KS or
total-variation distance of ``0.10``.  Numeric correlation deltas default to a
maximum absolute tolerance of ``0.15``.

Profiles contain source-derived marginal values and must be treated as
sensitive local state.  Generated rows are checked against unsalted, in-memory
SHA-256 fingerprints of every source row; sampling fails rather than
returning a copied source row when the source support cannot produce an unseen
combination.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from bisect import bisect_right
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from statistics import NormalDist
from typing import Any

DEFAULT_MARGINAL_TOLERANCE = 0.10
DEFAULT_CORRELATION_TOLERANCE = 0.15

_NORMAL = NormalDist()
_MAX_SAMPLE_ATTEMPTS = 32


@dataclass(frozen=True)
class ColumnDistribution:
    """Empirical distribution fitted for one source column.

    ``values`` contains sorted observations for numeric columns and distinct
    values for categorical columns.  ``probabilities`` is populated only for
    categorical columns.  The profile intentionally stays in memory and can
    contain source-derived values; it is not a shareable privacy artifact.
    """

    name: str
    kind: str
    values: tuple[Any, ...]
    probabilities: tuple[float, ...]
    missing_probability: float
    integral: bool = False


@dataclass(frozen=True)
class TabularProfile:
    """Fitted marginals, correlations, and source-row exclusion fingerprints."""

    columns: tuple[ColumnDistribution, ...]
    correlation_columns: tuple[str, ...]
    correlation_matrix: tuple[tuple[float, ...], ...]
    source_row_hashes: frozenset[str]
    source_row_count: int

    @property
    def column_names(self) -> tuple[str, ...]:
        """Return fitted column names in deterministic source order."""

        return tuple(column.name for column in self.columns)


def fit_tabular_profile(records: Any) -> TabularProfile:
    """Fit empirical marginals and numeric pairwise rank correlations.

    Args:
        records: A non-empty sequence of row mappings, a single row mapping,
            or a DataFrame-like object implementing ``to_dict("records")``.
            Supported cell values are strings, booleans, finite numbers, and
            ``None``.  NaN values are normalized to ``None``.

    Returns:
        A reusable in-memory profile for :func:`sample_synthetic_table`.

    Raises:
        TypeError: If the table shape or a cell value is unsupported.
        ValueError: If the table has no rows or columns, contains duplicate
            stringified column names, or contains an infinite numeric value.
    """

    rows, names = _coerce_table(records)
    columns = tuple(_fit_column(name, rows) for name in names)
    numeric_names = tuple(column.name for column in columns if column.kind == "numeric")
    correlation = _correlation_matrix(rows, numeric_names)
    hashes = frozenset(_row_hash(row, names) for row in rows)
    return TabularProfile(
        columns=columns,
        correlation_columns=numeric_names,
        correlation_matrix=correlation,
        source_row_hashes=hashes,
        source_row_count=len(rows),
    )


def sample_synthetic_table(
    profile: TabularProfile,
    *,
    rows: int | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Sample a synthetic table from ``profile`` without copying source rows.

    Numeric columns use a Gaussian rank-correlation model followed by
    stratified inverse empirical marginals.  Categorical columns use
    stratified empirical probabilities.  Every exact source-row fingerprint
    is rejected, including rows containing missing values.

    Args:
        profile: A profile returned by :func:`fit_tabular_profile`.
        rows: Number of rows to generate.  Defaults to the source row count.
        seed: Deterministic pseudo-random seed.

    Returns:
        A list of new row dictionaries in fitted column order.

    Raises:
        TypeError: If ``profile`` or ``rows`` has the wrong type.
        ValueError: If ``rows`` is negative.
        RuntimeError: If no source-row-free sample can be found.  This can
            happen when a small categorical table exhausts every possible row.
    """

    if not isinstance(profile, TabularProfile):
        raise TypeError("profile must be a TabularProfile")
    row_count = profile.source_row_count if rows is None else rows
    if isinstance(row_count, bool) or not isinstance(row_count, Integral):
        raise TypeError("rows must be an integer or None")
    row_count = int(row_count)
    if row_count < 0:
        raise ValueError("rows must be non-negative")
    if row_count == 0:
        return []

    rng = random.Random(seed)
    for _ in range(_MAX_SAMPLE_ATTEMPTS):
        sampled = _draw_table(profile, row_count, rng)
        _repair_source_matches(sampled, profile, rng)
        if not _source_matches(sampled, profile):
            return sampled
    raise RuntimeError(
        "Unable to generate a table without copying a source row; "
        "the fitted categorical support may exhaust all possible rows."
    )


def tabular_fidelity_report(
    source: Any,
    synthetic: Any,
    *,
    marginal_tolerance: float = DEFAULT_MARGINAL_TOLERANCE,
    correlation_tolerance: float = DEFAULT_CORRELATION_TOLERANCE,
) -> dict[str, Any]:
    """Compare synthetic and source marginals and numeric correlations.

    Numeric marginal distance is the two-sample Kolmogorov-Smirnov statistic;
    categorical marginal distance is total-variation distance.  Missingness is
    included in both measures.  The headline score is the mean of marginal
    similarities and, when numeric pairs exist, correlation similarity.

    Args:
        source: Source rows accepted by :func:`fit_tabular_profile`.
        synthetic: Synthetic rows with exactly the source column set.
        marginal_tolerance: Maximum accepted per-column distance in ``[0, 1]``.
        correlation_tolerance: Maximum accepted absolute correlation delta in
            ``[0, 2]``.

    Returns:
        A deterministic JSON-serializable report.  ``passed`` also requires
        zero exact source rows in ``synthetic``.

    Raises:
        ValueError: If tolerances or table schemas are invalid.
    """

    _validate_tolerance("marginal_tolerance", marginal_tolerance, upper=1.0)
    _validate_tolerance("correlation_tolerance", correlation_tolerance, upper=2.0)
    source_rows, names = _coerce_table(source)
    synthetic_rows, synthetic_names = _coerce_table(synthetic)
    if set(synthetic_names) != set(names):
        raise ValueError("synthetic table must contain exactly the source columns")
    synthetic_rows = [{name: row[name] for name in names} for row in synthetic_rows]

    source_profile = fit_tabular_profile(source_rows)
    column_reports: dict[str, dict[str, Any]] = {}
    marginal_scores: list[float] = []
    for column in source_profile.columns:
        source_values = [row[column.name] for row in source_rows]
        synthetic_values = [row[column.name] for row in synthetic_rows]
        if column.kind == "numeric":
            distance = _numeric_marginal_distance(source_values, synthetic_values)
        else:
            distance = _categorical_distance(source_values, synthetic_values)
        score = max(0.0, 1.0 - distance)
        marginal_scores.append(score)
        column_reports[column.name] = {
            "kind": column.kind,
            "distance": float(distance),
            "score": float(score),
            "within_tolerance": bool(distance <= marginal_tolerance),
        }

    source_corr = _correlation_matrix(source_rows, source_profile.correlation_columns)
    synthetic_corr = _correlation_matrix(
        synthetic_rows, source_profile.correlation_columns
    )
    correlation_reports: list[dict[str, Any]] = []
    correlation_scores: list[float] = []
    for left in range(len(source_profile.correlation_columns)):
        for right in range(left + 1, len(source_profile.correlation_columns)):
            difference = abs(source_corr[left][right] - synthetic_corr[left][right])
            correlation_scores.append(max(0.0, 1.0 - difference / 2.0))
            correlation_reports.append(
                {
                    "columns": [
                        source_profile.correlation_columns[left],
                        source_profile.correlation_columns[right],
                    ],
                    "source": float(source_corr[left][right]),
                    "synthetic": float(synthetic_corr[left][right]),
                    "difference": float(difference),
                    "within_tolerance": bool(difference <= correlation_tolerance),
                }
            )

    components = list(marginal_scores)
    if correlation_scores:
        components.append(sum(correlation_scores) / len(correlation_scores))
    copied_row_count = sum(
        _row_hash(row, names) in source_profile.source_row_hashes
        for row in synthetic_rows
    )
    return {
        "schema_version": 1,
        "source_row_count": len(source_rows),
        "synthetic_row_count": len(synthetic_rows),
        "score": float(sum(components) / len(components)),
        "marginal_tolerance": float(marginal_tolerance),
        "correlation_tolerance": float(correlation_tolerance),
        "columns": column_reports,
        "correlations": correlation_reports,
        "copied_row_count": int(copied_row_count),
        "passed": bool(
            copied_row_count == 0
            and all(item["within_tolerance"] for item in column_reports.values())
            and all(item["within_tolerance"] for item in correlation_reports)
        ),
    }


def _coerce_table(data: Any) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    dataframe_records = _maybe_dataframe_records(data)
    if dataframe_records is not None:
        data = dataframe_records
    if isinstance(data, Mapping):
        raw_rows = [data]
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        raw_rows = list(data)
    else:
        raise TypeError("records must be a row mapping or sequence of row mappings")
    if not raw_rows:
        raise ValueError("records must contain at least one row")
    if not all(isinstance(row, Mapping) for row in raw_rows):
        raise TypeError("every table row must be a mapping")

    names: list[str] = []
    original_names: dict[str, Any] = {}
    for row in raw_rows:
        for key in row:
            name = str(key)
            if name in original_names and original_names[name] != key:
                raise ValueError(f"duplicate stringified column name: {name!r}")
            if name not in original_names:
                original_names[name] = key
                names.append(name)
    if not names:
        raise ValueError("records must contain at least one column")

    rows: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        string_row = {str(key): value for key, value in raw_row.items()}
        rows.append({name: _normalize_cell(string_row.get(name)) for name in names})
    return rows, tuple(names)


def _maybe_dataframe_records(data: Any) -> list[Mapping[str, Any]] | None:
    to_dict = getattr(data, "to_dict", None)
    if to_dict is None or isinstance(data, Mapping):
        return None
    try:
        records = to_dict("records")
    except TypeError:
        return None
    if isinstance(records, list) and all(isinstance(row, Mapping) for row in records):
        return records
    return None


def _normalize_cell(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Real):
        numeric = float(value)
        if math.isnan(numeric):
            return None
        if not math.isfinite(numeric):
            raise ValueError("numeric cells must be finite")
        if isinstance(value, Integral):
            return int(value)
        return numeric
    raise TypeError("table cells must be strings, booleans, finite numbers, or None")


def _fit_column(name: str, rows: Sequence[Mapping[str, Any]]) -> ColumnDistribution:
    observed = [row[name] for row in rows]
    nonmissing = [value for value in observed if value is not None]
    missing_probability = 1.0 - len(nonmissing) / len(observed)
    is_numeric = bool(nonmissing) and all(
        isinstance(value, Real) and not isinstance(value, bool) for value in nonmissing
    )
    if is_numeric:
        integral = all(isinstance(value, Integral) for value in nonmissing)
        values = tuple(sorted(float(value) for value in nonmissing))
        return ColumnDistribution(
            name=name,
            kind="numeric",
            values=values,
            probabilities=(),
            missing_probability=missing_probability,
            integral=integral,
        )

    counts = Counter(_typed_value_key(value) for value in nonmissing)
    category_by_key: dict[tuple[str, Any], Any] = {}
    for value in nonmissing:
        category_by_key.setdefault(_typed_value_key(value), value)
    categories = tuple(
        category_by_key[key]
        for key in dict.fromkeys(_typed_value_key(value) for value in nonmissing)
    )
    total = len(nonmissing)
    probabilities = (
        tuple(counts[_typed_value_key(value)] / total for value in categories)
        if total
        else ()
    )
    return ColumnDistribution(
        name=name,
        kind="categorical",
        values=categories,
        probabilities=probabilities,
        missing_probability=missing_probability,
    )


def _correlation_matrix(
    rows: Sequence[Mapping[str, Any]], names: Sequence[str]
) -> tuple[tuple[float, ...], ...]:
    size = len(names)
    if not size:
        return ()
    complete = [
        row
        for row in rows
        if all(
            row.get(name) is not None
            and isinstance(row.get(name), Real)
            and not isinstance(row.get(name), bool)
            for name in names
        )
    ]
    if len(complete) < 2:
        return _identity_matrix(size)

    ranked = [_normal_scores([float(row[name]) for row in complete]) for name in names]
    matrix: list[list[float]] = [[0.0] * size for _ in range(size)]
    for left in range(size):
        matrix[left][left] = 1.0
        for right in range(left + 1, size):
            value = _pearson(ranked[left], ranked[right])
            matrix[left][right] = value
            matrix[right][left] = value
    return tuple(tuple(value for value in row) for row in matrix)


def _normal_scores(values: Sequence[float]) -> list[float]:
    ranks = _average_ranks(values)
    count = len(ranks)
    return [_NORMAL.inv_cdf((rank - 0.5) / count) for rank in ranks]


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        average = (position + 1 + end) / 2.0
        for index in order[position:end]:
            ranks[index] = average
        position = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_scale == 0.0 or right_scale == 0.0:
        return 0.0
    return max(-1.0, min(1.0, numerator / (left_scale * right_scale)))


def _identity_matrix(size: int) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(1.0 if row == column else 0.0 for column in range(size))
        for row in range(size)
    )


def _draw_table(
    profile: TabularProfile, row_count: int, rng: random.Random
) -> list[dict[str, Any]]:
    sampled = [dict.fromkeys(profile.column_names) for _ in range(row_count)]
    latent = _correlated_normals(
        profile.correlation_matrix,
        row_count,
        rng,
    )
    numeric_positions = {
        name: index for index, name in enumerate(profile.correlation_columns)
    }

    for column in profile.columns:
        if column.kind == "numeric":
            _assign_numeric_column(
                sampled,
                column,
                [row[numeric_positions[column.name]] for row in latent],
                rng,
            )
        else:
            values = _stratified_categorical_values(column, row_count, rng)
            rng.shuffle(values)
            for row, value in zip(sampled, values, strict=True):
                row[column.name] = value
    return sampled


def _correlated_normals(
    matrix: Sequence[Sequence[float]],
    row_count: int,
    rng: random.Random,
) -> list[list[float]]:
    if not matrix:
        return [[] for _ in range(row_count)]
    cholesky = _cholesky(matrix)
    output: list[list[float]] = []
    for _ in range(row_count):
        independent = [rng.gauss(0.0, 1.0) for _ in matrix]
        output.append(
            [
                sum(
                    cholesky[row][column] * independent[column]
                    for column in range(row + 1)
                )
                for row in range(len(matrix))
            ]
        )
    return output


def _cholesky(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    size = len(matrix)
    factor = [[0.0] * size for _ in range(size)]
    for row in range(size):
        for column in range(row + 1):
            residual = matrix[row][column] - sum(
                factor[row][index] * factor[column][index] for index in range(column)
            )
            if row == column:
                factor[row][column] = math.sqrt(max(residual, 0.0))
            elif factor[column][column] > 1e-12:
                factor[row][column] = residual / factor[column][column]
            elif abs(residual) > 1e-8:
                return _cholesky(_shrunk_correlation(matrix))
    return factor


def _shrunk_correlation(
    matrix: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(
            1.0 if row == column else 0.99 * float(matrix[row][column])
            for column in range(len(matrix))
        )
        for row in range(len(matrix))
    )


def _assign_numeric_column(
    sampled: list[dict[str, Any]],
    column: ColumnDistribution,
    latent: Sequence[float],
    rng: random.Random,
) -> None:
    row_count = len(sampled)
    nonmissing_count = _allocated_nonmissing_count(
        row_count, column.missing_probability
    )
    indexes = list(range(row_count))
    rng.shuffle(indexes)
    nonmissing_indexes = indexes[:nonmissing_count]
    nonmissing_indexes.sort(key=latent.__getitem__)
    values = [
        _empirical_quantile(
            column.values,
            (position + rng.random()) / nonmissing_count,
            integral=column.integral,
        )
        for position in range(nonmissing_count)
    ]
    values.sort()
    for index, value in zip(nonmissing_indexes, values, strict=True):
        sampled[index][column.name] = value


def _allocated_nonmissing_count(row_count: int, missing_probability: float) -> int:
    expected = row_count * (1.0 - missing_probability)
    return min(row_count, max(0, int(math.floor(expected + 0.5))))


def _empirical_quantile(
    values: Sequence[float], probability: float, *, integral: bool
) -> int | float:
    if len(values) == 1:
        value = values[0]
    else:
        position = probability * (len(values) - 1)
        lower = int(math.floor(position))
        upper = min(lower + 1, len(values) - 1)
        fraction = position - lower
        value = values[lower] * (1.0 - fraction) + values[upper] * fraction
    return int(round(value)) if integral else float(value)


def _stratified_categorical_values(
    column: ColumnDistribution,
    row_count: int,
    rng: random.Random,
) -> list[Any]:
    cumulative: list[tuple[float, Any]] = []
    threshold = column.missing_probability
    if threshold:
        cumulative.append((threshold, None))
    remaining = 1.0 - column.missing_probability
    for value, probability in zip(column.values, column.probabilities, strict=True):
        threshold += remaining * probability
        cumulative.append((threshold, value))
    if not cumulative:
        return [None] * row_count
    cumulative[-1] = (1.0, cumulative[-1][1])
    return [
        _categorical_quantile(
            cumulative,
            (position + rng.random()) / row_count,
        )
        for position in range(row_count)
    ]


def _categorical_quantile(
    cumulative: Sequence[tuple[float, Any]], probability: float
) -> Any:
    for threshold, value in cumulative:
        if probability < threshold:
            return value
    return cumulative[-1][1]


def _repair_source_matches(
    sampled: list[dict[str, Any]],
    profile: TabularProfile,
    rng: random.Random,
) -> None:
    names = profile.column_names
    for _ in range(max(1, len(sampled) * len(names))):
        matches = _source_matches(sampled, profile)
        if not matches:
            return
        changed = False
        for left in matches:
            partners = [index for index in range(len(sampled)) if index != left]
            rng.shuffle(partners)
            columns = list(names)
            rng.shuffle(columns)
            for right in partners:
                for name in columns:
                    if sampled[left][name] == sampled[right][name]:
                        continue
                    sampled[left][name], sampled[right][name] = (
                        sampled[right][name],
                        sampled[left][name],
                    )
                    left_safe = (
                        _row_hash(sampled[left], names) not in profile.source_row_hashes
                    )
                    right_safe = (
                        _row_hash(sampled[right], names)
                        not in profile.source_row_hashes
                    )
                    if left_safe and right_safe:
                        changed = True
                        break
                    sampled[left][name], sampled[right][name] = (
                        sampled[right][name],
                        sampled[left][name],
                    )
                if changed:
                    break
            if changed:
                break
        if not changed:
            return


def _source_matches(
    sampled: Sequence[Mapping[str, Any]], profile: TabularProfile
) -> list[int]:
    return [
        index
        for index, row in enumerate(sampled)
        if _row_hash(row, profile.column_names) in profile.source_row_hashes
    ]


def _row_hash(row: Mapping[str, Any], names: Sequence[str]) -> str:
    payload = [[name, _typed_value(row.get(name))] for name in names]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _typed_value(value: Any) -> list[Any]:
    if value is None:
        return ["none", None]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", value]
    if isinstance(value, float):
        return ["float", value.hex()]
    return ["str", str(value)]


def _numeric_marginal_distance(
    source: Sequence[Any], synthetic: Sequence[Any]
) -> float:
    source_missing = sum(value is None for value in source) / len(source)
    synthetic_missing = sum(value is None for value in synthetic) / len(synthetic)
    source_numeric = sorted(float(value) for value in source if value is not None)
    synthetic_nonmissing = [value for value in synthetic if value is not None]
    if not all(
        isinstance(value, Real) and not isinstance(value, bool)
        for value in synthetic_nonmissing
    ):
        return 1.0
    synthetic_numeric = sorted(float(value) for value in synthetic_nonmissing)
    if not source_numeric or not synthetic_numeric:
        ks_distance = 0.0 if source_numeric == synthetic_numeric else 1.0
    else:
        ks_distance = _ks_distance(source_numeric, synthetic_numeric)
    return max(abs(source_missing - synthetic_missing), ks_distance)


def _ks_distance(left: Sequence[float], right: Sequence[float]) -> float:
    points = sorted(set(left) | set(right))
    return max(
        abs(
            bisect_right(left, point) / len(left)
            - bisect_right(right, point) / len(right)
        )
        for point in points
    )


def _categorical_distance(source: Sequence[Any], synthetic: Sequence[Any]) -> float:
    source_counts = Counter(_typed_value_key(value) for value in source)
    synthetic_counts = Counter(_typed_value_key(value) for value in synthetic)
    support = set(source_counts) | set(synthetic_counts)
    return 0.5 * sum(
        abs(
            source_counts[value] / len(source)
            - synthetic_counts[value] / len(synthetic)
        )
        for value in support
    )


def _typed_value_key(value: Any) -> tuple[str, Any]:
    typed = _typed_value(value)
    return str(typed[0]), typed[1]


def _validate_tolerance(name: str, value: float, *, upper: float) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a number between 0 and {upper}")
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= upper:
        raise ValueError(f"{name} must be between 0 and {upper}")


__all__ = [
    "DEFAULT_CORRELATION_TOLERANCE",
    "DEFAULT_MARGINAL_TOLERANCE",
    "ColumnDistribution",
    "TabularProfile",
    "fit_tabular_profile",
    "sample_synthetic_table",
    "tabular_fidelity_report",
]
