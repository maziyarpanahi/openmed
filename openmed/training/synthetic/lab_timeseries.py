"""Synthetic lab time series governed by caller-declared numerical constraints."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass, replace
from itertools import islice

_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_UNIT = re.compile(r"[A-Za-z][A-Za-z0-9./^%-]{0,31}")


def _number(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("invalid synthetic lab numeric constraint")
    if not math.isfinite(value) or abs(value) > 1e12:
        raise ValueError("invalid synthetic lab numeric constraint")
    return 0.0 if value == 0 else float(value)


@dataclass(frozen=True, slots=True)
class LabIntervention:
    """A declared synthetic additive effect from one sample index onward."""

    index: int
    effect: float
    intervention_code: str

    def __post_init__(self) -> None:
        if type(self.index) is not int or not 0 <= self.index < 10000:
            raise ValueError("invalid synthetic lab intervention index")
        if not isinstance(self.intervention_code, str) or not _CODE.fullmatch(
            self.intervention_code
        ):
            raise ValueError("invalid synthetic lab intervention code")
        object.__setattr__(self, "effect", _number(self.effect))


@dataclass(frozen=True, slots=True)
class LabSeriesSpec:
    """A finite synthetic numerical recipe, never a clinical reference interval."""

    analyte: str
    unit: str
    count: int
    lower: float
    upper: float
    baseline: float
    trend_per_step: float = 0.0
    jitter: float = 0.0
    max_step_change: float = 1e12
    start_day: int = 0
    step_days: int = 1
    missing_indices: tuple[int, ...] = ()
    interventions: tuple[LabIntervention, ...] = ()

    def __post_init__(self) -> None:
        for code, pattern in ((self.analyte, _CODE), (self.unit, _UNIT)):
            if not isinstance(code, str) or not pattern.fullmatch(code):
                raise ValueError("invalid synthetic lab metadata code")
        if type(self.count) is not int or not 1 <= self.count <= 10000:
            raise ValueError("invalid synthetic lab sample count")
        if type(self.start_day) is not int or not -100000 <= self.start_day <= 100000:
            raise ValueError("invalid synthetic lab starting day")
        if type(self.step_days) is not int or not 1 <= self.step_days <= 10000:
            raise ValueError("invalid synthetic lab sampling interval")
        for name in (
            "lower",
            "upper",
            "baseline",
            "trend_per_step",
            "jitter",
            "max_step_change",
        ):
            object.__setattr__(self, name, _number(getattr(self, name)))
        if self.lower >= self.upper or self.jitter < 0 or self.max_step_change < 0:
            raise ValueError("invalid synthetic lab range or variability")
        missing = tuple(islice(self.missing_indices, self.count + 1))
        if len(missing) > self.count or any(
            type(i) is not int or not 0 <= i < self.count for i in missing
        ):
            raise ValueError("invalid synthetic lab missingness")
        if len(set(missing)) != len(missing):
            raise ValueError("duplicate synthetic lab missing index")
        object.__setattr__(self, "missing_indices", tuple(sorted(missing)))
        interventions = tuple(islice(self.interventions, 1001))
        if len(interventions) > 1000 or any(
            not isinstance(i, LabIntervention) or i.index >= self.count
            for i in interventions
        ):
            raise ValueError("invalid synthetic lab interventions")
        keys = [(i.index, i.intervention_code) for i in interventions]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate synthetic lab intervention")
        object.__setattr__(
            self,
            "interventions",
            tuple(
                sorted(
                    interventions,
                    key=lambda i: (i.index, i.intervention_code, i.effect),
                )
            ),
        )
        expected = [self.expected(i) for i in range(self.count)]
        if any(
            v - self.jitter < self.lower or v + self.jitter > self.upper
            for v in expected
        ):
            raise ValueError("synthetic lab recipe exceeds declared range")
        if any(
            abs(b - a) + 2 * self.jitter > self.max_step_change
            for a, b in zip(expected, expected[1:])
        ):
            raise ValueError("synthetic lab recipe exceeds declared change limit")

    def expected(self, index: int) -> float:
        """Return the synthetic expected trend plus declared intervention effects."""
        return (
            self.baseline
            + index * self.trend_per_step
            + sum(i.effect for i in self.interventions if i.index <= index)
        )

    @property
    def digest(self) -> str:
        """Fingerprint complete recipe metadata for reproducible evidence binding."""
        value = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "interventions"
        }
        value["interventions"] = [
            {"index": i.index, "effect": i.effect, "code": i.intervention_code}
            for i in self.interventions
        ]
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class LabSample:
    """A synthetic sample in relative days with explicit observed/missing state."""

    index: int
    day: int
    unit: str
    value: float | None


@dataclass(frozen=True, slots=True)
class LabSeries:
    """A bounded synthetic series; negative controls are explicitly labeled."""

    recipe_digest: str
    seed: int
    samples: tuple[LabSample, ...]
    split: str = "synthetic"


def generate_lab_series(spec: LabSeriesSpec, *, seed: int = 0) -> LabSeries:
    """Generate a deterministic local fixture with exact declared missingness."""
    if (
        not isinstance(spec, LabSeriesSpec)
        or type(seed) is not int
        or not 0 <= seed <= 2**63 - 1
    ):
        raise ValueError("invalid synthetic lab generation settings")
    rng = random.Random(seed)
    missing = set(spec.missing_indices)
    rows = []
    for index in range(spec.count):
        value = spec.expected(index) + rng.uniform(-spec.jitter, spec.jitter)
        rows.append(
            LabSample(
                index,
                spec.start_day + index * spec.step_days,
                spec.unit,
                None if index in missing else value,
            )
        )
    series = LabSeries(spec.digest, seed, tuple(rows))
    if validate_lab_series(spec, series):
        raise ValueError("synthetic lab generation failed declared constraints")
    return series


def validate_lab_series(spec: LabSeriesSpec, series: LabSeries) -> tuple[str, ...]:
    """Independently check shape, units, time, missingness, range and declared effects."""
    if not isinstance(spec, LabSeriesSpec) or not isinstance(series, LabSeries):
        raise ValueError("invalid synthetic lab validation input")
    errors = set()
    if series.recipe_digest != spec.digest:
        errors.add("recipe_mismatch")
    if series.split != "synthetic":
        errors.add("negative_control_split")
    if not isinstance(series.samples, tuple) or len(series.samples) > 10000:
        raise ValueError("invalid synthetic lab sample collection")
    if len(series.samples) != spec.count:
        errors.add("sample_count")
    previous = None
    for index, row in enumerate(series.samples):
        if index >= spec.count:
            break
        if not isinstance(row, LabSample):
            errors.add("sample_type")
            continue
        if (
            type(row.index) is not int
            or type(row.day) is not int
            or row.index != index
            or row.day != spec.start_day + index * spec.step_days
        ):
            errors.add("sample_time")
        if row.unit != spec.unit:
            errors.add("unit_mismatch")
        if index in spec.missing_indices:
            if row.value is not None:
                errors.add("missingness_mismatch")
            previous = None
            continue
        if row.value is None:
            errors.add("missingness_mismatch")
            previous = None
            continue
        try:
            value = _number(row.value)
        except ValueError:
            errors.add("invalid_value")
            previous = None
            continue
        if not spec.lower <= value <= spec.upper:
            errors.add("range_violation")
        if abs(value - spec.expected(index)) > spec.jitter + 1e-8:
            errors.add("trend_or_intervention_violation")
        if previous is not None and abs(value - previous) > spec.max_step_change + 1e-8:
            errors.add("change_limit_violation")
        previous = value
    return tuple(sorted(errors))


def lab_negative_control(
    spec: LabSeriesSpec, series: LabSeries, *, kind: str
) -> LabSeries:
    """Produce a named invalid fixture that the normal validation gate rejects."""
    if not isinstance(kind, str) or kind not in {
        "range",
        "unit",
        "time",
        "missingness",
    }:
        raise ValueError("unknown synthetic lab negative control")
    if validate_lab_series(spec, series):
        raise ValueError("negative control requires a valid synthetic series")
    rows = list(series.samples)
    index = next((i for i, r in enumerate(rows) if r.value is not None), None)
    if index is None:
        raise ValueError("negative control requires an observed sample")
    row = rows[index]
    if kind == "range":
        rows[index] = replace(row, value=spec.upper + max(1, abs(spec.upper)))
    elif kind == "unit":
        rows[index] = replace(row, unit="invalid_unit")
    elif kind == "time":
        rows[index] = replace(row, day=row.day + 1)
    else:
        rows[index] = replace(row, value=None)
    return replace(series, samples=tuple(rows), split="negative_control")
