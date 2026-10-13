# Synthetic lab time series

`openmed.training.synthetic.lab_timeseries` generates offline numerical fixtures
under caller-declared units, ranges, trends, variability, relative-day schedules,
missing indices and additive intervention effects. No clinical reference ranges,
physiological model, proprietary terminology or actual patient records are
bundled. These are evaluator fixtures, not a claim of physiological fidelity.

Create `LabSeriesSpec` with a controlled analyte code, unit, count, lower/upper
bounds and baseline. Optional `LabIntervention(index, effect, intervention_code)`
changes the expected trajectory from its sample index onward. Every planned
sample, including the full jitter envelope, must fit the declared range and step
change limit. Impossible recipes reject; values are never silently clamped.
Declared missing indices remain explicitly `None` while the underlying synthetic
trajectory continues. Relative days can be shifted without changing intervals.

`generate_lab_series(spec, seed=...)` uses a private seeded RNG, records the complete
recipe digest, and returns bounded immutable samples. It is deterministic within
the documented Python runtime; it makes no cross-runtime random-number guarantee.
`validate_lab_series` independently checks recipe binding, sample count/time,
units, exact missingness, ranges, trends, intervention effects and change limits.
Controlled error categories do not contain source values or private paths.

`lab_negative_control(..., kind="range" | "unit" | "time" | "missingness")` emits
one deliberate corruption with `negative_control` split. The normal validator
rejects that split and the underlying corruption independently. Keep those
fixtures outside normal synthetic training splits to test the evaluator.
Samples are limited to 10,000 and interventions to 1,000; no files or network
requests are created implicitly. Recipe constraints are declarations rather than
scientific evidence; clinical review is required for realistic parameter choices.
