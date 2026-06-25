"""Robustness evaluation for perturbed benchmark fixtures."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from openmed.eval.harness import (
    BenchmarkFixture,
    ModelRunner,
    load_fixtures,
    run_benchmark,
)
from openmed.eval.metrics import EvalSpan
from openmed.eval.report import BenchmarkReport

Edit = tuple[int, int, str]
FixturePerturber = Callable[[BenchmarkFixture, random.Random], BenchmarkFixture]


@dataclass(frozen=True)
class Perturbation:
    """Named, seedable fixture perturbation."""

    name: str
    apply: FixturePerturber

    def __call__(
        self,
        fixture: BenchmarkFixture,
        rng: random.Random,
    ) -> BenchmarkFixture:
        """Apply the perturbation to one benchmark fixture."""
        return self.apply(fixture, rng)


PerturbationLike = str | Perturbation


@dataclass(frozen=True)
class RobustnessVariant:
    """Benchmark report for one perturbation and its clean-run deltas."""

    name: str
    report: BenchmarkReport
    deltas: Mapping[str, float]
    seed: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready variant payload."""
        return {
            "deltas": dict(self.deltas),
            "report": self.report.to_dict(),
            "seed": self.seed,
        }


@dataclass(frozen=True)
class RobustnessReport:
    """Clean benchmark report plus per-perturbation benchmark deltas."""

    clean: BenchmarkReport
    variants: tuple[RobustnessVariant, ...]
    seed: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready robustness report payload."""
        return {
            "clean": self.clean.to_dict(),
            "seed": self.seed,
            "variants": {variant.name: variant.to_dict() for variant in self.variants},
        }

    def variant(self, name: str) -> RobustnessVariant:
        """Return the named perturbation variant."""
        for variant in self.variants:
            if variant.name == name:
                return variant
        raise KeyError(name)


def identity_perturbation(name: str = "identity") -> Perturbation:
    """Return a no-op perturbation for baseline delta checks."""
    return Perturbation(name=name, apply=lambda fixture, rng: fixture)


def character_typo_perturbation(
    *,
    probability: float = 0.08,
    name: str = "character_typo",
) -> Perturbation:
    """Return a deterministic character typo perturbation."""

    def apply(fixture: BenchmarkFixture, rng: random.Random) -> BenchmarkFixture:
        text = fixture.text
        positions = [index for index, char in enumerate(text) if char.isalnum()]
        edits = [
            (index, index + 1, _typo_replacement(text[index], rng))
            for index in _select_positions(positions, probability, rng)
        ]
        return _perturb_fixture(fixture, edits)

    return Perturbation(name=name, apply=apply)


def ocr_noise_perturbation(
    *,
    probability: float = 0.08,
    name: str = "ocr_noise",
) -> Perturbation:
    """Return a deterministic OCR-confusion perturbation."""

    def apply(fixture: BenchmarkFixture, rng: random.Random) -> BenchmarkFixture:
        candidates: list[Edit] = []
        text = fixture.text
        for index, char in enumerate(text):
            replacement = _OCR_CONFUSIONS.get(char)
            if replacement is not None:
                candidates.append((index, index + 1, replacement))
            if text.startswith("rn", index):
                candidates.append((index, index + 2, "m"))
            if char == "m":
                candidates.append((index, index + 1, "rn"))
        return _perturb_fixture(
            fixture,
            _select_non_overlapping_edits(candidates, probability, rng),
        )

    return Perturbation(name=name, apply=apply)


def case_flip_perturbation(
    *,
    probability: float = 0.08,
    name: str = "case_flip",
) -> Perturbation:
    """Return a deterministic casing perturbation."""

    def apply(fixture: BenchmarkFixture, rng: random.Random) -> BenchmarkFixture:
        text = fixture.text
        positions = [index for index, char in enumerate(text) if char.isalpha()]
        edits = [
            (index, index + 1, text[index].swapcase())
            for index in _select_positions(positions, probability, rng)
        ]
        return _perturb_fixture(fixture, edits)

    return Perturbation(name=name, apply=apply)


def whitespace_noise_perturbation(
    *,
    probability: float = 0.08,
    name: str = "whitespace_noise",
) -> Perturbation:
    """Return a deterministic whitespace-artifact perturbation."""

    def apply(fixture: BenchmarkFixture, rng: random.Random) -> BenchmarkFixture:
        text = fixture.text
        candidates = [
            (index, index + 1, _whitespace_replacement(text[index], rng))
            for index, char in enumerate(text)
            if char.isspace()
        ]
        if not candidates:
            candidates = [
                (index, index + 1, f"{char} ")
                for index, char in enumerate(text)
                if char.isalnum()
            ]
        return _perturb_fixture(
            fixture,
            _select_non_overlapping_edits(candidates, probability, rng),
        )

    return Perturbation(name=name, apply=apply)


DEFAULT_PERTURBATIONS: tuple[Perturbation, ...] = (
    character_typo_perturbation(),
    ocr_noise_perturbation(),
    case_flip_perturbation(),
    whitespace_noise_perturbation(),
)


def perturb_fixture(
    fixture: BenchmarkFixture,
    perturbation: str | Perturbation,
    *,
    seed: int = 0,
) -> BenchmarkFixture:
    """Perturb one fixture with deterministic span-offset re-projection."""
    resolved = _coerce_perturbation(perturbation)
    return resolved(fixture, random.Random(seed))


def robustness_report(
    model: str | ModelRunner,
    suite: str | Path | Iterable[BenchmarkFixture],
    perturbations: (
        PerturbationLike
        | Sequence[PerturbationLike]
        | Mapping[str, PerturbationLike]
        | None
    ) = None,
    seed: int = 0,
    *,
    suite_name: str | None = None,
    device: str = "cpu",
    runner: ModelRunner | None = None,
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    confidence_intervals: bool = False,
    ci_resamples: int = 1000,
    ci_alpha: float = 0.05,
    ci_seed: int = 0,
) -> RobustnessReport:
    """Score clean and perturbed suite variants with leakage/recall/F1 deltas.

    Args:
        model: Model name, or a benchmark runner callable. When a callable is
            supplied and ``runner`` is omitted, it is used as the runner.
        suite: Fixture path, named suite, or an iterable of benchmark fixtures.
        perturbations: Perturbation names or objects. ``None`` uses the default
            typo, OCR, casing, and whitespace perturbations.
        seed: Base seed used to make each perturbation reproducible.
        suite_name: Optional report suite name for iterable fixture inputs.
        device: Device label passed through to the benchmark harness.
        runner: Optional injected model runner.
        generated_at: Optional benchmark timestamp.
        metadata: Optional metadata merged into clean and perturbed reports.
        confidence_intervals: Whether to request harness bootstrap intervals.
        ci_resamples: Bootstrap resample count.
        ci_alpha: Bootstrap interval alpha.
        ci_seed: Bootstrap seed.

    Returns:
        A clean report plus one report per perturbation with metric deltas.
    """
    fixtures, resolved_suite_name = _resolve_suite(suite, suite_name=suite_name)
    model_name, model_runner = _resolve_model_runner(model, runner)
    base_metadata = dict(metadata or {})
    clean_report = run_benchmark(
        fixtures,
        suite=resolved_suite_name,
        model_name=model_name,
        device=device,
        runner=model_runner,
        generated_at=generated_at,
        metadata={
            **base_metadata,
            "robustness": {"role": "clean", "seed": seed},
        },
        confidence_intervals=confidence_intervals,
        ci_resamples=ci_resamples,
        ci_alpha=ci_alpha,
        ci_seed=ci_seed,
    )

    variants: list[RobustnessVariant] = []
    for index, perturbation in enumerate(_resolve_perturbations(perturbations)):
        variant_seed = seed + index
        rng = random.Random(variant_seed)
        perturbed_fixtures = [perturbation(fixture, rng) for fixture in fixtures]
        variant_report = run_benchmark(
            perturbed_fixtures,
            suite=resolved_suite_name,
            model_name=model_name,
            device=device,
            runner=model_runner,
            generated_at=generated_at,
            metadata={
                **base_metadata,
                "robustness": {
                    "base_suite": resolved_suite_name,
                    "perturbation": perturbation.name,
                    "role": "perturbed",
                    "seed": variant_seed,
                },
            },
            confidence_intervals=confidence_intervals,
            ci_resamples=ci_resamples,
            ci_alpha=ci_alpha,
            ci_seed=ci_seed,
        )
        variants.append(
            RobustnessVariant(
                name=perturbation.name,
                report=variant_report,
                deltas=_metric_deltas(clean_report.metrics, variant_report.metrics),
                seed=variant_seed,
            )
        )
    return RobustnessReport(
        clean=clean_report,
        variants=tuple(variants),
        seed=seed,
    )


_OCR_CONFUSIONS: Mapping[str, str] = {
    "0": "O",
    "1": "l",
    "5": "S",
    "8": "B",
    "B": "8",
    "I": "1",
    "O": "0",
    "S": "5",
    "l": "1",
    "o": "0",
}
_TYPO_FALLBACK = "abcdefghijklmnopqrstuvwxyz0123456789"


def _perturb_fixture(
    fixture: BenchmarkFixture,
    edits: Iterable[Edit],
) -> BenchmarkFixture:
    text, spans = _apply_edits(fixture.text, fixture.gold_spans, edits)
    return replace(fixture, text=text, gold_spans=spans)


def _apply_edits(
    text: str,
    spans: Sequence[EvalSpan],
    edits: Iterable[Edit],
) -> tuple[str, tuple[EvalSpan, ...]]:
    safe_edits = _safe_edits(text, spans, edits)
    replacements = list(text)
    for start, end, replacement in safe_edits:
        replacements[start] = replacement
        for index in range(start + 1, end):
            replacements[index] = ""

    offset_map = [0] * (len(text) + 1)
    output: list[str] = []
    position = 0
    offset_map[0] = 0
    for index, replacement in enumerate(replacements):
        output.append(replacement)
        position += len(replacement)
        offset_map[index + 1] = position

    perturbed_text = "".join(output)
    projected = tuple(_project_span(span, perturbed_text, offset_map) for span in spans)
    _validate_projected_spans(perturbed_text, projected)
    return perturbed_text, projected


def _safe_edits(
    text: str,
    spans: Sequence[EvalSpan],
    edits: Iterable[Edit],
) -> list[Edit]:
    boundaries = {
        boundary
        for span in spans
        for boundary in (span.start, span.end)
        if 0 <= boundary <= len(text)
    }
    safe: list[Edit] = []
    cursor = 0
    for start, end, replacement in sorted(edits, key=lambda item: (item[0], item[1])):
        if start < cursor or start < 0 or end <= start or end > len(text):
            continue
        if text[start:end] == replacement:
            continue
        if any(start < boundary < end for boundary in boundaries):
            continue
        safe.append((start, end, replacement))
        cursor = end
    return safe


def _project_span(
    span: EvalSpan,
    text: str,
    offset_map: Sequence[int],
) -> EvalSpan:
    start = offset_map[span.start]
    end = offset_map[span.end]
    return replace(span, start=start, end=end, text=text[start:end])


def _validate_projected_spans(text: str, spans: Sequence[EvalSpan]) -> None:
    for span in spans:
        if span.start < 0 or span.end < span.start or span.end > len(text):
            raise ValueError(f"perturbed gold span has invalid offsets: {span!r}")
        if span.text != text[span.start : span.end]:
            raise ValueError(f"perturbed gold span has drifted offsets: {span!r}")


def _select_positions(
    positions: Sequence[int],
    probability: float,
    rng: random.Random,
) -> list[int]:
    _validate_probability(probability)
    if not positions:
        return []
    selected = [position for position in positions if rng.random() < probability]
    if not selected and probability > 0.0:
        selected = [positions[rng.randrange(len(positions))]]
    return selected


def _select_non_overlapping_edits(
    candidates: Sequence[Edit],
    probability: float,
    rng: random.Random,
) -> list[Edit]:
    _validate_probability(probability)
    if not candidates:
        return []
    selected = [edit for edit in candidates if rng.random() < probability]
    if not selected and probability > 0.0:
        selected = [candidates[rng.randrange(len(candidates))]]

    edits: list[Edit] = []
    cursor = -1
    for start, end, replacement in sorted(
        selected, key=lambda item: (item[0], item[1])
    ):
        if start < cursor:
            continue
        edits.append((start, end, replacement))
        cursor = end
    return edits


def _validate_probability(probability: float) -> None:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("perturbation probability must be between 0 and 1")


def _typo_replacement(char: str, rng: random.Random) -> str:
    if char.isdigit():
        return str((int(char) + rng.randrange(1, 10)) % 10)
    candidates = [
        candidate for candidate in _TYPO_FALLBACK if candidate != char.lower()
    ]
    replacement = candidates[rng.randrange(len(candidates))]
    if char.isupper() and replacement.isalpha():
        return replacement.upper()
    return replacement


def _whitespace_replacement(char: str, rng: random.Random) -> str:
    choices = (char * 2, "\n", "\t", f"{char}\t")
    return choices[rng.randrange(len(choices))]


def _resolve_model_runner(
    model: str | ModelRunner,
    runner: ModelRunner | None,
) -> tuple[str, ModelRunner | None]:
    if runner is None and callable(model) and not isinstance(model, str):
        return getattr(model, "__name__", "model"), model
    return str(model), runner


def _resolve_suite(
    suite: str | Path | Iterable[BenchmarkFixture],
    *,
    suite_name: str | None,
) -> tuple[list[BenchmarkFixture], str]:
    if isinstance(suite, (str, Path)):
        path = Path(suite)
        if path.exists():
            return load_fixtures(path), suite_name or path.stem
        suite_key = str(suite)
        if suite_key == "golden":
            from openmed.eval.golden import load_benchmark_fixtures

            return load_benchmark_fixtures(), suite_name or suite_key
        from openmed.eval.suites import load_suite_fixtures

        return load_suite_fixtures(suite_key), suite_name or suite_key
    return list(suite), suite_name or "custom"


def _resolve_perturbations(
    perturbations: (
        PerturbationLike
        | Sequence[PerturbationLike]
        | Mapping[str, PerturbationLike]
        | None
    ),
) -> tuple[Perturbation, ...]:
    if perturbations is None:
        return DEFAULT_PERTURBATIONS
    if isinstance(perturbations, (str, Perturbation)):
        return (_coerce_perturbation(perturbations),)
    if isinstance(perturbations, Mapping):
        return tuple(
            _coerce_perturbation(perturbation, name=name)
            for name, perturbation in perturbations.items()
        )
    return tuple(_coerce_perturbation(perturbation) for perturbation in perturbations)


def _coerce_perturbation(
    perturbation: str | Perturbation,
    *,
    name: str | None = None,
) -> Perturbation:
    if isinstance(perturbation, Perturbation):
        if name is not None and name != perturbation.name:
            return Perturbation(name=name, apply=perturbation.apply)
        return perturbation
    key = perturbation.lower().replace("-", "_")
    aliases = {
        "case": "case_flip",
        "noop": "identity",
        "ocr": "ocr_noise",
        "typo": "character_typo",
        "whitespace": "whitespace_noise",
    }
    key = aliases.get(key, key)
    for candidate in (*DEFAULT_PERTURBATIONS, identity_perturbation()):
        if candidate.name == key:
            if name is not None and name != candidate.name:
                return Perturbation(name=name, apply=candidate.apply)
            return candidate
    raise ValueError(f"unknown robustness perturbation: {perturbation!r}")


def _metric_deltas(
    clean_metrics: Mapping[str, Any],
    perturbed_metrics: Mapping[str, Any],
) -> dict[str, float]:
    return {
        "f1": _metric_value(perturbed_metrics, "exact_span_f1", "f1")
        - _metric_value(clean_metrics, "exact_span_f1", "f1"),
        "leakage": _metric_value(perturbed_metrics, "leakage", "overall")
        - _metric_value(clean_metrics, "leakage", "overall"),
        "recall": _metric_value(perturbed_metrics, "character_recall", "rate")
        - _metric_value(clean_metrics, "character_recall", "rate"),
    }


def _metric_value(metrics: Mapping[str, Any], *path: str) -> float:
    value: Any = metrics
    for key in path:
        if not isinstance(value, Mapping):
            raise KeyError(".".join(path))
        value = value[key]
    return float(value)


__all__ = [
    "DEFAULT_PERTURBATIONS",
    "Perturbation",
    "RobustnessReport",
    "RobustnessVariant",
    "case_flip_perturbation",
    "character_typo_perturbation",
    "identity_perturbation",
    "ocr_noise_perturbation",
    "perturb_fixture",
    "robustness_report",
    "whitespace_noise_perturbation",
]
