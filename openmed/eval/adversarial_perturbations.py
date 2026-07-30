"""Adversarial Unicode-evasion robustness suite.

The typo/OCR/casing robustness suite (``openmed.eval.robustness``) covers
natural perturbations. This module adds deterministic *adversarial* Unicode
evasions an attacker uses to slip identifiers past a detector: homoglyph
substitution, zero-width/invisible character insertion, combining-mark
injection, and bidi-control wrapping.

Every operator is seedable and produces byte-identical output across runs and
processes. The suite generates perturbed variants of synthetic fixtures, scores
the PHI-leakage delta against the clean baseline per operator (and per label),
and enforces a fail-closed leakage-delta budget. It runs entirely offline over
in-memory fixtures with no external assets.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from openmed.eval.harness import BenchmarkFixture, ModelRunner, default_model_runner
from openmed.eval.metrics import EvalSpan, LeakageMetrics, compute_leakage_rate
from openmed.eval.robustness import (
    Edit,
    Perturbation,
    _perturb_fixture,
    _resolve_model_runner,
    _resolve_suite,
    _run_predictions,
    _select_positions,
)

FixturePerturber = Callable[[BenchmarkFixture, random.Random], BenchmarkFixture]
PerturbationLike = str | Perturbation

DEFAULT_ADVERSARIAL_PERTURBATION_PROBABILITY = 0.5
DEFAULT_LEAKAGE_DELTA_BUDGET = 0.0

# Curated Latin -> confusable maps built in-code (Greek/Cyrillic look-alikes).
_HOMOGLYPH_SUBSTITUTIONS: Mapping[str, str] = {
    "A": "\u0391",  # Greek capital alpha
    "B": "\u0392",  # Greek capital beta
    "C": "\u0421",  # Cyrillic capital es
    "E": "\u0395",  # Greek capital epsilon
    "H": "\u0397",  # Greek capital eta
    "I": "\u0399",  # Greek capital iota
    "K": "\u039a",  # Greek capital kappa
    "M": "\u039c",  # Greek capital mu
    "N": "\u039d",  # Greek capital nu
    "O": "\u039f",  # Greek capital omicron
    "P": "\u03a1",  # Greek capital rho
    "T": "\u03a4",  # Greek capital tau
    "X": "\u03a7",  # Greek capital chi
    "a": "\u0430",  # Cyrillic small a
    "c": "\u0441",  # Cyrillic small es
    "e": "\u0435",  # Cyrillic small ie
    "i": "\u0456",  # Cyrillic small dotless i
    "o": "\u03bf",  # Greek small omicron
    "p": "\u0440",  # Cyrillic small er
    "x": "\u0445",  # Cyrillic small ha
}
# Zero-width / invisible code points inserted between visible characters.
_ZERO_WIDTH_CHARS: tuple[str, ...] = (
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\u2060",  # word joiner
    "\ufeff",  # zero-width no-break space
)
# Combining marks appended to a base character to split its cluster.
_COMBINING_MARKS: tuple[str, ...] = (
    "\u0301",  # combining acute accent
    "\u0323",  # combining dot below
    "\u0308",  # combining diaeresis
    "\u0331",  # combining macron below
)
# Bidi-control (prefix, suffix) wrappers applied around a span.
_BIDI_WRAPPERS: tuple[tuple[str, str], ...] = (
    ("\u202e", "\u202c"),  # RLO ... PDF
    ("\u202d", "\u202c"),  # LRO ... PDF
    ("\u2067", "\u2069"),  # RLI ... PDI
    ("\u2066", "\u2069"),  # LRI ... PDI
)


@dataclass(frozen=True)
class AdversarialPerturbationVariant:
    """Leakage and leakage-delta record for one adversarial operator."""

    name: str
    leakage: LeakageMetrics
    delta_overall: float
    delta_by_label: Mapping[str, float]
    flagged: bool
    flagged_labels: tuple[str, ...]
    seed: int
    perturbed_documents: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready variant payload keyed by operator and label."""
        return {
            "delta_by_label": dict(sorted(self.delta_by_label.items())),
            "delta_overall": self.delta_overall,
            "flagged": self.flagged,
            "flagged_labels": list(self.flagged_labels),
            "leakage": self.leakage.to_dict(),
            "perturbed_documents": self.perturbed_documents,
            "seed": self.seed,
        }


class AdversarialPerturbationGateError(RuntimeError):
    """Raised when an operator raises leakage above the configured budget."""


@dataclass(frozen=True)
class AdversarialPerturbationReport:
    """Clean-vs-perturbed leakage report with a fail-closed delta gate."""

    model_name: str
    suite: str
    device: str
    seed: int
    leakage_delta_budget: float
    clean: LeakageMetrics
    variants: tuple[AdversarialPerturbationVariant, ...]
    violations: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Return whether every operator stayed within the leakage budget."""
        return not self.violations

    def variant(self, name: str) -> AdversarialPerturbationVariant:
        """Return the named operator variant."""
        for variant in self.variants:
            if variant.name == name:
                return variant
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready robustness-degradation report payload."""
        return {
            "clean": self.clean.to_dict(),
            "device": self.device,
            "leakage_delta_budget": self.leakage_delta_budget,
            "model_name": self.model_name,
            "passed": self.passed,
            "seed": self.seed,
            "suite": self.suite,
            "variants": {variant.name: variant.to_dict() for variant in self.variants},
            "violations": list(self.violations),
        }


def homoglyph_substitution_perturbation(
    *,
    probability: float = DEFAULT_ADVERSARIAL_PERTURBATION_PROBABILITY,
    name: str = "homoglyph_substitution",
) -> Perturbation:
    """Return a deterministic Latin-to-confusable homoglyph perturbation."""

    def apply(fixture: BenchmarkFixture, rng: random.Random) -> BenchmarkFixture:
        text = fixture.text
        edits: list[Edit] = []
        for span in fixture.gold_spans:
            positions = [
                index
                for index in range(span.start, span.end)
                if text[index] in _HOMOGLYPH_SUBSTITUTIONS
            ]
            edits.extend(
                (index, index + 1, _HOMOGLYPH_SUBSTITUTIONS[text[index]])
                for index in _select_positions(positions, probability, rng)
            )
        return _perturb_fixture(fixture, edits)

    return Perturbation(name=name, apply=apply)


def zero_width_injection_perturbation(
    *,
    probability: float = DEFAULT_ADVERSARIAL_PERTURBATION_PROBABILITY,
    name: str = "zero_width_injection",
) -> Perturbation:
    """Return a deterministic zero-width/invisible-character perturbation."""

    def apply(fixture: BenchmarkFixture, rng: random.Random) -> BenchmarkFixture:
        text = fixture.text
        edits: list[Edit] = []
        for span in fixture.gold_spans:
            positions = [
                index
                for index in range(span.start, span.end - 1)
                if text[index].isalnum() and text[index + 1].isalnum()
            ]
            for index in _select_positions(positions, probability, rng):
                invisible = _ZERO_WIDTH_CHARS[rng.randrange(len(_ZERO_WIDTH_CHARS))]
                edits.append((index, index + 1, f"{text[index]}{invisible}"))
        return _perturb_fixture(fixture, edits)

    return Perturbation(name=name, apply=apply)


def combining_mark_injection_perturbation(
    *,
    probability: float = DEFAULT_ADVERSARIAL_PERTURBATION_PROBABILITY,
    name: str = "combining_mark_injection",
) -> Perturbation:
    """Return a deterministic combining-mark splitting perturbation."""

    def apply(fixture: BenchmarkFixture, rng: random.Random) -> BenchmarkFixture:
        text = fixture.text
        edits: list[Edit] = []
        for span in fixture.gold_spans:
            positions = [
                index for index in range(span.start, span.end) if text[index].isalpha()
            ]
            for index in _select_positions(positions, probability, rng):
                mark = _COMBINING_MARKS[rng.randrange(len(_COMBINING_MARKS))]
                edits.append((index, index + 1, f"{text[index]}{mark}"))
        return _perturb_fixture(fixture, edits)

    return Perturbation(name=name, apply=apply)


def bidi_control_wrapping_perturbation(
    *,
    name: str = "bidi_control_wrapping",
) -> Perturbation:
    """Return a deterministic bidi-control wrapping perturbation.

    Each gold span is wrapped in a seed-chosen bidirectional override pair
    (for example RLO ... PDF), a byte-level evasion that leaves the rendered
    glyphs unchanged while corrupting the underlying identifier.
    """

    def apply(fixture: BenchmarkFixture, rng: random.Random) -> BenchmarkFixture:
        text = fixture.text
        edits: list[Edit] = []
        for span in fixture.gold_spans:
            if span.end <= span.start:
                continue
            prefix, suffix = _BIDI_WRAPPERS[rng.randrange(len(_BIDI_WRAPPERS))]
            first = span.start
            last = span.end - 1
            if first == last:
                edits.append((first, first + 1, f"{prefix}{text[first]}{suffix}"))
                continue
            edits.append((first, first + 1, f"{prefix}{text[first]}"))
            edits.append((last, last + 1, f"{text[last]}{suffix}"))
        return _perturb_fixture(fixture, edits)

    return Perturbation(name=name, apply=apply)


DEFAULT_ADVERSARIAL_PERTURBATIONS: tuple[Perturbation, ...] = (
    homoglyph_substitution_perturbation(),
    zero_width_injection_perturbation(),
    combining_mark_injection_perturbation(),
    bidi_control_wrapping_perturbation(),
)


def adversarial_perturbation_report(
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
    leakage_delta_budget: float = DEFAULT_LEAKAGE_DELTA_BUDGET,
    raise_on_violation: bool = False,
) -> AdversarialPerturbationReport:
    """Score per-operator PHI-leakage deltas under adversarial Unicode evasions.

    Args:
        model: Model name, or a benchmark runner callable. When a callable is
            supplied and ``runner`` is omitted, it is used as the runner.
        suite: Fixture path, named suite, or an iterable of benchmark fixtures.
        perturbations: Operator names or objects. ``None`` uses the default
            homoglyph, zero-width, combining-mark, and bidi-control operators.
        seed: Base seed making each operator reproducible.
        suite_name: Optional report suite name for iterable fixture inputs.
        device: Device label passed through to leakage scoring.
        runner: Optional injected model runner.
        leakage_delta_budget: Maximum tolerated increase in overall leakage per
            operator. An operator whose leakage delta exceeds this budget is
            flagged and recorded as a gate violation (fail-closed at ``0.0``).
        raise_on_violation: When true, raise instead of returning a failing
            report.

    Returns:
        A clean-baseline leakage report plus one variant per operator with the
        leakage delta keyed by operator and label and a fail-closed gate.
    """
    if leakage_delta_budget < 0.0:
        raise ValueError("leakage_delta_budget must be non-negative")

    fixtures, resolved_suite_name = _resolve_suite(suite, suite_name=suite_name)
    model_name, model_runner = _resolve_model_runner(model, runner)
    base_runner = model_runner or default_model_runner

    clean = _suite_leakage(fixtures, base_runner, model_name, device)

    variants: list[AdversarialPerturbationVariant] = []
    violations: list[str] = []
    for index, perturbation in enumerate(_resolve_perturbations(perturbations)):
        variant_seed = seed + index
        rng = random.Random(variant_seed)
        perturbed = [perturbation(fixture, rng) for fixture in fixtures]
        perturbed_documents = sum(
            1
            for original, variant in zip(fixtures, perturbed)
            if variant.text != original.text
        )
        leakage = _suite_leakage(perturbed, base_runner, model_name, device)
        delta_overall = leakage.overall - clean.overall
        delta_by_label = _leakage_delta_by_label(clean, leakage)
        flagged_labels = tuple(
            label
            for label, delta in sorted(delta_by_label.items())
            if delta > leakage_delta_budget
        )
        flagged = delta_overall > leakage_delta_budget
        if flagged:
            violations.append(perturbation.name)
        variants.append(
            AdversarialPerturbationVariant(
                name=perturbation.name,
                leakage=leakage,
                delta_overall=delta_overall,
                delta_by_label=delta_by_label,
                flagged=flagged,
                flagged_labels=flagged_labels,
                seed=variant_seed,
                perturbed_documents=perturbed_documents,
            )
        )

    report = AdversarialPerturbationReport(
        model_name=model_name,
        suite=resolved_suite_name,
        device=device,
        seed=seed,
        leakage_delta_budget=leakage_delta_budget,
        clean=clean,
        variants=tuple(variants),
        violations=tuple(violations),
    )
    if raise_on_violation and not report.passed:
        joined = ", ".join(report.violations)
        raise AdversarialPerturbationGateError(
            f"adversarial-perturbation leakage gate failed: {joined}"
        )
    return report


def _suite_leakage(
    fixtures: Sequence[BenchmarkFixture],
    runner: ModelRunner,
    model_name: str,
    device: str,
) -> LeakageMetrics:
    """Aggregate gold/predicted spans across a suite and score leakage."""
    gold: list[EvalSpan] = []
    predicted: list[EvalSpan] = []
    document_offset = 0
    for fixture in fixtures:
        for span in fixture.gold_spans:
            gold.append(
                replace(
                    span,
                    start=document_offset + span.start,
                    end=document_offset + span.end,
                )
            )
        for prediction in _run_predictions(fixture, runner, model_name, device):
            predicted.append(
                replace(
                    prediction,
                    start=document_offset + prediction.start,
                    end=document_offset + prediction.end,
                )
            )
        document_offset += len(fixture.text) + 1
    return compute_leakage_rate(gold, predicted, default_device=device)


def _leakage_delta_by_label(
    clean: LeakageMetrics,
    perturbed: LeakageMetrics,
) -> dict[str, float]:
    """Return per-label leakage deltas over labels observed in either run.

    ``LeakageMetrics.by_label`` enumerates the full canonical label set with a
    zero rate for absent labels; the report is keyed only on labels that carry
    gold PHI graphemes in the clean or perturbed suite.
    """
    labels = {
        label
        for counts in (clean.total_chars_by_label, perturbed.total_chars_by_label)
        for label, total in counts.items()
        if total > 0
    }
    return {
        label: perturbed.by_label.get(label, 0.0) - clean.by_label.get(label, 0.0)
        for label in sorted(labels)
    }


def _resolve_perturbations(
    perturbations: (
        PerturbationLike
        | Sequence[PerturbationLike]
        | Mapping[str, PerturbationLike]
        | None
    ),
) -> tuple[Perturbation, ...]:
    if perturbations is None:
        return DEFAULT_ADVERSARIAL_PERTURBATIONS
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
        "bidi": "bidi_control_wrapping",
        "combining": "combining_mark_injection",
        "combining_mark": "combining_mark_injection",
        "homoglyph": "homoglyph_substitution",
        "zero_width": "zero_width_injection",
        "zerowidth": "zero_width_injection",
    }
    key = aliases.get(key, key)
    for candidate in DEFAULT_ADVERSARIAL_PERTURBATIONS:
        if candidate.name == key:
            if name is not None and name != candidate.name:
                return Perturbation(name=name, apply=candidate.apply)
            return candidate
    raise ValueError(f"unknown adversarial perturbation: {perturbation!r}")


__all__ = [
    "DEFAULT_ADVERSARIAL_PERTURBATIONS",
    "DEFAULT_ADVERSARIAL_PERTURBATION_PROBABILITY",
    "DEFAULT_LEAKAGE_DELTA_BUDGET",
    "AdversarialPerturbationGateError",
    "AdversarialPerturbationReport",
    "AdversarialPerturbationVariant",
    "adversarial_perturbation_report",
    "bidi_control_wrapping_perturbation",
    "combining_mark_injection_perturbation",
    "homoglyph_substitution_perturbation",
    "zero_width_injection_perturbation",
]
