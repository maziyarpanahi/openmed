"""Robustness evaluation for perturbed benchmark fixtures."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from openmed.core.labels import normalize_label
from openmed.core.script_detect import normalize_for_pii_detection
from openmed.eval.harness import (
    BenchmarkFixture,
    ModelRunner,
    default_model_runner,
    load_fixtures,
    run_benchmark,
)
from openmed.eval.metrics import EvalSpan, normalize_eval_spans
from openmed.eval.report import BenchmarkReport

Edit = tuple[int, int, str]
FixturePerturber = Callable[[BenchmarkFixture, random.Random], BenchmarkFixture]

DIRECT_IDENTIFIER_LABELS = frozenset(
    {
        "ACCOUNT_NUMBER",
        "AGE",
        "API_KEY",
        "BUILDING_NUMBER",
        "CREDIT_CARD",
        "DATE",
        "DATE_OF_BIRTH",
        "EMAIL",
        "FIRST_NAME",
        "GPS_COORDINATES",
        "IBAN",
        "ID_NUM",
        "LAST_NAME",
        "LOCATION",
        "MIDDLE_NAME",
        "PERSON",
        "PHONE",
        "SSN",
        "STREET_ADDRESS",
        "TIME",
        "URL",
        "USERNAME",
        "ZIPCODE",
    }
)
DEFAULT_ATTACK_CLASSES: tuple[str, ...] = (
    "homoglyph",
    "zero_width",
    "combining_mark",
    "segmentation",
    "targeted_typo",
)
DEFAULT_ATTACK_DISTANCE_BUDGET = 2
DEFAULT_ATTACK_BEAM_WIDTH = 4
DEFAULT_ADVERSARIAL_RECALL_FLOOR = 0.99


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


@dataclass(frozen=True)
class AdversarialAttackArtifact:
    """PHI-free reproduction record for one worst-case span attack."""

    fixture_id: str
    span_index: int
    label: str
    original_start: int
    original_end: int
    attacked_start: int
    attacked_end: int
    perturbation_classes: tuple[str, ...]
    distance: int
    distance_budget: int
    seed: int
    beam_width: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe artifact without raw PHI surfaces."""
        return {
            "attacked_end": self.attacked_end,
            "attacked_start": self.attacked_start,
            "beam_width": self.beam_width,
            "distance": self.distance,
            "distance_budget": self.distance_budget,
            "fixture_id": self.fixture_id,
            "label": self.label,
            "original_end": self.original_end,
            "original_start": self.original_start,
            "perturbation_classes": list(self.perturbation_classes),
            "seed": self.seed,
            "span_index": self.span_index,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AdversarialAttackArtifact":
        """Build an artifact from its serialized representation."""
        return cls(
            fixture_id=str(value["fixture_id"]),
            span_index=int(value["span_index"]),
            label=normalize_label(str(value["label"])),
            original_start=int(value["original_start"]),
            original_end=int(value["original_end"]),
            attacked_start=int(value["attacked_start"]),
            attacked_end=int(value["attacked_end"]),
            perturbation_classes=tuple(
                str(item) for item in value.get("perturbation_classes", [])
            ),
            distance=int(value["distance"]),
            distance_budget=int(value["distance_budget"]),
            seed=int(value["seed"]),
            beam_width=int(value.get("beam_width", DEFAULT_ATTACK_BEAM_WIDTH)),
        )


@dataclass(frozen=True)
class AdversarialRobustnessReport:
    """Worst-case adversarial de-identification robustness result."""

    model_name: str
    suite: str
    device: str
    seed: int
    recall_floor: float
    distance_budget: int
    beam_width: int
    artifacts: tuple[AdversarialAttackArtifact, ...]
    metrics: Mapping[str, Any]
    generated_at: str | None = None
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready report with sanitized attack artifacts."""
        return {
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "device": self.device,
            "distance_budget": self.distance_budget,
            "fixture_count": len(self.artifacts),
            "generated_at": self.generated_at,
            "metadata": dict(self.metadata or {}),
            "metrics": dict(self.metrics),
            "model_name": self.model_name,
            "recall_floor": self.recall_floor,
            "seed": self.seed,
            "suite": self.suite,
        }


@dataclass(frozen=True)
class _AttackOperation:
    perturbation_class: str
    start: int
    end: int
    replacement: str
    cost: int = 1

    @property
    def edit(self) -> Edit:
        return (self.start, self.end, self.replacement)


@dataclass(frozen=True)
class _AttackState:
    fixture: BenchmarkFixture
    distance: int
    perturbation_classes: tuple[str, ...]
    miss_score: float


@dataclass(frozen=True)
class _AdversarialCase:
    artifact: AdversarialAttackArtifact
    fixture: BenchmarkFixture
    pre_defense_miss_score: float


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


def unicode_defended_runner(runner: ModelRunner | None = None) -> ModelRunner:
    """Wrap a runner with the Unicode defense used before PII detection."""
    base_runner = runner or default_model_runner

    def run(fixture: BenchmarkFixture, model_name: str, device: str) -> Iterable[Any]:
        normalization = normalize_for_pii_detection(fixture.text)
        if not normalization.changed and not normalization.mixed_script:
            return base_runner(fixture, model_name, device)

        normalized_fixture = replace(fixture, text=normalization.text)
        raw_predictions = base_runner(normalized_fixture, model_name, device)
        predictions = normalize_eval_spans(
            raw_predictions,
            default_language=fixture.language,
            default_device=device,
            source_text=normalization.text,
        )
        remapped = []
        for prediction in predictions:
            start, end = normalization.remap_span(prediction.start, prediction.end)
            remapped.append(
                replace(
                    prediction,
                    start=start,
                    end=end,
                    text=fixture.text[start:end],
                    metadata={
                        **dict(prediction.metadata),
                        "unicode_defense": normalization.to_metadata(),
                    },
                )
            )
        return remapped

    return run


def adversarial_robustness_report(
    model: str | ModelRunner,
    suite: str | Path | Iterable[BenchmarkFixture],
    *,
    seed: int = 0,
    distance_budget: int = DEFAULT_ATTACK_DISTANCE_BUDGET,
    beam_width: int = DEFAULT_ATTACK_BEAM_WIDTH,
    attack_classes: Sequence[str] = DEFAULT_ATTACK_CLASSES,
    labels: Iterable[str] | None = DIRECT_IDENTIFIER_LABELS,
    recall_floor: float = DEFAULT_ADVERSARIAL_RECALL_FLOOR,
    suite_name: str | None = None,
    device: str = "cpu",
    runner: ModelRunner | None = None,
    defended_runner: ModelRunner | None = None,
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> AdversarialRobustnessReport:
    """Search for worst-case PHI misses and score recall under attack.

    The serialized report stores only PHI-free attack artifacts: fixture ids,
    offsets, labels, perturbation classes, distance, and seeds.
    """
    if distance_budget < 1:
        raise ValueError("distance_budget must be at least 1")
    if beam_width < 1:
        raise ValueError("beam_width must be at least 1")

    fixtures, resolved_suite_name = _resolve_suite(suite, suite_name=suite_name)
    model_name, model_runner = _resolve_model_runner(model, runner)
    base_runner = model_runner or default_model_runner
    defense_runner = defended_runner or unicode_defended_runner(base_runner)
    label_filter = (
        None
        if labels is None
        else frozenset(normalize_label(str(label)) for label in labels)
    )

    best_by_label: dict[str, _AdversarialCase] = {}
    for fixture_index, fixture in enumerate(fixtures):
        for span_index, span in enumerate(fixture.gold_spans):
            label = normalize_label(span.label, lang=span.language)
            if label_filter is not None and label not in label_filter:
                continue
            span_seed = seed + (fixture_index * 1009) + (span_index * 9176)
            state = _search_span_attack(
                fixture,
                span_index,
                model_name=model_name,
                device=device,
                runner=base_runner,
                seed=span_seed,
                distance_budget=distance_budget,
                beam_width=beam_width,
                attack_classes=tuple(attack_classes),
            )
            if state.distance == 0:
                continue
            artifact = _artifact_for_state(
                original=fixture,
                state=state,
                span_index=span_index,
                seed=span_seed,
                distance_budget=distance_budget,
                beam_width=beam_width,
            )
            case = _AdversarialCase(
                artifact=artifact,
                fixture=state.fixture,
                pre_defense_miss_score=state.miss_score,
            )
            current = best_by_label.get(artifact.label)
            if current is None or _case_rank(case) > _case_rank(current):
                best_by_label[artifact.label] = case

    cases = tuple(best_by_label[label] for label in sorted(best_by_label))
    metrics = _adversarial_metrics(
        cases,
        model_name=model_name,
        device=device,
        pre_defense_runner=base_runner,
        post_defense_runner=defense_runner,
        recall_floor=recall_floor,
        distance_budget=distance_budget,
    )
    return AdversarialRobustnessReport(
        model_name=model_name,
        suite=resolved_suite_name,
        device=device,
        seed=seed,
        recall_floor=recall_floor,
        distance_budget=distance_budget,
        beam_width=beam_width,
        artifacts=tuple(case.artifact for case in cases),
        metrics={"adversarial_robustness": metrics},
        generated_at=generated_at,
        metadata=metadata,
    )


def replay_adversarial_attack(
    fixture: BenchmarkFixture,
    artifact: AdversarialAttackArtifact | Mapping[str, Any],
    model: str | ModelRunner,
    *,
    device: str = "cpu",
    runner: ModelRunner | None = None,
    attack_classes: Sequence[str] = DEFAULT_ATTACK_CLASSES,
) -> BenchmarkFixture:
    """Replay a sanitized attack artifact from its seed and source fixture."""
    resolved_artifact = (
        artifact
        if isinstance(artifact, AdversarialAttackArtifact)
        else AdversarialAttackArtifact.from_mapping(artifact)
    )
    model_name, model_runner = _resolve_model_runner(model, runner)
    state = _search_span_attack(
        fixture,
        resolved_artifact.span_index,
        model_name=model_name,
        device=device,
        runner=model_runner or default_model_runner,
        seed=resolved_artifact.seed,
        distance_budget=resolved_artifact.distance_budget,
        beam_width=resolved_artifact.beam_width,
        attack_classes=tuple(attack_classes),
    )
    return state.fixture


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


_HOMOGLYPH_ATTACKS: Mapping[str, str] = {
    "A": "\u0391",
    "B": "\u0392",
    "C": "\u0421",
    "E": "\u0395",
    "H": "\u0397",
    "I": "\u0399",
    "K": "\u039a",
    "M": "\u039c",
    "N": "\u039d",
    "O": "\u039f",
    "P": "\u03a1",
    "T": "\u03a4",
    "X": "\u03a7",
    "a": "\u0430",
    "c": "\u0441",
    "e": "\u0435",
    "i": "\u0456",
    "o": "\u03bf",
    "p": "\u0440",
    "x": "\u0445",
}


def _search_span_attack(
    fixture: BenchmarkFixture,
    span_index: int,
    *,
    model_name: str,
    device: str,
    runner: ModelRunner,
    seed: int,
    distance_budget: int,
    beam_width: int,
    attack_classes: tuple[str, ...],
) -> _AttackState:
    base_score = _target_miss_score(fixture, span_index, runner, model_name, device)
    best = _AttackState(
        fixture=fixture,
        distance=0,
        perturbation_classes=(),
        miss_score=base_score,
    )
    beam = (best,)

    for depth in range(distance_budget):
        expanded: list[_AttackState] = []
        for state_index, state in enumerate(beam):
            rng = random.Random(seed + depth * 104729 + state_index * 15485863)
            operations = _candidate_attack_operations(
                state.fixture,
                span_index,
                rng=rng,
                attack_classes=attack_classes,
            )
            for operation in operations:
                distance = state.distance + operation.cost
                if distance > distance_budget:
                    continue
                attacked = _perturb_fixture(state.fixture, [operation.edit])
                if attacked.text == state.fixture.text:
                    continue
                score = _target_miss_score(
                    attacked,
                    span_index,
                    runner,
                    model_name,
                    device,
                )
                expanded.append(
                    _AttackState(
                        fixture=attacked,
                        distance=distance,
                        perturbation_classes=(
                            *state.perturbation_classes,
                            operation.perturbation_class,
                        ),
                        miss_score=score,
                    )
                )

        if not expanded:
            break
        expanded.sort(key=_state_rank, reverse=True)
        if _state_rank(expanded[0]) > _state_rank(best):
            best = expanded[0]
        beam = tuple(expanded[:beam_width])

    return best


def _candidate_attack_operations(
    fixture: BenchmarkFixture,
    span_index: int,
    *,
    rng: random.Random,
    attack_classes: tuple[str, ...],
) -> list[_AttackOperation]:
    span = fixture.gold_spans[span_index]
    text = fixture.text
    operations: list[_AttackOperation] = []
    enabled = set(attack_classes)
    for index in range(span.start, span.end):
        char = text[index]
        if "homoglyph" in enabled and char in _HOMOGLYPH_ATTACKS:
            operations.append(
                _AttackOperation(
                    "homoglyph", index, index + 1, _HOMOGLYPH_ATTACKS[char]
                )
            )
        if "zero_width" in enabled and char.isalnum():
            operations.append(
                _AttackOperation("zero_width", index, index + 1, f"{char}\u200b")
            )
        if "combining_mark" in enabled and char.isalpha():
            operations.append(
                _AttackOperation("combining_mark", index, index + 1, f"{char}\u0301")
            )
        if (
            "segmentation" in enabled
            and char.isalnum()
            and index + 1 < span.end
            and text[index + 1].isalnum()
        ):
            operations.append(
                _AttackOperation("segmentation", index, index + 1, f"{char}-")
            )
        if "targeted_typo" in enabled and char.isalnum():
            operations.append(
                _AttackOperation(
                    "targeted_typo",
                    index,
                    index + 1,
                    _typo_replacement(char, rng),
                )
            )

    keyed = sorted(
        enumerate(operations),
        key=lambda item: (
            _attack_class_priority(item[1].perturbation_class),
            item[1].start,
            item[0],
        ),
    )
    grouped: list[_AttackOperation] = [operation for _, operation in keyed]
    # Keep deterministic tie-breaking but vary same-class candidates by seed.
    for start in range(0, len(grouped), 8):
        block = grouped[start : start + 8]
        rng.shuffle(block)
        grouped[start : start + 8] = block
    return grouped


def _artifact_for_state(
    *,
    original: BenchmarkFixture,
    state: _AttackState,
    span_index: int,
    seed: int,
    distance_budget: int,
    beam_width: int,
) -> AdversarialAttackArtifact:
    original_span = original.gold_spans[span_index]
    attacked_span = state.fixture.gold_spans[span_index]
    return AdversarialAttackArtifact(
        fixture_id=original.fixture_id,
        span_index=span_index,
        label=normalize_label(original_span.label, lang=original_span.language),
        original_start=original_span.start,
        original_end=original_span.end,
        attacked_start=attacked_span.start,
        attacked_end=attacked_span.end,
        perturbation_classes=tuple(state.perturbation_classes),
        distance=state.distance,
        distance_budget=distance_budget,
        seed=seed,
        beam_width=beam_width,
    )


def _adversarial_metrics(
    cases: Sequence[_AdversarialCase],
    *,
    model_name: str,
    device: str,
    pre_defense_runner: ModelRunner,
    post_defense_runner: ModelRunner,
    recall_floor: float,
    distance_budget: int,
) -> dict[str, Any]:
    pre = _score_cases(cases, model_name, device, pre_defense_runner)
    post = _score_cases(cases, model_name, device, post_defense_runner)
    post_violations = {
        label: recall
        for label, recall in post["recall_under_attack_by_label"].items()
        if label in DIRECT_IDENTIFIER_LABELS and recall < recall_floor
    }
    return {
        "artifact_count": len(cases),
        "distance_budget": distance_budget,
        "post_defense_leaked_chars": post["leaked_chars"],
        "post_defense_leaked_chars_by_label": post["leaked_chars_by_label"],
        "post_defense_miss_count": post["miss_count"],
        "post_defense_recall_under_attack": post["recall_under_attack"],
        "post_defense_recall_under_attack_by_label": post[
            "recall_under_attack_by_label"
        ],
        "pre_defense_miss_count": pre["miss_count"],
        "pre_defense_recall_under_attack": pre["recall_under_attack"],
        "pre_defense_recall_under_attack_by_label": pre["recall_under_attack_by_label"],
        "recall_floor": recall_floor,
        "violations": post_violations,
    }


def _score_cases(
    cases: Sequence[_AdversarialCase],
    model_name: str,
    device: str,
    runner: ModelRunner,
) -> dict[str, Any]:
    covered_by_label: dict[str, int] = {}
    total_by_label: dict[str, int] = {}
    leaked_by_label: dict[str, int] = {}
    miss_count = 0
    for case in cases:
        span = case.fixture.gold_spans[case.artifact.span_index]
        predictions = _run_predictions(case.fixture, runner, model_name, device)
        covered = _covered_char_count(span, predictions)
        leaked = max(span.length - covered, 0)
        label = normalize_label(span.label, lang=span.language)
        covered_by_label[label] = covered_by_label.get(label, 0) + covered
        total_by_label[label] = total_by_label.get(label, 0) + span.length
        leaked_by_label[label] = leaked_by_label.get(label, 0) + leaked
        if leaked > 0:
            miss_count += 1

    total = sum(total_by_label.values())
    covered_total = sum(covered_by_label.values())
    leaked_total = sum(leaked_by_label.values())
    return {
        "leaked_chars": leaked_total,
        "leaked_chars_by_label": dict(sorted(leaked_by_label.items())),
        "miss_count": miss_count,
        "recall_under_attack": _safe_rate(covered_total, total, 1.0),
        "recall_under_attack_by_label": {
            label: _safe_rate(covered_by_label.get(label, 0), total_chars, 1.0)
            for label, total_chars in sorted(total_by_label.items())
        },
        "total_chars": total,
        "total_chars_by_label": dict(sorted(total_by_label.items())),
    }


def _target_miss_score(
    fixture: BenchmarkFixture,
    span_index: int,
    runner: ModelRunner,
    model_name: str,
    device: str,
) -> float:
    span = fixture.gold_spans[span_index]
    predictions = _run_predictions(fixture, runner, model_name, device)
    covered = _covered_char_count(span, predictions)
    return 1.0 - _safe_rate(covered, span.length, 1.0)


def _run_predictions(
    fixture: BenchmarkFixture,
    runner: ModelRunner,
    model_name: str,
    device: str,
) -> tuple[EvalSpan, ...]:
    return tuple(
        normalize_eval_spans(
            runner(fixture, model_name, device),
            default_language=fixture.language,
            default_device=device,
            source_text=fixture.text,
        )
    )


def _covered_char_count(span: EvalSpan, predictions: Sequence[EvalSpan]) -> int:
    covered: set[int] = set()
    for prediction in predictions:
        if normalize_label(
            prediction.label, lang=prediction.language
        ) != normalize_label(
            span.label,
            lang=span.language,
        ):
            continue
        start = max(span.start, prediction.start)
        end = min(span.end, prediction.end)
        if end > start:
            covered.update(range(start, end))
    return len(covered)


def _safe_rate(
    numerator: int | float, denominator: int | float, default: float
) -> float:
    return default if denominator == 0 else float(numerator) / float(denominator)


def _state_rank(state: _AttackState) -> tuple[float, int, int, tuple[int, ...]]:
    return (
        state.miss_score,
        -state.distance,
        -len(state.perturbation_classes),
        tuple(-_attack_class_priority(item) for item in state.perturbation_classes),
    )


def _case_rank(case: _AdversarialCase) -> tuple[float, int, str]:
    return (
        case.pre_defense_miss_score,
        -case.artifact.distance,
        case.artifact.label,
    )


def _attack_class_priority(name: str) -> int:
    priority = {attack: index for index, attack in enumerate(DEFAULT_ATTACK_CLASSES)}
    return priority.get(name, len(priority))


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
    "DEFAULT_ADVERSARIAL_RECALL_FLOOR",
    "DEFAULT_ATTACK_BEAM_WIDTH",
    "DEFAULT_ATTACK_CLASSES",
    "DEFAULT_ATTACK_DISTANCE_BUDGET",
    "DEFAULT_PERTURBATIONS",
    "DIRECT_IDENTIFIER_LABELS",
    "AdversarialAttackArtifact",
    "AdversarialRobustnessReport",
    "Perturbation",
    "RobustnessReport",
    "RobustnessVariant",
    "adversarial_robustness_report",
    "case_flip_perturbation",
    "character_typo_perturbation",
    "identity_perturbation",
    "ocr_noise_perturbation",
    "perturb_fixture",
    "replay_adversarial_attack",
    "robustness_report",
    "unicode_defended_runner",
    "whitespace_noise_perturbation",
]
