"""Multilingual ConText coverage reporting for release and eval surfaces.

The deterministic ConText eval (``run_context_multilingual_eval``) scores every
language that has fixture rows. That leaves one failure mode unguarded: a cue
pack can ship in the lexicon module, advertise healthy cue counts through
``clinical_context_lexicon_stats``, and never be exercised by a single fixture
row. The eval's ``context_gate_passed`` flag is computed over the scored
languages only, so it stays green in that situation, and the macro-F1 helper
returns ``1.0`` for an empty label sequence -- a perfect score for a language
nobody tested.

This module reconciles the two sides. It joins the declared cue packs against
the fixture rows and fails closed with ``registered_without_fixtures`` whenever
a pack has no fixtures behind it.

The declared language set is derived by scanning the lexicon module for
``ClinicalCueLexicon`` constants rather than by reading the mutable runtime
registry. The registry is process-global and can be extended at runtime by any
caller, which would make the gate's authority depend on import and test
ordering. Runtime-only registrations are reported as advisory
``unscored_languages`` instead.

Scope boundary: this module reports whether a cue pack is exercised by
fixtures, not how well its cues perform. A pack can be fully covered here
and still be linguistically wrong. A language with no pack of its own is
absent from the summary and resolves through the documented English
fallback; that absence is deliberate, not a coverage gap.

Reports are offline and raw-text-free: they carry language codes, counts,
trap names, scores, and verdicts, never fixture text.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from openmed.eval.harness import (
    DEFAULT_CONTEXT_MULTILINGUAL_FIXTURE,
    load_context_multilingual_fixtures,
    run_context_multilingual_eval,
)

CONTEXT_COVERAGE = "context_coverage"
CONTEXT_COVERAGE_SCHEMA_VERSION = 1

#: Assertion axes scored by the multilingual ConText eval.
SCORED_AXES: tuple[str, ...] = ("negation", "temporality", "uncertainty")

#: Cue-count axes reported by ``clinical_context_lexicon_stats``.
CUE_AXES: tuple[str, ...] = (
    "negation",
    "pseudo_negation",
    "historical",
    "hypothetical",
    "recent",
    "uncertainty",
    "backward",
    "scope_terminators",
    "conjunction_terminators",
)

#: Fixture traps every declared language must exercise.
REQUIRED_TRAPS: tuple[str, ...] = (
    "affirmed",
    "double_negation",
    "historical",
    "hypothetical",
    "negated",
    "pseudo_negation",
)

#: Cue axes that may never be empty for a declared language pack.
REQUIRED_CUE_AXES: tuple[str, ...] = ("negation", "uncertainty")

#: Language a code without its own cue pack resolves through.
FALLBACK_LANGUAGE = "en"

#: Headline failure: a declared cue pack that no fixture row exercises.
REASON_REGISTERED_WITHOUT_FIXTURES = "registered_without_fixtures"
REASON_FIXTURE_LANGUAGE_NOT_REGISTERED = "fixture_language_not_registered"
REASON_MISSING_REQUIRED_TRAP = "missing_required_trap"
REASON_EMPTY_AXIS_CUE_PACK = "empty_axis_cue_pack"
REASON_MACRO_F1_BELOW_THRESHOLD = "macro_f1_below_threshold"


def builtin_clinical_cue_languages() -> tuple[str, ...]:
    """Return the language codes declared as cue-pack constants in the lexicon.

    The codes come from the ``ClinicalCueLexicon`` constants defined in
    ``openmed.clinical.lexicons.context_cues``, not from the runtime registry,
    so a pack registered at runtime by a caller or a test cannot change what
    the coverage gate expects to find fixtures for.
    """

    from openmed.clinical.lexicons import context_cues

    return tuple(
        sorted(
            {
                value.language
                for value in vars(context_cues).values()
                if isinstance(value, context_cues.ClinicalCueLexicon)
            }
        )
    )


@dataclass(frozen=True)
class ContextLanguageCoverage:
    """Cue-pack and fixture coverage for one ConText language."""

    language: str
    declared: bool
    registered: bool
    fixture_case_count: int
    traps: tuple[str, ...]
    missing_traps: tuple[str, ...]
    cue_counts: Mapping[str, int]
    total_cues: int
    macro_f1: Mapping[str, float | None]
    meets_thresholds: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, raw-text-free mapping for this language."""

        return {
            "language": self.language,
            "declared": self.declared,
            "registered": self.registered,
            "fixture_case_count": self.fixture_case_count,
            "traps": list(self.traps),
            "missing_traps": list(self.missing_traps),
            "cue_counts": {axis: int(self.cue_counts[axis]) for axis in CUE_AXES},
            "total_cues": self.total_cues,
            "macro_f1": {axis: self.macro_f1[axis] for axis in SCORED_AXES},
            "meets_thresholds": self.meets_thresholds,
        }


@dataclass(frozen=True)
class ContextCoverageGateFailure:
    """Raw-text-free failure emitted by the ConText coverage gate."""

    language: str
    reason: str
    detail: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic mapping for this failure."""

        return {
            "language": self.language,
            "reason": self.reason,
            "detail": dict(self.detail),
        }


@dataclass(frozen=True)
class ContextCoverageReport:
    """Deterministic multilingual ConText coverage summary and gate verdict."""

    eval_suite: str
    fixture_count: int
    languages: tuple[str, ...]
    declared_languages: tuple[str, ...]
    thresholds: Mapping[str, float]
    per_language: tuple[ContextLanguageCoverage, ...]
    unscored_languages: tuple[str, ...]
    passed: bool
    failures: tuple[ContextCoverageGateFailure, ...] = field(default_factory=tuple)
    schema_version: int = CONTEXT_COVERAGE_SCHEMA_VERSION

    def coverage_for(self, language: str) -> ContextLanguageCoverage | None:
        """Return the per-language coverage row for *language* if present."""

        for coverage in self.per_language:
            if coverage.language == language:
                return coverage
        return None

    def failure_reasons(self) -> tuple[str, ...]:
        """Return the sorted, de-duplicated gate failure reasons."""

        return tuple(sorted({failure.reason for failure in self.failures}))

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, raw-text-free report mapping."""

        return {
            "suite": CONTEXT_COVERAGE,
            "schema_version": self.schema_version,
            "eval_suite": self.eval_suite,
            "fixture_count": self.fixture_count,
            "languages": list(self.languages),
            "declared_languages": list(self.declared_languages),
            "thresholds": {
                axis: float(self.thresholds[axis]) for axis in sorted(self.thresholds)
            },
            "per_language": [coverage.to_dict() for coverage in self.per_language],
            "unscored_languages": list(self.unscored_languages),
            "passed": self.passed,
            "failures": [failure.to_dict() for failure in self.failures],
        }

    def to_markdown(self) -> str:
        """Render a byte-stable Markdown coverage summary."""

        lines = [
            "# Multilingual ConText Coverage",
            "",
            "| Field | Value |",
            "| --- | --- |",
            f"| Suite | `{CONTEXT_COVERAGE}` |",
            f"| Eval Suite | `{self.eval_suite}` |",
            f"| Fixture Cases | `{self.fixture_count}` |",
            f"| Declared Languages | `{_join(self.declared_languages)}` |",
            f"| Verdict | `{'pass' if self.passed else 'fail'}` |",
            f"| Schema | `{self.schema_version}` |",
            "",
            "## Thresholds",
            "",
            "| Axis | Macro-F1 |",
            "| --- | ---: |",
        ]
        for axis in sorted(self.thresholds):
            lines.append(f"| `{axis}` | {_format_score(self.thresholds[axis])} |")

        lines.extend(
            [
                "",
                "## Coverage by Language",
                "",
                "| Language | Cases | Cues | Negation | Temporality | "
                "Uncertainty | Traps | Status |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for coverage in self.per_language:
            scores = " | ".join(
                _format_score(coverage.macro_f1[axis]) for axis in SCORED_AXES
            )
            # Count required traps only: a language may exercise extra trap
            # kinds, and a "7/6" cell would read as a reporting bug.
            covered_traps = len(REQUIRED_TRAPS) - len(coverage.missing_traps)
            traps = f"{covered_traps}/{len(REQUIRED_TRAPS)}"
            status = "ok" if coverage.meets_thresholds else "gap"
            lines.append(
                f"| `{coverage.language}` | {coverage.fixture_case_count} | "
                f"{coverage.total_cues} | {scores} | {traps} | {status} |"
            )

        lines.extend(
            [
                "",
                f"A language without a declared cue pack is absent from this table "
                f"and resolves through the documented `{FALLBACK_LANGUAGE}` "
                f"fallback. Absence is a deliberate fallback, not a coverage gap. "
                f"This table reports whether a pack is exercised by fixtures, not "
                f"how well its cues perform.",
            ]
        )
        lines.extend(["", "## Unscored Registered Languages", ""])
        lines.extend(_bullets(f"`{code}`" for code in self.unscored_languages))
        lines.extend(["", "## Failures", ""])
        lines.extend(
            _bullets(
                f"`{failure.language}`: {failure.reason}" for failure in self.failures
            )
        )
        return "\n".join(lines) + "\n"

    def write_markdown(self, path: str | Path) -> Path:
        """Write the byte-stable Markdown coverage summary to *path*."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown(), encoding="utf-8")
        return output_path


def run_context_coverage(
    path: str | Path = DEFAULT_CONTEXT_MULTILINGUAL_FIXTURE,
    *,
    languages: Sequence[str] | None = None,
) -> ContextCoverageReport:
    """Summarise multilingual ConText cue and fixture coverage.

    Args:
        path: Optional fixture path. Defaults to the committed synthetic
            multilingual ConText fixture.
        languages: Optional override of the declared language set. Defaults to
            the cue-pack constants declared in the lexicon module. Callers may
            inject a set to exercise the gate.

    Returns:
        A deterministic coverage report joining declared cue packs against
        fixture rows, plus the gate verdict and raw-text-free failures.
    """

    from openmed.clinical.lexicons import available_clinical_cue_languages

    eval_report = run_context_multilingual_eval(path)
    _, fixtures = load_context_multilingual_fixtures(path)

    declared = (
        tuple(sorted(set(languages)))
        if languages is not None
        else builtin_clinical_cue_languages()
    )
    cue_stats = eval_report.metrics["context_lexicon_coverage"]
    macro_scores = eval_report.metrics["context_macro_f1"]
    thresholds = {
        axis: float(value)
        for axis, value in eval_report.metrics["context_thresholds"].items()
    }

    case_counts: Counter[str] = Counter()
    traps_by_language: dict[str, set[str]] = {}
    for row in fixtures:
        language = str(row.get("language") or "en")
        case_counts[language] += 1
        trap = row.get("trap")
        if trap:
            traps_by_language.setdefault(language, set()).add(str(trap))

    all_languages = tuple(sorted(set(declared) | set(case_counts)))
    per_language: list[ContextLanguageCoverage] = []
    failures: list[ContextCoverageGateFailure] = []

    for language in all_languages:
        case_count = case_counts.get(language, 0)
        registered = language in cue_stats
        cue_counts = (
            {axis: int(cue_stats[language].get(axis, 0)) for axis in CUE_AXES}
            if registered
            else dict.fromkeys(CUE_AXES, 0)
        )
        traps = tuple(sorted(traps_by_language.get(language, set())))
        missing_traps = tuple(trap for trap in REQUIRED_TRAPS if trap not in traps)

        # A language without fixture rows is never scored, even if the eval
        # happens to carry a key for it: an empty label sequence macro-averages
        # to a perfect 1.0 and would advertise coverage nobody verified.
        scored = macro_scores.get(language) if case_count else None
        macro_f1: dict[str, float | None] = {
            axis: (float(scored[axis]) if scored and axis in scored else None)
            for axis in SCORED_AXES
        }
        meets_thresholds = case_count > 0 and all(
            macro_f1[axis] is not None and macro_f1[axis] >= thresholds[axis]
            for axis in SCORED_AXES
            if axis in thresholds
        )

        per_language.append(
            ContextLanguageCoverage(
                language=language,
                declared=language in declared,
                registered=registered,
                fixture_case_count=case_count,
                traps=traps,
                missing_traps=missing_traps,
                cue_counts=cue_counts,
                total_cues=sum(cue_counts.values()),
                macro_f1=macro_f1,
                meets_thresholds=meets_thresholds,
            )
        )

        if language in declared and case_count == 0:
            failures.append(
                ContextCoverageGateFailure(
                    language=language,
                    reason=REASON_REGISTERED_WITHOUT_FIXTURES,
                    detail={
                        "fixture_case_count": 0,
                        "registered": registered,
                        "total_cues": sum(cue_counts.values()),
                        "required_traps": list(REQUIRED_TRAPS),
                    },
                )
            )
            continue

        if case_count and not registered:
            failures.append(
                ContextCoverageGateFailure(
                    language=language,
                    reason=REASON_FIXTURE_LANGUAGE_NOT_REGISTERED,
                    detail={"fixture_case_count": case_count},
                )
            )
        if case_count and missing_traps:
            failures.append(
                ContextCoverageGateFailure(
                    language=language,
                    reason=REASON_MISSING_REQUIRED_TRAP,
                    detail={"missing_traps": list(missing_traps)},
                )
            )
        if registered:
            empty_axes = [axis for axis in REQUIRED_CUE_AXES if not cue_counts[axis]]
            if empty_axes:
                failures.append(
                    ContextCoverageGateFailure(
                        language=language,
                        reason=REASON_EMPTY_AXIS_CUE_PACK,
                        detail={"empty_axes": empty_axes},
                    )
                )
        if case_count and not meets_thresholds:
            failures.append(
                ContextCoverageGateFailure(
                    language=language,
                    reason=REASON_MACRO_F1_BELOW_THRESHOLD,
                    detail={
                        "macro_f1": {
                            axis: macro_f1[axis]
                            for axis in SCORED_AXES
                            if axis in thresholds
                        },
                        "thresholds": {
                            axis: thresholds[axis]
                            for axis in SCORED_AXES
                            if axis in thresholds
                        },
                    },
                )
            )

    unscored = tuple(
        sorted(
            code
            for code in available_clinical_cue_languages()
            if not case_counts.get(code)
        )
    )

    return ContextCoverageReport(
        eval_suite=eval_report.suite,
        fixture_count=eval_report.fixture_count,
        languages=all_languages,
        declared_languages=declared,
        thresholds=thresholds,
        per_language=tuple(per_language),
        unscored_languages=unscored,
        passed=not failures,
        failures=tuple(failures),
    )


def assert_context_coverage_gate(
    path: str | Path = DEFAULT_CONTEXT_MULTILINGUAL_FIXTURE,
    *,
    languages: Sequence[str] | None = None,
) -> ContextCoverageReport:
    """Return a passing coverage report or raise with raw-text-free diagnostics."""

    report = run_context_coverage(path, languages=languages)
    if not report.passed:
        offenders = ", ".join(
            sorted(
                {f"{failure.language}:{failure.reason}" for failure in report.failures}
            )
        )
        raise AssertionError(f"multilingual ConText coverage gate failed: {offenders}")
    return report


def context_coverage_metadata() -> dict[str, Any]:
    """Return raw-text-free metadata describing the ConText coverage summary."""

    return {
        "suite": CONTEXT_COVERAGE,
        "schema_version": CONTEXT_COVERAGE_SCHEMA_VERSION,
        "eval_suite": "context_multilingual",
        "synthetic": True,
        "scored_axes": list(SCORED_AXES),
        "cue_axes": list(CUE_AXES),
        "required_traps": list(REQUIRED_TRAPS),
        "required_cue_axes": list(REQUIRED_CUE_AXES),
        "fallback_language": FALLBACK_LANGUAGE,
    }


def _join(values: Sequence[str]) -> str:
    return ", ".join(values) if values else "none"


def _bullets(values: Any) -> list[str]:
    rendered = [f"- {value}" for value in values]
    return rendered or ["- none"]


def _format_score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


__all__ = [
    "CONTEXT_COVERAGE",
    "CONTEXT_COVERAGE_SCHEMA_VERSION",
    "CUE_AXES",
    "FALLBACK_LANGUAGE",
    "REASON_EMPTY_AXIS_CUE_PACK",
    "REASON_FIXTURE_LANGUAGE_NOT_REGISTERED",
    "REASON_MACRO_F1_BELOW_THRESHOLD",
    "REASON_MISSING_REQUIRED_TRAP",
    "REASON_REGISTERED_WITHOUT_FIXTURES",
    "REQUIRED_CUE_AXES",
    "REQUIRED_TRAPS",
    "SCORED_AXES",
    "ContextCoverageGateFailure",
    "ContextCoverageReport",
    "ContextLanguageCoverage",
    "assert_context_coverage_gate",
    "builtin_clinical_cue_languages",
    "context_coverage_metadata",
    "run_context_coverage",
]
